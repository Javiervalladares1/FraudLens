"""Ejecutar desde la raíz: python -m scripts.train. Todas las métricas se calculan."""
import copy, hashlib, json, math, platform, random, subprocess, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,log_loss
from sklearn.preprocessing import StandardScaler
from src.model import FraudMLP,feature_array,export_network,FEATURES
from src.value import Assumptions,business,outcomes,select_threshold

ROOT=Path(__file__).resolve().parents[1]
SEED=23045
def save(path,obj):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def metrics(y,p,t):
    c=outcomes(y,p,t);tp,fp,fn,tn=[c[k] for k in ['tp','fp','fn','tn']]
    return dict(**c,threshold=float(t),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
                f1=2*tp/max(2*tp+fp+fn,1),accuracy=(tp+tn)/len(y),
                ap=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)),
                nll=float(log_loss(y,p,labels=[0,1])),brier=float(np.mean((p-y)**2)),
                business=business(c))

def main():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    path=ROOT/'data/raw/creditcard.csv';raw=pd.read_csv(path)
    assert list(raw.columns)==['Time']+[f'V{i}' for i in range(1,29)]+['Amount','Class']
    assert raw.isna().sum().sum()==0 and set(raw.Class.unique())=={0,1}
    raw['source_row']=np.arange(len(raw))
    # Duplicados de todas las entradas, incluyendo tiempo. Se conserva la primera aparición.
    cols=['Time']+[f'V{i}' for i in range(1,29)]+['Amount']
    assert raw.groupby(cols,dropna=False)['Class'].nunique().max()==1
    df=raw.drop_duplicates(cols).sort_values(['Time','source_row'],kind='stable').reset_index(drop=True)
    # Cortes temporales sin separar transacciones con el mismo segundo.
    bounds=[float(df.Time.quantile(q)) for q in [.6,.7,.8]]
    groups=np.searchsorted(bounds,df.Time.to_numpy(),side='right')
    names=['train','calibration','validation','test']
    splits={name:np.flatnonzero(groups==i) for i,name in enumerate(names)}
    x=feature_array(df);y=df.Class.to_numpy(dtype=np.float32)
    scaler=StandardScaler().fit(x[splits['train']]);xs=scaler.transform(x).astype(np.float32)
    X={n:torch.from_numpy(xs[idx]) for n,idx in splits.items()}
    Y={n:torch.from_numpy(y[idx]) for n,idx in splits.items()}
    joblib.dump(scaler,ROOT/'models/scaler.joblib')
    eda=dict(raw_rows=len(raw),clean_rows=len(df),removed_duplicates=len(raw)-len(df),
             raw_frauds=int(raw.Class.sum()),clean_frauds=int(y.sum()),nulls=0,
             amount=df.Amount.describe(percentiles=[.01,.5,.95,.99]).to_dict(),
             split={n:dict(rows=len(idx),frauds=int(y[idx].sum()),prevalence=float(y[idx].mean()),
                           time_min=float(df.iloc[idx].Time.min()),time_max=float(df.iloc[idx].Time.max()))
                    for n,idx in splits.items()},
             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),features=FEATURES)
    save('results/eda.json',eda)
    pd.DataFrame({'source_row':df.source_row,'split':[names[g] for g in groups]}).to_csv(ROOT/'data/processed/split_manifest.csv',index=False)
    print('DATOS',json.dumps(eda['split']),flush=True)
    summary={};histories={};exports={};predictions={};a=Assumptions()
    configs=[dict(name='A',hidden=[32,16],dropout=0.0,epochs=12,lr=.001,weight=1.0),
             dict(name='B',hidden=[64,32,16],dropout=.15,epochs=20,lr=.001,weight=float(np.sqrt((Y['train']==0).sum()/(Y['train']==1).sum())))]
    for cfg in configs:
        torch.manual_seed(SEED);model=FraudMLP(cfg['hidden'],cfg['dropout'])
        prior=float(Y['train'].mean()); model.layers[-1].bias.data.fill_(math.log(prior/(1-prior))+math.log(cfg['weight']))
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
        lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg['weight']))
        history=[];best=-1;best_epoch=0;state=None;start=time.perf_counter()
        for epoch in range(1,cfg['epochs']+1):
            model.train();order=torch.randperm(len(X['train']));total=0
            for ix in order.split(1024):
                opt.zero_grad();loss=lossfn(model(X['train'][ix]),Y['train'][ix]);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(ix)
            model.eval()
            with torch.inference_mode():vp=torch.sigmoid(model(X['validation'])).numpy()
            ap=float(average_precision_score(Y['validation'].numpy(),vp))
            nll=float(log_loss(Y['validation'].numpy(),np.clip(vp,1e-7,1-1e-7),labels=[0,1]))
            history.append(dict(epoch=epoch,train_loss=total/len(order),validation_ap=ap,validation_nll=nll))
            if ap>best:best=ap;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
            print(f"{cfg['name']} epoch={epoch} loss={history[-1]['train_loss']:.5f} valAP={ap:.5f}",flush=True)
        elapsed=time.perf_counter()-start;model.load_state_dict(state);model.eval()
        # Platt sobre un bloque independiente, sin ponderar ni usar prueba.
        with torch.inference_mode():cal_logit=model(X['calibration']).numpy()
        calibrator=LogisticRegression(C=100.0,max_iter=1000).fit(cal_logit.reshape(-1,1),y[splits['calibration']])
        probs={}
        for name in names:
            with torch.inference_mode():logit=model(X[name]).numpy()
            probs[name]=calibrator.predict_proba(logit.reshape(-1,1))[:,1]
        t=select_threshold(y[splits['validation']],probs['validation'],a)
        lat=[]
        with torch.inference_mode():
            for _ in range(50):model(X['test'][:1])
            for _ in range(500):
                begin=time.perf_counter();model(X['test'][:1]);lat.append((time.perf_counter()-begin)*1000)
            begin=time.perf_counter();model(X['test']);batch_elapsed=time.perf_counter()-begin
        summary[cfg['name']]=dict(config=cfg,best_epoch=best_epoch,training_seconds=elapsed,
            parameter_count=sum(p.numel() for p in model.parameters()),threshold=t,
            calibration=dict(coef=float(calibrator.coef_[0,0]),intercept=float(calibrator.intercept_[0])),
            latency_ms_p50=float(np.median(lat)),latency_ms_p95=float(np.percentile(lat,95)),
            throughput_per_second=len(X['test'])/batch_elapsed,
            training_cost_usd_estimate=elapsed/3600*.1,
            metrics={n:metrics(y[splits[n]],probs[n],t) for n in names},
            test_at_05=metrics(y[splits['test']],probs['test'],.5))
        torch.save(dict(config=cfg,state_dict=model.state_dict()),ROOT/f'models/model_{cfg["name"]}.pt')
        joblib.dump(calibrator,ROOT/f'models/calibration_{cfg["name"]}.joblib')
        exports[cfg['name']]=dict(layers=export_network(model),calibration=summary[cfg['name']]['calibration'],threshold=t)
        predictions[cfg['name']]=probs;histories[cfg['name']]=history
    # Referencia clásica entrenada solo con el primer bloque temporal.
    start=time.perf_counter();lr=LogisticRegression(C=1,max_iter=2000).fit(xs[splits['train']],y[splits['train']])
    duration=time.perf_counter()-start
    cal=LogisticRegression(C=100,max_iter=1000).fit(lr.decision_function(xs[splits['calibration']]).reshape(-1,1),y[splits['calibration']])
    bp={n:cal.predict_proba(lr.decision_function(xs[idx]).reshape(-1,1))[:,1] for n,idx in splits.items()}
    t=select_threshold(y[splits['validation']],bp['validation'])
    summary['LR']=dict(training_seconds=duration,threshold=t,metrics={n:metrics(y[splits[n]],bp[n],t) for n in names})
    joblib.dump(dict(model=lr,calibration=cal),ROOT/'models/logistic.joblib');predictions['LR']=bp
    # Selección entre redes únicamente con validación. Test se usa una vez para informar.
    selected=max(['A','B'],key=lambda n:summary[n]['metrics']['validation']['business']['net_value'])
    testy=y[splits['test']].astype(int);testframe=df.iloc[splits['test']].copy()
    testframe.to_csv(ROOT/'data/processed/test.csv',index=False)
    preds=pd.DataFrame(dict(source_row=testframe.source_row.to_numpy(),y=testy,Amount=testframe.Amount.to_numpy()))
    for name in predictions:preds[name]=predictions[name]['test']
    preds.to_csv(ROOT/'results/test_predictions.csv',index=False)
    # Bootstrap pareado de filas con semilla fija. El intervalo refleja muestreo, no costos ni drift.
    rng=np.random.default_rng(SEED);values={n:[] for n in predictions};aps={n:[] for n in predictions}
    for iteration in range(400):
        ix=rng.integers(0,len(testy),len(testy)); yy=testy[ix]
        for name in predictions:
            pp=predictions[name]['test'][ix]
            values[name].append(business(outcomes(yy,pp,summary[name]['threshold']))['net_value'])
            if iteration<100:aps[name].append(float(average_precision_score(yy,pp)))
    for name in predictions:
        summary[name]['value_ci95']=np.percentile(values[name],[2.5,97.5]).tolist()
        summary[name]['ap_ci95']=np.percentile(aps[name],[2.5,97.5]).tolist()
    difference=np.asarray(values['B'])-np.asarray(values['A'])
    summary['delta_B_A']=dict(value=summary['B']['metrics']['test']['business']['net_value']-summary['A']['metrics']['test']['business']['net_value'],
        ci95=np.percentile(difference,[2.5,97.5]).tolist())
    # Sensibilidad de una variable por vez, política fija. Ningún umbral se optimiza en test.
    sensitivity=[]
    for param in ['monthly_transactions','loss_per_fraud','prevention_rate','false_alarm_cost','review_cost','monthly_hosting']:
        for factor in [.5,.8,1,1.2,1.5]:
            base=Assumptions();value=getattr(base,param)*factor
            if param=='prevention_rate':value=min(value,1)
            setattr(base,param,value)
            sensitivity.append(dict(parameter=param,factor=factor,value=value,
                **{n:business(outcomes(testy,predictions[n]['test'],summary[n]['threshold']),base)['net_value'] for n in predictions}))
    save('results/sensitivity.json',sensitivity)
    # Muestras públicas de prueba: se revela la etiqueta después de clasificar.
    examples=[]
    for label in [0,1]:
        subset=testframe[testframe.Class==label].head(8)
        for _,row in subset.iterrows():
            examples.append(dict(id=f"TX-{int(row.source_row):06d}",features={k:float(row[k]) for k in [f'V{i}' for i in range(1,29)]+['Amount']},label=int(row.Class)))
    # Auditoría por monto sin atribuir el efecto a grupos demográficos ausentes.
    slices=[]
    amounts=testframe.Amount.to_numpy()
    for label,mask in [('monto <= 20',amounts<=20),('20 < monto <= 100',(amounts>20)&(amounts<=100)),('monto > 100',amounts>100)]:
        row=dict(group=label,n=int(mask.sum()),frauds=int(testy[mask].sum()))
        for n in ['A','B']:row[n]=metrics(testy[mask],predictions[n]['test'][mask],summary[n]['threshold'])
        slices.append(row)
    save('results/subgroups.json',slices)
    # Explicación local por oclusión de una variable al promedio de entrenamiento.
    example_ix=np.flatnonzero(testy==1)[0];xx=xs[splits['test']][example_ix]
    chosen=exports[selected]
    def infer_np(z):
        for i,layer in enumerate(chosen['layers']):
            z=np.asarray(layer['weight'])@z+np.asarray(layer['bias'])
            if i<len(chosen['layers'])-1:z=np.maximum(z,0)
        return float(expit(z.item()*chosen['calibration']['coef']+chosen['calibration']['intercept']))
    basep=infer_np(xx);explain=[]
    for i,k in enumerate(FEATURES):
        z=xx.copy();z[i]=0;explain.append(dict(feature=k,delta=basep-infer_np(z)))
    save('results/explanation.json',dict(source_row=int(testframe.iloc[example_ix].source_row),probability=basep,method='occlusion_to_training_mean',features=sorted(explain,key=lambda t:abs(t['delta']),reverse=True)))
    # Casos para probar paridad de la inferencia JS y Python sin depender de PyTorch en el navegador.
    parity=[]
    for ex in examples:
        index=int(np.flatnonzero(testframe.source_row.to_numpy()==int(ex['id'][3:]))[0])
        parity.append(dict(features=ex['features'],expected={n:float(predictions[n]['test'][index]) for n in ['A','B']}))
    artifact=dict(feature_names=FEATURES,scaler=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist()),
        networks=exports,selected=selected,assumptions=a.__dict__,examples=examples,
        feature_ranges=dict(min=np.min(x[splits['train']],axis=0).tolist(),max=np.max(x[splits['train']],axis=0).tolist()))
    save('site/dist/model.json',artifact);save('tests/parity_cases.json',parity)
    save('results/history.json',histories)
    save('results/metrics.json',dict(seed=SEED,selected=selected,assumptions=a.__dict__,models=summary,
        hardware=dict(platform=platform.platform(),processor=platform.processor(),torch_threads=4,device='CPU'),
        bootstrap=dict(value_replicates=400,ap_replicates=100,method='paired_iid_rows',seed=SEED)))
    save('site/dist/metrics.json',json.loads((ROOT/'results/metrics.json').read_text()))
    print('FINAL',json.dumps({n:dict(AP=summary[n]['metrics']['test']['ap'],value=summary[n]['metrics']['test']['business']['net_value'],threshold=summary[n]['threshold']) for n in ['A','B','LR']}),flush=True)
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
