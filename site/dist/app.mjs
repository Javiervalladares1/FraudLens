import {predict,explain,summarize,transform} from './inference.mjs';
const $=id=>document.getElementById(id);
const money=x=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(x);
const pct=x=>`${(100*x).toFixed(1)}%`;
let artifact,metrics,rows,currentExample=null,threshold;
function safeAssumptions(){
  const base={...artifact.assumptions};
  const defs=[['volume','monthly_transactions',1,1e8],['loss','loss_per_fraud',0,1e6],['false-cost','false_alarm_cost',0,1e5],['efficacy','prevention_rate',0,100]];
  for(const [id,key,min,max] of defs){
    const input=$(id),value=Number(input.value);
    if(input.value===''||!Number.isFinite(value)||value<min||value>max){input.setCustomValidity(`Ingresa un valor entre ${min} y ${max}.`);input.reportValidity();return null;}
    input.setCustomValidity('');base[key]=key==='prevention_rate'?value/100:value;
  }
  return base;
}
function chooseExample(){
  currentExample=artifact.examples.find(x=>x.id===$('sample').value);
  $('transaction').value=JSON.stringify(currentExample.features,null,2);
  $('truth').hidden=true;$('input-error').textContent='';
  $('result').hidden=true;$('empty-result').hidden=false;
}
function analyze(){
  try{
    const row=JSON.parse($('transaction').value),version=$('version').value;
    const begin=performance.now(),p=predict(row,artifact,version),elapsed=performance.now()-begin;
    const net=artifact.networks[version],flag=p>=net.threshold;
    $('input-error').textContent='';$('result').hidden=false;$('empty-result').hidden=true;
    $('probability').textContent=pct(p);$('meter-fill').style.width=`${100*p}%`;
    $('meter-threshold').style.left=`${100*net.threshold}%`;
    $('decision').textContent=flag?'Enviar a revisión':'No generar alerta';
    $('decision').classList.toggle('alert',flag);
    $('decision-note').textContent=flag?'El puntaje supera el umbral económico elegido en validación. Un analista debe confirmar el caso.':'El puntaje está debajo del umbral. Esta decisión puede omitir fraude y no constituye una garantía.';
    $('decision-threshold').textContent=net.threshold.toFixed(5);
    $('inference-time').textContent=`${elapsed.toFixed(2)} ms`;
    const standardized=transform(row,artifact),raw=standardized.map((z,i)=>z*artifact.scaler.scale[i]+artifact.scaler.mean[i]);
    const outside=raw.filter((v,i)=>v<artifact.feature_ranges.min[i]||v>artifact.feature_ranges.max[i]).length;
    $('range-note').textContent=outside?`${outside} variable(s) fuera del rango de entrenamiento. Conviene revisar la entrada y su procedencia.`:'';
    const importance=explain(row,artifact,version).slice(0,4),max=Math.max(...importance.map(x=>Math.abs(x.delta)),1e-8);
    $('explanation').replaceChildren(...importance.map(x=>{
      const div=document.createElement('div');div.className='explain-row';
      const name=document.createElement('span');name.textContent=x.feature;name.style.minWidth='45px';
      const bar=document.createElement('span');bar.className='bar';const i=document.createElement('i');i.style.width=`${100*Math.abs(x.delta)/max}%`;bar.append(i);
      const num=document.createElement('b');num.textContent=`${x.delta>=0?'+':''}${(100*x.delta).toFixed(2)} pp`;div.append(name,bar,num);return div;
    }));
    $('truth').hidden=true;
    const matches=currentExample&&Object.keys(currentExample.features).every(k=>row[k]===currentExample.features[k]);
    $('reveal').disabled=!matches;
    $('reveal').textContent=matches?'Revelar etiqueta de referencia':'Entrada editada: etiqueta desconocida';
    $('reveal').onclick=()=>{$('truth').textContent=`Etiqueta del dataset: ${currentExample.label?'fraude':'legítima'}. Se consulta después de predecir.`;$('truth').hidden=false;};
  }catch(error){$('input-error').textContent=error.message;$('result').hidden=true;$('empty-result').hidden=false;}
}
function drawChart(a){
  const W=560,H=245,left=54,right=15,top=15,bottom=35;
  // Muestreo más denso cerca de los umbrales bajos usados por las políticas.
  const ts=[0,...Array.from({length:80},(_,i)=>10**(-5+5*i/79)),threshold];ts.sort((a,b)=>a-b);
  const points={A:[],B:[]};
  for(const name of ['A','B'])for(const t of ts)points[name].push([t,summarize(rows,name,t,a).value]);
  const values=Object.values(points).flat().map(x=>x[1]),ymin=Math.min(0,...values),ymax=Math.max(1,...values);
  const yy=v=>top+(ymax-v)/(ymax-ymin)*(H-top-bottom);
  const xx=t=>left+(Math.log10(Math.max(t,1e-5))+5)/5*(W-left-right);
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('role','img');svg.setAttribute('aria-label','Valor mensual por umbral. El eje horizontal usa escala logarítmica.');
  const add=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;svg.append(e);return e;};
  for(let i=0;i<4;i++){
    const value=ymin+(ymax-ymin)*i/3,y=yy(value);
    add('line',{x1:left,y1:y,x2:W-right,y2:y,stroke:'#e5ebe5','stroke-width':1});
    add('text',{x:left-7,y:y+3,'text-anchor':'end',fill:'#7b8c8c','font-size':9},`${Math.round(value/1000)}k`);
  }
  for(const t of [.00001,.001,.01,.1,1])add('text',{x:xx(t),y:H-15,'text-anchor':'middle',fill:'#7b8c8c','font-size':9},t.toString());
  for(const name of ['A','B'])add('polyline',{points:points[name].map(([t,v])=>`${xx(t)},${yy(v)}`).join(' '),fill:'none',stroke:name==='A'?'#318981':'#ba5b35','stroke-width':2});
  add('line',{x1:xx(threshold),y1:top,x2:xx(threshold),y2:H-bottom,stroke:'#172f3a','stroke-dasharray':'4 4','stroke-width':1});
  add('text',{x:left,y:10,fill:'#7b8c8c','font-size':9},'USD / mes');
  $('value-chart').replaceChildren(svg);
}
function updateScenario(){
  const a=safeAssumptions();if(!a)return;
  threshold=Number($('threshold').value);$('threshold-value').textContent=threshold.toFixed(4);
  const stats=summarize(rows,artifact.selected,threshold,a);
  $('recall').textContent=pct(stats.recall);$('recall-detail').textContent=`${stats.tp} de ${stats.tp+stats.fn} fraudes detectados en prueba.`;
  $('net').textContent=money(stats.value);$('fps').textContent=stats.fp.toLocaleString('en-US');
  $('fp-detail').textContent=`${Math.round(stats.alerts)} alertas estimadas / mes.`;
  $('business-details').replaceChildren(...[['Pérdida evitada',stats.avoided],['Falsas alarmas',stats.friction],['Revisión + operación',stats.reviews+stats.operation]].map(([label,val])=>{const e=document.createElement('span'),b=document.createElement('b');b.textContent=money(val);e.append(b,document.createTextNode(label));return e;}));
  const tbody=$('comparison');tbody.replaceChildren();
  for(const name of ['A','B','LR']){
    const m=metrics.models[name],s=summarize(rows,name,m.threshold,a),tr=document.createElement('tr');if(name===artifact.selected)tr.className='selected-row';
    for(const text of [name==='LR'?'LR · Referencia':`${name} · ${name===artifact.selected?'Seleccionado':'MLP ponderado'}`,m.metrics.test.ap.toFixed(3),pct(s.precision),pct(s.recall),s.f1.toFixed(3),m.threshold.toFixed(5),money(s.value)]){const td=document.createElement('td');td.textContent=text;tr.append(td);}tbody.append(tr);
  }
  drawChart(a);
}
async function init(){
  const responses=await Promise.all(['model.json','metrics.json','test_scores.json'].map(x=>fetch(x)));
  for(const response of responses)if(!response.ok)throw new Error('No se pudieron cargar los archivos del experimento.');
  [artifact,metrics,rows]=await Promise.all(responses.map(x=>x.json()));
  $('chosen').textContent=`Versión ${artifact.selected} · MLP base`;
  $('ap').textContent=metrics.models[artifact.selected].metrics.test.ap.toFixed(3);
  $('version').value=artifact.selected;$('sample').replaceChildren();
  for(const ex of artifact.examples){const opt=document.createElement('option');opt.value=ex.id;opt.textContent=`${ex.id} · Amount ${ex.features.Amount.toFixed(2)}`;$('sample').append(opt);}
  for(const id of ['sample','analyze','load-sample'])$(id).disabled=false;
  $('sample').onchange=chooseExample;$('load-sample').onclick=chooseExample;$('analyze').onclick=analyze;$('version').onchange=()=>{$('result').hidden=true;$('empty-result').hidden=false;};
  $('threshold').value=artifact.networks[artifact.selected].threshold;
  for(const id of ['volume','loss','false-cost','efficacy']){
    $(id).addEventListener('input',updateScenario);
    $(id).addEventListener('change',updateScenario);
  }
  $('threshold').addEventListener('input',updateScenario);
  $('reset').onclick=()=>{$('volume').value=artifact.assumptions.monthly_transactions;$('loss').value=artifact.assumptions.loss_per_fraud;$('false-cost').value=artifact.assumptions.false_alarm_cost;$('efficacy').value=artifact.assumptions.prevention_rate*100;$('threshold').value=artifact.networks[artifact.selected].threshold;updateScenario();};
  chooseExample();updateScenario();
}
init().catch(error=>{$('load-error').hidden=false;$('load-error').textContent=error.message;console.error(error);});
