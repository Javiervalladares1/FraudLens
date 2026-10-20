"""Inferencia batch sin modificar archivos de entrada. Requiere V1..V28 y Amount."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from src.model import feature_array
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('--output',required=True);ap.add_argument('--model',choices=['A','B'],default='A');args=ap.parse_args()
    if Path(args.input).resolve()==Path(args.output).resolve():raise ValueError('El resultado debe guardarse en otro archivo.')
    frame=pd.read_csv(args.input);required=[f'V{i}' for i in range(1,29)]+['Amount']
    if not set(required).issubset(frame):raise ValueError('Faltan columnas V1..V28 o Amount.')
    values=frame[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (frame.Amount<0).any():raise ValueError('Las entradas deben ser finitas y Amount no negativo.')
    artifact=json.loads((ROOT/'site/dist/model.json').read_text());net=artifact['networks'][args.model]
    x=(feature_array(frame)-artifact['scaler']['mean'])/artifact['scaler']['scale']
    for i,layer in enumerate(net['layers']):
        x=x@np.asarray(layer['weight']).T+layer['bias']
        if i<len(net['layers'])-1:x=np.maximum(x,0)
    p=expit(x[:,0]*net['calibration']['coef']+net['calibration']['intercept'])
    out=pd.DataFrame({'probability':p,'alert':p>=net['threshold'],'model':args.model,'threshold':net['threshold']})
    out.to_csv(args.output,index=False);print(f'{len(out)} predicciones guardadas en {args.output}')
if __name__=='__main__':main()
