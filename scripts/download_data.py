"""Descarga la copia usada por el tutorial oficial de TensorFlow."""
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
URL='https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
if __name__=='__main__':
    target=ROOT/'data/raw/creditcard.csv'; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): urllib.request.urlretrieve(URL,target)
    print(target)
