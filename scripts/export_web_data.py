from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    df=pd.read_csv(ROOT/'results/test_predictions.csv')
    (ROOT/'site/dist/test_scores.json').write_text(json.dumps(df[['y','A','B','LR']].to_dict('records'),separators=(',',':')))
    # Identidad y despliegue del sitio existente.
    path=ROOT/'site/.openai/hosting.json';manifest=json.loads(path.read_text())
    manifest['static']={'directory':'dist'};path.write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
