"""Empaqueta código y entregables, sin datos completos, credenciales ni entorno virtual."""
from pathlib import Path
import json,shutil,subprocess,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    dest=ROOT/'output/FraudLens_Proyecto';dest.mkdir(exist_ok=True)
    ignored=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store','node_modules','.git')
    for name in ['src','scripts','tests','models','docs','results']:
        shutil.copytree(ROOT/name,dest/name,dirs_exist_ok=True,ignore=ignored)
    for name in ['README.md','requirements.txt','.gitignore']:
        shutil.copy2(ROOT/name,dest/name)
    shutil.copytree(ROOT/'site/dist',dest/'site/dist',dirs_exist_ok=True)
    (dest/'data').mkdir(exist_ok=True)
    for name in ['README.md','sample_inputs.csv']:shutil.copy2(ROOT/'data'/name,dest/'data'/name)
    for name in ['pdf','presentacion','video']:
        shutil.copytree(ROOT/'output'/name,dest/'output'/name,dirs_exist_ok=True)
    meta=dict(project='FraudLens',authors=[dict(name='Javier Valladares',student_id='23045'),dict(name='Ian Cumes',student_id='23236')],
              demo='https://fraudlens-javier-ian.javiervalladares.chatgpt.site',demo_access='owner_private',
              source_commit='2393a356879f9cd3a3494c310fc555c768155024',github_public_url=None,
              disclaimer='Los costos y el valor monetario son estimaciones de escenario. Las métricas se calcularon en el benchmark.')
    (dest/'entrega.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    subprocess.run(['git','init','-b','main'],cwd=dest,check=True,capture_output=True)
    subprocess.run(['git','add','.'],cwd=dest,check=True,capture_output=True)
    status=subprocess.run(['git','status','--porcelain'],cwd=dest,check=True,capture_output=True,text=True)
    if status.stdout:
        subprocess.run(['git','commit','-m','Proyecto FraudLens: modelos, evaluación, demo y entregables'],cwd=dest,check=True,capture_output=True)
    archive=ROOT/'output/FraudLens_Proyecto_Completo.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(dest.rglob('*')):
            if not file.is_file() or '.git' in file.relative_to(dest).parts:continue
            z.write(file,Path('FraudLens_Proyecto')/file.relative_to(dest))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert any(n.endswith('FraudLens_Demostracion.mp4') for n in z.namelist())
        assert not any('/.venv/' in n or '/data/raw/' in n or '/.git/' in n for n in z.namelist())
        count=len(z.namelist())
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=count,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),local_repository=str(dest)),indent=2))
if __name__=='__main__':main()
