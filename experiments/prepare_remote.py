"""Prepare immutable inline bootstrap; does not submit any cloud operation."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parent
TASK='acp-me2502-chemgpt-20261113-01'
old='/zzc/260010120/cci_control/acp-tests/acp-explore-chemgpt-profile-20261104-05'
remote='/zzc/260010120/cci_control/acp-tests/'+TASK
files={str(p.relative_to(ROOT)):base64.b64encode(p.read_bytes()).decode() for p in [ROOT/'contract.json',*sorted((ROOT/'src').glob('*.py'))]}
models=ROOT.parent/'proposal/implementation/chemgpt'
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in models.iterdir() if p.is_file()}
bootstrap='''import base64,hashlib,json,pathlib,shutil,subprocess,sys,os,time,zipfile,zlib
ROOT=pathlib.Path(REMOTE)
if subprocess.check_output(['findmnt','-n','--mountpoint','/zzc','--output','TARGET'],text=True).strip()!='/zzc':raise RuntimeError('AFS unavailable')
ROOT.mkdir(exist_ok=False,parents=True)
for rel,b64 in FILES.items():
 p=ROOT/rel;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(base64.b64decode(b64))
source=pathlib.Path(OLD)
model=ROOT/'model';model.mkdir()
for name,h in MANIFEST.items():
 p=source/'model'/name
 if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError('prior AFS input mismatch '+name)
 shutil.copyfile(p,model/name)
wheel=source/'selfies-1.0.4-py3-none-any.whl'
if hashlib.sha256(wheel.read_bytes()).hexdigest()!='e41009c1b5512d87ed8ea4d2a52ea9d833fbe6ddeb9ea34990b4a4d77b5b037c':raise RuntimeError('SELFIES mismatch')
zipfile.ZipFile(wheel).extractall(ROOT/'vendor')
py='/root/anaconda3/envs/ddc_encoding/bin/python'
env=dict(os.environ,PYTHONPATH=str(ROOT/'vendor'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
print('ME2502_INPUTS_VERIFIED '+json.dumps({'task':TASK,'files':len(FILES),'model_files':len(MANIFEST),'afs':str(ROOT)}),flush=True)
subprocess.run([py,str(ROOT/'src/check_theory.py')],env=env,check=True,timeout=60)
res=subprocess.run([py,'-u',str(ROOT/'src/study.py'),'--model-dir',str(model),'--output',str(ROOT/'results'),'--contract',str(ROOT/'contract.json'),'--device','cuda','--phase','all'],env=env,timeout=13000)
cp=ROOT/'results/campaign.json'
if cp.exists():
 raw=cp.read_bytes();print('ME2502_CAMPAIGN_B64 '+base64.b64encode(zlib.compress(raw,9)).decode(),flush=True)
print('ME2502_FINISHED '+json.dumps({'returncode':res.returncode,'task':TASK}),flush=True)
raise SystemExit(res.returncode)
'''
prefix='\n'.join(f'{k}={v!r}' for k,v in dict(REMOTE=remote,OLD=old,TASK=TASK,FILES=files,MANIFEST=manifest).items())
(ROOT/'remote_bootstrap.py').write_text(prefix+'\n'+bootstrap)
meta=dict(task=TASK,remote=remote,files={k:hashlib.sha256(base64.b64decode(v)).hexdigest() for k,v in files.items()},model_files=manifest,bootstrap_sha256=hashlib.sha256((ROOT/'remote_bootstrap.py').read_bytes()).hexdigest())
(ROOT/'evidence/input_manifest.json').write_text(json.dumps(meta,indent=2))
print(json.dumps({'task':TASK,'bootstrap_bytes':(ROOT/'remote_bootstrap.py').stat().st_size,'remote':remote}))
