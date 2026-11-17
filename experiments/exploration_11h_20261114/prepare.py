"""Freeze extension source and an offline bootstrap; no cloud operation."""
import hashlib,json,zlib,base64
from pathlib import Path
ROOT=Path(__file__).resolve().parent
c=json.loads((ROOT/'contract.json').read_text());task=c['task_id'];remote='/zzc/260010120/cci_control/acp-tests/'+task
old='/zzc/260010120/cci_control/acp-tests/acp-me2502-chemgpt-20261113-01'
files={str(p.relative_to(ROOT)):p.read_text() for p in [ROOT/'contract.json',*sorted((ROOT/'src').glob('*.py'))]}
model=ROOT.parent.parent/'proposal/implementation/chemgpt'
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model.iterdir() if p.is_file()}
old_manifest=json.loads((ROOT.parent/'evidence/input_manifest.json').read_text())['files']
code='''import base64,hashlib,io,json,os,pathlib,shutil,subprocess,sys,tarfile,time,zlib,zipfile
from datetime import datetime,timezone
r=pathlib.Path(REMOTE);old=pathlib.Path(OLD)
if subprocess.check_output(['findmnt','-n','--mountpoint','/zzc','--output','TARGET'],text=True).strip()!='/zzc':raise RuntimeError('AFS unavailable')
r.mkdir(parents=True,exist_ok=False)
for rel,text in FILES.items():
 p=r/rel;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(text)
for rel,h in OLD_SOURCE_HASHES.items():
 if hashlib.sha256((old/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('prior source hash mismatch '+rel)
assert len(list((old/'results').rglob('result.json')))==24,'prior raw evidence incomplete'
def archive(root,dest):
 paths=[root/'contract.json']+list((root/'src').glob('*.py'))
 for p in (root/'results').rglob('*.json'):paths.append(p)
 for p in (root/'results').rglob('*.jsonl'):paths.append(p)
 with tarfile.open(dest,'w:gz') as t:
  for p in sorted(set(paths)):
   if p.is_file():t.add(p,arcname=str(p.relative_to(root)),recursive=False)
 return {'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
meta=archive(old,r/'prior_full_evidence.tar.gz')
print('EXT_PRIOR_EXPORT '+json.dumps(meta),flush=True)
m=r/'model';m.mkdir()
for name,h in MODEL_HASHES.items():
 p=old/'model'/name
 if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError('model input mismatch '+name)
 shutil.copyfile(p,m/name)
w=pathlib.Path('/zzc/260010120/cci_control/acp-tests/acp-explore-chemgpt-profile-20261104-05/selfies-1.0.4-py3-none-any.whl')
if hashlib.sha256(w.read_bytes()).hexdigest()!='e41009c1b5512d87ed8ea4d2a52ea9d833fbe6ddeb9ea34990b4a4d77b5b037c':raise RuntimeError('SELFIES input mismatch')
zipfile.ZipFile(w).extractall(r/'vendor')
py='/root/anaconda3/envs/ddc_encoding/bin/python';env=dict(os.environ,PYTHONPATH=str(r/'vendor'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
print('EXT_INPUTS_VERIFIED '+json.dumps({'task':TASK,'source_files':len(FILES)}),flush=True)
subprocess.run([py,str(r/'src/check_theory.py')],env=env,check=True,timeout=60)
remaining=int((datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds())-180
if remaining<300:raise RuntimeError('authorized deadline too close')
proc=subprocess.run([py,'-u',str(r/'src/campaign.py'),'--root',str(r/'results'),'--model-dir',str(m),'--contract',str(r/'contract.json')],env=env,timeout=remaining)
meta=archive(r,r/'extension_full_evidence.tar.gz');print('EXT_EXPORT_READY '+json.dumps(meta),flush=True)
cp=r/'results/campaign.json'
if cp.exists():print('EXT_CAMPAIGN_B64 '+base64.b64encode(zlib.compress(cp.read_bytes(),9)).decode(),flush=True)
print('EXT_HOLD_FOR_COLLECTION',flush=True)
until=min(time.monotonic()+120,time.monotonic()+max(0,(datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds()-10))
while time.monotonic()<until and not (r/'collection_ack.json').exists():time.sleep(1)
print('EXT_EXIT '+str(proc.returncode),flush=True)
raise SystemExit(proc.returncode)
'''
prefix='\n'.join(f'{k}={v!r}' for k,v in dict(REMOTE=remote,OLD=old,TASK=task,FILES=files,MODEL_HASHES=hashes,OLD_SOURCE_HASHES=old_manifest,DEADLINE=c['absolute_deadline']).items())
(ROOT/'bootstrap.py').write_text(prefix+'\n'+code)
manifest={'task':task,'remote':remote,'source_files':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()},'bootstrap_sha256':hashlib.sha256((ROOT/'bootstrap.py').read_bytes()).hexdigest(),'absolute_deadline':c['absolute_deadline'],'model_hashes':hashes}
(ROOT/'evidence/input_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'task':task,'planned_runs':108,'bootstrap_bytes':(ROOT/'bootstrap.py').stat().st_size,'compressed_command_chars':len(base64.b64encode(zlib.compress((ROOT/'bootstrap.py').read_bytes(),9)))}))
