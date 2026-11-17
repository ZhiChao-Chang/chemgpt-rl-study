"""Prepare a task-specific ACP bootstrap; public pinned inputs, no CCI upload.

This file only writes local source/recipe. It does not CREATE cloud resources.
"""
from pathlib import Path
import base64
import hashlib
import json
import zlib

ROOT = Path(__file__).resolve().parent
TASK = 'acp-explore-chemgpt-profile-20261104-02'
profile = (ROOT / 'profile_h100.py').read_bytes()
download = json.loads(Path('/tmp/chemgpt_proposal_20261104/download_manifest.json').read_text())
recipe = {
    'task_id': TASK,
    'remote_directory': '/zzc/260010120/cci_control/acp-tests/' + TASK,
    'profile_source_b64': base64.b64encode(profile).decode(),
    'profile_source_sha256': hashlib.sha256(profile).hexdigest(),
    'public_model_files': download['files'],
    'selfies_wheel': download['selfies_wheel'],
    'max_total_seconds': 1200,
    'max_profile_seconds': 600,
    'packages': ['transformers==4.51.3', 'tokenizers==0.21.4', 'rdkit==2025.9.6'],
    'scope': 'H100 resource profiling on development objective only; no formal campaign or saved weights',
}
(ROOT / 'acp_profile_recipe.json').write_text(json.dumps(recipe, indent=2))
encoded_recipe = base64.b64encode(zlib.compress(json.dumps(recipe).encode(),9)).decode()
source = '''import base64,hashlib,json,os,pathlib,subprocess,sys,time,urllib.request,zipfile,zlib
TASK_RECIPE = json.loads(zlib.decompress(base64.b64decode(RECIPE_B64)))
t0=time.monotonic()
root=pathlib.Path(TASK_RECIPE['remote_directory'])
def event(name, **kw):
 print(json.dumps({'event':name,'elapsed_seconds':round(time.monotonic()-t0,3),**kw}),flush=True)
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def fetch(item,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(3):
  try:
   with urllib.request.urlopen(item['url'],timeout=40) as r,p.with_suffix(p.suffix+'.part').open('wb') as f:
    size=0
    while True:
     b=r.read(1024*1024)
     if not b: break
     size+=len(b)
     if size>80*1024*1024: raise RuntimeError('input size cap')
     f.write(b)
   p.with_suffix(p.suffix+'.part').replace(p)
   if h(p)!=item['sha256']: raise RuntimeError('input hash mismatch')
   if p.stat().st_size!=item['bytes']: raise RuntimeError('input size mismatch')
   return
  except Exception:
   if attempt==2: raise
try:
 mount=subprocess.run(['findmnt','-n','--mountpoint','/zzc','--output','TARGET'],capture_output=True,text=True,timeout=10)
 if mount.returncode or mount.stdout.strip()!='/zzc': raise RuntimeError('AFS mount not verified')
 root.mkdir(parents=True,exist_ok=False)
 scratch=pathlib.Path('/tmp')/TASK_RECIPE['task_id'];scratch.mkdir(exist_ok=False)
 (root/'recipe.json').write_text(json.dumps(TASK_RECIPE,indent=2))
 source=base64.b64decode(TASK_RECIPE['profile_source_b64'])
 if hashlib.sha256(source).hexdigest()!=TASK_RECIPE['profile_source_sha256']: raise RuntimeError('source hash mismatch')
 (root/'profile_h100.py').write_bytes(source)
 event('bootstrap_started',task_id=TASK_RECIPE['task_id'])
 base=pathlib.Path('/root/anaconda3/envs/deepdds/bin/python')
 if not base.is_file(): raise RuntimeError('verified base Python missing')
 subprocess.run([str(base),'-m','venv','--system-site-packages',str(scratch/'venv')],check=True,stdout=subprocess.DEVNULL,timeout=90)
 py=scratch/'venv/bin/python'
 with (root/'dependency_install.log').open('w') as log:
  subprocess.run([str(py),'-m','pip','--disable-pip-version-check','install','--index-url','https://pypi.org/simple','--no-cache-dir',*TASK_RECIPE['packages']],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 event('isolated_dependencies_ready')
 for item in TASK_RECIPE['public_model_files']: fetch(item,root/'model'/item['name'])
 wheel=TASK_RECIPE['selfies_wheel'];wp=root/wheel['name'];fetch(wheel,wp)
 with zipfile.ZipFile(wp) as z:
  vendor=scratch/'vendor';vendor.mkdir()
  for member in z.infolist():
   if not (vendor/member.filename).resolve().is_relative_to(vendor.resolve()): raise RuntimeError('wheel path escape')
  z.extractall(vendor)
 event('inputs_hash_verified')
 out=root/'profile.json'
 argv=[str(py),'-u',str(root/'profile_h100.py'),'--model-dir',str(root/'model'),'--vendor-dir',str(scratch/'vendor'),'--out',str(out),'--device','cuda','--batch-size','32','--steps','4','--max-length','256','--threads','4','--stress-shapes']
 result=subprocess.run(argv,timeout=TASK_RECIPE['max_profile_seconds'])
 report=json.loads(out.read_text())
 raw=json.dumps(report,separators=(',',':')).encode()
 event('profile_finished',returncode=result.returncode,status=report['status'])
 print('H100_PROFILE_RESULT_B64 '+base64.b64encode(zlib.compress(raw,9)).decode(),flush=True)
 summary={'gpu':report.get('gpu'),'runtime':report.get('runtime'),'status':report['status'],'remote_directory':str(root),'methods':{k:{field:v.get(field) for field in ['status','mean_measured_step_seconds','max_peak_allocated_bytes','max_peak_reserved_bytes']} for k,v in report['methods'].items()},'synthetic_stress':report.get('synthetic_stress')}
 print('H100_PROFILE_SUMMARY '+json.dumps(summary,separators=(',',':')),flush=True)
 raise SystemExit(result.returncode)
except Exception as exc:
 error={'status':'BOOTSTRAP_OR_PROFILE_FAILED','type':type(exc).__name__,'message':str(exc),'elapsed_seconds':time.monotonic()-t0}
 if root.exists(): (root/'bootstrap_error.json').write_text(json.dumps(error,indent=2))
 print('H100_PROFILE_ERROR '+json.dumps(error,separators=(',',':')),flush=True)
 raise
'''.replace('RECIPE_B64',repr(encoded_recipe))
(ROOT/'acp_bootstrap.py').write_text(source)
compile(source,'acp_bootstrap.py','exec')
print(json.dumps({'task_id':TASK,'bootstrap_sha256':hashlib.sha256(source.encode()).hexdigest(),'source_sha256':recipe['profile_source_sha256'],'bytes':len(source)}))
