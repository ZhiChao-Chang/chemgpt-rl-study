"""Read completed own-run evidence over authenticated ACP terminal, no checkpoint downloads."""
import base64,hashlib,io,json,os,pty,re,select,shlex,subprocess,sys,tarfile,time,tty,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CONTROL=Path('/home/zhangzhichao/codex-cci-control');sys.path.insert(0,str(CONTROL))
import acp_ctl as acp
_,j=acp.load(CONTROL/'state/acp/jobs/acp-me2502-chemgpt-20261113-01.json')
job=acp.describe(j);assert job['uid']=='2096259930704a57a666199368b4e3ed' and job['state']=='RUNNING'
ws=[w for w in acp.workers_api(j)['workers'] if w['phase']=='Running'];assert len(ws)==1
w=ws[0]
code="""import pathlib,tarfile,io,base64,hashlib,json,sys,importlib.metadata,platform
r=pathlib.Path('/zzc/260010120/cci_control/acp-tests/acp-me2502-chemgpt-20261113-01')
files=[r/'contract.json',r/'results/environment.json',r/'results/technical_gate.json']+list((r/'src').glob('*.py'))
for result in (r/'results').rglob('result.json'):
 files+=list(result.parent.glob('*.json'))+list(result.parent.glob('*.jsonl'))
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as t:
 for p in sorted(set(files)):
  if p.is_file():t.add(p,arcname=str(p.relative_to(r)),recursive=False)
 sys.path.insert(0,str(r/'vendor'))
 versions={}
 for package in ['torch','transformers','tokenizers','numpy','rdkit','selfies']:
  try:versions[package]=importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:versions[package]=None
 runtime=json.dumps({'python':platform.python_version(),'packages':versions,'scope':'metadata sampled by evidence collector, vendor SELFIES path first; not a new training run'}).encode()
 info=tarfile.TarInfo('runtime_snapshot.json');info.size=len(runtime);t.addfile(info,io.BytesIO(runtime))
d=b.getvalue()
print('ME2502_BEGIN '+json.dumps({'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}),flush=True)
for off in range(0,len(d),32):print('ME2502_DATA '+base64.b64encode(d[off:off+32]).decode(),flush=True)
print('ME2502_END',flush=True)
"""
b64=base64.b64encode(zlib.compress(code.encode(),9)).decode();py='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(b64)+')))'
cmd='/root/anaconda3/envs/ddc_encoding/bin/python -u -c '+shlex.quote(py)+'\n';assert len(cmd)<3900
args=[acp.SCO,'--profile',j['config']['sco_profile'],'--region',j['config']['region'],'acp','jobs','exec',j['task_id'],'--workspace-name',j['config']['workspace'],'--worker-name',w['name']]
master,slave=pty.openpty();tty.setraw(slave);child=subprocess.Popen(args,stdin=slave,stdout=slave,stderr=slave,start_new_session=True,close_fds=True);os.close(slave);os.set_blocking(master,False)
buf=b'';start=time.monotonic()
def pump():
 global buf
 if select.select([master],[],[],.2)[0]:
  try:buf+=os.read(master,65536)
  except BlockingIOError:pass
 if len(buf)>32*1024*1024:raise RuntimeError('bounded evidence transfer exceeded')
def wait(marker,seconds):
 until=time.monotonic()+seconds
 while marker not in buf:
  pump()
  if child.poll() is not None:raise RuntimeError('terminal exited before marker')
  if time.monotonic()>until:raise TimeoutError('native read timeout')
try:
 until=time.monotonic()+12
 while time.monotonic()<until:
  pump()
  if re.search(rb'[#\$] ?',buf):break
 os.write(master,b"stty -echo\nprintf 'ME2502_SHELL_READY\\n'\n");wait(b'ME2502_SHELL_READY',20);buf=b''
 os.write(master,cmd.encode());wait(b'ME2502_END',120)
 buf=re.sub(rb'\x1b\[[0-?]*[ -/]*[@-~]',b'',buf)
 header=re.search(rb'ME2502_BEGIN (\{[^\r\n]+\})',buf);assert header
 meta=json.loads(header.group(1));blocks=re.findall(rb'ME2502_DATA ([A-Za-z0-9+/=]+)',buf)
 data=b''.join(base64.b64decode(x) for x in blocks)
 assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
 dest=ROOT/'collected_remote';dest.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  for m in tar.getmembers():
   target=(dest/m.name).resolve();assert target.is_relative_to(dest.resolve()) and m.isfile()
   content=tar.extractfile(m).read();target.parent.mkdir(parents=True,exist_ok=True)
   target.write_bytes(content)
 receipt=dict(status='COLLECTED_HASH_VERIFIED',job_uid=job['uid'],worker_uid=w['uid'],archive_sha256=meta['sha256'],bytes=len(data),completed_runs=len(list((dest/'results').rglob('result.json'))),elapsed_s=time.monotonic()-start)
 acp.save(ROOT/'evidence/native_collection.json',receipt);print(json.dumps(receipt))
 os.write(master,b'exit\n')
finally:
 if child.poll() is None:
  child.terminate()
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:child.kill();child.wait()
 os.close(master)
