"""Incremental authenticated native-terminal collection for this one owned worker."""
import argparse,base64,fcntl,hashlib,io,json,os,pty,re,select,shlex,subprocess,sys,tarfile,time,tty,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CONTROL=Path('/home/zhangzhichao/codex-cci-control');sys.path.insert(0,str(CONTROL))
import acp_ctl as acp
ap=argparse.ArgumentParser();ap.add_argument('--prior',action='store_true');a=ap.parse_args()
lock=(ROOT/'evidence/collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sub=json.loads((ROOT/'evidence/submission.json').read_text());_,j=acp.load(Path(sub['journal']))
job=acp.describe(j);assert job['uid']==sub['job_uid'] and job['state']=='RUNNING'
ws=[w for w in acp.workers_api(j)['workers'] if w['phase']=='Running'];assert len(ws)==1;w=ws[0]
dest=ROOT.parent/'collected_remote' if a.prior else ROOT/'collected'
known=[str(p.parent.relative_to(dest/'results')) for p in (dest/'results').rglob('result.json')]
remote='/zzc/260010120/cci_control/acp-tests/'+sub['task']
code='''import base64,hashlib,io,json,pathlib,tarfile,select,sys,time,zlib
r=pathlib.Path(REMOTE)
if PRIOR:r=pathlib.Path('/zzc/260010120/cci_control/acp-tests/acp-me2502-chemgpt-20261113-01')
files=[r/'contract.json']+list((r/'src').glob('*.py'))+list((r/'results').glob('*.json'))
candidates=[p for p in sorted((r/'results').rglob('result.json')) if str(p.parent.relative_to(r/'results')) not in KNOWN][:3]
for result in candidates:files+=list(result.parent.glob('*.json'))+list(result.parent.glob('*.jsonl'))
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as t:
 for p in sorted(set(files)):
  if p.is_file():t.add(p,arcname=str(p.relative_to(r)),recursive=False)
d=b.getvalue()
assert len(d)<16*1024*1024,'export too large'
print('RLX_BEGIN '+json.dumps({'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}),flush=True)
for index,start in enumerate(range(0,len(d),32768)):
 block=d[start:start+32768]
 for retry in range(3):
  for off in range(0,len(block),32):
   print('RLX_DATA '+base64.b64encode(block[off:off+32]).decode())
   if off%4096==0:time.sleep(.002)
  print('RLX_BLOCK_END %d %d %08x'%(index,len(block),zlib.crc32(block)),flush=True)
  if not select.select([sys.stdin],[],[],25)[0]:raise RuntimeError('collection ack timeout')
  ack=sys.stdin.readline().strip()
  if ack=='RLX_OK '+str(index):break
  if ack!='RLX_RETRY '+str(index):raise RuntimeError('invalid ack')
 else:raise RuntimeError('collection retry limit')
print('RLX_END',flush=True)
'''
code='\n'.join(f'{k}={v!r}' for k,v in dict(REMOTE=remote,PRIOR=a.prior,KNOWN=known).items())+'\n'+code
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
 if len(buf)>32*1024*1024:raise RuntimeError('bounded native transfer exceeded')
def wait(marker,seconds):
 until=time.monotonic()+seconds
 while marker not in buf:
  pump()
  if child.poll() is not None:raise RuntimeError('terminal exited before marker')
  if time.monotonic()>until:raise TimeoutError('native transfer timeout')
def frame(pattern,seconds):
 global buf
 until=time.monotonic()+seconds
 while True:
  m=re.search(pattern,buf)
  if m:
   before=buf[:m.start()];groups=m.groups();buf=buf[m.end():];return before,groups
  pump()
  if child.poll() is not None or time.monotonic()>until:raise TimeoutError('native frame timeout')
try:
 until=time.monotonic()+10
 while time.monotonic()<until:
  pump()
  if re.search(rb'[#\$] ?',buf):break
 os.write(master,b'stty -echo\n');time.sleep(.4);os.write(master,b"printf 'RLX_SHELL_READY\\n'\n");wait(b'RLX_SHELL_READY',20);buf=b''
 os.write(master,cmd.encode());_,groups=frame(rb'RLX_BEGIN (\{[^\r\n]+\})\r?\n',30);meta=json.loads(groups[0]);parts=[]
 for index in range((meta['bytes']+32767)//32768):
  for retry in range(3):
   packet,fields=frame(rb'RLX_BLOCK_END (\d+) (\d+) ([0-9a-f]{8})\r?\n',30)
   assert int(fields[0])==index
   packet=re.sub(rb'\x1b\[[0-?]*[ -/]*[@-~]',b'',packet)
   try:
    data_part=b''.join(base64.b64decode(x,validate=True) for x in re.findall(rb'RLX_DATA ([A-Za-z0-9+/=]+)',packet))
    ok=len(data_part)==int(fields[1]) and zlib.crc32(data_part)==int(fields[2],16)
   except ValueError:ok=False
   os.write(master,('RLX_'+('OK' if ok else 'RETRY')+' '+str(index)+'\n').encode())
   if ok:parts.append(data_part);break
  else:raise RuntimeError('native packet failed verification')
 wait(b'RLX_END',20);data=b''.join(parts)
 assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
 dest.mkdir(parents=True,exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  for member in tar.getmembers():
   target=(dest/member.name).resolve();assert target.is_relative_to(dest.resolve()) and member.isfile()
   content=tar.extractfile(member).read();target.parent.mkdir(parents=True,exist_ok=True)
   if a.prior and target.exists() and target.read_bytes()!=content:raise RuntimeError('prior evidence conflict: '+member.name)
   tmp=target.with_name(target.name+'.incoming');tmp.write_bytes(content);tmp.replace(target)
 receipt=dict(status='COLLECTED_HASH_VERIFIED',prior=a.prior,archive_sha256=meta['sha256'],bytes=len(data),job_uid=job['uid'],worker_uid=w['uid'],completed_raw_runs=len(list((dest/'results').rglob('result.json'))),elapsed_s=time.monotonic()-start)
 prior_name='prior_collection.json' if receipt['completed_raw_runs']==24 else 'prior_collection_progress.json'
 acp.save(ROOT/'evidence'/(prior_name if a.prior else 'collection.json'),receipt);print(json.dumps(receipt))
 os.write(master,b'exit\n')
finally:
 if child.poll() is None:
  child.terminate()
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:child.kill();child.wait()
 os.close(master)
