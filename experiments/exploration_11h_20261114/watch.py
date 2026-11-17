"""Bounded own-job observer, incremental collection and local ledger verification."""
import base64,json,re,subprocess,sys,time,zlib
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CONTROL=Path('/home/zhangzhichao/codex-cci-control');sys.path.insert(0,str(CONTROL))
import acp_ctl as acp
sub=json.loads((ROOT/'evidence/submission.json').read_text());_,j=acp.load(Path(sub['journal']))
deadline=datetime.fromisoformat(sub['absolute_deadline']);reader=None;last=None;verified={}
log=ROOT/'evidence/remote.log';handle=log.open('a',buffering=1)
def extract():
    text=log.read_text(errors='replace');found={}
    for m in re.finditer(r'EXT_RESULT (\{[^\r\n]+\})',text):
        try:d=json.loads(m.group(1))
        except ValueError:continue
        found[f"{d['phase']}/{d['task']}__{d['method']}__s{d['seed']}"]=d
    acp.save(ROOT/'evidence/log_results.json',{'results':list(found.values())})
    for m in re.finditer(r'EXT_CAMPAIGN_B64 ([A-Za-z0-9+/=]+)',text):
        try:acp.save(ROOT/'evidence/remote_campaign.json',json.loads(zlib.decompress(base64.b64decode(m.group(1)))))
        except Exception:pass
    return len(found),'EXT_HOLD_FOR_COLLECTION' in text
def local_update():
    for p in (ROOT/'collected/results').rglob('result.json'):
        key=str(p.parent.relative_to(ROOT/'collected/results'))
        if key not in verified:
            v=subprocess.run(['/home/zhangzhichao/miniconda3/envs/ddc_inference/bin/python',str(ROOT/'verify_extension.py'),str(p.parent)],capture_output=True,text=True,timeout=90)
            verified[key]=json.loads(v.stdout) if v.returncode==0 else {'status':'FAIL','exit_code':v.returncode}
    acp.save(ROOT/'evidence/verification.json',{'verified_runs':len(verified),'runs':verified})
    subprocess.run([sys.executable,str(ROOT/'analyze.py')],check=True,timeout=60)
try:
 while datetime.now(timezone.utc)<deadline+timedelta(minutes=5):
  try:
   job=acp.describe(j);assert job['uid']==sub['job_uid'];ws=acp.workers_api(j)['workers'];n,hold=extract()
   acp.save(ROOT/'evidence/status.json',dict(observed=acp.stamp(),state=job['state'],uid=job['uid'],workers=ws,log_results=n,absolute_deadline=sub['absolute_deadline']))
   brief=(job['state'],n)
   if brief!=last:print(json.dumps({'state':brief[0],'log_results':n}),flush=True);last=brief
   candidates=[w for w in ws if w['phase'] in ['Running','Succeeded','Failed']]
   if candidates and (reader is None or reader.poll() is not None):
    w=candidates[0];args=[acp.SCO,'--profile',j['config']['sco_profile'],'--region',j['config']['region'],'acp','jobs','stream-logs',j['task_id'],'--workspace-name',j['config']['workspace'],'--follow='+('true' if job['state'] not in ['SUCCEEDED','FAILED','SUSPENDED','DELETED'] else 'false'),'--worker-name',w['name'],'--container-name',w['containers'][0]['name']]
    reader=subprocess.Popen(args,stdout=handle,stderr=handle)
   if job['state']=='RUNNING':
    if not (ROOT/'evidence/prior_collection.json').exists():
     v=subprocess.run([sys.executable,str(ROOT/'collect.py'),'--prior'],capture_output=True,text=True,timeout=160)
     if v.returncode==0:
      chk=subprocess.run([sys.executable,str(ROOT.parent/'verify_ledgers.py'),str(ROOT.parent/'collected_remote/results')],capture_output=True,text=True,check=True,timeout=90)
      (ROOT.parent/'evidence/remote_ledger_check.json').write_text(chk.stdout);print('PRIOR_FULL_EVIDENCE_VERIFIED',flush=True)
    raw=len(list((ROOT/'collected/results').rglob('result.json')))
    if n>raw or hold:
     v=subprocess.run([sys.executable,str(ROOT/'collect.py')],capture_output=True,text=True,timeout=160)
     if v.returncode==0:local_update()
   if job['state'] in ['SUCCEEDED','FAILED','SUSPENDED','DELETED']:
    if reader:
     try:reader.wait(timeout=10)
     except subprocess.TimeoutExpired:pass
    extract();local_update();print('WATCH_TERMINAL '+job['state'],flush=True);break
  except Exception as exc:
   print(json.dumps({'observation_retry':type(exc).__name__}),flush=True)
  time.sleep(10 if 'hold' in locals() and hold else 25)
finally:
 if reader and reader.poll() is None:
  reader.terminate()
  try:reader.wait(timeout=5)
  except subprocess.TimeoutExpired:reader.kill();reader.wait()
 handle.close()
