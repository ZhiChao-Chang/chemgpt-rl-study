"""Absolute deadline guard; never creates/restarts a resource."""
from datetime import datetime,timezone
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CONTROL=Path('/home/zhangzhichao/codex-cci-control');sys.path.insert(0,str(CONTROL))
import acp_ctl as acp
sub=json.loads((ROOT/'evidence/submission.json').read_text());path,j=acp.load(Path(sub['journal']));deadline=datetime.fromisoformat(sub['absolute_deadline'])
acp.save(ROOT/'evidence/guard.json',dict(status='ARMED',uid=sub['job_uid'],deadline=sub['absolute_deadline']))
print('GUARD_ARMED '+sub['absolute_deadline'],flush=True)
while True:
 try:
  job=acp.describe(j);assert job['uid']==sub['job_uid']
  if job['state'] in ['SUCCEEDED','FAILED','SUSPENDED','DELETED']:
   acp.save(ROOT/'evidence/guard.json',dict(status='TERMINAL_NO_STOP_NEEDED',state=job['state'],uid=job['uid']));break
  if datetime.now(timezone.utc)>=deadline:
   _,j=acp.load(path);acp.dispatch_once(path,j,'stop',['acp','jobs','stop',j['task_id'],'--workspace-name',j['config']['workspace']])
   acp.save(ROOT/'evidence/guard.json',dict(status='DEADLINE_STOP_DISPATCHED_NEEDS_GET',uid=job['uid']))
   for _ in range(18):
    job=acp.describe(j);ws=acp.workers_api(j);acp.save(ROOT/'evidence/release.json',dict(uid=job['uid'],state=job['state'],workers=ws))
    if job['state']=='SUSPENDED' and all(w['phase']=='Deleted' for w in ws['workers']):break
    time.sleep(10)
   break
 except Exception as exc:
  print(json.dumps({'guard_read_retry':type(exc).__name__}),flush=True)
  # dispatch_once journals unknown actions and will not resend a stop.
 time.sleep(20)
