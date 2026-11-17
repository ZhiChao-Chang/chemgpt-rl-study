"""Own-job absolute four-hour guard, no restart or resource replacement."""
from datetime import datetime,timezone,timedelta
import json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CONTROL=Path('/home/zhangzhichao/codex-cci-control');sys.path.insert(0,str(CONTROL))
import acp_ctl as acp
path,j=acp.load(CONTROL/'state/acp/jobs/acp-me2502-chemgpt-20261113-01.json')
assert j['uid']=='2096259930704a57a666199368b4e3ed'
job=acp.describe(j);assert job['uid']==j['uid']
deadline=datetime.fromisoformat(job['create_time'])+timedelta(hours=4)
acp.save(ROOT/'evidence/budget_guard.json',dict(status='ARMED',uid=j['uid'],absolute_deadline=deadline.isoformat(),scope='one owned ACP job; includes startup/queue time conservatively'))
print('BUDGET_GUARD_ARMED '+deadline.isoformat(),flush=True)
while datetime.now(timezone.utc)<deadline:
    job=acp.describe(j);assert job['uid']==j['uid']
    if job['state'] in ['SUCCEEDED','FAILED','SUSPENDED','DELETED']:
        acp.save(ROOT/'evidence/budget_guard.json',dict(status='TERMINAL_NO_STOP_NEEDED',uid=j['uid'],state=job['state'],absolute_deadline=deadline.isoformat()))
        print('BUDGET_GUARD_TERMINAL '+job['state'],flush=True);raise SystemExit(0)
    time.sleep(min(30,max(0,(deadline-datetime.now(timezone.utc)).total_seconds())))
job=acp.describe(j);assert job['uid']==j['uid']
if job['state'] not in ['SUCCEEDED','FAILED','SUSPENDED','DELETED']:
    # Fixed intent, dispatch once; if outcome unknown, only describe original identity.
    _,j=acp.load(path)
    acp.dispatch_once(path,j,'stop',['acp','jobs','stop',j['task_id'],'--workspace-name',j['config']['workspace']])
    acp.save(ROOT/'evidence/budget_guard.json',dict(status='DEADLINE_STOP_DISPATCHED_NEEDS_GET',uid=j['uid'],absolute_deadline=deadline.isoformat()))
    for _ in range(12):
        job=acp.describe(j);assert job['uid']==j['uid'];ws=acp.workers_api(j)
        acp.save(ROOT/'evidence/budget_release.json',dict(uid=j['uid'],state=job['state'],workers=ws,observed=acp.stamp()))
        if job['state']=='SUSPENDED' and all(w['phase']=='Deleted' for w in ws['workers']):break
        time.sleep(10)
