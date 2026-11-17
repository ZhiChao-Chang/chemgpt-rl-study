"""Bounded read-only observer; never creates/restarts/stops any cloud job."""
import base64
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import zlib
ROOT=Path(__file__).resolve().parent;CONTROL=Path('/home/zhangzhichao/codex-cci-control');sys.path.insert(0,str(CONTROL))
import acp_ctl as acp
_,j=acp.load(CONTROL/'state/acp/jobs/acp-me2502-chemgpt-20261113-01.json')
log=ROOT/'evidence/remote.log';handle=log.open('a',buffering=1);reader=None;t0=time.monotonic();last=None
def extract():
    text=log.read_text(errors='replace')
    found={}
    for m in re.finditer(r'RUN_RESULT (\{[^\r\n]+\})',text):
        try:d=json.loads(m.group(1))
        except json.JSONDecodeError:continue
        key=f"{d['task']}__{d['method']}__s{d['seed']}";found[key]=d
    if found:acp.save(ROOT/'evidence/remote_results.json',{'results':list(found.values()),'scope':'summaries from native logs; full ledgers remain AFS'})
    for m in re.finditer(r'ME2502_CAMPAIGN_B64 ([A-Za-z0-9+/=]+)',text):
        try:acp.save(ROOT/'evidence/remote_campaign.json',json.loads(zlib.decompress(base64.b64decode(m.group(1)))))
        except Exception:continue
    return len(found)
try:
    while time.monotonic()-t0<15000:
        job=acp.describe(j);assert job['uid']==j['uid']
        ws=acp.workers_api(j).get('workers',[])
        state=dict(observed_utc=acp.stamp(),uid=job['uid'],state=job['state'],workers=ws,elapsed_observer_s=time.monotonic()-t0,results_collected=extract())
        acp.save(ROOT/'evidence/remote_status.json',state)
        brief=(job['state'],tuple(w['phase'] for w in ws),state['results_collected'])
        if brief!=last:print(json.dumps({'state':brief[0],'workers':brief[1],'results_collected':brief[2]}),flush=True);last=brief
        running=[w for w in ws if w['phase'] in ['Running','Succeeded','Failed']]
        if running and (reader is None or reader.poll() is not None):
            w=running[0]
            args=[acp.SCO,'--profile',j['config']['sco_profile'],'--region',j['config']['region'],'acp','jobs','stream-logs',j['task_id'],
                  '--workspace-name',j['config']['workspace'],'--follow='+('true' if job['state']=='RUNNING' else 'false'),
                  '--worker-name',w['name'],'--container-name',w['containers'][0]['name']]
            reader=subprocess.Popen(args,stdout=handle,stderr=handle)
        if job['state'] in ['SUCCEEDED','FAILED','SUSPENDED','DELETED']:
            if reader:
                try:reader.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
            extract();print('OBSERVER_TERMINAL '+job['state'],flush=True);break
        time.sleep(20)
finally:
    if reader and reader.poll() is None:
        reader.terminate()
        try:reader.wait(timeout=5)
        except subprocess.TimeoutExpired:reader.kill();reader.wait()
    handle.close()
