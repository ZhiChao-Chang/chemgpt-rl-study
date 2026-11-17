"""Read-only bounded observer for the one already-created profile job."""
from pathlib import Path
import base64
import json
import re
import subprocess
import sys
import time
import zlib

ROOT=Path(__file__).resolve().parent
CONTROL=Path('/home/zhangzhichao/codex-cci-control')
sys.path.insert(0,str(CONTROL))
import acp_ctl as acp
path,j=acp.load(CONTROL/'state/acp/jobs/acp-explore-chemgpt-profile-20261104-03.json')
log_path=ROOT/'remote_native_logs.txt'
log_file=log_path.open('a',buffering=1)
reader=None
t0=time.monotonic();last=None
def read_result():
    if not log_path.exists():return False
    data=log_path.read_text(errors='replace')
    matches=re.findall(r'H100_PROFILE_RESULT_B64\s+([A-Za-z0-9+/=]+)',data)
    for blob in reversed(matches):
        try:
            report=json.loads(zlib.decompress(base64.b64decode(blob)))
            (ROOT/'actual_h100_profile.json').write_text(json.dumps(report,indent=2))
            return True
        except Exception:pass
    return False

try:
    while time.monotonic()-t0<1800:
        job=acp.describe(j)
        workers=acp.workers_api(j)
        snapshot={'observed_UTC':acp.stamp(),'uid':job['uid'],'state':job['state'],
                  'workers':workers.get('workers',[]),'elapsed_observer_seconds':time.monotonic()-t0}
        (ROOT/'runtime_status.json').write_text(json.dumps(snapshot,indent=2))
        print(json.dumps({'state':job['state'],'workers':[x['phase'] for x in workers.get('workers',[])],
                          'elapsed_seconds':round(time.monotonic()-t0)}),flush=True)
        candidates=[w for w in workers.get('workers',[]) if w['phase'] in ['Running','Succeeded','Failed']]
        if candidates and (reader is None or reader.poll() is not None):
            w=candidates[0]
            follow='true' if job['state']=='RUNNING' else 'false'
            args=[acp.SCO,'--profile',j['config']['sco_profile'],'--region',j['config']['region'],
                  'acp','jobs','stream-logs',j['task_id'],'--workspace-name',j['config']['workspace'],
                  '--follow='+follow,'--worker-name',w['name'],'--container-name',w['containers'][0]['name']]
            reader=subprocess.Popen(args,stdout=log_file,stderr=log_file)
        if read_result():
            print('H100_RESULT_COLLECTED',flush=True)
        if job['state'] in ['SUCCEEDED','FAILED','SUSPENDED','DELETED']:
            if reader is not None:
                try:reader.wait(timeout=15)
                except subprocess.TimeoutExpired:pass
            collected=read_result()
            print(json.dumps({'terminal':job['state'],'result_collected':collected}),flush=True)
            break
        time.sleep(20)
finally:
    if reader is not None and reader.poll() is None:
        reader.terminate()
        try:reader.wait(timeout=5)
        except subprocess.TimeoutExpired:reader.kill();reader.wait()
    log_file.close()
