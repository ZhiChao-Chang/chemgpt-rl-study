"""Submit exactly one owned ACP worker with a pinned image and absolute deadline."""
import base64,hashlib,json,shlex,sys,zlib
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CONTROL=Path('/home/zhangzhichao/codex-cci-control');sys.path.insert(0,str(CONTROL))
import acp_ctl as acp
assert json.loads((ROOT/'evidence/extension_gate.json').read_text())['status']=='PASS'
meta=json.loads((ROOT/'evidence/input_manifest.json').read_text());config=json.loads((CONTROL/'profiles/cci.json').read_text());task=meta['task']
_,old=acp.load(CONTROL/'state/acp/jobs/acp-me2502-chemgpt-20261113-01.json');oldjob=acp.describe(old)
assert oldjob['uid']=='2096259930704a57a666199368b4e3ed' and oldjob['state']=='SUCCEEDED'
path,j=acp.plan(config,task,str(ROOT/'src/campaign.py'))
if j['phase']!='PLANNED':raise RuntimeError(j['phase'])
s=j['selected'];assert s['gpu_count']==1 and s['pool_name'] in ['share-cluster','computing-cluster-01g-02']
assert j['mount']==old['mount'],'AFS mount identity changed'
raw=(ROOT/'bootstrap.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==meta['bootstrap_sha256']
j.update(prior_default_probe_intent_sha256=j['intent_sha256'],image=old['image'],image_digest=old['image_digest'],
 launcher_path=str(ROOT/'bootstrap.py'),launcher_sha256=meta['bootstrap_sha256'],
 authorization='User explicitly grants eleven hours autonomous ME2502 RL exploration, 2026-11-14; single GPU',
 absolute_deadline=meta['absolute_deadline'],input_manifest=meta,phase='PLANNED_RL_EXTENSION',
 limits={'gpu_count':1,'maximum_gpu_hours':11,'automatic_new_attempts':False})
fields=['task_id','config','selected','image','image_digest','mount','source_path','source_sha256','remote_directory','launcher_path','launcher_sha256','framework','resume_from']
j['intent_sha256']=acp.digest({k:j[k] for k in fields})
b64=base64.b64encode(zlib.compress(raw,9)).decode();code='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(b64)+')))'
left=int((datetime.fromisoformat(meta['absolute_deadline'])-datetime.now(timezone.utc)).total_seconds())-30
assert 300<left<=39600
command=f'timeout --signal=TERM --kill-after=20s {left}s /root/anaconda3/envs/deepdds/bin/python -u -c '+shlex.quote(code)
j['command_sha256']=hashlib.sha256(command.encode()).hexdigest();acp.save(path,j)
report=acp.preflight(config);acp.budget_gate(report);assert acp.choose_single(report)==s
assert not any(x['name']==task for x in acp.pages_cli(config,'acp',name=task))
args=['acp','jobs','create','--workspace-name',config['workspace'],'--name',task,'--job-name',task,'--aec2-name',s['pool_name'],
 '--training-framework','pytorch','--worker-nodes','1','--worker-spec',s['spec'],'--priority','NORMAL','--quota-type','reserved',
 '--container-image-url',j['image'],'--az',s['zone'],'--vpc-id',s['vpc_id'],'--storage-mount',j['mount']['id']+':'+j['mount']['mount_path'],'--command',command]
res=acp.dispatch_once(path,j,'create',args);print(json.dumps({'journal':str(path),'dispatch':res}),flush=True)
job=acp.describe(j);j.update(uid=job['uid'],server_id=job['id'],phase='SUBMITTED_RL_EXTENSION');acp.save(path,j)
(ROOT/'evidence/submission.json').write_text(json.dumps({'job_uid':job['uid'],'task':task,'journal':str(path),'absolute_deadline':meta['absolute_deadline']},indent=2))
print(json.dumps({'uid':job['uid'],'state':job['state'],'deadline':meta['absolute_deadline']}),flush=True)
