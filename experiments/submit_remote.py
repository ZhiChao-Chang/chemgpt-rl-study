"""One journaled ACP create for the explicitly authorized ME2502 study."""
import base64,hashlib,json,shlex,sys,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONTROL=Path('/home/zhangzhichao/codex-cci-control');sys.path.insert(0,str(CONTROL))
import acp_ctl as acp
meta=json.loads((ROOT/'evidence/input_manifest.json').read_text());task=meta['task']
config=json.loads((CONTROL/'profiles/cci.json').read_text())
path,j=acp.plan(config,task,str(ROOT/'src/study.py'))
if j['phase']!='PLANNED':raise RuntimeError(j['phase'])
s=j['selected']
assert s['pool_name'] in ['share-cluster','computing-cluster-01g-02'] and s['gpu_count']==1
raw=(ROOT/'remote_bootstrap.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==meta['bootstrap_sha256']
j.update(prior_default_probe_intent_sha256=j['intent_sha256'],launcher_path=str(ROOT/'remote_bootstrap.py'),launcher_sha256=meta['bootstrap_sha256'],
 authorization='2026-11-13 user explicitly delegates RL final project experiments and necessary servers',
 limits={'gpu_count':1,'hard_worker_seconds':13200,'maximum_gpu_hours':4},phase='PLANNED_CHEMGPT_STUDY',input_manifest=meta)
fields=['task_id','config','selected','image','image_digest','mount','source_path','source_sha256','remote_directory','launcher_path','launcher_sha256','framework','resume_from']
j['intent_sha256']=acp.digest({k:j[k] for k in fields})
b64=base64.b64encode(zlib.compress(raw,9)).decode()
code='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(b64)+')))'
command='timeout --signal=TERM --kill-after=30s 13200s /root/anaconda3/envs/deepdds/bin/python -u -c '+shlex.quote(code)
j['command_sha256']=hashlib.sha256(command.encode()).hexdigest();acp.save(path,j)
report=acp.preflight(config);acp.budget_gate(report)
assert acp.choose_single(report)==s
assert not any(x['name']==task for x in acp.pages_cli(config,'acp',name=task))
args=['acp','jobs','create','--workspace-name',config['workspace'],'--name',task,'--job-name',task,'--aec2-name',s['pool_name'],
 '--training-framework','pytorch','--worker-nodes','1','--worker-spec',s['spec'],'--priority','NORMAL','--quota-type','reserved',
 '--container-image-url',j['image'],'--az',s['zone'],'--vpc-id',s['vpc_id'],'--storage-mount',j['mount']['id']+':'+j['mount']['mount_path'],'--command',command]
result=acp.dispatch_once(path,j,'create',args)
print(json.dumps({'journal':str(path),'task':task,'dispatch':result}),flush=True)
job=acp.describe(j);j.update(uid=job['uid'],server_id=job['id'],phase='SUBMITTED_CHEMGPT_STUDY');acp.save(path,j)
print(json.dumps({'uid':job['uid'],'state':job['state'],'selected':s}),flush=True)
