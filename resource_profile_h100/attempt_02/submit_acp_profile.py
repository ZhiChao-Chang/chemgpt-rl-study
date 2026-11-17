"""One explicit, journaled CREATE for a bounded user-authorized H100 profile."""
from pathlib import Path
import base64
import hashlib
import json
import shlex
import sys
import zlib

CONTROL = Path('/home/zhangzhichao/codex-cci-control')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(CONTROL))
import acp_ctl as acp

recipe=json.loads((ROOT/'acp_profile_recipe.json').read_text())
task=recipe['task_id']
config=json.loads((CONTROL/'profiles/cci.json').read_text())
# plan() freezes source image/AFS/scope and performs complete fresh preflight.
path,j=acp.plan(config,task,str(ROOT/'profile_h100.py'))
if j['phase']!='PLANNED':
    raise RuntimeError('Preflight did not admit a profile plan: '+j['phase'])
selected=j['selected']
if selected['pool_name'] not in ['share-cluster','computing-cluster-01g-02'] or selected['gpu_count']!=1:
    raise RuntimeError('Only a single full H100 candidate is allowed; no 5090 fallback')
bootstrap=(ROOT/'acp_bootstrap.py').read_bytes()
j['prior_default_probe_intent_sha256']=j['intent_sha256']
j['launcher_path']=str(ROOT/'acp_bootstrap.py')
j['launcher_sha256']=hashlib.sha256(bootstrap).hexdigest()
j['profile_recipe_sha256']=hashlib.sha256((ROOT/'acp_profile_recipe.json').read_bytes()).hexdigest()
j['input_transport']='Verified bootstrap source in CREATE command; pinned public asset downloads verified on worker; no CCI upload'
j['authorization']='Current user explicitly requested H100 to determine course-project resource requirements'
j['retry_from']={'task_id':'acp-explore-chemgpt-profile-20261104-01','uid':'504dcf43724a47b5a100e906db927822','reason':'AFS venv creation timeout before profiling; terminal FAILED and Worker Failed, budget release confirmed by fresh preflight'}
j['limits']={'full_h100_cards':1,'whole_job_hard_timeout_seconds':1200,'measurement_timeout_seconds':600,'allocator_cap_fraction':.24,'output_model_weights':False}
j['phase']='PLANNED_RESOURCE_PROFILE'
fields=['task_id','config','selected','image','image_digest','mount','source_path','source_sha256','remote_directory','launcher_path','launcher_sha256','framework','resume_from']
j['intent_sha256']=acp.digest({k:j[k] for k in fields})
encoded=base64.b64encode(zlib.compress(bootstrap,9)).decode()
python_code="import base64,zlib;exec(zlib.decompress(base64.b64decode("+repr(encoded)+")))"
command='timeout --signal=TERM --kill-after=30s 1200s /root/anaconda3/envs/deepdds/bin/python -u -c '+shlex.quote(python_code)
j['profile_command_sha256']=hashlib.sha256(command.encode()).hexdigest()
acp.save(path,j)
# Fresh gate again immediately before dispatch; no mutation has happened yet.
report=acp.preflight(config)
acp.budget_gate(report)
if acp.choose_single(report)!=selected:
    raise RuntimeError('Resource selection changed; do not submit this fixed intent')
if any(x['name']==task for x in acp.pages_cli(config,'acp',name=task)):
    raise RuntimeError('Job already exists; query, never create again')
assert hashlib.sha256((ROOT/'acp_bootstrap.py').read_bytes()).hexdigest()==j['launcher_sha256']
args=['acp','jobs','create','--workspace-name',config['workspace'],'--name',task,'--job-name',task,
      '--aec2-name',selected['pool_name'],'--training-framework','pytorch','--worker-nodes','1',
      '--worker-spec',selected['spec'],'--priority','NORMAL','--quota-type','reserved',
      '--container-image-url',j['image'],'--az',selected['zone'],'--vpc-id',selected['vpc_id'],
      '--storage-mount',j['mount']['id']+':'+j['mount']['mount_path'],'--command',command]
result=acp.dispatch_once(path,j,'create',args)
print(json.dumps({'journal':str(path),'task':task,'selected':selected,'dispatch':result}),flush=True)
try:
    job=acp.describe(j)
    j['uid']=job['uid'];j['server_id']=job['id'];j['phase']='SUBMITTED_PROFILE_NEEDS_RUNTIME_ACCEPTANCE'
    acp.save(path,j)
    print(json.dumps({'task':task,'uid':job['uid'],'state':job['state']}),flush=True)
except Exception as exc:
    print(json.dumps({'task':task,'post_dispatch_get':'unknown; query original ID only','error_type':type(exc).__name__}),flush=True)
