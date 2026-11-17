"""Fixed extension schedule with an absolute user deadline, no adaptive tuning."""
import argparse,json,time,traceback,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import study as s
import torch
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--model-dir',required=True);ap.add_argument('--contract',required=True)
    a=ap.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);c=json.loads(Path(a.contract).read_text())
    seconds=(datetime.fromisoformat(c['absolute_deadline'])-datetime.now(timezone.utc)).total_seconds()
    deadline=time.monotonic()+seconds-180
    if seconds<300:raise RuntimeError('insufficient authorized time remains')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');device=torch.device('cuda:0')
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(0)
    md=Path(a.model_dir);assert s.rt.sha256_file(md/'pytorch_model.bin')==s.rt.WEIGHT_SHA256
    model,info=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,torch_dtype=torch.float32,weights_only=True,output_loading_info=True)
    assert not info.get('missing_keys') and not info.get('mismatched_keys')
    assert all(k.endswith('.attn.attention.bias') or k.endswith('.attn.attention.masked_bias') for k in info.get('unexpected_keys',[]))
    model.eval();tok=Tokenizer.from_file(str(md/'tokenizer.json'))
    import importlib.metadata
    packages={x:importlib.metadata.version(x) for x in ['torch','transformers','tokenizers','numpy','rdkit','selfies']}
    s.save(root/'environment.json',dict(python=platform.python_version(),packages=packages,gpu=torch.cuda.get_device_name(0),model_sha256=s.rt.WEIGHT_SHA256,contract=c))
    s.validate_model(model,tok,device,root)
    schedule=[]
    for p in c['phases']:
        for seed in p['seeds']:
            for task in p['tasks']:
                for condition in p['conditions']:schedule.append(dict(phase=p['name'],task=task,seed=seed,condition=condition))
    s.save(root/'schedule.json',schedule);results=[];failures=[];started=set()
    for item in schedule:
        if s.STOP or time.monotonic()>deadline-180:break
        p=next(p for p in c['phases'] if p['name']==item['phase']);cond=item['condition']
        settings={**c['fixed_training'],**cond,'beta_override':cond['beta'],'phase_name':p['name']}
        phase_root=root/p['name'];phase_root.mkdir(exist_ok=True)
        run_id=f"{p['name']}/{item['task']}__{cond['id']}__s{item['seed']}";started.add(run_id)
        s.save(root/'progress.json',dict(status='RUNNING',current=run_id,completed=len(results),failed=len(failures),planned=len(schedule)))
        try:
            r=s.run_one(model,tok,device,phase_root,item['task'],cond['id'],item['seed'],p,settings,deadline)
            results.append(r);print('EXT_RESULT '+json.dumps(r),flush=True)
            if r['status']!='COMPLETE':break
        except Exception as exc:
            failure=dict(run_id=run_id,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            failures.append(failure);s.save(root/'failure.json',failure)
            print('EXT_FAILURE '+json.dumps({k:v for k,v in failure.items() if k!='traceback'}),flush=True)
            # Preserve failure and stop; do not silently drop or replace a seed.
            break
        s.save(root/'campaign.json',dict(status='RUNNING',results=results,failures=failures,planned=len(schedule)))
    planned_ids=[f"{i['phase']}/{i['task']}__{i['condition']['id']}__s{i['seed']}" for i in schedule]
    status='COMPLETE' if len(results)==len(schedule) else ('FAILED' if failures else 'BUDGET_STOP')
    final=dict(status=status,results=results,failures=failures,planned=len(schedule),not_started=[x for x in planned_ids if x not in started])
    s.save(root/'campaign.json',final);print('EXT_CAMPAIGN '+json.dumps({'status':status,'completed':len(results),'planned':len(schedule)}),flush=True)

if __name__=='__main__':main()
