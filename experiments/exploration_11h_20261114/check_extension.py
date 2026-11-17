"""Real-model edge test for distinct KL/length/reward interventions."""
import json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path[:0]=[str(ROOT/'src'),str(BASE/'vendor')]
import study
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM
torch.set_num_threads(2);torch.set_num_interop_threads(1)
r={'reward':.6,'length':100,'valid':False,'identity':None}
assert study.objective_terms(r,{'length_cost':.003})['training_reward']==.3
r2={'reward':.6,'length':10,'valid':True,'identity':'CCO'}
a=study.objective_terms(r2,{'transform':'upper_tpsa','upper_tpsa':150,'tpsa_sigma':30});assert a['tpsa_factor']==1 and a['training_reward']==.6
b=study.objective_terms(r2,{'transform':'upper_tpsa','upper_tpsa':0,'tpsa_sigma':30})
assert 0<b['training_reward']<.6 and r2['reward']==.6
assert study.objective_terms({'reward':0.,'length':256,'valid':False},{'length_cost':.003})['training_reward']==-.768
c=json.loads((ROOT/'contract.json').read_text());md=BASE.parent/'proposal/implementation/chemgpt'
model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,torch_dtype=torch.float32,weights_only=True).eval();tok=Tokenizer.from_file(str(md/'tokenizer.json'))
out=ROOT/'evidence/edge_runs';out.mkdir(exist_ok=False)
for i,cond in enumerate([c['phases'][0]['conditions'][2],c['phases'][1]['conditions'][3],c['phases'][1]['conditions'][5]]):
    settings={**c['fixed_training'],**cond,'beta_override':cond['beta'],'phase_name':'edge'}
    limits={'unique_budget':5,'max_attempts':5,'max_updates':1,'evaluation_attempts':4,'run_wall_seconds':120}
    z=study.run_one(model,tok,torch.device('cpu'),out,'fexofenadine',cond['id'],9001+i,limits,settings,time.monotonic()+180)
    assert z['beta']==cond['beta'] and z['length_cost']==cond['length_cost'] and z['status']=='COMPLETE'
    ev=[json.loads(x) for x in (out/f"fexofenadine__{cond['id']}__s{9001+i}"/'evaluation.jsonl').read_text().splitlines()]
    assert len(ev)==4 and all(abs(x['training_reward']-(x['reward']*x['tpsa_factor']-x['length_penalty']))<1e-12 for x in ev)
    assert abs(z['terminal_mean_training_reward']-sum(x['training_reward'] for x in ev)/4)<1e-10
    assert z['terminal_full_vocabulary_prefix_kl']>=-1e-6
res={'status':'PASS','scope':'three real-model single-update branches, original MPO separated from transformed training return, terminal KL recorded','planned_runs':sum(len(p['tasks'])*len(p['seeds'])*len(p['conditions']) for p in c['phases'])}
(ROOT/'evidence/extension_gate.json').write_text(json.dumps(res,indent=2));print(json.dumps(res))
