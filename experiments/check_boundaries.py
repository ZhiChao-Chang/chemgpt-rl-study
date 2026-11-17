"""Real-model small budget/tail cases and deterministic reward-cache semantics."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'vendor'),str(ROOT/'src')]
import study
import torch
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
torch.set_num_threads(2);torch.set_num_interop_threads(1)
base=ROOT.parent/'proposal/implementation/chemgpt'
tok=Tokenizer.from_file(str(base/'tokenizer.json'))
src=ROOT/'results/cpu_gate/development/perindopril__frozen__s11/search.jsonl'
row=next(json.loads(l) for l in src.read_text().splitlines() if json.loads(l)['valid'])
s=study.Scorer('perindopril',tok)
a=s.decode(row['token_ids'],True);b=s.decode(row['token_ids'],True);c=s.decode(row['token_ids'],False)
assert a['reward']==b['reward'] and not a['cache_hit'] and b['cache_hit'] and s.physical_calls==1
assert c['reward']==0 and not c['valid'] and c['failure']=='truncated'
model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.float32,weights_only=True).eval()
contract=json.loads((ROOT/'contract.json').read_text());out=ROOT/'results/boundary_tests';out.mkdir(exist_ok=False)
cases=[]
for n in [1,5]:
    lim={'unique_budget':n,'max_attempts':n,'max_updates':1,'evaluation_attempts':1,'run_wall_seconds':120}
    d=study.run_one(model,tok,torch.device('cpu'),out,'perindopril','reinforce_kl',900+n,lim,contract['fixed_training'],time.monotonic()+180)
    assert d['attempts']==n and d['unique_queries']<=n and d['updates']==1 and d['evaluation']['attempts']==1
    ck=torch.load(out/f'perindopril__reinforce_kl__s{900+n}/checkpoint.pt',weights_only=False,map_location='cpu')
    assert all(k in ck for k in ['actor','optimizer','python_rng','numpy_rng','torch_rng','seen','curve'])
    cases.append({'tail_batch':n,'attempts':d['attempts'],'unique':d['unique_queries'],'status':'PASS'})
result={'status':'PASS','cache_duplicate_preserves_reward':True,'truncation_zero':True,'cases':cases,'checkpoint':'contains model/Adam/RNG/ledger state; no automatic cross-worker recovery claimed'}
(ROOT/'evidence/boundary_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
