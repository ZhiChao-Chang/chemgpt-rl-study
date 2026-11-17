"""Verify original metrics plus intervention arithmetic from a completed run ledger."""
import json,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path[:0]=[str(ROOT/'src'),str(BASE/'vendor')]
import study
from tokenizers import Tokenizer
run=Path(sys.argv[1]);d=json.loads((run/'result.json').read_text());cfg=json.loads((run/'config.json').read_text());s=cfg['settings']
tok=Tokenizer.from_file(str(BASE.parent/'proposal/implementation/chemgpt/tokenizer.json'));replay=study.Scorer(d['task'],tok);max_replay_error=0.
check=subprocess.run([sys.executable,str(BASE/'verify_ledgers.py'),str(run)],capture_output=True,text=True,check=True)
for file in ['search.jsonl','evaluation.jsonl']:
    for line in (run/file).read_text().splitlines():
        r=json.loads(line);assert 0<=r['reward']<=1 and 0<=r['tpsa_factor']<=1
        original=replay.decode(r['token_ids'],r['natural_eos'])
        assert original['valid']==r['valid'] and original['identity']==r['identity']
        max_replay_error=max(max_replay_error,abs(original['reward']-r['reward']))
        assert max_replay_error<1e-10
        assert abs(r['length_penalty']-s.get('length_cost',0.)*r['length'])<1e-12
        assert abs(r['transformed_mpo']-r['reward']*r['tpsa_factor'])<1e-12
        assert abs(r['training_reward']-(r['transformed_mpo']-r['length_penalty']))<1e-12
        if s.get('transform','raw')=='raw':assert r['tpsa_factor']==1
ev=[json.loads(l) for l in (run/'evaluation.jsonl').read_text().splitlines()]
assert abs(sum(r['training_reward'] for r in ev)/len(ev)-d['terminal_mean_training_reward'])<1e-12
assert abs(sum(r['trajectory_logratio'] for r in ev)/len(ev)-d['terminal_mean_sample_trajectory_kl'])<1e-10
assert d['beta']==s['beta_override'] and d['length_cost']==s.get('length_cost',0)
print(json.dumps({'status':'PASS','run':str(run),'phase':d['phase'],'method':d['method'],'seed':d['seed'],'original_metric_check':json.loads(check.stdout)['status'],'exact_token_reward_replay_max_error':max_replay_error}))
