"""Independent recomputation from raw attempts; no imports from training code."""
import heapq
import json
from pathlib import Path
import statistics
import sys

root=Path(sys.argv[1]);reports=[]
for p in sorted(root.rglob('result.json')):
    d=json.loads(p.read_text());out=p.parent;seen={};heap=[];ys=[];attempts=0
    for l in (out/'search.jsonl').read_text().splitlines():
        r=json.loads(l);attempts+=1;assert r['attempt']==attempts
        fresh=r['valid'] and r['identity'] not in seen
        assert fresh==r['new_unique']
        if not r['valid']:assert r['reward']==0
        if fresh:
            seen[r['identity']]=r['reward'];heapq.heappush(heap,r['reward'])
            if len(heap)>10:heapq.heappop(heap)
            ys.append(sum(heap)/10)
        assert len(seen)==r['unique_queries']
    assert attempts==d['attempts'] and len(seen)==d['unique_queries']
    auc=sum(ys)/len(ys) if ys else 0
    assert abs(auc-d['auc_achieved'])<1e-12
    assert abs((ys[-1] if ys else 0)-d['top10'])<1e-12
    ev=[json.loads(l) for l in (out/'evaluation.jsonl').read_text().splitlines()]
    assert len(ev)==d['evaluation']['attempts']
    assert abs(statistics.mean(r['reward'] for r in ev)-d['evaluation']['mean_reward'])<1e-12
    assert abs(sum(r['valid'] for r in ev)/len(ev)-d['evaluation']['validity'])<1e-12
    assert len({r['identity'] for r in ev if r['valid']})==d['evaluation']['unique_valid']
    steps=[json.loads(l) for l in (out/'steps.jsonl').read_text().splitlines()]
    assert sum('loss' in r for r in steps)==d['updates']
    if d['method']=='frozen':assert d['updates']==0
    config=json.loads((out/'config.json').read_text())
    assert len(seen)<=config['limits']['unique_budget']
    assert attempts<=config['limits']['max_attempts']
    reports.append(dict(path=str(out),status='PASS',attempts=attempts,unique=len(seen),evaluation=len(ev)))
assert reports,'No completed runs to verify'
print(json.dumps({'status':'PASS','verified_runs':len(reports),'runs':reports},indent=2))
