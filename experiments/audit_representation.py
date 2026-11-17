"""Replay exact token ledgers; separately quantify canonical-SMILES reread sensitivity."""
import hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
import study
from tokenizers import Tokenizer
from rdkit import Chem
from rdkit.Chem import Crippen
tok=Tokenizer.from_file(str(ROOT.parent/'proposal/implementation/chemgpt/tokenizer.json'))
class ReplayScorer(study.Scorer):
    def components(self,mol):
        self.last_raw_mol=Chem.Mol(mol)
        return super().components(mol)
out=ROOT/'evidence/representation_audit';out.mkdir(exist_ok=True);all_reports=[];start=time.monotonic()
for p in sorted((ROOT/'collected_remote/results').rglob('result.json')):
    saved=json.loads(p.read_text());task=saved['task'];s=ReplayScorer(task,tok);counter_cache={};edges=[];mismatches=[];stats={}
    for kind in ['search','evaluation']:
        rows=[json.loads(l) for l in (p.parent/(kind+'.jsonl')).read_text().splitlines()]
        orig=[];canonical=[];seen={};seen_counter={}
        for i,r in enumerate(rows):
            is_new=r['valid'] and r['identity'] not in s.cache
            replay=s.decode(r['token_ids'],r['natural_eos'])
            err=abs(replay['reward']-r['reward'])
            if replay['valid']!=r['valid'] or replay['identity']!=r['identity'] or err>1e-10:
                mismatches.append({'ledger':kind,'row':i+1,'reward_error':err})
            orig.append(r['reward'])
            if not r['valid']:canonical.append(0.);continue
            ident=r['identity']
            if ident not in counter_cache:
                m=Chem.MolFromSmiles(ident);c=study.Scorer.components(s,m)
                cr=math.prod(c)**(1/len(c));counter_cache[ident]=cr
                delta=max(abs(a-b) for a,b in zip(c,r['components']))
                raw_logp=Crippen.MolLogP(s.last_raw_mol) if is_new else None
                if delta>1e-10:
                    edges.append({'ledger':kind,'row':i+1,'identity':ident,'component_delta':delta,'original_components':r['components'],'reread_components':c,
                                  'original_reward':r['reward'],'reread_reward':cr,'raw_logP':raw_logp,'reread_logP':Crippen.MolLogP(m),
                                  'canonical_idempotent':Chem.MolToSmiles(m,isomericSmiles=True)==ident})
            canonical.append(counter_cache[ident]);seen.setdefault(ident,r['reward']);seen_counter.setdefault(ident,counter_cache[ident])
        stats[kind]={'attempts':len(rows),'original_mean':sum(orig)/len(orig),'canonical_reread_mean':sum(canonical)/len(canonical),
                     'mean_absolute_reward_change':sum(abs(a-b) for a,b in zip(orig,canonical))/len(rows),
                     'max_absolute_reward_change':max(abs(a-b) for a,b in zip(orig,canonical)),
                     'original_unique_top10':sum(sorted(seen.values(),reverse=True)[:10])/10,
                     'reread_unique_top10':sum(sorted(seen_counter.values(),reverse=True)[:10])/10}
    report={'run':str(p.parent.relative_to(ROOT/'collected_remote/results')),'exact_token_replay':'PASS' if not mismatches else 'FAIL',
            'replay_mismatches':mismatches,'canonical_descriptor_edges':edges,'stats':stats,
            'ledger_sha256':{k:hashlib.sha256((p.parent/(k+'.jsonl')).read_bytes()).hexdigest() for k in ['search','evaluation']}}
    name=p.parent.name+'.json';(out/name).write_text(json.dumps(report,indent=2));all_reports.append(report)
    (out/'progress.json').write_text(json.dumps({'completed_runs':len(all_reports),'elapsed_s':time.monotonic()-start},indent=2))
    print(json.dumps({'run':report['run'],'exact_replay':report['exact_token_replay'],'canonical_edges':len(edges),'terminal_mean_abs_change':stats['evaluation']['mean_absolute_reward_change']}),flush=True)
summary={'status':'PASS' if all(r['exact_token_replay']=='PASS' for r in all_reports) else 'FAIL','checked_runs':len(all_reports),
         'edge_runs':sum(bool(r['canonical_descriptor_edges']) for r in all_reports),'edge_identities':sum(len(r['canonical_descriptor_edges']) for r in all_reports),
         'max_terminal_mean_absolute_change':max(r['stats']['evaluation']['mean_absolute_reward_change'] for r in all_reports),
         'boundary':'Exact token replay validates recorded protocol. Canonical reread is a counterfactual representation-sensitivity check, not a replacement of original rewards. No samples excluded.',
         'runs':[{k:r[k] for k in ['run','exact_token_replay','stats']} for r in all_reports]}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='runs'}),flush=True)
