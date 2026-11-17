#!/usr/bin/env python3
"""Rebuild presentation numerics from the curated research corpus; never train."""
import collections,csv,hashlib,heapq,json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
APP=ROOT/'experiments/visual_report'
LABELS={'frozen':'冻结采样','reinforce':'REINFORCE','reinforce_kl':'REINFORCE + KL','rf':'REINFORCE','kl_0003':'KL β=0.003','kl_001':'KL β=0.01','kl_003':'KL β=0.03','kl_01':'KL β=0.1','length_0001':'长度惩罚 0.001','length_0003':'长度惩罚 0.003','upper_tpsa150':'TPSA 软约束'}
PHASES={'core':'基础对照','kl_dose':'KL 强度','mechanism':'机制干预','long_budget':'大预算'}

def read_verified(spec):
 p=(ROOT/spec['path']).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Source path escapes research repository')
 raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Source identity changed: '+spec['path'])
 return raw

def records(raw):
 return [json.loads(line) for line in raw.splitlines() if line.strip()]

def main():
 manifest_path=ROOT/'data/study_manifest.json';manifest=json.loads(manifest_path.read_text())
 snap=json.loads((APP/'src/data.json').read_text());runs=[];curves=[];cdf=[];groups=collections.defaultdict(list)
 details={s['path']:json.loads(read_verified(s)) for s in manifest['descriptor_evidence']}
 core_details={r['run']:r['diagnostics'] for r in details['experiments/evidence/cases/cases.json']['runs']}
 ext_details=details['experiments/exploration_11h_20261114/evidence/detail_metrics.json']
 case_index=collections.defaultdict(list)
 for c in snap['queries']['cases']['rows']:
  case_index[c['phase'],c['task'],c['condition'],c['seed']].append(c)
 checked_cases=0;max_auc_error=0.
 for entry in manifest['runs']:
  files=entry['files'];r=json.loads(read_verified(files['result']));cfg=json.loads(read_verified(files['config']))
  ev=records(read_verified(files['evaluation']));search=records(read_verified(files['search']))
  phase,task,method,seed=(entry[k] for k in ['phase','task','condition','seed'])
  assert r['status']=='COMPLETE' and r['task']==task and r['method']==method and r['seed']==seed
  assert len(ev)==r['evaluation']['attempts']
  terminal_mean=statistics.mean(x['reward'] for x in ev)
  assert abs(terminal_mean-r['evaluation']['mean_reward'])<1e-12
  run_name=pathlib.Path(files['result']['path']).parent.name
  if phase=='core':
   detail=core_details[run_name];props=detail['properties_valid_outputs'];mw=props['MW']['mean'];tpsa=props['TPSA']['mean'];div=detail['fingerprint_diversity_all_unique_pairs']
  else:
   detail=ext_details[f'{phase}/{run_name}'];mw=detail['valid_property_means']['MW'];tpsa=detail['valid_property_means']['TPSA'];div=detail['diversity_unique_full_pairs']
  row=dict(id=entry['id'],phase=phase,phaseLabel=PHASES[phase],task=task,condition=method,methodLabel=LABELS[method],seed=seed,
   beta=cfg['beta'],uniqueBudget=cfg['limits']['unique_budget'],queries=r['unique_queries'],attempts=r['attempts'],updates=r['updates'],evaluationAttempts=len(ev),
   terminalReward=r['evaluation']['mean_reward'],auc=r['auc_achieved'],aucFixed=r['auc_fixed_budget_carry_forward'],top10=r['top10'],maxReward=r['max_reward'],
   validity=r['evaluation']['validity'],repeat=r['evaluation']['repeat_fraction_among_valid'],meanActions=statistics.mean(x['length'] for x in ev),meanMW=mw,meanTPSA=tpsa,
   diversity=div,terminalKL=r.get('terminal_mean_sample_trajectory_kl'),trainingReward=r.get('terminal_mean_training_reward'),wallSeconds=r['search_wall_s'],stopReason=r['stop_reason'],
   sourceResult=pathlib.Path(files['result']['path']).relative_to('experiments').as_posix(),resultSha256=files['result']['sha256'])
  seen=set();heap=[];ys=[];grid={1,10,25,50,100,250,500,*range(750,row['uniqueBudget']+1,250),row['uniqueBudget']}
  for x in search:
   if not x['valid'] or x['identity'] in seen:continue
   seen.add(x['identity']);heapq.heappush(heap,x['reward'])
   if len(heap)>10:heapq.heappop(heap)
   score=sum(heap)/10;ys.append(score)
   if len(seen) in grid:curves.append(dict(runId=row['id'],phase=phase,task=task,condition=method,methodLabel=LABELS[method],seed=seed,queries=len(seen),score=score))
  assert len(seen)==r['unique_queries']
  auc_error=abs(sum(ys)/len(ys)-row['auc']);max_auc_error=max(max_auc_error,auc_error);assert auc_error<1e-12
  for i in range(41):
   threshold=i/40;cdf.append(dict(runId=row['id'],phase=phase,task=task,condition=method,methodLabel=LABELS[method],seed=seed,threshold=threshold,cumulativeFraction=sum(x['reward']<=threshold for x in ev)/len(ev)))
  for c in case_index[phase,task,method,seed]:
   ledger=ev if c['sourceLedger']=='evaluation.jsonl' else search;x=ledger[c['sourceRow']-1]
   assert x['identity']==c['smiles'] and abs(x['reward']-c['reward'])<1e-12
   assert x['components']==c['components']
   checked_cases+=1
  runs.append(row);groups[phase,task,method].append(row)
 summary=[]
 for (phase,task,method),rs in groups.items():
  out={k:rs[0][k] for k in ['phase','phaseLabel','task','condition','methodLabel','beta','uniqueBudget']};out['n']=len(rs)
  for k in ['terminalReward','auc','aucFixed','top10','validity','meanActions','meanMW','meanTPSA','diversity','terminalKL','trainingReward']:
   vs=[r[k] for r in rs if r[k] is not None];out[k]=statistics.mean(vs) if vs else None;out[k+'SD']=statistics.stdev(vs) if len(vs)>1 else None
  summary.append(out)
 assert len(runs)==126 and checked_cases==252
 actual=dict(collections.Counter(r['phase'] for r in runs));assert actual==manifest['population']
 # Preserve authored prose, stable app identity and independent molecular assets.
 for key,value in [('runs',runs),('summary',summary),('curves',curves),('cdf',cdf)]:snap['queries'][key]['rows']=value
 snap['report']['asOf']=manifest['evidence_cutoff'];snap['study']['sourceManifest']='data/study_manifest.json';snap['study']['sourceManifestSha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
 (APP/'src/data.json').write_text(json.dumps(snap,ensure_ascii=False))
 for file,rs in [('run_metrics.csv',runs),('summary_metrics.csv',summary)]:
  with (ROOT/'data'/file).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 receipt={'schema':'chemgpt-data-build-v1','selected_runs':len(runs),'source_files_checked':len(runs)*4+len(manifest['descriptor_evidence']),'seed_groups':len(summary),'case_records_checked':checked_cases,'max_independent_auc_error':max_auc_error,'source_manifest_sha256':snap['study']['sourceManifestSha256'],'numerical_data_sha256':hashlib.sha256(json.dumps({'runs':runs,'summary':summary,'curves':curves,'cdf':cdf},sort_keys=True).encode()).hexdigest(),'operations':'read immutable records; recompute numerics; no training, model loading or cloud jobs'}
 (ROOT/'data/report_data_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
