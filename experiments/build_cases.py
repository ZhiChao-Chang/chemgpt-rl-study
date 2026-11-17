"""Source-backed case gallery and full-pair terminal diagnostics; no training or selection."""
import collections
import hashlib
import html
import json
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
import study
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem,Crippen,Descriptors,rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

OUT=ROOT/'evidence/cases';OUT.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'collected_remote/results/main'
LABELS={'fexofenadine':['AP similarity / 0.8 (clipped)','TPSA >= 90 modifier','logP <= 4 modifier'],
        'osimertinib':['FCFP4 similarity / 0.8 (clipped)','ECFP6 <= 0.85 modifier','TPSA >= 100 modifier','logP <= 1 modifier']}
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def describe(r):
    m=Chem.MolFromSmiles(r['identity'])
    return {'identity':r['identity'],'reward':r['reward'],'components':r['components'],
            'MW':Descriptors.MolWt(m),'logP':Crippen.MolLogP(m),'TPSA':rdMolDescriptors.CalcTPSA(m),
            'heavy_atoms':m.GetNumHeavyAtoms(),'aromatic_rings':rdMolDescriptors.CalcNumAromaticRings(m),
            'HBD':rdMolDescriptors.CalcNumHBD(m),'HBA':rdMolDescriptors.CalcNumHBA(m),'token_length':r['length'],
            'source_row':r['_row'],'source_ledger':r['_ledger']}
def draw(smiles):
    name='molecule_'+hashlib.sha256(smiles.encode()).hexdigest()[:20]
    m=Chem.MolFromSmiles(smiles)
    d=rdMolDraw2D.MolDraw2DSVG(650,350);d.drawOptions().padding=.08;d.DrawMolecule(m);d.FinishDrawing()
    (OUT/(name+'.svg')).write_text(d.GetDrawingText())
    if not (OUT/'example.png').exists():
        d2=rdMolDraw2D.MolDraw2DCairo(900,480);d2.DrawMolecule(m);d2.FinishDrawing();(OUT/'example.png').write_bytes(d2.GetDrawingText())
    return name+'.svg'
def quantiles(xs):
    s=sorted(xs)
    if not s:return None
    return {'mean':statistics.mean(s),'p10':s[round(.1*(len(s)-1))],'median':statistics.median(s),'p90':s[round(.9*(len(s)-1))]}

result=[];sections=[]
for path in sorted(SRC.glob('*/result.json')):
    base=path.parent;summary=json.loads(path.read_text());task=summary['task'];method=summary['method'];seed=summary['seed']
    ledgers={}
    for which in ['search','evaluation']:
        rs=read(base/(which+'.jsonl'))
        for i,r in enumerate(rs):r.update(_row=i+1,_ledger=which+'.jsonl')
        ledgers[which]=rs
    search=ledgers['search'];ev=ledgers['evaluation'];valid=[r for r in ev if r['valid']]
    # Recompute component scores using the previously validated pinned-formula adapter.
    scorer=study.Scorer(task,None);max_error=0.
    for r in valid:
        c=scorer.components(Chem.MolFromSmiles(r['identity']))
        max_error=max(max_error,max(abs(a-b) for a,b in zip(c,r['components'])))
    audit_path=ROOT/'evidence/representation_audit'/f'{base.name}.json'
    exact_replay=False
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        exact_replay=audit['exact_token_replay']=='PASS' and all(audit['ledger_sha256'][k]==digest(base/(k+'.jsonl')) for k in ['search','evaluation'])
    if max_error>1e-10 and not exact_replay:raise RuntimeError(f'canonical descriptor sensitivity without verified token replay {base.name}: {max_error}')
    distinct={r['identity']:r for r in search if r['valid']}
    best=sorted(distinct.values(),key=lambda r:(-r['reward'],r['identity']))[0] if distinct else None
    median=statistics.median(r['reward'] for r in valid) if valid else None
    typical=min(valid,key=lambda r:(abs(r['reward']-median),r['identity'],r['_row'])) if valid else None
    trade=sorted([r for r in distinct.values() if min(r['components'])<.5],key=lambda r:(-r['reward'],r['identity']))
    failure=next((r for r in search if not r['valid']),None)
    chosen={k:describe(r) if r else None for k,r in [('typical_terminal',typical),('best_discovered',best),('component_tradeoff',trade[0] if trade else None)]}
    freq=collections.Counter(r['identity'] for r in valid);ids=sorted(freq)
    first={}
    for r in valid:first.setdefault(r['identity'],r)
    fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,nBits=2048) for s in ids]
    sim_sum=0.;weighted_distance=0.;pair_count=0;closest=None;max_sim=-1.
    for i in range(1,len(ids)):
        similarities=DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        sim_sum+=sum(similarities);pair_count+=len(similarities)
        for j,sim in enumerate(similarities):
            weighted_distance+=freq[ids[i]]*freq[ids[j]]*(1-sim)
            if sim>max_sim:max_sim=sim;closest=(ids[j],ids[i])
    properties=[describe(r) for r in valid]
    diagnostics={'valid_count':len(valid),'unique_count':len(ids),'all_unique_pair_count':pair_count,
                 'unique_scaffolds_including_acyclic_bucket':summary['evaluation']['unique_scaffolds'],
                 'acyclic_unique':summary['evaluation']['acyclic_unique'],
                 'fingerprint_diversity_all_unique_pairs':1-sim_sum/pair_count if pair_count else None,
                 'fingerprint_diversity_all_valid_attempt_pairs':weighted_distance/(len(valid)*(len(valid)-1)/2) if len(valid)>1 else None,
                 'reward_all_attempts':quantiles([r['reward'] for r in ev]),
                 'components_valid_outputs':{label:quantiles([r['components'][j] for r in valid]) for j,label in enumerate(LABELS[task])},
                 'properties_valid_outputs':{name:quantiles([r[name] for r in properties]) for name in ['MW','logP','TPSA','heavy_atoms','aromatic_rings','token_length']},
                 'component_saturation_fraction_valid':{label:sum(r['components'][j]>=1-1e-12 for r in valid)/len(valid) if valid else None for j,label in enumerate(LABELS[task])},
                 'token_lengths_all_attempts':quantiles([r['length'] for r in ev]),
                 'component_recompute_max_abs_error':max_error,
                 'exact_token_replay_verified':exact_replay,
                 'descriptor_basis':'Structural descriptors/diversity use canonical-SMILES reread; recorded reward/components use original token-decoded graph. Representation-sensitivity audit quantifies rare differences; original rewards unchanged.',
                 'no_repeat_observed':len(valid)==len(ids),
                 'failure_count_search':sum(not r['valid'] for r in search),
                 'joint_components_ge_08':sum(min(r['components'])>=.8 for r in valid)/len(ev)}
    chosen['closest_terminal_pair']={'similarity':max_sim,'molecules':[describe(first[s]) for s in closest]} if closest else None
    chosen['first_failure']={k:v for k,v in failure.items() if k not in ['components','_ledger']} if failure else None
    steps=read(base/'steps.jsonl');annotated=[];previous=0
    for st in steps:
        n=st['attempts']-previous;tokens=sum(r['length'] for r in search[previous:st['attempts']]);previous=st['attempts']
        annotated.append((st,n,tokens))
    late=annotated[-20:];count=sum(n for st,n,nt in late);token_count=sum(nt for st,n,nt in late)
    diagnostics['late_search_drift']={'batches':len(late),'attempts':count,'actions':token_count,
        'sample_trajectory_logratio_mean':sum(st['sample_trajectory_kl']*n for st,n,nt in late)/count if count else None,
        'full_vocabulary_prefix_kl_token_weighted_mean':sum(st['full_vocabulary_prefix_kl']*nt for st,n,nt in late)/token_count if token_count else None,
        'interpretation':'Last up-to-20 search batches, weighted by actual batch size/actions. Mixture of successive policies BEFORE their updates; not final-policy molecular KL.'}
    entry={'run':base.name,'task':task,'method':method,'seed':seed,'source_sha256':{which:digest(base/(which+'.jsonl')) for which in ledgers},
           'selection_protocol_sha256':digest(ROOT/'case_protocol.json'),'source_result_sha256':digest(path),'cases':chosen,'diagnostics':diagnostics}
    result.append(entry)
    cards=[]
    for role in ['typical_terminal','best_discovered','component_tradeoff']:
        r=chosen[role]
        duplicate_tradeoff=(role=='component_tradeoff' and r is not None and chosen['best_discovered'] is not None and r['identity']==chosen['best_discovered']['identity'])
        if duplicate_tradeoff:continue
        if not r:cards.append('<article><h3>'+role+'</h3><p>无符合预定规则的样本。</p></article>');continue
        asset=draw(r['identity']);components=' · '.join(f'{html.escape(k)}: {v:.3f}' for k,v in zip(LABELS[task],r['components']))
        title=role
        if role=='best_discovered' and chosen['component_tradeoff'] is not None and chosen['component_tradeoff']['identity']==r['identity']:
            title+=' · also component_tradeoff'
        source=f'../../collected_remote/results/main/{base.name}/{r["source_ledger"]}'
        cards.append(f"<article><h3>{title}</h3><img src='{asset}'><p><b>MPO {r['reward']:.4f}</b> · MW {r['MW']:.1f} · logP {r['logP']:.2f} · TPSA {r['TPSA']:.1f}</p><p>{components}</p><small><a href='{source}'>{r['source_ledger']}</a> 第 {r['source_row']} 行</small><details><summary>canonical SMILES</summary><code>{html.escape(r['identity'])}</code></details></article>")
    if closest:
        pair=chosen['closest_terminal_pair'];images=''.join(f"<img src='{draw(r['identity'])}'>" for r in pair['molecules'])
        cards.append(f"<article><h3>closest_terminal_pair</h3>{images}<p>Tanimoto {max_sim:.4f}；这是最相似的一对，不代表整体分布。</p></article>")
    fail=chosen['first_failure']
    cards.append('<article><h3>first_failure</h3><pre>'+html.escape(json.dumps(fail,ensure_ascii=False,indent=2) if fail else '本次搜索没有无效或截断样本；不虚构失败案例。')+'</pre></article>')
    div=diagnostics['fingerprint_diversity_all_unique_pairs'];div_text=f'{div:.4f}' if div is not None else '无法估计（不足两个独特分子）'
    stats=f"全体独特终态分子的平均指纹距离：{div_text}（{pair_count:,} 对，无 256 个样本截断）；有效输出 {len(valid)}/{len(ev)}，独特分子 {len(ids)}。"
    sections.append(f"<section><h2>{html.escape(base.name)}</h2><p>{stats}</p><div class='grid'>{''.join(cards)}</div></section>")
contract=json.loads((ROOT/'contract.json').read_text())['main']
expected={f'{t}__{m}__s{s}' for t in contract['tasks'] for m in contract['methods'] for s in contract['seeds']}
missing=sorted(expected-{r['run'] for r in result})
payload={'status':'PARTIAL_MAIN_CASES' if missing else 'ALL_MAIN_CASES','completed_raw_runs':len(result),'planned_main_runs':18,'awaiting_raw_ledgers':missing,
         'scope':'Descriptive selection with declared rules; not independent efficacy validation or hyperparameter selection','runs':result}
(OUT/'cases.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>分子案例与失败分析</title><style>body{font:16px/1.8 system-ui;max-width:1300px;margin:30px auto;padding:20px;color:#26354a}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}article{border:1px solid #dce2ea;border-radius:10px;padding:16px;overflow-wrap:anywhere}img{width:100%;height:auto}pre{white-space:pre-wrap;max-height:360px;overflow:auto;font-size:12px}h3{font-size:16px}section{margin:36px 0;padding-top:10px;border-top:2px solid #dce2ea}aside{padding:20px;background:#eef3fa}small{color:#64748b}code{overflow-wrap:anywhere}</style><h1>分子案例：典型输出、最好候选、性质取舍与失败</h1>'''
page+=f'<p>已覆盖取得完整账本的 {len(result)}/18 次正式运行。其余尚未纳入，不能视为失败或零分。</p>'
page+='<aside>规则在已知首组汇总性能后、查看和挑选具体分子前制定，属于探索性案例分析，不是主实验前预注册。全部方法与种子陆续展示。典型案例取有效输出奖励中位数附近；最好候选和最近邻对明确属于极值。属性均为计算代理值，不代表结合、药效、安全性、可合成性或新颖性。结构图由 RDKit 从原始 canonical SMILES 绘制。</aside>'
page+='<p><a href="cases.json">完整案例与来源哈希</a> · <a href="../../case_protocol.json">预定选择规则</a>。所有分子和分量可追溯到原始账本行号。</p>'
page+='<p>评分按原始token解码图保留，图示结构和结构描述符按canonical SMILES重读。少数复杂环结构存在表示敏感性；<a href="../representation_audit/summary.json">完整token回放与敏感性审计</a>已量化其影响，没有删除样本或替换原评分。</p>'
if missing:page+='<details><summary>尚待取得完整原始账本的运行</summary><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in missing)+'</ul></details>'
page+=''.join(sections)+'</html>'
(OUT/'index.html').write_text(page)
print(json.dumps({'status':payload['status'],'runs':len(result),'gallery':str(OUT/'index.html')}))
