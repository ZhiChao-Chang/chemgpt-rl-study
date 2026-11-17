"""Build a reviewed browser snapshot from immutable local RL evidence."""
import base64,collections,hashlib,heapq,json,math,statistics
from datetime import datetime,timezone
from pathlib import Path
from rdkit import Chem,RDLogger
from rdkit.Chem import Crippen,Descriptors,rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog('rdApp.*')
BASE=Path(__file__).resolve().parents[1];EXT=BASE/'exploration_11h_20261114';APP=BASE/'visual_report'
ASSETS=APP/'src/content/assets';ASSETS.mkdir(parents=True,exist_ok=True)
PHASES={'core':'基础对照','kl_dose':'KL 强度','mechanism':'机制干预','long_budget':'大预算'}
LABELS={'frozen':'冻结采样','reinforce':'REINFORCE','reinforce_kl':'REINFORCE + KL','rf':'REINFORCE',
 'kl_0003':'KL β=0.003','kl_001':'KL β=0.01','kl_003':'KL β=0.03','kl_01':'KL β=0.1',
 'length_0001':'长度惩罚 0.001','length_0003':'长度惩罚 0.003','upper_tpsa150':'TPSA 软约束'}
CORE_CASES={r['run']:r for r in json.loads((BASE/'evidence/cases/cases.json').read_text())['runs']}
DETAILS=json.loads((EXT/'evidence/detail_metrics.json').read_text())
RUNS=[];CURVES=[];CDF=[];CASE_ROWS=[];MOLECULES={};FILES=[];AUC_ERRORS=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open() as f:
        for line in f:yield json.loads(line)
def draw(smiles):
    m=Chem.MolFromSmiles(smiles);d=rdMolDraw2D.MolDraw2DSVG(760,410);d.drawOptions().padding=.08;d.DrawMolecule(m);d.FinishDrawing()
    return d.GetDrawingText()
def add_case(r,phase,task,method,seed,role,line,ledger):
    if not r:return
    ident=r['identity'];m=Chem.MolFromSmiles(ident);mol_id=hashlib.sha256(ident.encode()).hexdigest()[:20]
    MOLECULES.setdefault(mol_id,{'smiles':ident,'svg':draw(ident)})
    CASE_ROWS.append(dict(id=f'{phase}:{task}:{method}:{seed}:{role}',molId=mol_id,phase=phase,task=task,condition=method,methodLabel=LABELS[method],seed=seed,role=role,
      reward=r['reward'],components=r['components'],MW=Descriptors.MolWt(m),TPSA=rdMolDescriptors.CalcTPSA(m),logP=Crippen.MolLogP(m),actions=r.get('length',r.get('token_length')),
      smiles=ident,trainingReward=r.get('training_reward'),tpsaFactor=r.get('tpsa_factor'),sourceRow=line,sourceLedger=ledger,
      descriptorBasis='canonical SMILES reread; original recorded MPO preserved'))

roots=[('core',BASE/'collected_remote/results/main'),(None,EXT/'collected/results')]
for fixed,root in roots:
 for p in sorted(root.rglob('result.json')):
    r=json.loads(p.read_text());cfg=json.loads((p.parent/'config.json').read_text());phase=fixed or r['phase'];task=r['task'];method=r['method'];seed=r['seed']
    if r['status']!='COMPLETE':raise RuntimeError('incomplete run '+str(p))
    ident=f'{phase}:{task}:{method}:{seed}';ev=list(read(p.parent/'evaluation.jsonl'));valid=[x for x in ev if x['valid']]
    if phase=='core':
        detail=CORE_CASES[p.parent.name]['diagnostics'];props=detail['properties_valid_outputs']
        mean_actions=statistics.mean(x['length'] for x in ev);mw=props['MW']['mean'];tpsa=props['TPSA']['mean'];div=detail['fingerprint_diversity_all_unique_pairs']
    else:
        detail=DETAILS[f'{phase}/{p.parent.name}'];mean_actions=detail['mean_actions_all_attempts'];mw=detail['valid_property_means']['MW'];tpsa=detail['valid_property_means']['TPSA'];div=detail['diversity_unique_full_pairs']
    row=dict(id=ident,phase=phase,phaseLabel=PHASES[phase],task=task,condition=method,methodLabel=LABELS[method],seed=seed,
       beta=cfg['beta'],uniqueBudget=cfg['limits']['unique_budget'],queries=r['unique_queries'],attempts=r['attempts'],updates=r['updates'],evaluationAttempts=len(ev),
       terminalReward=r['evaluation']['mean_reward'],auc=r['auc_achieved'],aucFixed=r['auc_fixed_budget_carry_forward'],top10=r['top10'],maxReward=r['max_reward'],
       validity=r['evaluation']['validity'],repeat=r['evaluation']['repeat_fraction_among_valid'],meanActions=mean_actions,meanMW=mw,meanTPSA=tpsa,
       diversity=div,terminalKL=r.get('terminal_mean_sample_trajectory_kl'),trainingReward=r.get('terminal_mean_training_reward'),wallSeconds=r['search_wall_s'],stopReason=r['stop_reason'],
       sourceResult=str(p.relative_to(BASE)),resultSha256=digest(p))
    RUNS.append(row);FILES.append({'name':str(p.relative_to(BASE)),'sha256':digest(p)})
    assert len(ev)==r['evaluation']['attempts']
    assert abs(statistics.mean(x['reward'] for x in ev)-row['terminalReward'])<1e-12
    # Independently reconstruct discovery curve, including initial zero-padded top10.
    seen=set();heap=[];ys=[];best=None;bestline=None
    grid={1,10,25,50,100,250,500,*range(750,row['uniqueBudget']+1,250)};grid.add(row['uniqueBudget'])
    for line,x in enumerate(read(p.parent/'search.jsonl'),1):
        if not x['valid'] or x['identity'] in seen:continue
        seen.add(x['identity']);heapq.heappush(heap,x['reward'])
        if len(heap)>10:heapq.heappop(heap)
        value=sum(heap)/10;ys.append(value)
        if len(seen) in grid:CURVES.append(dict(runId=ident,phase=phase,task=task,condition=method,methodLabel=LABELS[method],seed=seed,queries=len(seen),score=value))
        if best is None or (-x['reward'],x['identity'])<(-best['reward'],best['identity']):best=x;bestline=line
    error=abs(sum(ys)/len(ys)-row['auc']);assert error<1e-12;AUC_ERRORS.append(error)
    rewards=sorted(x['reward'] for x in ev)
    for i in range(41):
        threshold=i/40;fraction=sum(v<=threshold for v in rewards)/len(rewards)
        CDF.append(dict(runId=ident,phase=phase,task=task,condition=method,methodLabel=LABELS[method],seed=seed,threshold=threshold,cumulativeFraction=fraction))
    med=statistics.median(x['reward'] for x in valid)
    ix,typ=min(((i,x) for i,x in enumerate(ev,1) if x['valid']),key=lambda a:(abs(a[1]['reward']-med),a[1]['identity'],a[0]))
    add_case(typ,phase,task,method,seed,'typical',ix,'evaluation.jsonl');add_case(best,phase,task,method,seed,'best',bestline,'search.jsonl')
assert len(RUNS)==126 and sum(r['phase']=='core' for r in RUNS)==18
groups=collections.defaultdict(list)
for r in RUNS:groups[r['phase'],r['task'],r['condition']].append(r)
SUMMARY=[]
for (phase,task,method),rows in groups.items():
    d={'phase':phase,'phaseLabel':PHASES[phase],'task':task,'condition':method,'methodLabel':LABELS[method],'n':len(rows),'beta':rows[0]['beta'],'uniqueBudget':rows[0]['uniqueBudget']}
    for k in ['terminalReward','auc','aucFixed','top10','validity','meanActions','meanMW','meanTPSA','diversity','terminalKL','trainingReward']:
        vs=[r[k] for r in rows if r[k] is not None];d[k]=statistics.mean(vs) if vs else None;d[k+'SD']=statistics.stdev(vs) if len(vs)>1 else None
    SUMMARY.append(d)

files=FILES+[{'name':'case_protocol.json','sha256':digest(BASE/'case_protocol.json')},{'name':'exploration_11h_20261114/contract.json','sha256':digest(EXT/'contract.json')}]
def source(title,definitions,subset=None):
    return {'provider':'ME2502 project experiment artifacts','title':title,'files':subset or files,'metricDefinitions':definitions,
       'evidenceFlow':[{'title':'原始文件读取','detail':'读取所有已完成运行的result/config/search/evaluation JSON；保留文件身份及SHA256。'},
                       {'title':'独立复算','detail':'从首次有效canonical分子重新构建top10轨迹与AUC；终态均值由全体独立采样（含无效零分）复算。'},
                       {'title':'分组','detail':'同一阶段、任务、条件下按独立训练种子计算均值与样本标准差；不同预算和队列不混合。'}]}
snapshot=json.loads((APP/'src/data.json').read_text());snapshot.update(title='ChemGPT 强化学习研究',surface='report',status='computed',generatedAt=datetime.now(timezone.utc).isoformat(),buildStatus='creating',filters=[],
 report={'asOf':'2026-11-15','presentationOnly':True},study={'coreRuns':18,'extensionRuns':108,'totalRuns':126,'developmentRuns':6,'tasks':2,'seedCounts':{'core':3,'kl_dose':5,'mechanism':5,'long_budget':3},
  'experimentStatus':'全部计算与独立验收完成','finalReportStatus':'待撰写定稿','slidesStatus':'待制作正式演示稿','rehearsalStatus':'待三名成员实际演练',
  'model':'ncfrey/ChemGPT-4.7M','revision':'7438a282460b3038e17a27e25b85b1376e9a23e2','cutoff':'2026-11-15','referenceRepo':'https://github.com/ZhiChao-Chang/canine-cd20-design'},queries={
 'runs':{'rows':RUNS,'source':source('逐种子实验结果',[{'label':'终态平均奖励','definition':'训练结束后独立生成的全部尝试平均原始MPO；无效/截断为0。'},{'label':'AUC','definition':'top10独特候选平均奖励对首次有效canonical查询数的右端点积分除以预算；不足十个时补0。'},{'label':'统计单位','definition':'训练种子，而不是单个终态样本。'}])},
 'summary':{'rows':SUMMARY,'source':source('同协议种子均值',[{'label':'均值与SD','definition':'每个阶段×任务×条件内，独立种子的算术均值与样本标准差。SD不是置信区间。'}])},
 'curves':{'rows':CURVES,'source':source('候选发现曲线',[{'label':'查询','definition':'每次运行首次遇见的有效canonical分子；重复与无效另记attempt。'}])},
 'cdf':{'rows':CDF,'source':source('独立采样奖励分布',[{'label':'累计比例','definition':'每次运行奖励不高于横轴阈值的全部终态采样尝试比例。'}])},
 'cases':{'rows':CASE_ROWS,'source':source('固定规则分子案例',[{'label':'典型输出','definition':'有效终态样本中最接近奖励中位数的样本。不是所有性质的多维中位数。'},{'label':'最高分候选','definition':'搜索中最高原MPO的独特分子，平局按canonical字符串。'},{'label':'描述符','definition':'结构图、MW/TPSA/logP使用canonical重读；原评分保留。表示敏感性独立审计，无样本删除。'}])}})
(APP/'src/data.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
(ASSETS/'molecules.json').write_text(json.dumps(MOLECULES,ensure_ascii=False))
receipt={'source_runs':len(RUNS),'case_rows':len(CASE_ROWS),'curves':len(CURVES),'max_independent_auc_error':max(AUC_ERRORS),'sources':files,'source_scope':'18 original main +108 extension; development excluded from comparisons','reference_style':'canine-cd20-design; public report and local authored presentation code reviewed; no antibody experiment data copied'}
(BASE/'visualization_work/snapshot_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k not in ['sources']},ensure_ascii=False))
