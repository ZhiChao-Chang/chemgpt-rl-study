"""Keep empirical observations, mathematical facts and hypotheses distinct."""
import hashlib,html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
cases=json.loads((ROOT/'evidence/cases/cases.json').read_text())['runs']
claims=[
 {'id':'T1','type':'mathematical_identity_and_exact_check','claim':'固定确定性解码下，轨迹KL分解为输出KL与条件编码KL。','evidence':'evidence/theory.json','boundary':'标准KL链式法则的项目应用；不声称新通用定理或已估计全分子KL。'},
 {'id':'T2','type':'mathematical_counterexample','claim':'优化期望奖励不一般保证best-of-B、top-k或独特候选发现改善。','evidence':'evidence/objective_mismatch.json','boundary':'例子证明不存在一般蕴含；不直接证明真实训练的具体机制。'}]
observations=[]
for task,seed in sorted({(r['task'],r['seed']) for r in cases}):
    rows={r['method']:r for r in cases if r['task']==task and r['seed']==seed}
    if set(rows)!={'frozen','reinforce','reinforce_kl'}:continue
    obs={'task':task,'seed':seed,'methods':{}}
    for method,r in rows.items():
        result=json.loads((ROOT/'collected_remote/results/main'/r['run']/'result.json').read_text());d=r['diagnostics']
        obs['methods'][method]={
          'search_auc':result['auc_achieved'],'terminal_mean_reward':result['evaluation']['mean_reward'],
          'median_MW':d['properties_valid_outputs']['MW']['median'],
          'median_TPSA':d['properties_valid_outputs']['TPSA']['median'],
          'median_tokens':d['properties_valid_outputs']['token_length']['median'],
          'fingerprint_diversity_all_unique_pairs':d['fingerprint_diversity_all_unique_pairs'],
          'unique_scaffolds':d['unique_scaffolds_including_acyclic_bucket'],
          'validity':result['evaluation']['validity'],'repetition_among_valid':result['evaluation']['repeat_fraction_among_valid'],
          'late_search_drift':d['late_search_drift'],'normalized_components':d['components_valid_outputs'],
          'source_result_sha256':r['source_result_sha256']}
    observations.append(obs)
claims.extend([
 {'id':'E1','type':'empirical_observation_pending_full_matrix','claim':'比较终态均值提升与搜索AUC提升是否同幅度。','evidence':'evidence/claim_ledger.json#matched_seed_observations','boundary':'全部匹配种子逐项列出；不以单种子或样本级伪重复声称显著。'},
 {'id':'E2','type':'exploratory_empirical_diagnostic','claim':'检查奖励分量改善与MW、TPSA和生成长度漂移是否共同出现。','evidence':'evidence/cases/cases.json','boundary':'性质诊断是观察，不把分量均值变化当成总奖励的精确因果分解。'},
 {'id':'E3','type':'exploratory_empirical_diagnostic','claim':'比较KL对策略偏离和输出性质漂移的影响，同时单独报告各多样性指标。','evidence':'evidence/cases/cases.json','boundary':'训练末期KL来自相继策略的最后若干batch，不是终态分子分布KL；不能据此保证所有多样性指标改善。'},
 {'id':'E4','type':'exploratory_conditional_summary','claim':'在共同支持的粗长度分箱内，检查不同方法的奖励差异。','evidence':'evidence/length_analysis.html','boundary':'长度是策略影响的结果，分箱不是因果调整；保留覆盖率与残余差异说明。'},
 {'id':'H1','type':'hypothesis_not_established','claim':'模型可能更多利用了奖励中容易满足且无上限的性质，而未同等改善目标相似度。','evidence':'evidence/cases/index.html','boundary':'需要跨种子、跨任务及聚焦干预；现有分析不足以称为reward hacking或证明因果。'}])
data={'status':'PROVISIONAL','matched_seed_count':len(observations),'claim_records':claims,'matched_seed_observations':observations,
      'historical_hypothesis':'最初提出重复塌缩的可能性；必须报告实际重复率，若未出现则保留这个负结果，不把性质漂移偷换成已证明的重复塌缩。',
      'case_protocol_sha256':hashlib.sha256((ROOT/'case_protocol.json').read_bytes()).hexdigest()}
(ROOT/'evidence/claim_ledger.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
cards=''.join(f"<section><h2>{c['id']} · {html.escape(c['type'])}</h2><p>{html.escape(c['claim'])}</p><p><b>边界：</b>{html.escape(c['boundary'])}</p><a href='../{html.escape(c['evidence'],quote=True)}'>证据</a></section>" for c in claims)
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>主张与证据台账</title><style>body{font:16px/1.85 system-ui;max-width:950px;margin:40px auto;padding:20px;color:#26354a}section{padding:20px 0;border-top:1px solid #ddd}h2{font-size:18px}</style><h1>每个结论能说到哪一步？</h1><p>数学事实、实现检查、经验观察和未证实解释分开记录。当前匹配种子组数：'+str(len(observations))+'。</p>'+cards+'</html>'
(ROOT/'evidence/claim_ledger.html').write_text(page)
print(json.dumps({'matched_seed_groups':len(observations),'claims':len(claims)}))
