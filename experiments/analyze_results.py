"""Descriptive seed-level summaries; never treats 1024 samples as 1024 training seeds."""
import html,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/cpu_gate/development'
if source.is_file():
    data=json.loads(source.read_text())['results']
    if any(d['task']!='perindopril' for d in data):data=[d for d in data if d['task']!='perindopril']
else:data=[json.loads(p.read_text()) for p in sorted(source.rglob('result.json'))]
groups={}
for d in data:groups.setdefault((d['task'],d['method']),[]).append(d)
rows=[];summary=[]
for (task,method),ds in sorted(groups.items()):
    row=dict(task=task,method=method,seeds=[d['seed'] for d in ds],runs=len(ds))
    for name,fn in [('auc',lambda d:d['auc_achieved']),('top10',lambda d:d['top10']),('evaluation_mean_reward',lambda d:d['evaluation']['mean_reward']),('validity',lambda d:d['evaluation']['validity'])]:
        vals=[fn(d) for d in ds];row[name]={'mean':statistics.mean(vals),'sd':statistics.stdev(vals) if len(vals)>1 else None,'values':vals}
    summary.append(row)
    rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [task,method,len(ds),f"{row['auc']['mean']:.4f}",f"{row['top10']['mean']:.4f}",f"{row['evaluation_mean_reward']['mean']:.4f}",f"{row['validity']['mean']:.2%}"] )+'</tr>')
label='Development / exploratory' if all(d['task']=='perindopril' for d in data) else 'Main comparison / current completed runs'
stem=('development_h100_summary' if source.is_file() else 'development_summary') if label.startswith('Development') else 'main_summary'
out=ROOT/'evidence'/stem
out.with_suffix('.json').write_text(json.dumps({'label':label,'source':str(source),'groups':summary,'statistical_unit':'independent training seed','inference':'descriptive; no significance claim from two/three seeds'},indent=2))
theory=json.loads((ROOT/'evidence/theory.json').read_text())
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ME2502 实验进展</title><style>body{{font:16px/1.8 system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#243045}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse;width:100%}}code{{white-space:pre-wrap}}aside{{background:#edf3fa;padding:18px}}</style>
<h1>ChemGPT 强化学习 · 实验进展</h1><p>{html.escape(label)} · 已完成 {len(data)} 次运行</p>
<aside>问题：固定评分预算下，RL 改善候选发现还是终态生成分布？这两种结果必须分开；KL 的收益不预设。下表是已完成运行的描述性均值，不能作为显著性结论。</aside>
<table><tr><th>任务</th><th>方法</th><th>种子数</th><th>发现 AUC</th><th>终点 top10</th><th>独立采样均值</th><th>有效率</th></tr>{''.join(rows)}</table>
<h2>理论与实现</h2><p>在固定解码映射 f 与共同支持下，轨迹 KL = 分子输出 KL + 按分子概率加权的条件编码 KL。它说明轨迹约束比输出约束更强，不证明真实训练的分子 KL 已被估计。</p>
<p>精确枚举四条轨迹：策略梯度误差 {theory['gradient_error']:.3g}，LOO 误差 {theory['loo_error']:.3g}，KL 分解误差 {theory['chain_rule_error']:.3g}。均为小环境正确性证据，不是分子性能证据。</p>
<h2>如何解释当前结果</h2><p>分别检查逐种子差异、全预算曲线、终态独立采样、有效率与结构覆盖。小幅均值改善不能替代种子稳定性；best-found 改善不能替代生成分布改善。所有记录仍需独立账本复算。</p>
<p>协议、代码和证据位于本目录；原 proposal 与旧验证保留。远程日志汇总不替代原始账本验收；每次收取后的核验范围见 remote_ledger_check.json。正式结论须以 H100 固定输入版本运行及完整证据为准。</p></html>'''
out.with_suffix('.html').write_text(page)
print(json.dumps({'report':str(out.with_suffix('.html')),'completed_runs':len(data),'groups':len(summary)}))
