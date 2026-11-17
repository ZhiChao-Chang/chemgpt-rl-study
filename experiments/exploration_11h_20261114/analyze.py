"""Extension descriptive tables: never pool protocols or select favorable seeds."""
import html,json,statistics,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
subprocess.run([__import__('sys').executable,str(ROOT/'strict_replay.py')],check=True,timeout=240)
subprocess.run(['/home/zhangzhichao/miniconda3/envs/ddc_inference/bin/python',str(ROOT/'derive.py')],check=True,timeout=120)
subprocess.run(['/home/zhangzhichao/miniconda3/envs/ddc_inference/bin/python',str(ROOT/'plot.py')],check=True,timeout=120)
vf=ROOT/'evidence/verification.json'
if vf.exists():
    v=json.loads(vf.read_text());v['checked_runs']=len(v['runs']);v['verified_runs']=sum(r['status']=='PASS' for r in v['runs'].values());vf.write_text(json.dumps(v,indent=2))
source=ROOT/'collected/results';rows=[json.loads(p.read_text()) for p in source.rglob('result.json')]
groups={}
for r in rows:groups.setdefault((r['phase'],r['task'],r['method']),[]).append(r)
out=[];trs=[]
for (phase,task,method),rs in sorted(groups.items()):
    rs.sort(key=lambda r:r['seed']);d={'phase':phase,'task':task,'method':method,'n':len(rs),'seeds':[r['seed'] for r in rs]}
    d['stop_reasons']=dict((reason,sum(r['stop_reason']==reason for r in rs)) for reason in sorted({r['stop_reason'] for r in rs}))
    for k,f in [('auc',lambda r:r['auc_achieved']),('auc_fixed',lambda r:r['auc_fixed_budget_carry_forward']),('queries',lambda r:r['unique_queries']),('raw_reward',lambda r:(r.get('evaluation') or {}).get('mean_reward')),('training_reward',lambda r:r['terminal_mean_training_reward']),('terminal_kl',lambda r:r['terminal_mean_sample_trajectory_kl']),('validity',lambda r:(r.get('evaluation') or {}).get('validity'))]:
        vals=[f(r) for r in rs];available=[v for v in vals if v is not None];d[k]={'mean':statistics.mean(available) if available else None,'sd':statistics.stdev(available) if len(available)>1 else None,'values':vals,'available':len(available)}
    fmt=lambda v: f'{v:.4f}' if v is not None else 'missing'
    out.append(d);trs.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [phase,task,method,len(rs),fmt(d['queries']['mean']),fmt(d['auc_fixed']['mean']),fmt(d['raw_reward']['mean']),fmt(d['training_reward']['mean']),fmt(d['terminal_kl']['mean'])])+'</tr>')
c=json.loads((ROOT/'contract.json').read_text());planned=sum(len(p['tasks'])*len(p['seeds'])*len(p['conditions']) for p in c['phases'])
summary={'completed_raw_runs':len(rows),'planned':planned,'groups':out,'interpretation':'separate extension cohorts/protocols; independent seeds, descriptive only; no efficacy claim'}
(ROOT/'evidence/summary.json').write_text(json.dumps(summary,indent=2))
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RL 十一小时探索</title><style>body{font:16px/1.8 system-ui;max-width:1250px;margin:40px auto;padding:20px;color:#25354a}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}aside{padding:20px;background:#edf3fa}</style><h1>ME2502 · 独立扩展实验</h1><aside>原始18次正式比较保持不变。这一轮在见过原结果后制定，使用新种子检验KL强度、长度/奖励约束和较大预算；不能冒称原实验前预注册。修改后的训练目标与原始MPO分开报告。</aside>'''
page+=f'<p>已取得完整原始账本：{len(rows)}/{planned}；截止：{c["absolute_deadline"]}。未完成的组不计零分。</p>'
page+='<table><tr><th>阶段</th><th>任务</th><th>条件</th><th>种子数</th><th>实际查询均值</th><th>固定预算MPO AUC*</th><th>终态原MPO均值</th><th>修改后训练回报</th><th>终态轨迹KL估计</th></tr>'+''.join(trs)+'</table><p>* 未达固定预算的曲线延续最后观察值。逐组同时保留实际查询数、停止原因与实际预算AUC，不能将早停隐藏成完成全部查询。缺失终态指标标missing。</p>'
page+='<p>不同阶段预算、终态样本数及目标不同，分别比较；KL为token轨迹相对固定prior的采样估计，可有方差，不能当作canonical分子KL。所有初始模型和评分定义相同，实际停止原因保留。</p>'
for f in json.loads((ROOT/'evidence/figures.json').read_text()):page+=f'<img style="max-width:100%" src="{f}">'
page+='<p>图中浅线为已完成的独立种子，深线为可用种子均值；不同点可能尚未齐种子，不用于提前选参数。</p><p><a href="contract.json">固定协议</a> · <a href="evidence/status.json">运行状态</a> · <a href="evidence/verification.json">账本核验</a> · <a href="evidence/summary.json">逐种子数据</a></p></html>'
(ROOT/'index.html').write_text(page)
print(json.dumps({'completed':len(rows),'planned':planned}))
