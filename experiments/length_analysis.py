"""Exploratory within-length summaries. Length is post-treatment, no causal attribution."""
import html,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent;SRC=ROOT/'collected_remote/results/main'
METHODS=['frozen','reinforce','reinforce_kl'];BINS=[(1,32),(33,64),(65,96),(97,128),(129,256)]
data={}
for p in SRC.glob('*/result.json'):
    d=json.loads(p.read_text());rows=[json.loads(s) for s in (p.parent/'evaluation.jsonl').read_text().splitlines()]
    data[(d['task'],d['seed'],d['method'])]=rows
result=[]
for task,seed in sorted({(t,s) for t,s,m in data}):
    if not all((task,seed,m) in data for m in METHODS):continue
    records=[]
    for lo,hi in BINS:
        row={'low':lo,'high':hi,'methods':{}}
        for m in METHODS:
            allrows=data[task,seed,m];xs=[r for r in allrows if r['valid'] and lo<=r['length']<=hi]
            row['methods'][m]={'n':len(xs),'fraction_all_attempts':len(xs)/len(allrows),
                               'reward_mean':statistics.mean(r['reward'] for r in xs) if xs else None,
                               'length_mean':statistics.mean(r['length'] for r in xs) if xs else None}
        row['common_support_min10']=all(row['methods'][m]['n']>=10 for m in METHODS)
        records.append(row)
    eligible=[r for r in records if r['common_support_min10']]
    denom=sum(r['methods']['frozen']['n'] for r in eligible)
    standardized={};coverage={}
    for m in METHODS:
        standardized[m]=sum(r['methods']['frozen']['n']/denom*r['methods'][m]['reward_mean'] for r in eligible) if denom else None
        coverage[m]=sum(r['methods'][m]['n'] for r in eligible)/len(data[task,seed,m])
    result.append({'task':task,'seed':seed,'bins':records,'frozen_bin_weighted_common_support_mean':standardized,'common_support_all_attempt_coverage':coverage,
                   'raw_all_attempt_mean':{m:statistics.mean(r['reward'] for r in data[task,seed,m]) for m in METHODS}})
out={'status':'EXPLORATORY','bin_rule':'fixed 32-token blocks up to128;129-256 tail. Each eligible bin >=10 valid samples in EVERY method; baseline bin weights renormalized over eligible bins.',
     'causal_boundary':'Token length is an outcome/mediator of the policy; conditioning on it is not a causal adjustment. Coarse bins also leave residual length differences. This analysis cannot prove how much gain is caused by length.',
     'primary_unchanged':'All-attempt primary terminal metrics and original search endpoints unchanged. Excluded-bin coverage explicitly reported, no samples removed from primary analysis.',
     'runs':result}
(ROOT/'evidence/length_analysis.json').write_text(json.dumps(out,indent=2))
parts=[]
for d in result:
    rows=[]
    for b in d['bins']:
        vals=[]
        for m in METHODS:
            x=b['methods'][m];v=x['reward_mean'];vals.append(f"n={x['n']}, mean={v:.4f}" if v is not None else 'n=0, mean=missing')
        rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [f"{b['low']}-{b['high']}",*vals,b['common_support_min10']])+'</tr>')
    parts.append(f"<h2>{d['task']} · seed {d['seed']}</h2><table><tr><th>长度</th><th>Frozen</th><th>REINFORCE</th><th>RF+KL</th><th>共同支持≥10</th></tr>{''.join(rows)}</table><p>共同支持按冻结策略分箱权重标准化：{html.escape(str(d['frozen_bin_weighted_common_support_mean']))}</p><p>各条件在共同支持区域的样本覆盖率：{html.escape(str(d['common_support_all_attempt_coverage']))}</p>")
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>长度分层诊断</title><style>body{font:16px/1.8 system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#25354a}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}aside{background:#fff3d9;padding:20px}</style><h1>奖励提升是否只伴随生成长度变化？</h1><aside>探索性描述分析。长度本身是策略影响的结果，按长度分组不构成因果识别；粗分箱仍有残余长度差异。这里只检查不同长度区域内的奖励表现，不能据此把总收益分解成“长度造成”和“算法造成”。</aside><p>所有方法原始评价保持不变；仅展示三个条件都已取得账本的种子。固定分箱，且只有每个方法均≥10个有效样本的分箱用于标准化；完整分箱与覆盖率均列出。</p>'''+''.join(parts)+'</html>'
(ROOT/'evidence/length_analysis.html').write_text(page)
print(json.dumps({'complete_matched_seeds':len(result),'output':'evidence/length_analysis.html'}))
