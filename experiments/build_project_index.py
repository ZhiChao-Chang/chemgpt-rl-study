"""Render rubric evidence matrix, presentation outline and rehearsal questions."""
import html,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;e=html.escape
subprocess.run([sys.executable,str(ROOT/'length_analysis.py')],check=True,timeout=60)
subprocess.run([sys.executable,str(ROOT/'build_claim_ledger.py')],check=True,timeout=60)
m=json.loads((ROOT/'rubric_matrix.json').read_text());qa=json.loads((ROOT/'defense_questions.json').read_text());plan=json.loads((ROOT/'presentation_plan.json').read_text())
sections=[]
for i,c in enumerate(m['criteria'],1):
    evidence=' · '.join(f"<a href='{e(p,quote=True)}'>{e(p)}</a>" for p in c['evidence'])
    sections.append(f"<section><h2>{i}. {e(c['criterion'])} · {c['weight_percent']}%</h2><p><b>当前：</b>{e(c['current'])}</p><p><b>完成标准：</b>{e(c['target'])}</p><ul>"+''.join(f'<li>{e(x)}</li>' for x in c['deliverables'])+'</ul><p><b>尚需完成：</b>'+e('；'.join(c['remaining']))+'</p><p>'+evidence+'</p></section>')
questions=''.join(f"<details><summary>{i}. {e(q['question'])}</summary><p>{e(q['answer'])}</p><a href='{e(q['evidence'],quote=True)}'>对应材料</a></details>" for i,q in enumerate(qa['questions'],1))
slides=''.join(f"<tr><td>{s['n']}</td><td>{e(s['title'])}</td><td>{e(s['purpose'])}</td></tr>" for s in plan['slides'])
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ME2502 项目证据与展示</title><style>body{font:16px/1.85 system-ui;max-width:1050px;margin:40px auto;padding:24px;color:#233044}h1{font-size:29px}h2{font-size:21px}section{border-top:1px solid #dde4ee;padding:20px 0}a{color:#175b9a;overflow-wrap:anywhere}aside{background:#ecf3fa;padding:20px;border-radius:10px}td{padding:10px;border-bottom:1px solid #ddd}details{padding:12px 0;border-bottom:1px solid #ddd}summary{cursor:pointer;font-weight:600}img{max-width:100%}</style><h1>ME2502 · 按五项评分要求建立证据</h1>'''
page+='<aside>完整主线：问题与目标 → 正确实现 → 固定预算对照 → 分布与案例 → 可支持的解释和局限。所有主张需要相应证据；不以漂亮案例代替整体实验，不虚构贡献或新理论。</aside>'
page+='<p>官网于 2026-11-13 复核：项目占课程成绩 50%；五项各占项目成绩 20%；报告截止 2026-12-06；第13–14周全员参与线下展示与问答，具体场次和时长尚未在该公告公布。</p>'
page+='<p><a href="evidence/main_summary.html">实时正式结果汇总</a> · <a href="evidence/cases/index.html">分子案例与失败</a> · <a href="evidence/length_analysis.html">长度分层诊断</a> · <a href="evidence/claim_ledger.html">主张与证据边界</a> · <a href="method_notes.html">数学与方法</a> · <a href="evidence/project_requirements.json">官网原要求</a> · <a href="evidence/remote_status.json">运行状态</a></p>'
case=ROOT/'evidence/cases/cases.json'
if case.exists():
    data=json.loads(case.read_text());page+=f"<p>当前案例覆盖取得原始账本的 {data['completed_raw_runs']}/18 次正式运行。</p>"
    tasks=sorted({r['task'] for r in data['runs']})
    for task in tasks:
        for kind in ['property_shifts','component_shifts']:
            f=f'evidence/cases/{task}_{kind}.png'
            if (ROOT/f).exists():page+=f"<img src='{f}' alt='{task} {kind}'>"
page+=''.join(sections)+'<section><h2>展示主线（建议结构，不是老师要求的页数或时长）</h2><table>'+slides+'</table></section>'
page+='<section><h2>答辩练习：需要成员实际讲清楚</h2><p>以下是准备材料，不表示任何成员已完成排练或掌握全部内容。计划分工与实际工作分开记录。</p>'+questions+'</section></html>'
(ROOT/'project_review.html').write_text(page)
print('Built project_review.html from rubric/source/defense records.')
