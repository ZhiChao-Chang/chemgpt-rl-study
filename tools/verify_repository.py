#!/usr/bin/env python3
"""Check immutable results, aggregation and the browser artifact without training."""
import hashlib,json,pathlib,re,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1];APP=ROOT/'experiments/visual_report'
def main():
 d=json.loads((APP/'src/data.json').read_text());runs=d['queries']['runs']['rows'];groups=d['queries']['summary']['rows'];assert len(runs)==126
 for r in runs:
  p=ROOT/'experiments'/r['sourceResult'];assert p.is_file(),p
  assert hashlib.sha256(p.read_bytes()).hexdigest()==r['resultSha256'],p
 for s in groups:
  rs=[r for r in runs if (r['phase'],r['task'],r['condition'])==(s['phase'],s['task'],s['condition'])];assert len(rs)==s['n']
  for k in ['terminalReward','auc','top10','meanActions','meanMW','meanTPSA']:
   assert abs(statistics.mean(r[k] for r in rs)-s[k])<1e-12,(s,k)
   assert abs(statistics.stdev(r[k] for r in rs)-s[k+'SD'])<1e-12,(s,k)
 mf=json.loads((ROOT/'docs/publication-manifest.json').read_text());html=(ROOT/'docs/index.html').read_bytes()
 assert hashlib.sha256(html).hexdigest()==mf['sha256']
 assert not re.search(rb'<meta[^>]+data-app-local-thread[^>]*>',html)
 assert len(json.loads((APP/'src/content/assets/pair_comparisons.json').read_text()))==306
 print(json.dumps({'status':'PASS','source_results':126,'aggregated_groups':len(groups),'molecule_pairs':306,'website_sha256':mf['sha256']}))
if __name__=='__main__':main()
