#!/usr/bin/env python3
"""Build the static research report with repository-local dependencies only."""
import argparse,hashlib,json,os,pathlib,re,shutil,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
APP=ROOT/'experiments/visual_report'
def main():
 p=argparse.ArgumentParser();p.add_argument('--skip-build',action='store_true');a=p.parse_args()
 subprocess.run(['python3',str(ROOT/'tools/prepare_report_data.py')],cwd=ROOT,check=True)
 if not a.skip_build:
  env={k:v for k,v in os.environ.items() if k not in {'CODEX_SESSION_ID','CODEX_THREAD_ID'}}
  subprocess.run(['npm','run','build'],cwd=APP,env=env,check=True)
 html=(APP/'dist/index.html').read_bytes()
 if re.search(rb'<meta[^>]+data-app-local-thread[^>]*>',html):raise SystemExit('Refusing local-session metadata in website. Rebuild without CODEX_SESSION_ID/CODEX_THREAD_ID.')
 (ROOT/'docs').mkdir(exist_ok=True)
 (ROOT/'docs/index.html').write_bytes(html)
 (ROOT/'docs/.nojekyll').touch()
 (ROOT/'docs/data').mkdir(exist_ok=True)
 for name in ['run_metrics.csv','summary_metrics.csv','report_data_receipt.json']:
  shutil.copyfile(ROOT/'data'/name,ROOT/'docs/data'/name)
 snap=json.loads((APP/'src/data.json').read_text());runs=snap['queries']['runs']['rows']
 manifest={'artifact':'ChemGPT RL research report','repository':'ZhiChao-Chang/chemgpt-rl-study','evidence_cutoff':snap['report']['asOf'],'runs':len(runs),'artifact_path':'index.html','bytes':len(html),'sha256':hashlib.sha256(html).hexdigest(),'source':'experiments/visual_report','presentation':'read-only audience view; 3D illustrative conformers only','source_commit':os.environ.get('GITHUB_SHA'),'source_manifest_sha256':hashlib.sha256((ROOT/'data/study_manifest.json').read_bytes()).hexdigest(),'data_build_receipt':'data/report_data_receipt.json'}
 (ROOT/'docs/publication-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'website':'docs/index.html','bytes':len(html),'sha256':manifest['sha256'],'runs':len(runs)}))
if __name__=='__main__':main()
