"""Fill strict token-replay checks for early extension runs, cached by ledger hashes."""
import fcntl,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;out=ROOT/'evidence/strict_replay.json'
lock=(ROOT/'evidence/strict_replay.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    print(json.dumps({'strict_replay':'already running'}));raise SystemExit(0)
saved=json.loads(out.read_text()) if out.exists() else {}
for p in (ROOT/'collected/results').rglob('result.json'):
    if json.loads(p.read_text()).get('status')!='COMPLETE':continue
    key=str(p.parent.relative_to(ROOT/'collected/results'));h=hashlib.sha256((p.parent/'search.jsonl').read_bytes()+(p.parent/'evaluation.jsonl').read_bytes()).hexdigest()
    if saved.get(key,{}).get('ledger_sha256')==h:continue
    check=subprocess.run(['/home/zhangzhichao/miniconda3/envs/ddc_inference/bin/python',str(ROOT/'verify_extension.py'),str(p.parent)],capture_output=True,text=True,timeout=150)
    rec=json.loads(check.stdout) if check.returncode==0 else {'status':'FAIL','exit_code':check.returncode}
    rec['ledger_sha256']=h;saved[key]=rec
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(saved,indent=2));tmp.replace(out)
print(json.dumps({'strict_replay_pass':sum(r['status']=='PASS' for r in saved.values()),'checked':len(saved)}))
