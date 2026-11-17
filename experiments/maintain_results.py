"""Bounded local evidence collector; only reads this owned job, never starts resources."""
from datetime import datetime,timezone
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
until=datetime.fromisoformat('2026-11-14T03:10:00+08:00');last=-1;collected=6
while datetime.now(timezone.utc)<until:
    try:
        status=json.loads((ROOT/'evidence/remote_status.json').read_text())
        result=ROOT/'evidence/remote_results.json'
        if result.exists():
            n=len(json.loads(result.read_text())['results'])
            if n!=last:
                subprocess.run([sys.executable,str(ROOT/'analyze_results.py'),str(result)],check=True,timeout=30)
                last=n
            if status['state']=='RUNNING' and n>=collected+3:
                done=subprocess.run([sys.executable,str(ROOT/'collect_native.py')],capture_output=True,text=True,timeout=150)
                if done.returncode==0:
                    collected=json.loads((ROOT/'evidence/native_collection.json').read_text())['completed_runs']
                    check=subprocess.run([sys.executable,str(ROOT/'verify_ledgers.py'),str(ROOT/'collected_remote/results')],capture_output=True,text=True,timeout=60)
                    if check.returncode==0:(ROOT/'evidence/remote_ledger_check.json').write_text(check.stdout)
                    print(json.dumps({'collected_runs':collected,'ledger_verified':check.returncode==0}),flush=True)
                else:print(json.dumps({'collection':'read failed; retry later','exit_code':done.returncode}),flush=True)
        if status['state'] in ['SUCCEEDED','FAILED','SUSPENDED','DELETED']:
            print('EVIDENCE_WATCHER_TERMINAL '+status['state'],flush=True);break
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        print(json.dumps({'read_retry':type(exc).__name__}),flush=True)
    time.sleep(30)
