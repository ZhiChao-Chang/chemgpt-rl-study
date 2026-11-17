"""Refresh local case/rubric artifacts as already-collected ledgers arrive; bounded, no network."""
from datetime import datetime,timezone
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
deadline=datetime.fromisoformat('2026-11-14T03:10:00+08:00');last=-1
while datetime.now(timezone.utc)<deadline:
    files=list((ROOT/'collected_remote/results/main').glob('*/result.json'))
    if files and len(files)!=last:
        try:
            for script in ['build_cases.py','plot_cases.py','build_project_index.py']:
                subprocess.run([sys.executable,str(ROOT/script)],check=True,timeout=180)
            last=len(files)
            (ROOT/'evidence/material_refresh.json').write_text(json.dumps({'status':'UPDATED','raw_main_runs':last,'planned':18},indent=2))
            if last==18:break
        except (OSError,subprocess.SubprocessError) as exc:
            print(json.dumps({'status':'RETRY','type':type(exc).__name__}),flush=True)
    status=ROOT/'evidence/remote_status.json'
    if status.exists():
        try:
            if json.loads(status.read_text())['state'] in ['SUCCEEDED','FAILED','SUSPENDED','DELETED']:
                print(json.dumps({'status':'STOP_AT_REMOTE_TERMINAL','raw_main_runs':last}),flush=True);break
        except (ValueError,KeyError):pass
    time.sleep(30)
