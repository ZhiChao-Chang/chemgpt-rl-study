import base64,hashlib,json,pathlib,subprocess,time,zlib
R={'task_id': 'acp-explore-chemgpt-profile-20261104-04', 'remote_directory': '/zzc/260010120/cci_control/acp-tests/acp-explore-chemgpt-profile-20261104-04', 'bundle': {'path': '/tmp/chemgpt_h100_profile_20261104/input_bundle.tar.gz', 'bytes': 19254733, 'sha256': 'aeed1d4af540874b14eb3b7265a9321b072abfcc01c9fc22897aa799369e31f3', 'files': [{'path': 'profile_h100.py', 'bytes': 23839, 'sha256': '951d6fd36287f404cd02712dd97e138dd59257adca4d3f3914c3fd4e7abe0e0f'}, {'path': 'model/config.json', 'bytes': 1284, 'sha256': '73a9eb113e31b2ac8a6b070469b749dbaa8110f72a197bea7899d93db462e8ba'}, {'path': 'model/tokenizer.json', 'bytes': 10789, 'sha256': 'fee4dad651e2edb49afb5d6aa3e4fea437e08c740c80b6d1d6e1909ad2b30100'}, {'path': 'model/tokenizer_config.json', 'bytes': 260, 'sha256': 'ee5b5daf3a4f292292f944d06fa34ca00c6f4b18aec68478c55ef223311ee145'}, {'path': 'model/special_tokens_map.json', 'bytes': 2, 'sha256': '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a'}, {'path': 'model/README.md', 'bytes': 1942, 'sha256': '5732275e9233a8b8ca8f690b6beb624da9ff5b3e7906f7d446a8c6ff45597195'}, {'path': 'model/pytorch_model.bin', 'bytes': 70836865, 'sha256': 'a8a39c5ea1d50c450632e003d21c90f847d815a801934b03a12b02010b74d21b'}, {'path': 'selfies-1.0.4-py3-none-any.whl', 'bytes': 30220, 'sha256': 'e41009c1b5512d87ed8ea4d2a52ea9d833fbe6ddeb9ea34990b4a4d77b5b037c'}]}, 'max_total_seconds': 1200, 'max_profile_seconds': 600, 'scope': 'Network-free H100 profile using native authenticated terminal upload; no installs'}
root=pathlib.Path(R['remote_directory']);t0=time.monotonic()
def event(name,**kw): print(json.dumps({'event':name,'elapsed_seconds':time.monotonic()-t0,**kw}),flush=True)
try:
 m=subprocess.run(['findmnt','-n','--mountpoint','/zzc','--output','TARGET'],capture_output=True,text=True,timeout=10)
 if m.returncode or m.stdout.strip()!='/zzc': raise RuntimeError('AFS mount not verified')
 root.mkdir(parents=True,exist_ok=False)
 (root/'recipe.json').write_text(json.dumps(R,indent=2))
 event('waiting_for_native_input',task_id=R['task_id'])
 while not (root/'input_ready.json').is_file():
  if time.monotonic()-t0>600: raise RuntimeError('native input deadline exceeded')
  time.sleep(1)
 ready=json.loads((root/'input_ready.json').read_text())
 if ready['bundle_sha256']!=R['bundle']['sha256']: raise RuntimeError('wrong uploaded bundle')
 for item in R['bundle']['files']:
  f=root/item['path']
  if hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('input file hash mismatch')
 event('native_input_hashes_verified')
 py='/root/anaconda3/envs/ddc_encoding/bin/python';out=root/'profile.json'
 result=subprocess.run([py,'-u',str(root/'profile_h100.py'),'--model-dir',str(root/'model'),'--vendor-dir',str(root/'vendor'),'--out',str(out),'--device','cuda','--batch-size','32','--steps','4','--max-length','256','--threads','4','--stress-shapes'],timeout=600)
 report=json.loads(out.read_text());raw=json.dumps(report,separators=(',',':')).encode()
 print('H100_PROFILE_RESULT_B64 '+base64.b64encode(zlib.compress(raw,9)).decode(),flush=True)
 event('profile_finished',status=report['status'],returncode=result.returncode)
 print('H100_PROFILE_SUMMARY '+json.dumps({'gpu':report.get('gpu'),'status':report['status'],'methods':{k:{field:v.get(field) for field in ['status','mean_measured_step_seconds','max_peak_allocated_bytes','max_peak_reserved_bytes']} for k,v in report['methods'].items()}}),flush=True)
 raise SystemExit(result.returncode)
except Exception as exc:
 error={'status':'FAILED','type':type(exc).__name__,'message':str(exc),'elapsed_seconds':time.monotonic()-t0}
 if root.exists():(root/'bootstrap_error.json').write_text(json.dumps(error,indent=2))
 print('H100_PROFILE_ERROR '+json.dumps(error),flush=True)
 raise
