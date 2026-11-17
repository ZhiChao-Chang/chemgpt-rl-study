import urllib.request,json,pathlib,time
p=pathlib.Path('/tmp/tamgen_feasibility')
def get(url):
 for i in range(4):
  try:return urllib.request.urlopen(url,timeout=25).read()
  except Exception:
   if i==3: raise
   time.sleep(1)
repo='microsoft/TamGen';d=json.loads(get('https://api.github.com/repos/'+repo));c=json.loads(get('https://api.github.com/repos/'+repo+'/commits/'+d['default_branch']));t=json.loads(get('https://api.github.com/repos/'+repo+'/git/trees/'+c['sha']+'?recursive=1'))
(p/'microsoft_metadata.json').write_text(json.dumps({'repo':d,'commit':c,'tree':t},indent=2));print(c['sha'],c['commit']['committer']['date'])
paths=['README.md','setup_env.sh','setup.py','LICENSE','fairseq/models/tgt2drug/transformer_3d.py','fairseq/tasks/molecule/translation_coord.py','fairseq/criterions/label_smoothed_cross_entropy_with_vae.py','fairseq/sequence_generator.py','fairseq/search.py','fairseq/models/transformer.py','scripts/train.sh','scripts/generate.sh','TamGen_Demo.py','scripts/build_data/prepare_pdb_ids_center.py','data/README.md','model/README.md','fairseq/models/fairseq_model.py','fairseq/tasks/fairseq_task.py','fairseq/criterions/label_smoothed_cross_entropy.py','generate.py','customized_example/README.MD']
for path in paths:
 try:b=get('https://raw.githubusercontent.com/'+repo+'/'+c['sha']+'/'+path)
 except Exception as e:print(path,str(e));continue
 f=p/'source'/path;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b);print(path,len(b))
