exec(open('/tmp/tamgen_feasibility/fetch.py').read().split("repo='microsoft/TamGen'")[0])
d=json.loads((p/'microsoft_metadata.json').read_text());sha=d['commit']['sha']
paths=['fairseq/modules/pointcloud/distance_attention.py','fairseq/modules/pointcloud/point_transformer_layer_multihead.py','fairseq/molecule_utils/external/fairseq_dataset_build_utils.py','fairseq/molecule_utils/coordinate_utils/binding_site_utils.py','fairseq/data/molecule/language_pair_coord_dataset.py','fairseq/molecule_utils/basic/smiles_utils.py','fairseq/checkpoint_utils.py','fairseq/sequence_scorer.py','gpt_model/README.md']
for path in paths:
 try:b=get('https://raw.githubusercontent.com/microsoft/TamGen/'+sha+'/'+path)
 except Exception as e:print(path,str(e));continue
 f=p/'source'/path;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b);print(path,len(b),flush=True)
for path in ['setup_env.sh','fairseq/models/tgt2drug/transformer_3d.py','fairseq/sequence_generator.py','fairseq/criterions/label_smoothed_cross_entropy_with_vae.py']:
 b=get('https://raw.githubusercontent.com/SigmaGenX/TamGen/858e35eb9f6d5e1f9877f14911b939bf33cf6d49/'+path);f=p/'sigma_source'/path;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
try:
 b=get('https://zenodo.org/api/records/13751391');(p/'zenodo_metadata.json').write_bytes(b);d=json.loads(b);print('ZENODO LICENSE',d['metadata'].get('license'));print([(f['key'],f['size']) for f in d['files']])
except Exception as e:print('ZENODO',str(e))
