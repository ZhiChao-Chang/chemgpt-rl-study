"""Incremental terminal physicochemical and full-pair diversity metrics; local CPU only."""
import collections,hashlib,json,statistics
from pathlib import Path
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem,Crippen,Descriptors,rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*');ROOT=Path(__file__).resolve().parent
out=ROOT/'evidence/detail_metrics.json';cache=json.loads(out.read_text()) if out.exists() else {}
for result in (ROOT/'collected/results').rglob('result.json'):
    path=result.parent/'evaluation.jsonl'
    if not path.exists():continue
    key=str(result.parent.relative_to(ROOT/'collected/results'));h=hashlib.sha256(path.read_bytes()).hexdigest()
    if cache.get(key,{}).get('source_sha256')==h:continue
    rows=[json.loads(x) for x in path.read_text().splitlines()];valid=[r for r in rows if r['valid']]
    freq=collections.Counter(r['identity'] for r in valid);ids=sorted(freq);mols=[Chem.MolFromSmiles(i) for i in ids]
    props={i:{'MW':Descriptors.MolWt(m),'TPSA':rdMolDescriptors.CalcTPSA(m),'logP':Crippen.MolLogP(m)} for i,m in zip(ids,mols)}
    fps=[AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048) for m in mols]
    dist=0.;weighted=0.;pairs=0
    for i in range(1,len(fps)):
        sims=DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i]);dist+=sum(1-s for s in sims);pairs+=len(sims)
        weighted+=sum(freq[ids[i]]*freq[ids[j]]*(1-s) for j,s in enumerate(sims))
    d={'source_sha256':h,'attempts':len(rows),'valid':len(valid),'unique':len(ids),'mean_actions_all_attempts':statistics.mean(r['length'] for r in rows) if rows else None,
       'diversity_unique_full_pairs':dist/pairs if pairs else None,
       'diversity_valid_attempt_full_pairs':weighted/(len(valid)*(len(valid)-1)/2) if len(valid)>1 else None,
       'unique_scaffolds_including_acyclic':len({MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols}),
       'valid_property_means':{k:statistics.mean(props[r['identity']][k] for r in valid) if valid else None for k in ['MW','TPSA','logP']},
       'valid_property_medians':{k:statistics.median(props[r['identity']][k] for r in valid) if valid else None for k in ['MW','TPSA','logP']}}
    cache[key]=d
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(cache,indent=2));tmp.replace(out)
print(json.dumps({'derived_runs':len(cache)}))
