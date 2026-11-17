"""Compare adapter components against pinned original modifier/fingerprint modules."""
import importlib.util
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'vendor'),str(ROOT/'src')]
import study
from rdkit import Chem,DataStructs
from rdkit.Chem import Crippen,rdMolDescriptors
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
source=ROOT.parent/'proposal/evidence/guacamol'
mods=load('original_modifiers',source/'score_modifier.py')
fps=load('original_fingerprints',source/'utils/fingerprints.py')
smiles=['CCO','c1ccccc1','C1CCCCC1','O=C(O)c1ccccc1',*study.REFS.values()]
ledger=ROOT/'results/cpu_gate/development/perindopril__frozen__s11/search.jsonl'
if ledger.exists():smiles.extend(json.loads(l)['identity'] for l in ledger.read_text().splitlines()[:32])
errors=[];count=0
for task in study.REFS:
    s=study.Scorer(task,None);ref=Chem.MolFromSmiles(study.REFS[task])
    def sim(m,k):return DataStructs.TanimotoSimilarity(fps.get_fingerprint(m,k),fps.get_fingerprint(ref,k))
    for sm in smiles:
        if not sm:continue
        m=Chem.MolFromSmiles(sm)
        if task=='perindopril':expected=[sim(m,'ECFP4'),mods.GaussianModifier(2,.5)(rdMolDescriptors.CalcNumAromaticRings(m))]
        elif task=='fexofenadine':expected=[mods.ClippedScoreModifier(.8)(sim(m,'AP')),mods.MaxGaussianModifier(90,10)(rdMolDescriptors.CalcTPSA(m)),mods.MinGaussianModifier(4,1)(Crippen.MolLogP(m))]
        else:expected=[mods.ClippedScoreModifier(.8)(sim(m,'FCFP4')),mods.MinGaussianModifier(.85,.1)(sim(m,'ECFP6')),mods.MaxGaussianModifier(100,10)(rdMolDescriptors.CalcTPSA(m)),mods.MinGaussianModifier(1,1)(Crippen.MolLogP(m))]
        actual=s.components(m);error=max(abs(float(a)-float(b)) for a,b in zip(expected,actual));errors.append(error);count+=1
assert max(errors)<1e-12,max(errors)
result=dict(status='PASS',case_task_pairs=count,max_component_abs_error=max(errors),source_commit=study.rt.GUACAMOL_COMMIT,
            scope='Original pinned fingerprint and modifier modules; task assembly checked against standard_benchmarks.py; not importing the complete benchmark dependency graph')
(ROOT/'evidence/scorer_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
