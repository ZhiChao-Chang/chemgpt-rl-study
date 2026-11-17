"""Illustrative isolated-molecule conformers for the viewer; no docking or activity inference."""
import json,sys
from pathlib import Path
from rdkit import Chem,RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
BASE=Path(__file__).resolve().parents[1];APP=BASE/'visual_report';f=APP/'src/content/assets/molecules.json';molecules=json.loads(f.read_text())
cases=json.loads((APP/'src/data.json').read_text())['queries']['cases']['rows']
selected=sorted({r['molId'] for r in cases if r['phase']=='core'})
receipt=[]
for key in selected:
    if 'conformerStatus' in molecules[key]:continue
    try:
        m=Chem.AddHs(Chem.MolFromSmiles(molecules[key]['smiles']));p=AllChem.ETKDGv3();p.randomSeed=20261117;p.timeout=3;p.maxIterations=100;p.numThreads=1
        code=AllChem.EmbedMolecule(m,p)
        if code!=0:raise RuntimeError('ETKDG did not produce coordinates within the bound')
        # Coordinates are for inspection only, deliberately not scored as a structure prediction.
        m=Chem.RemoveHs(m);molecules[key].update(sdf=Chem.MolToMolBlock(m),conformerStatus='available',conformerProvenance='RDKit ETKDGv3; seed20261117; isolated illustrative conformer; no docking, experimental structure, or affinity inference')
    except Exception as e:molecules[key].update(conformerStatus='unavailable',conformerError=type(e).__name__)
    receipt.append({'molId':key,'status':molecules[key]['conformerStatus']})
f.write_text(json.dumps(molecules,ensure_ascii=False))
ref=json.loads(Path('/home/zhangzhichao/projects/antibody/presentation/src/content/assets/structures.json').read_text())
lib=APP/'src/content/assets/molecular_viewer_library.json';d=json.loads(lib.read_text());d.update(license=ref['license'],version=ref['libraryVersion']);lib.write_text(json.dumps(d))
(BASE/'visualization_work/conformer_receipt.json').write_text(json.dumps({'scope':'illustrative coordinates, not evidence of binding or generated model outputs','requested':len(selected),'available':sum(molecules[k].get('conformerStatus')=='available' for k in selected),'details':receipt},indent=2))
print('Illustrative conformers:',sum(molecules[k].get('conformerStatus')=='available' for k in selected),'/',len(selected))
