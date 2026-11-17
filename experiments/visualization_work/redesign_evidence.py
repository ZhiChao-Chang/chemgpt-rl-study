import json,itertools,hashlib,time
from pathlib import Path
from rdkit import Chem,RDLogger
from rdkit.Chem import rdFMCS,rdDepictor,rdMolAlign
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog('rdApp.*')
B=Path(__file__).resolve().parents[1];A=B/'visual_report';D=A/'src/content/assets';p=A/'src/data.json';snap=json.loads(p.read_text());snap['buildStatus']='updating';snap['authoredRevision']=2
core=json.loads((B/'evidence/cases/cases.json').read_text())['runs'];components=[];props=[]
for r in core:
 for ix,(k,v) in enumerate(r['diagnostics']['components_valid_outputs'].items()):
  components.append(dict(task=r['task'],condition=r['method'],seed=r['seed'],component=ix,definition=k,score=v['mean']))
 for k,v in r['diagnostics']['properties_valid_outputs'].items():
  props.append(dict(task=r['task'],condition=r['method'],seed=r['seed'],property=k,**v))
snap['queries']['components']={'rows':components,'source':{'provider':'Local experiment evidence','title':'核心实验逐种子奖励分量','metricDefinitions':{'score':'先对每次运行的有效终末样本取分量均值，再跨训练种子汇总；与含无效零分的原始MPO均值口径不同。'}}}
snap['queries']['properties']={'rows':props,'source':{'provider':'Local experiment evidence','title':'核心实验逐种子分子性质分布','metricDefinitions':{'median':'有效终末样本的性质中位数；p10/p90为同一次运行内部的分位数，不是置信区间。'}}}
p.write_text(json.dumps(snap,ensure_ascii=False))
refs={'fexofenadine':'CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4','osimertinib':'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34'}
def svg(m,hi=(),color=(.23,.48,.79)):
 d=rdMolDraw2D.MolDraw2DSVG(780,470);d.drawOptions().padding=.08;d.DrawMolecule(m,highlightAtoms=list(hi),highlightAtomColors={x:color for x in hi});d.FinishDrawing();return d.GetDrawingText()
(D/'references.json').write_text(json.dumps({k:{'smiles':v,'svg':svg(Chem.MolFromSmiles(v))} for k,v in refs.items()}))
mols=json.loads((D/'molecules.json').read_text());cases=[c for c in snap['queries']['cases']['rows'] if c['phase']=='core'];pairs={};start=time.time()
for task in refs:
 for seed in [None]:
  for role in [None]:
   cs=[c for c in cases if c['task']==task]
   for a,b in itertools.combinations(cs,2):
    key='|'.join(sorted([a['molId'],b['molId']]))
    if key in pairs:continue
    ma,mb=Chem.MolFromSmiles(a['smiles']),Chem.MolFromSmiles(b['smiles'])
    res=rdFMCS.FindMCS([ma,mb],timeout=1,ringMatchesRingOnly=True,completeRingsOnly=True,matchValences=True,bondCompare=rdFMCS.BondCompare.CompareOrderExact)
    patt=Chem.MolFromSmarts(res.smartsString);ia=ma.GetSubstructMatch(patt) if patt else ();ib=mb.GetSubstructMatch(patt) if patt else ()
    ca=[i for i in range(ma.GetNumAtoms()) if i not in ia];cb=[i for i in range(mb.GetNumAtoms()) if i not in ib]
    rdDepictor.Compute2DCoords(ma)
    if len(ia)>=3:
     try:rdDepictor.GenerateDepictionMatching2DStructure(mb,ma,atomMap=list(zip(ia,ib)))
     except Exception:rdDepictor.Compute2DCoords(mb)
    else:rdDepictor.Compute2DCoords(mb)
    rec={'a':a['molId'],'b':b['molId'],'commonAtoms':len(ia),'atomsA':ma.GetNumAtoms(),'atomsB':mb.GetNumAtoms(),'searchComplete':not res.canceled,'svgA':svg(ma,ca,(.84,.9,.98)),'svgB':svg(mb,cb,(.85,.94,.88)),'sdfA':None,'sdfB':None,'alignment':'2D shared-subgraph depiction; generated molecules are not parent/mutant pairs'}
    if len(ia)>=3 and mols[a['molId']].get('sdf') and mols[b['molId']].get('sdf'):
     try:
      m3a=Chem.MolFromMolBlock(mols[a['molId']]['sdf'],removeHs=True);m3b=Chem.MolFromMolBlock(mols[b['molId']]['sdf'],removeHs=True)
      rms=rdMolAlign.AlignMol(m3b,m3a,atomMap=list(zip(ib,ia)));rec.update(sdfA=Chem.MolToMolBlock(m3a),sdfB=Chem.MolToMolBlock(m3b),commonRMSD=rms)
     except Exception:pass
    pairs[key]=rec
(D/'pair_comparisons.json').write_text(json.dumps(pairs,ensure_ascii=False));print(json.dumps({'core_components':len(components),'property_rows':len(props),'pairs':len(pairs),'seconds':time.time()-start}))
