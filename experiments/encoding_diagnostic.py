"""Fixed molecule panel, alternate encodings: finite-subset diagnostic, not full molecular KL."""
import copy,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'vendor'),str(ROOT/'src')]
import study
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM
from rdkit import Chem,rdBase
import selfies

torch.set_num_threads(2);torch.set_num_interop_threads(1)
base=ROOT.parent/'proposal/implementation/chemgpt'
tok=Tokenizer.from_file(str(base/'tokenizer.json'))
source=ROOT/'results/cpu_gate/development/perindopril__frozen__s11/search.jsonl'
records=[json.loads(x) for x in source.read_text().splitlines()]
panel=[]
for r in records:
    if r['valid'] and r['identity'] not in panel:panel.append(r['identity'])
    if len(panel)==8:break
rdBase.SeedRandomNumberGenerator(713)
variants=[]
for smiles in panel:
    mol=Chem.MolFromSmiles(smiles);seqs=set()
    # Fixed panel uses frozen baseline discovery order, independent of trained outcomes.
    alternatives=[Chem.MolToSmiles(mol)]+[Chem.MolToSmiles(mol,canonical=False,doRandom=True) for _ in range(24)]
    for sm in alternatives:
        try:
            sf=selfies.encoder(sm);ids=tok.encode(sf).ids
            if 0 in ids or ids[0]!=2 or ids[-1]!=3 or len(ids)>257:continue
            back=selfies.decoder(tok.decode(ids[1:-1],skip_special_tokens=True).replace(' ',''))
            m=Chem.MolFromSmiles(back)
            if m is None or Chem.MolToSmiles(m,isomericSmiles=True)!=smiles:continue
            seqs.add(tuple(ids[1:]))
        except (ValueError,TypeError,IndexError):continue
    if len(seqs)>1:variants.append(dict(identity=smiles,encodings=[list(x) for x in sorted(seqs)[:8]]))
assert variants,'No compatible alternate encodings'
template=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.float32,weights_only=True).eval()
def lps(model,seqs):
    n=max(map(len,seqs));seq=torch.full((len(seqs),n),3,dtype=torch.long);mask=torch.zeros_like(seq,dtype=torch.bool)
    for i,s in enumerate(seqs):seq[i,:len(s)]=torch.tensor(s);mask[i,:len(s)]=True
    with torch.no_grad():return study.rt.sequence_logprob(model,seq,mask).double()
out=[]
for method in ['frozen','reinforce','reinforce_kl']:
    cp=ROOT/f'results/cpu_gate/development/perindopril__{method}__s11/checkpoint.pt'
    state=torch.load(cp,map_location='cpu',weights_only=False);actor=copy.deepcopy(template);actor.load_state_dict(state['actor']);actor.eval()
    for item in variants:
        a=lps(actor,item['encodings']);b=lps(template,item['encodings']);q=a.softmax(0);q0=b.softmax(0)
        k=float((q*(q.log()-q0.log())).sum())
        out.append(dict(method=method,identity=item['identity'],variant_count=len(a),conditional_KL_within_fixed_subset=k,
                        actor_subset_probability=float(a.logsumexp(0).exp()),reference_subset_probability=float(b.logsumexp(0).exp())))
result=dict(status='COMPLETE',label='CPU exploratory checkpoints; fixed frozen-discovery panel; diagnostic only',
            interpretation='All encodings in each panel decode to the same canonical molecule. Conditional probabilities are renormalized ONLY over this finite encoding set. This is neither full conditional encoding KL nor molecular output KL, and not evidence of reward improvement.',
            panel=variants,measurements=out)
(ROOT/'evidence/encoding_diagnostic.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'status':'COMPLETE','molecules':len(variants),'rows':len(out),'scope':'finite encoding subset only'}))
