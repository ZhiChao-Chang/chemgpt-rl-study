"""Small CPU interface check; no optimizer, RL training, or target selection."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
from email.parser import Parser

ROOT = Path(__file__).resolve().parent
vendor = ROOT / 'vendor'
vendor.mkdir(exist_ok=True)
wheel = next(ROOT.glob('selfies-1.0.4-*.whl'))
with zipfile.ZipFile(wheel) as z:
    wheel_meta = Parser().parsestr(z.read('selfies-1.0.4.dist-info/METADATA').decode())
    for item in z.infolist():
        target = (vendor / item.filename).resolve()
        if not target.is_relative_to(vendor.resolve()):
            raise RuntimeError('invalid wheel path')
    z.extractall(vendor)
sys.path.insert(0, str(vendor))

import torch
import transformers
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
import selfies
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.manual_seed(20261104)
t0 = time.monotonic()
model, info = AutoModelForCausalLM.from_pretrained(
    ROOT / 'chemgpt', local_files_only=True, torch_dtype=torch.float32,
    weights_only=True, output_loading_info=True)
model.eval()
assert not info['missing_keys'], info['missing_keys']
assert not info.get('mismatched_keys'), info.get('mismatched_keys')
assert all(k.endswith('.attn.attention.bias') or k.endswith('.attn.attention.masked_bias')
           for k in info['unexpected_keys']), info['unexpected_keys']
tokenizer = Tokenizer.from_file(str(ROOT / 'chemgpt/tokenizer.json'))
vocab = tokenizer.get_vocab()
assert len(vocab) == model.config.vocab_size == 684
assert all(vocab[k] == v for k, v in {'[UNK]':0, '[PAD]':1, '[CLS]':2, '[SEP]':3, '[MASK]':4}.items())
assert tokenizer.encode('[C][O]').ids[0] == 2 and tokenizer.encode('[C][O]').ids[-1] == 3
print('loaded; params',sum(p.numel() for p in model.parameters()),flush=True)

N, MAX = 32, 128
input_ids = torch.full((N, 1), 2, dtype=torch.long)
past = None
alive = torch.ones(N, dtype=torch.bool)
tokens, masks = [], []
sample_lp = torch.zeros(N)
blocked_mass = []
@torch.no_grad()
def step(input_ids, past, length):
    out = model(input_ids=input_ids, attention_mask=torch.ones(N,length,dtype=torch.long),
                token_type_ids=torch.zeros_like(input_ids), past_key_values=past, use_cache=True)
    return out.logits[:, -1], out.past_key_values

for t in range(MAX):
    logits, past = step(input_ids, past, t+1)
    blocked_mass.append(float(logits.softmax(-1)[:, [0,1,2,4]].sum(-1).mean()))
    logits[:, [0,1,2,4]] = -torch.inf
    lp = logits.log_softmax(-1)
    token = torch.multinomial(lp.exp(),1).squeeze(1)
    masks.append(alive.clone())
    sample_lp += lp.gather(-1,token[:,None]).squeeze(-1) * alive
    token = torch.where(alive,token,torch.full_like(token,3))
    tokens.append(token)
    alive &= token != 3
    input_ids = token[:,None]
    if not alive.any():
        break
seq = torch.stack(tokens,1)
mask = torch.stack(masks,1)
x = torch.cat([torch.full((N,1),2),seq[:,:-1]],1)
with torch.no_grad():
    logits = model(input_ids=x, attention_mask=mask.long(),token_type_ids=torch.zeros_like(x),use_cache=False).logits
    logits[..., [0,1,2,4]] = -torch.inf
    teacher_lp = (logits.log_softmax(-1).gather(-1,seq[...,None]).squeeze(-1) * mask).sum(-1)
delta = float((sample_lp-teacher_lp).abs().max())
assert delta < 5e-4, delta
rows=[]
for i,row in enumerate(seq):
    ids=row.tolist()
    if 3 in ids:
        ids=ids[:ids.index(3)]
    decoded=tokenizer.decode(ids,skip_special_tokens=True)
    # WordPiece decoding separates chemical units with spaces; SELFIES strings
    # have no whitespace. Keep raw decode and this deterministic normalization.
    sftext=decoded.replace(' ','')
    smi=None;error=None;valid=False;canonical=None
    try:
        smi=selfies.decoder(sftext)
        mol=Chem.MolFromSmiles(smi) if smi else None
        valid=(mol is not None and mol.GetNumAtoms()>0 and len(Chem.GetMolFrags(mol))==1)
        if valid: canonical=Chem.MolToSmiles(mol,isomericSmiles=True)
    except Exception as exc:
        error=type(exc).__name__
    rows.append({'sample':i,'natural_sep':not bool(alive[i]),'raw_wordpiece_decode':decoded,
                 'selfies':sftext,'smiles':smi,'valid_connected':valid,'canonical':canonical,'error':error})
# Check differentiability on a fixed sequence, without changing any weights.
tiny_x=x[:2,:16];tiny_y=seq[:2,:16];tiny_mask=mask[:2,:16]
model.zero_grad(set_to_none=True)
out=model(input_ids=tiny_x,attention_mask=tiny_mask.long(),token_type_ids=torch.zeros_like(tiny_x),use_cache=False)
masked=out.logits.clone();masked[..., [0,1,2,4]]=-torch.inf
nll=-(masked.log_softmax(-1).gather(-1,tiny_y[...,None]).squeeze(-1)*tiny_mask).sum(-1).mean()
nll.backward()
norm=torch.sqrt(sum(p.grad.float().square().sum() for p in model.parameters() if p.grad is not None))
assert torch.isfinite(norm) and norm>0
model.zero_grad(set_to_none=True)
summary={'status':'PASS_INTERFACE_ONLY','scope':'no target reward, optimizer update, or efficacy claim',
         'model':'ncfrey/ChemGPT-4.7M','revision':'7438a282460b3038e17a27e25b85b1376e9a23e2',
         'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,
         'transformers':transformers.__version__,
         'selfies_distribution_version':wheel_meta['Version'],
         'selfies_module_version':getattr(selfies,'__version__','unknown'),
         'selfies_wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),
         'device':'cpu','threads':2,'attempts':N,'max_generated_tokens':MAX,
         'naturally_terminated':sum(r['natural_sep'] for r in rows),
         'valid_connected':sum(r['valid_connected'] for r in rows),
         'unique_valid_connected':len({r['canonical'] for r in rows if r['valid_connected']}),
         'sample_vs_teacherforce_max_logprob_error':delta,'gradient_norm_no_optimizer':float(norm),
         'loading_missing_keys':info['missing_keys'],'loading_unexpected_legacy_buffers':info['unexpected_keys'],
         'special_ids':{'bos':2,'eos':3,'pad':1,'blocked':[0,1,2,4]},
         'token_type_ids':'all zeros in both rollout and teacher forcing; consistent with tokenizer output convention; historical train matrices not inspected',
         'kernel':'temperature1, full softmax over same static allowed tokens, no top-k/top-p',
         'mean_blocked_token_mass':sum(blocked_mass)/len(blocked_mass),
         'decode':'WordPiece decode skip specials, remove spaces, SELFIES1.0.4, RDKit connected nonempty',
         'elapsed_seconds':time.monotonic()-t0}
(ROOT/'interface_summary.json').write_text(json.dumps(summary,indent=2))
(ROOT/'interface_samples.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(summary,indent=2),flush=True)
