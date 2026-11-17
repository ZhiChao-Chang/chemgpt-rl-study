"""Budgeted on-policy ChemGPT study. Local files only; explicit run ledgers."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import selfies
import model_runtime as rt

rt.torch = torch
RDLogger.DisableLog('rdApp.*')
STOP = False

def stop_signal(*_):
    global STOP
    STOP = True

signal.signal(signal.SIGTERM, stop_signal)
signal.signal(signal.SIGINT, stop_signal)

def save(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)); t.replace(p)

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def gaussian(x, mu, sigma): return math.exp(-.5 * ((x-mu)/sigma)**2)
def upper(x, mu, sigma): return gaussian(max(x,mu),mu,sigma)
def lower(x, mu, sigma): return gaussian(min(x,mu),mu,sigma)

REFS = {
    'perindopril': 'O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC',
    'fexofenadine': 'CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4',
    'osimertinib': 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34'
}

def fp(m, kind):
    if kind == 'AP': return AllChem.GetAtomPairFingerprint(m,maxLength=10)
    return AllChem.GetMorganFingerprint(m,3 if kind=='ECFP6' else 2,useFeatures=(kind=='FCFP4'))

class Scorer:
    def __init__(self, task, tok):
        self.task, self.tok = task, tok
        m = Chem.MolFromSmiles(REFS[task])
        self.refs = {k:fp(m,k) for k in ['AP','ECFP4','ECFP6','FCFP4']}
        self.cache = {}; self.physical_calls = 0

    def components(self, mol):
        sim = lambda k: float(DataStructs.TanimotoSimilarity(fp(mol,k),self.refs[k]))
        if self.task=='perindopril':
            return [sim('ECFP4'),gaussian(rdMolDescriptors.CalcNumAromaticRings(mol),2,.5)]
        if self.task=='fexofenadine':
            return [min(sim('AP')/.8,1.),lower(rdMolDescriptors.CalcTPSA(mol),90,10),upper(Crippen.MolLogP(mol),4,1)]
        return [min(sim('FCFP4')/.8,1.),upper(sim('ECFP6'),.85,.1),lower(rdMolDescriptors.CalcTPSA(mol),100,10),upper(Crippen.MolLogP(mol),1,1)]

    def decode(self, ids, ended):
        length = ids.index(rt.EOS)+1 if rt.EOS in ids else len(ids)
        body = ids[:length-1] if ended else ids
        rec = dict(token_ids=ids[:length],length=length,natural_eos=bool(ended),identity=None,reward=0.,components=None,valid=False,cache_hit=False)
        if not ended:
            rec['failure']='truncated'; return rec
        try:
            sf = self.tok.decode(body,skip_special_tokens=True).replace(' ','')
            sm = selfies.decoder(sf)
            mol = Chem.MolFromSmiles(sm) if sm else None
            if mol is None or mol.GetNumAtoms()==0 or len(Chem.GetMolFrags(mol))!=1:
                rec['failure']='invalid_or_disconnected'; return rec
            ident = Chem.MolToSmiles(mol,isomericSmiles=True)
            rec.update(identity=ident,valid=True,cache_hit=ident in self.cache)
            if ident not in self.cache:
                c = self.components(mol); r = math.prod(c)**(1/len(c))
                if not math.isfinite(r) or not 0<=r<=1: raise RuntimeError('reward outside [0,1]')
                self.cache[ident] = (r,c); self.physical_calls += 1
            rec['reward'],rec['components'] = self.cache[ident]
        except (ValueError,KeyError,IndexError) as exc:
            rec['failure']=type(exc).__name__
        return rec

def output_metrics(recs):
    ids = [r['identity'] for r in recs if r['valid']]
    unique = sorted(set(ids)); mols = [Chem.MolFromSmiles(s) for s in unique]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048) for m in mols]
    # Fixed deterministic sample cap; record actual number for quadratic diversity.
    fps = fps[:256]; sims = []
    for i in range(1,len(fps)): sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i]))
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    return dict(attempts=len(recs),mean_reward=float(np.mean([r['reward'] for r in recs])),validity=len(ids)/len(recs),
                unique_valid=len(unique),unique_valid_per_attempt=len(unique)/len(recs),repeat_fraction_among_valid=1-len(unique)/len(ids) if ids else None,
                unique_scaffolds=len(set(scaffolds)),acyclic_unique=sum(s=='' for s in scaffolds),
                fingerprint_diversity_unique_lexicographic_cap256=1-float(np.mean(sims)) if sims else None,
                diversity_pairs=len(sims),fraction_reward_ge_08=sum(r['reward']>=.8 for r in recs)/len(recs),
                joint_components_ge_08=sum(bool(r['components']) and min(r['components'])>=.8 for r in recs)/len(recs))

def likelihoods_and_kl(actor, ref, seq, mask):
    x=torch.cat([torch.full((seq.shape[0],1),rt.BOS,dtype=seq.dtype,device=seq.device),seq[:,:-1]],1)
    kw=dict(input_ids=x,attention_mask=mask.long(),token_type_ids=torch.zeros_like(x),use_cache=False)
    alp=rt.masked_logprobs(actor(**kw).logits)
    with torch.no_grad():rlp=rt.masked_logprobs(ref(**kw).logits)
    actor_lp=(alp.gather(-1,seq[...,None]).squeeze(-1)*mask).sum(-1)
    ref_lp=(rlp.gather(-1,seq[...,None]).squeeze(-1)*mask).sum(-1)
    keep=[i for i in range(alp.shape[-1]) if i not in rt.BLOCKED]
    with torch.no_grad():
        token_kl=(alp.detach()[...,keep].exp()*(alp.detach()[...,keep]-rlp[...,keep])).sum(-1)
        prefix_kl=(token_kl*mask).sum()/mask.sum()
    return actor_lp,ref_lp,prefix_kl

def validate_model(template, tok, device, out):
    seed(11); actor=copy.deepcopy(template).to(device).eval().requires_grad_(True)
    seq,mask,natural,lp=rt.rollout(actor,4,256,device)
    check=rt.sequence_logprob(actor,seq,mask)
    err=float((check.detach()-lp).abs().max()); assert err<.005,err
    padded=torch.cat([seq,torch.full((4,5),rt.EOS,device=device,dtype=torch.long)],1)
    pm=torch.cat([mask,torch.zeros((4,5),device=device,dtype=torch.bool)],1)
    perr=float((rt.sequence_logprob(actor,padded,pm)-check).abs().max());assert perr<.005,perr
    # Policy inference contract: no active dropout, fixed masks, reference disconnected.
    ref=copy.deepcopy(template).to(device).eval().requires_grad_(False)
    rlp=rt.sequence_logprob(ref,seq,mask).detach()
    loss=-((torch.tensor([.2,.4,.1,.3],device=device)-.01*(check.detach()-rlp)).detach()*check).mean()
    loss.backward(); gn=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True))
    assert gn>0 and not any(p.grad is not None for p in ref.parameters())
    scores={}
    for task in REFS:
        s=Scorer(task,tok);m=Chem.MolFromSmiles(REFS[task]);c=s.components(m)
        assert all(0<=x<=1 for x in c);scores[task]={'reference_components':c,'reference_reward':math.prod(c)**(1/len(c))}
    result=dict(status='PASS',sample_vs_recompute_max_abs=err,padding_invariance_max_abs=perr,finite_nonzero_gradient_norm=gn,reference_has_gradient=False,scorer_references=scores)
    save(out/'technical_gate.json',result);del actor,ref
    if device.type=='cuda':torch.cuda.empty_cache()
    return result

def run_one(template,tok,device,root,task,method,s,limits,settings,global_deadline):
    out=root/f'{task}__{method}__s{s}';out.mkdir(parents=True,exist_ok=False)
    seed(s); actor=copy.deepcopy(template).to(device).eval().requires_grad_(method!='frozen')
    ref=copy.deepcopy(template).to(device).eval().requires_grad_(False)
    reference_hash=rt.parameter_hash(ref)
    optimizer=None if method=='frozen' else torch.optim.Adam(actor.parameters(),lr=settings['lr'])
    scorer=Scorer(task,tok); seen={};curve=[];attempt=0;updates=0;ema=0.;start=time.monotonic()
    beta=settings['beta'] if method=='reinforce_kl' else 0.
    save(out/'config.json',dict(task=task,method=method,seed=s,limits=limits,settings=settings,beta=beta))
    ledger=(out/'search.jsonl').open('w');steps=(out/'steps.jsonl').open('w')

    def checkpoint():
        tmp=out/'checkpoint.tmp'
        torch.save(dict(actor=actor.state_dict(),optimizer=optimizer.state_dict() if optimizer else None,steps=updates,
                        attempts=attempt,unique=len(seen),ema=ema,scorer_cache=scorer.cache,seen=seen,curve=curve,
                        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                        config=json.loads((out/'config.json').read_text())),tmp)
        tmp.replace(out/'checkpoint.pt')

    reason='unknown'
    while True:
        elapsed=time.monotonic()-start
        if STOP or time.monotonic()>global_deadline-90:reason='global_deadline_or_signal';break
        if len(seen)>=limits['unique_budget']:reason='unique_budget';break
        if attempt>=limits['max_attempts']:reason='attempt_limit';break
        if optimizer and updates>=limits['max_updates']:reason='update_limit';break
        if elapsed>=limits.get('run_wall_seconds',1800):reason='wall_limit';break
        n=min(settings['batch_size'],limits['unique_budget']-len(seen),limits['max_attempts']-attempt)
        seq,mask,natural,sample_lp=rt.rollout(actor,n,settings['horizon'],device)
        recs=[scorer.decode(ids,e) for ids,e in zip(seq.cpu().tolist(),natural.cpu().tolist())]
        for rec in recs:
            attempt+=1;rec.update(attempt=attempt,update=updates,elapsed_s=time.monotonic()-start)
            fresh=rec['valid'] and rec['identity'] not in seen
            rec['new_unique']=fresh
            if fresh:
                seen[rec['identity']]=rec['reward']
                best=sorted(seen.values(),reverse=True)[:10]
                top10=sum(best)/10. # pad unavailable entries with zero, explicitly fixed
                curve.append(dict(query=len(seen),top10=top10,attempt=attempt,elapsed_s=rec['elapsed_s']))
            rec['unique_queries']=len(seen);ledger.write(json.dumps(rec)+'\n')
        reward=torch.tensor([r['reward'] for r in recs],device=device)
        with torch.set_grad_enabled(optimizer is not None):
            actor_lp,ref_lp,prefix_kl=likelihoods_and_kl(actor,ref,seq,mask)
        kl=sample_lp-ref_lp
        stats=dict(update=updates,attempts=attempt,unique=len(seen),mean_reward=float(reward.mean()),sample_trajectory_kl=float(kl.mean()),full_vocabulary_prefix_kl=float(prefix_kl),elapsed_s=time.monotonic()-start)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
            discrepancy=float((actor_lp.detach()-sample_lp).abs().max())
            if discrepancy>.005:raise RuntimeError(f'logprob mismatch {discrepancy}')
            g=(reward-beta*(actor_lp.detach()-ref_lp)).detach()
            b=(g.sum()-g)/(n-1) if n>1 else torch.full_like(g,ema)
            advantage=(g-b).detach();loss=-(advantage*actor_lp).mean();loss.backward()
            gn=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),settings['gradient_norm_clip'],error_if_nonfinite=True))
            optimizer.step();updates+=1;ema=.9*ema+.1*float(g.mean())
            stats.update(loss=float(loss.detach()),grad_norm=gn,advantage_variance=float(advantage.var(unbiased=False)),logprob_error=discrepancy)
        steps.write(json.dumps(stats)+'\n');steps.flush();ledger.flush()
        save(out/'progress.json',stats)
        if updates%25==0 and optimizer:checkpoint()
    ledger.close();steps.close();checkpoint()
    assert rt.parameter_hash(ref)==reference_hash
    search_elapsed=time.monotonic()-start
    # Independent deterministic evaluation RNG, no checkpoint selection or updates.
    seed(100000+s);evaluation=[]
    if not STOP and time.monotonic()<global_deadline-60:
        with (out/'evaluation.jsonl').open('w') as ef:
            for off in range(0,limits['evaluation_attempts'],32):
                if STOP or time.monotonic()>global_deadline-30:break
                n=min(32,limits['evaluation_attempts']-off)
                sq,mk,en,_=rt.rollout(actor,n,settings['horizon'],device)
                rs=[scorer.decode(ids,e) for ids,e in zip(sq.cpu().tolist(),en.cpu().tolist())]
                for r in rs:ef.write(json.dumps(r)+'\n')
                evaluation.extend(rs)
    ys=[x['top10'] for x in curve];B=limits['unique_budget']
    achieved_auc=sum(ys)/len(ys) if ys else 0.
    fixed_auc=(sum(ys)+(B-len(ys))*(ys[-1] if ys else 0.))/B
    result=dict(status='COMPLETE' if len(evaluation)==limits['evaluation_attempts'] else 'PARTIAL',task=task,method=method,seed=s,
                stop_reason=reason,attempts=attempt,updates=updates,unique_queries=len(seen),search_wall_s=search_elapsed,
                top10=ys[-1] if ys else 0.,auc_achieved=achieved_auc,auc_fixed_budget_carry_forward=fixed_auc,
                carry_forward_queries=B-len(ys),auc_rule='right-endpoint unit-query mean; top10 pads missing with zero',
                max_reward=max(seen.values()) if seen else 0.,physical_scoring_calls_search_and_eval=scorer.physical_calls,
                evaluation=output_metrics(evaluation) if evaluation else None,reference_unchanged=True,
                actor_parameter_hash=rt.parameter_hash(actor),peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
    save(out/'result.json',result);print('RUN_RESULT '+json.dumps(result),flush=True)
    del actor,ref,optimizer
    if device.type=='cuda':torch.cuda.empty_cache()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--contract',required=True);p.add_argument('--device',default='cuda');p.add_argument('--phase',choices=['gate','dev','all'],default='dev')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True);contract=json.loads(Path(a.contract).read_text())
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');device=torch.device(a.device)
    if device.type=='cuda':
        assert torch.cuda.is_available();assert 'H100' in torch.cuda.get_device_name(0)
    md=Path(a.model_dir);assert rt.sha256_file(md/'pytorch_model.bin')==rt.WEIGHT_SHA256
    template,info=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,torch_dtype=torch.float32,output_loading_info=True,weights_only=True)
    assert not info.get('missing_keys') and not info.get('mismatched_keys')
    assert all(k.endswith('.attn.attention.bias') or k.endswith('.attn.attention.masked_bias') for k in info.get('unexpected_keys',[])),info
    assert all(float(getattr(template.config,k,-1))==0 for k in ['attention_dropout','embed_dropout','resid_dropout'])
    template.eval();tok=Tokenizer.from_file(str(md/'tokenizer.json'))
    save(root/'environment.json',dict(torch=torch.__version__,device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,model_sha256=rt.WEIGHT_SHA256,contract=contract))
    validate_model(template,tok,device,root)
    if a.phase=='gate':return
    deadline=time.monotonic()+contract['resource_budget']['maximum_worker_seconds']-180
    all_results=[]
    for phase in ['development']+(['main'] if a.phase=='all' else []):
        limits=contract[phase];tasks=limits.get('tasks',[limits.get('task')]);(root/phase).mkdir(exist_ok=True)
        for task in tasks:
            for s in limits['seeds']:
                for method in limits['methods']:
                    if STOP or time.monotonic()>deadline-120:
                        save(root/'campaign.json',dict(status='BUDGET_STOP',results=all_results));return
                    result=run_one(template,tok,device,root/phase,task,method,s,limits,contract['fixed_training'],deadline)
                    all_results.append(result);save(root/'campaign.json',dict(status='RUNNING',results=all_results))
                    if result['status']!='COMPLETE':raise RuntimeError('incomplete run: retain evidence and stop')
        # Development is a technical gate, not selection for a positive result.
        if phase=='development':save(root/'development_gate.json',dict(status='PASS',runs=len(all_results),selection='fixed proposal hyperparameters; no main-task tuning'))
    save(root/'campaign.json',dict(status='COMPLETE',results=all_results))

if __name__=='__main__':main()
