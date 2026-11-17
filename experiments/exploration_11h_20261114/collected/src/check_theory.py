"""Exact enumeration: KL gradient, LOO, biased control and output KL chain rule."""
import itertools
import json
from pathlib import Path
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
z=torch.tensor([.2,-.1,.4,.05,-.3,.6],requires_grad=True)
r0=torch.tensor([-.1,.3,.1,-.2,.2,.4])
paths=list(itertools.product(range(2),repeat=2))
def logprob(theta,path):
    a,b=path
    return theta[:2].log_softmax(0)[a]+theta[2+2*a:4+2*a].log_softmax(0)[b]
l=torch.stack([logprob(z,t) for t in paths]);l0=torch.stack([logprob(r0,t) for t in paths])
p=l.exp();p0=l0.exp();reward=torch.tensor([.2,1.,.8,0.]);beta=.03
g=reward-beta*(l-l0)
j=(p*g).sum();exact=torch.autograd.grad(j,z,retain_graph=True)[0]
score=(p.detach()*g.detach()*l).sum();sg=torch.autograd.grad(score,z,retain_graph=True)[0]
loo=0.;selfmean=0.
for i,k in itertools.product(range(4),repeat=2):
    w=(p[i]*p[k]).detach();a=(g[i]-g[k]).detach()
    loo+=w*a*(l[i]-l[k])/2
    selfmean+=w*a*(l[i]-l[k])/4
lg=torch.autograd.grad(loo,z,retain_graph=True)[0]
bg=torch.autograd.grad(selfmean,z,retain_graph=True)[0]
assert torch.allclose(exact,sg,atol=1e-12)
assert torch.allclose(exact,lg,atol=1e-12)
assert torch.allclose(bg,exact/2,atol=1e-12) and not torch.allclose(bg,exact)
# A deterministic decoder merges the first two trajectories into one molecule.
groups=[[0,1],[2],[3]]
q=torch.stack([p[ix].sum() for ix in groups]);q0=torch.stack([p0[ix].sum() for ix in groups])
klt=(p*(l-l0)).sum();klm=(q*(q.log()-q0.log())).sum()
conditional=0.
for k,ix in enumerate(groups):
    cp=p[ix]/q[k];cp0=p0[ix]/q0[k]
    conditional+=q[k]*(cp*(cp.log()-cp0.log())).sum()
assert float(conditional)>0
assert torch.allclose(klt,klm+conditional,atol=1e-12)
result=dict(status='PASS',trajectory_count=4,independent_batches=16,
    gradient_error=float((exact-sg).abs().max()),loo_error=float((exact-lg).abs().max()),
    self_including_batch_mean_gradient_ratio=.5,negative_control='batch-mean baseline (N=2) halves expected gradient in this setting',
    trajectory_kl=float(klt.detach()),decoded_output_kl=float(klm.detach()),conditional_encoding_kl=float(conditional.detach()),
    chain_rule_error=float((klt-klm-conditional).abs().detach()),
    theorem='For fixed deterministic decoder f and common support: KL(P_tau||P0_tau)=KL(P_f||P0_f)+E_m KL(P(tau|m)||P0(tau|m)).',
    caveat='Exact finite example and identity; not an empirical estimate of molecular KL, nor evidence of improved drug efficacy.')
out=Path(__file__).resolve().parents[1]/'evidence/theory.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
