"""Exact finite-MDP check of sequence REINFORCE with reverse KL and LOO."""
import itertools
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
theta = torch.tensor([.2,-.1,.4,.05,-.3,.6], requires_grad=True)
ref = torch.tensor([-.1,.3,.1,-.2,.2,.4])
paths=list(itertools.product(range(2), repeat=2))

def logprob(z, path):
    a,b=path
    return z[:2].log_softmax(0)[a]+z[2+2*a:4+2*a].log_softmax(0)[b]

lp=torch.stack([logprob(theta,p) for p in paths])
lp0=torch.stack([logprob(ref,p) for p in paths])
probs=lp.exp()
reward=torch.tensor([.2,1.,.8,0.])
beta=.03
value=reward-beta*(lp-lp0)
objective=(probs*value).sum()
true_grad=torch.autograd.grad(objective,theta,retain_graph=True)[0]
surrogate=(probs.detach()*value.detach()*lp).sum()
surrogate_grad=torch.autograd.grad(surrogate,theta,retain_graph=True)[0]
# Expected gradient for independently sampled batch of two, LOO baseline.
loo_surrogate=0.
for i,j in itertools.product(range(4),repeat=2):
    weight=(probs[i]*probs[j]).detach()
    adv_i=(value[i]-value[j]).detach()
    adv_j=(value[j]-value[i]).detach()
    loo_surrogate += weight*(adv_i*lp[i]+adv_j*lp[j])/2
loo_grad=torch.autograd.grad(loo_surrogate,theta)[0]
error=float((true_grad-surrogate_grad).abs().max())
loo_error=float((true_grad-loo_grad).abs().max())
assert error<1e-12 and loo_error<1e-12
result={'status':'PASS','scope':'exact 2-step, 2-action enumerated toy MDP; not chemical efficacy',
        'beta':beta,'objective':'E_q[R]-beta*KL(q||q_ref)',
        'estimator':'stopgrad(R-beta*(logq-logq_ref)-LOO_baseline)*grad(logq)',
        'max_gradient_error_no_baseline':error,'max_gradient_error_LOO_batch2':loo_error,
        'true_gradient':true_grad.tolist(),'trajectory_count':4,'independent_batch_pairs':16}
(Path(__file__).parent/'gradient_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
