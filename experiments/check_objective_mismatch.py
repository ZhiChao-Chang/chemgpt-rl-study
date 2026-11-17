"""Exact counterexamples: expected reward need not improve best-of-B or unique top-k."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=.01;B=1000;k=10
rare_topk=(k-sum((k-n)*math.comb(B,n)*p**n*(1-p)**(B-n) for n in range(k)))/k
d={'status':'PASS','mean_vs_tail':{'policy_A':'reward always 0.6','policy_B':'reward 1 with probability .01, else 0',
   'mean_A':.6,'mean_B':p,'expected_max_A_B100':.6,'expected_max_B_B100':1-(1-p)**100,
   'expected_top10_A_B1000':.6,'expected_top10_B_B1000':rare_topk},
   'identity_collapse':{'policy_C':'always one identical molecule, reward1','policy_D':'at least ten distinct molecules each reward .8, once discovered',
   'per_attempt_mean_C':1.,'unique_top10_zero_padded_C':.1,'unique_top10_D_after_ten_discoveries':.8},
   'formula':'For iid bounded rewards under a FIXED policy: E[max(R1..RB)]=integral_0^1 (1-F(r)^B) dr; E[topk average]=integral_0^1 E[min(k,N_r)]/k dr, N_r~Binomial(B,1-F(r)).',
   'boundary':'Formula concerns iid raw attempts, NOT adaptive training or a budget counted in unique canonical molecules. Counterexamples show absence of a general implication, not actual model behavior.'}
assert d['mean_vs_tail']['mean_A']>p and d['mean_vs_tail']['expected_max_B_B100']>.6 and rare_topk>.6
(ROOT/'evidence/objective_mismatch.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
