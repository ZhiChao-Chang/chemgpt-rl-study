"""Exploratory figure: show each seed; search and terminal policy stay separate."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
ds=json.loads((ROOT/'evidence/development_summary.json').read_text())['groups']
methods=['frozen','reinforce','reinforce_kl'];names=['Frozen','REINFORCE','REINFORCE + KL']
fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
for ax,metric,title in zip(axes,['auc','evaluation_mean_reward'],['Candidate discovery: top-10 AUC','Terminal policy: mean reward']):
    for seed_index in range(2):
        ys=[next(d for d in ds if d['method']==m)[metric]['values'][seed_index] for m in methods]
        ax.plot(range(3),ys,'o-',label=f'Seed {[11,17][seed_index]}',alpha=.85,linewidth=1.5)
    ax.set_xticks(range(3),names,rotation=15);ax.set_title(title,fontsize=11);ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8)
fig.suptitle('Perindopril development only | 512 unique queries | 2 seeds',fontsize=12)
for ext in ['png','pdf']:fig.savefig(ROOT/'evidence'/f'development_metrics.{ext}',dpi=180)
print('Saved development_metrics.png / .pdf; exploratory, not main results.')
