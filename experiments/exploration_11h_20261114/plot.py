"""Scientific figures for completed extension runs, with seed-level observations."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
rows=[json.loads(p.read_text()) for p in (ROOT/'collected/results').rglob('result.json')]
details=json.loads((ROOT/'evidence/detail_metrics.json').read_text())
figs=[]
for task in ['fexofenadine','osimertinib']:
    rs=[r for r in rows if r['phase']=='kl_dose' and r['task']==task and r['method']!='frozen' and r.get('evaluation')]
    if not rs:continue
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    getters=[lambda r:r['evaluation']['mean_reward'],lambda r:r['terminal_mean_sample_trajectory_kl'],
             lambda r:details[f"{r['phase']}/{r['task']}__{r['method']}__s{r['seed']}"]['mean_actions_all_attempts']]
    for ax,getter,title in zip(axes,getters,['Terminal original MPO','Terminal trajectory KL estimate','Generated actions (mean)']):
        for seed in sorted({r['seed'] for r in rs}):
            x=sorted([r for r in rs if r['seed']==seed],key=lambda r:r['beta'])
            ax.plot([r['beta'] for r in x],[getter(r) for r in x],'o-',alpha=.25,linewidth=1)
        bs=sorted({r['beta'] for r in rs});ys=[statistics.mean(getter(r) for r in rs if r['beta']==b) for b in bs]
        ax.plot(bs,ys,'o-',color='#125d9c',linewidth=2,label='Available-seed mean')
        ax.set_xscale('symlog',linthresh=.002);ax.set_xticks([0,.003,.01,.03,.1],['0','.003','.01','.03','.1'])
        ax.set_xlabel('Trajectory KL beta');ax.set_title(title,fontsize=11);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18)
    axes[0].legend(frameon=False,fontsize=8);fig.suptitle(f'{task} | independent extension seeds | completed runs only',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(ROOT/'evidence'/f'{task}_kl_dose.{ext}',dpi=170)
    plt.close(fig);figs.append(f'evidence/{task}_kl_dose.png')
(ROOT/'evidence/figures.json').write_text(json.dumps(figs))
print(json.dumps({'figures':len(figs)}))
