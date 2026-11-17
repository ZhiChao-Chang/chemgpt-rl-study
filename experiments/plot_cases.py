"""Seed-level mechanistic figures from all available completed main-run ledgers."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'evidence/cases'
data=json.loads((OUT/'cases.json').read_text())['runs']
methods=['frozen','reinforce','reinforce_kl'];names=['Frozen','REINFORCE','RF + KL']
for task in sorted({d['task'] for d in data}):
    rows=[r for r in data if r['task']==task];seeds=sorted({r['seed'] for r in rows})
    figures=[('property_shifts',[('MW','Molecular weight (g/mol)'),('TPSA','TPSA (A²)'),('token_length','Generated actions incl. EOS')]),
             ('component_shifts',[(k,k) for k in rows[0]['diagnostics']['components_valid_outputs']])]
    for kind,keys in figures:
        fig,axes=plt.subplots(1,len(keys),figsize=(4*len(keys),3.8),layout='constrained')
        for ax,(key,title) in zip(axes,keys):
            for seed in seeds:
                xs=[];ys=[]
                for i,m in enumerate(methods):
                    r=next((r for r in rows if r['method']==m and r['seed']==seed),None)
                    if r:
                        vals=r['diagnostics']['properties_valid_outputs'] if kind=='property_shifts' else r['diagnostics']['components_valid_outputs']
                        if vals[key] is not None:
                            xs.append(i);ys.append(vals[key]['median' if kind=='property_shifts' else 'mean'])
                ax.plot(xs,ys,'o-',label=f'Seed {seed}',linewidth=1.3,alpha=.85)
            ax.set_xticks(range(3),names,rotation=15);ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
            ax.set_ylabel('Median of valid terminal outputs' if kind=='property_shifts' else 'Mean normalized component')
            if kind=='component_shifts':ax.set_ylim(0,1.05)
        axes[0].legend(frameon=False,fontsize=8)
        fig.suptitle(f'{task} | completed seeds only | terminal evaluation',fontsize=12)
        for ext in ['png','pdf']:fig.savefig(OUT/f'{task}_{kind}.{ext}',dpi=170)
        plt.close(fig)
print('Saved property/component figures; seed-level points, no significance claim.')
