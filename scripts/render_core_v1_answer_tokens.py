"""Render exact-token heatmap and budget coverage from saved core v1 analysis."""
import argparse
import json
import os
from pathlib import Path

from src.common import ROOT


def render(folder):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/aime-answer-tokens-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    data=json.loads((folder/'analysis.json').read_text())
    rows=data['rows'];seeds=data['seeds'];values=np.full((30,5),np.nan)
    for r in rows:
        if r['verified']:values[r['question']-1,seeds.index(r['seed'])]=r['tokens_to_winning_candidate']
    fig,(ax,counts)=plt.subplots(1,2,figsize=(10.8,13.5),gridspec_kw={'width_ratios':[5,1],'wspace':.10})
    cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#edf0f3')
    chart=ax.imshow(values,aspect='auto',cmap=cmap,vmin=0,vmax=max(r['tokens_to_winning_candidate'] for r in rows if r['verified']))
    for r in rows:
        x,y=seeds.index(r['seed']),r['question']-1
        if r['verified']:
            v=r['tokens_to_winning_candidate'];color='white' if v>7000 else '#172b3a'
            label=f'{v:,}'
            if r['verification_slot'] in (17,18):
                label+=f" · {r['verification_slot']}"
                ax.add_patch(Rectangle((x-.48,y-.47),.96,.94,fill=False,edgecolor='#d35637',linewidth=1.8))
            ax.text(x,y,label,ha='center',va='center',fontsize=8.5,color=color)
        else:ax.text(x,y,'—',ha='center',va='center',color='#8c98a0',fontsize=11)
    ax.set_xticks(range(5),[str(s) for s in seeds],fontsize=9)
    ax.xaxis.tick_top();ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False)
    ax.set_yticks(range(30),[f'Q{q:02d}' for q in range(1,31)],fontsize=9)
    ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,30,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.1);ax.tick_params(which='minor',bottom=False,left=False)
    for spine in ax.spines.values():spine.set_visible(False)
    ns=[q['verified_trials'] for q in data['by_question']]
    counts.barh(range(30),ns,color='#7798ab',height=.62)
    counts.set_ylim(29.5,-.5);counts.set_xlim(0,6.5);counts.set_yticks([]);counts.set_xticks([0,5],['0','5'])
    counts.set_title('Verified\n/ 5 trials',fontsize=10,pad=12)
    for y,n in enumerate(ns):counts.text(n+.15,y,str(n),va='center',fontsize=9,color='#344e5e')
    counts.spines[['top','right','left']].set_visible(False);counts.grid(axis='x',alpha=.18)
    fig.suptitle('Core v1 · output tokens to the winning candidate',fontsize=16,x=.5,y=.986)
    fig.text(.5,.957,'Exact streamed token IDs, including prior continuation output; prompt tokens excluded',ha='center',fontsize=10,color='#50616e')
    fig.text(.07,.035,'Red outlines mark verification slots 17 / 18; the slot follows the token count.\n— means no positive verdict before the global stop; it is not a zero-token result.',fontsize=10,color='#50616e')
    fig.subplots_adjust(top=.917,bottom=.082,left=.075,right=.98)
    cbar=fig.colorbar(chart,ax=[ax,counts],location='bottom',fraction=.022,pad=.020,aspect=60)
    cbar.set_label('Generated output tokens at candidate detection',fontsize=9)
    for ext in ('png','svg','pdf'):fig.savefig(folder/f'answer-token-distribution.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5.1));colors=['#2b67aa','#cf7d24','#24835a','#8661a8','#b84e60']
    for seed,color in zip(seeds,colors):
        vals=sorted(r['tokens_to_winning_candidate'] for r in rows if r['seed']==seed and r['verified'])
        ax.step([0]+vals,[0]+list(range(1,len(vals)+1)),where='post',color=color,label=str(seed),linewidth=1.8)
    for cap in (4096,8192):ax.axvline(cap,color='#748393',linestyle='--',linewidth=1)
    ax.text(4096,1,'4K',ha='right',color='#667887');ax.text(8192,1,'8K',ha='right',color='#667887')
    ax.axhline(18,color='#667887',linestyle=':',linewidth=1)
    ax.set_ylim(0,19);ax.set_xlim(0,13000);ax.set_yticks(range(0,19,2))
    ax.set_xlabel('Cumulative output tokens to the subsequently verified candidate')
    ax.set_ylabel('Retained observed wins per trial')
    ax.set_title('Which observed wins survive a cumulative token cap?')
    ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=9,loc='lower right')
    fig.text(.10,.015,'Descriptive retention of recorded wins; does not simulate replacement answers or changed decoding speed.',fontsize=9,color='#50616e')
    fig.tight_layout(rect=(0,.04,1,1))
    for ext in ('png','svg','pdf'):fig.savefig(folder/f'budget-coverage.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path,default=ROOT/'runs/analyses/core-v1-five-seeds-answer-tokens');render(parser.parse_args().folder)
