from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

base=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description='Render the recorded CartPole curve ensemble.')
parser.add_argument('--output-dir',type=Path,default=base/'docs/learning-curves')
args=parser.parse_args()
data=json.loads((base/'experiments/curve-analysis/results.json').read_text())
out=args.output_dir
out.mkdir(parents=True,exist_ok=True)
ink='#253044'; muted='#728096'; line='#e5eaf2'; blue='#305cff'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':ink,'axes.labelcolor':muted,'axes.edgecolor':line,'xtick.color':muted,'ytick.color':muted,'axes.titlesize':14,'axes.titleweight':'normal','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
steps_format=FuncFormatter(lambda n,_: '0' if n==0 else f'{int(n/1000)}k')
runs={run['seed']:run for run in data['runs']}
groups=sorted(data['clustering']['groups'],key=lambda group:group['n'],reverse=True)

def style(ax,x_max=81920):
    ax.set_ylim(0,525);ax.set_yticks([0,250,500]);ax.set_xlim(0,x_max)
    ax.set_xticks([0,20000,40000,60000,80000]);ax.xaxis.set_major_formatter(steps_format)
    ax.grid(axis='y',color=line,linewidth=.8);ax.set_axisbelow(True);ax.tick_params(length=0,pad=8)
    ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color(line)
    ax.set_xlabel('Training steps',labelpad=10)

fig,axes=plt.subplots(1,3,figsize=(14,5.1),sharey=True)
fig.subplots_adjust(left=.055,right=.985,bottom=.25,top=.70,wspace=.13)
fig.text(.055,.94,'Three representative learning-curve groups',size=19,weight='normal',color=ink)
fig.text(.055,.865,'128 fresh PPO runs · same settings · groups ranked by observed frequency',size=12,color=muted)
for i,(ax,g) in enumerate(zip(axes,groups)):
    for seed in g['memberSeeds']:
        points=runs[seed]['history'];ax.plot([p['steps'] for p in points],[p['score'] for p in points],color='#cad3e2',alpha=.30,lw=.65,zorder=1)
    q=g['quantiles'];x=np.array([p['steps'] for p in q]);ax.fill_between(x,[p['p10'] for p in q],[p['p90'] for p in q],color=blue,alpha=.09,lw=0,zorder=2)
    curve=g['curve'];ax.plot([p['steps'] for p in curve],[p['score'] for p in curve],color=blue,lw=2.4,zorder=3)
    style(ax);label={1:'Earlier rise',2:'Gradual rise',3:'Dip and recovery'}[g['id']]
    ax.set_title(f"{i+1}. {label} · {100*g['share']:.1f}%",loc='left',pad=21)
    ax.text(.0,1.015,f"{g['n']} of 128 runs · representative seed {g['medoidSeed']}",transform=ax.transAxes,size=10,color=muted,ha='left',va='bottom')
axes[0].set_ylabel('Recent average reward\n(last 20 episodes)',labelpad=10)
fig.text(.055,.075,'Blue: one actual representative run. Pale lines: other runs in that group. Shading: pointwise 10th–90th percentiles.',size=10,color=muted)
fig.text(.055,.032,'Three groups were imposed by curve similarity; these percentages do not establish three universal patterns.',size=10,color=muted)
fig.savefig(out/'three-representative-groups.png',dpi=160,bbox_inches='tight',pad_inches=.22)
fig.savefig(out/'three-representative-groups.svg',bbox_inches='tight',pad_inches=.22)
plt.close(fig)

fig,(ax,hist)=plt.subplots(1,2,figsize=(13.8,5.2),gridspec_kw={'width_ratios':[1.65,1]})
fig.subplots_adjust(left=.065,right=.98,bottom=.25,top=.74,wspace=.23)
fig.text(.065,.945,'A distribution of curves, rather than one guaranteed path',size=18,weight='normal',color=ink)
fig.text(.065,.87,'The same learner reaches the same stage at different speeds and can dip along the way.',size=12,color=muted)
for run in runs.values():
    points=run['history'];ax.plot([p['steps'] for p in points],[p['score'] for p in points],color='#a8b7d3',alpha=.2,lw=.65)
overall=data['overall'];x=[p['steps'] for p in overall]
ax.fill_between(x,[p['p10'] for p in overall],[p['p90'] for p in overall],color=blue,alpha=.12,lw=0)
ax.plot(x,[p['median'] for p in overall],color=blue,lw=2.4,label='Median of 128 runs')
style(ax);ax.set_ylabel('Recent average reward',labelpad=10);ax.legend(loc='lower right',frameon=False,fontsize=11)
ax.set_title('All 128 sampled learning curves',loc='left',pad=15)
h=data['checkpointHistogram'];scores=np.asarray(h['scores'],dtype=float)
hist.hist(scores,bins=np.arange(0,526,25),color=blue,alpha=.80,edgecolor='white',linewidth=.6)
hist.set_xlim(0,500);hist.set_xlabel('Recent average reward',labelpad=10);hist.set_ylabel('Number of runs',labelpad=10)
hist.set_title(f"At {h['steps']:,} training steps",loc='left',pad=15);hist.set_xticks([0,250,500]);hist.grid(axis='y',color=line,linewidth=.8);hist.set_axisbelow(True);hist.tick_params(length=0,pad=8);hist.spines['left'].set_visible(False)
fig.text(.065,.075,'Shading: pointwise 10th–90th percentiles, not a confidence interval. At 81,920 steps, all 128 recent averages equal 500.',size=10,color=muted)
fig.text(.065,.032,'These 128 curves are a sample of possible runs. Different algorithms, settings, budgets, or scores produce different distributions.',size=10,color=muted)
fig.savefig(out/'sampled-curve-distribution.png',dpi=160,bbox_inches='tight',pad_inches=.22)
fig.savefig(out/'sampled-curve-distribution.svg',bbox_inches='tight',pad_inches=.22)
plt.close(fig)
for name in ['three-representative-groups.svg','sampled-curve-distribution.svg']:
    svg=out/name
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
print(json.dumps({'files': ['three-representative-groups.png','sampled-curve-distribution.png'], 'groupCounts':[g['n'] for g in groups], 'groupSeeds':[g['medoidSeed'] for g in groups]}))
