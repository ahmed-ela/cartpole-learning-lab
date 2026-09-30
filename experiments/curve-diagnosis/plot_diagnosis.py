from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parents[2]
scratch=base/'experiments/curve-diagnosis'
parser=argparse.ArgumentParser(description='Render the recorded frozen-policy dip diagnosis.')
parser.add_argument('--output-dir',type=Path,default=base/'docs/learning-curves')
args=parser.parse_args()
out=args.output_dir
out.mkdir(parents=True,exist_ok=True)
ensemble=json.loads((base/'experiments/curve-analysis/results.json').read_text())
training=next(r for r in ensemble['runs'] if r['seed']==162066966)
evaluation=json.loads((scratch/'evaluation/summary.json').read_text())
holdout=json.loads((scratch/'evaluation/holdout-validation.json').read_text())
epochs=json.loads((scratch/'audit/epoch-summary.json').read_text())['updates'][0]['rows']
interventions=json.loads((scratch/'interventions/summary.json').read_text())

ink='#263349'; muted='#66758b'; grid='#e6ebf2'; blue='#305cff'; orange='#e96c45'; green='#14866d'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':ink,'axes.labelcolor':muted,
    'axes.edgecolor':grid,'xtick.color':muted,'ytick.color':muted,'figure.facecolor':'white',
    'axes.facecolor':'white','savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(13.8,9.7))
fig.subplots_adjust(left=.075,right=.965,top=.835,bottom=.20,hspace=.52,wspace=.34)
fig.text(.075,.952,'What caused this CartPole dip?',fontsize=21)
fig.text(.075,.906,'One replayed PPO run · harmful controller bias · a delayed chart · controlled checkpoint tests',fontsize=12,color=muted)

def style(ax,title,ylabel=None):
    ax.set_title(title,loc='left',fontsize=13,pad=15)
    ax.grid(axis='y',color=grid,lw=.8);ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False);ax.tick_params(length=0,pad=7)
    if ylabel:ax.set_ylabel(ylabel,labelpad=9)

ax=axs[0,0]
t=[h for h in training['history'] if 8<=h['update']<=30]
c=[h for h in evaluation['checkpoints'] if 8<=h['update']<=30]
ax.plot([h['update'] for h in t],[h['score'] for h in t],color='#8796ad',lw=2,label='Chart: last 20 training episodes')
ax.plot([h['update'] for h in c],[h['sampled']['returns']['mean'] for h in c],color=blue,lw=2.3,label='Fresh frozen-policy evaluation')
ax.fill_between([h['update'] for h in c],[h['sampled']['returns']['ci95'][0] for h in c],
    [h['sampled']['returns']['ci95'][1] for h in c],color=blue,alpha=.13,lw=0)
ax.axvspan(14,16,color=orange,alpha=.09,lw=0)
ax.annotate('Actual loss: 330 → 165',xy=(16,165.37),xytext=(17.4,105),fontsize=10,color=orange,
    arrowprops={'arrowstyle':'-','color':orange,'lw':1})
ax.annotate('Chart reaches its trough later',xy=(19,204.4),xytext=(19.8,265),fontsize=9,color=muted,
    arrowprops={'arrowstyle':'-','color':'#8796ad','lw':1})
ax.set_xlim(8,30);ax.set_ylim(75,525);ax.set_xticks([8,14,16,19,24,30]);ax.set_yticks([100,200,300,400,500])
ax.set_xlabel('PPO update (1,024 training steps each)',labelpad=9)
style(ax,'1. The chart lags the real regression','Mean survival steps')
ax.legend(loc='upper left',frameon=False,fontsize=9)

ax=axs[0,1]
x=np.array([r['epoch'] for r in epochs]);y=np.array([r['sampled']['returns']['mean'] for r in epochs])
ax.plot(x,y,color=blue,marker='o',lw=2,ms=4,label='Fresh performance')
ax.fill_between(x,[r['sampled']['returns']['ci95'][0] for r in epochs],
    [r['sampled']['returns']['ci95'][1] for r in epochs],color=blue,alpha=.12,lw=0)
ax.set_ylim(140,370);ax.set_yticks([150,200,250,300,350]);ax.set_xticks([0,2,4,6,8]);ax.set_xlabel('Optimization epochs within update 15',labelpad=9)
style(ax,'2. Optimizing the batch made behavior worse','Mean survival steps')
right=ax.twinx();right.plot(x,[r['batch']['surrogate'] for r in epochs],color=orange,lw=2,marker='s',ms=3,label='PPO batch surrogate')
right.set_ylim(-.002,.016);right.set_yticks([0,.005,.01,.015]);right.set_ylabel('Estimated batch objective',color=orange,labelpad=8)
right.tick_params(axis='y',colors=orange,length=0);right.spines['top'].set_visible(False);right.spines['left'].set_visible(False);right.spines['right'].set_visible(False)
ax.legend([ax.lines[0],right.lines[0]],['Fresh performance','PPO batch surrogate'],frameon=False,fontsize=9,loc='center right')

ax=axs[1,0]
old=next(r for r in holdout['original'] if r['update']==14)['summary']['returns']
new=next(r for r in holdout['original'] if r['update']==15)['summary']['returns']
centered_old=next(r for r in holdout['centered'] if r['update']==14)['summary']['returns']
centered_new=next(r for r in holdout['centered'] if r['update']==15)['summary']['returns']
summaries=[old,new,centered_old,centered_new];means=[r['mean'] for r in summaries]
colors=['#8796ad',orange,'#82b5a7',green]
bars=ax.bar(np.arange(4),means,color=colors,width=.63,zorder=2)
errors=np.array([[r['mean']-r['ci95'][0] for r in summaries],[r['ci95'][1]-r['mean'] for r in summaries]])
ax.errorbar(np.arange(4),means,yerr=errors,fmt='none',ecolor=ink,capsize=3,lw=1,zorder=3)
for b,m in zip(bars,means):ax.text(b.get_x()+b.get_width()/2,m+13,f'{m:.0f}',ha='center',fontsize=11)
ax.set_xticks(np.arange(4));ax.set_xticklabels(['Before\n(original)','After\n(original)','Before\n(bias removed)','After\n(bias removed)'],fontsize=9)
ax.set_ylim(0,420);ax.set_yticks([0,100,200,300,400]);ax.set_xlabel('Independent 2,000-episode holdout · checkpoints 14 → 15',labelpad=12,fontsize=10)
style(ax,'3. Removing only the bias nearly removes the drop','Mean survival steps')

ax=axs[1,1]
cps=[14,15,16,18,20,40]
cp_map={r['update']:r for r in evaluation['checkpoints']}
bottom=np.zeros(len(cps));cause_labels=[('cart','Cart boundary',orange),('pole','Pole falls','#f2bc61'),('both','Both thresholds','#a57bcb'),('cap','Survives to 500',green)]
for cause,label,color in cause_labels:
    values=np.array([cp_map[u]['sampled']['causes'][cause]['fraction']*100 for u in cps])
    ax.bar(np.arange(len(cps)),values,bottom=bottom,width=.63,color=color,label=label)
    bottom+=values
ax.set_xticks(np.arange(len(cps)));ax.set_xticklabels([str(u) for u in cps]);ax.set_ylim(0,120);ax.set_yticks([0,25,50,75,100]);ax.set_xlabel('Frozen checkpoint update',labelpad=9)
style(ax,'4. The cart runs out of track while balancing','Share of test episodes (%)')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False,ncol=2,fontsize=9)
ax.text(.02,.98,'At 15: all cart-only failures go left',transform=ax.transAxes,va='top',color=ink,fontsize=9)

fig.text(.075,.060,'Seed 162066966. Fresh evaluations: 1,500 matched episodes per checkpoint; epoch trace: 750; bias validation: 2,000 new episodes.',fontsize=9,color=muted)
fig.text(.075,.029,'95% intervals describe evaluation uncertainty for these frozen policies. The diagnosis is specific to this run; the source and app were unchanged.',fontsize=9,color=muted)
for suffix in ['png','svg']:
    fig.savefig(out/f'dip-diagnosis.{suffix}',dpi=160,bbox_inches='tight',pad_inches=.24)
plt.close(fig)
svg=out/'dip-diagnosis.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
print(json.dumps({'files':['dip-diagnosis.png','dip-diagnosis.svg']}))
