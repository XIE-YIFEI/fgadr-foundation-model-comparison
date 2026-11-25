# Figure 3 - exploratory sensitivity analyses of the test-set MA patch probe.
#
# Input: exp17_paired_differences.csv written by experiments/exp17_probe_sensitivity.py
# Output: the figure (PNG, 600 dpi) and a CSV of the plotted values (Data S9, sheet S9h).
#
# Usage (from the repository root):
#   python figures/figure3_probe_sensitivity.py \
#       <results_root>/experiment17_probe_sensitivity/exp17_paired_differences.csv \
#       "Figure 3.png" figure3_values.csv
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
d=pd.read_csv(sys.argv[1]); d=d[d.layer=='block24']
rows=[('Standard negatives (main analysis)','baseline','pooled'),('Hard negatives','hard_negatives','pooled'),('All foreground cells','all_cells','pooled'),
      ('Per-image AUROC','baseline','per_image'),('Lenient mask set','lenient','pooled'),('Stricter mask set','stricter','pooled'),('Centred overlapping crops','strict_centred','pooled')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#0b0b0b'})
fig,axes=plt.subplots(1,2,figsize=(6.5,3.4),sharey=True)
col={224:'#2a78d6',1344:'#eb6834'}; mk={224:'o',1344:'s'}; lab={224:'224 px input',1344:'Native resolution (1,344 px)'}
out=[]
for ax,(pair,title) in zip(axes,[('mae','MAE pair'),('dinov2','DINOv2 pair')]):
    for i,(name,var,kind) in enumerate(rows):
        for S_,off in [(224,-0.17),(1344,0.17)]:
            r=d[(d.variant==var)&(d.pair==pair)&(d.scale==S_)]
            if r.empty: continue
            r=r.iloc[0]; m,lo,hi=(r.pooled_diff,r.pooled_ci_low,r.pooled_ci_high) if kind=='pooled' else (r.per_image_diff,r.per_image_ci_low,r.per_image_ci_high)
            y=i+off; ax.plot([lo,hi],[y,y],color=col[S_],lw=1.6,solid_capstyle='round'); ax.plot(m,y,mk[S_],ms=5.5,color=col[S_],mec='white',mew=0.8,label=lab[S_] if i==0 else None)
            out.append(dict(pair=pair,analysis=name,scale=S_,diff=m,ci_low=lo,ci_high=hi))
    ax.axvline(0,color='#8a8984',lw=0.8,zorder=0); ax.set_title(title,fontsize=8.5,loc='left')
    ax.grid(axis='x',color='#e6e5e1',lw=0.6); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.set_xlim(-0.13,0.06); ax.set_xlabel('Retinal − generic AUROC (95% CI)')
axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([r[0] for r in rows]); axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=7.5,bbox_to_anchor=(0.62,1.0))
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig(sys.argv[2],dpi=600); pd.DataFrame(out).to_csv(sys.argv[3],index=False)
print(pd.DataFrame(out).round(3).to_string())
