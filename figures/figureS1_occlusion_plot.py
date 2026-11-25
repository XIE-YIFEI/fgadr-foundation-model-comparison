# Figure S1, step 2 - plot the lesion and control occlusion examples.
#
# Inputs: the .npz files from figures/figureS1_occlusion_prepare.py and
# exp18_per_image.csv written by experiments/exp18_attribution_occlusion.py
# (used for the lesion-minus-control change in expected grade, dE_specific).
#
# Usage (from the repository root):
#   python figures/figureS1_occlusion_plot.py <workdir> \
#       <results_root>/experiment18_attribution_occlusion/exp18_per_image.csv "Figure S1.png"
import numpy as np, cv2, sys, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
D,csv,outp=sys.argv[1],sys.argv[2],sys.argv[3]
d=pd.read_csv(csv)
rows=[('1165_1','Grade 1'),('0606_1','Grade 1'),('1740_3','Grade 4')]
W=200
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7})
fig,ax=plt.subplots(3,4,figsize=(6.5,5.3))
def win(c,H): y,x=c; y=int(np.clip(y-W//2,0,H-W)); x=int(np.clip(x-W//2,0,H-W)); return y,x
for r,(iid,gl) in enumerate(rows):
    z=np.load(f"{D}/{iid}.npz"); rgb,occ,les,ctl,o2=z['rgb'],z['occ'],z['les'],z['ctl'],z['o2']; H=rgb.shape[0]
    dens=cv2.boxFilter(occ.astype(np.float32),-1,(W,W)); c=np.unravel_index(dens.argmax(),dens.shape); y,x=win(c,H)
    dens2=cv2.boxFilter(o2.astype(np.float32),-1,(W,W)); c2=np.unravel_index(dens2.argmax(),dens2.shape); y2,x2=win(c2,H)
    a=ax[r,0]; a.imshow(rgb); 
    cn,_=cv2.findContours(occ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for k in cn: a.plot(k[:,0,0],k[:,0,1],color='#eda100',lw=0.5)
    a.add_patch(Rectangle((x,y),W,W,fill=False,ec='white',lw=0.8)); a.add_patch(Rectangle((x2,y2),W,W,fill=False,ec='#2a78d6',lw=0.8,ls='--'))
    sub=d[d.image_id==iid]; dino=sub[sub.pair=='dinov2'].dE_specific.mean(); mae=sub[sub.pair=='mae'].dE_specific.mean()
    a.set_ylabel(f"{iid} ({gl})\nΔE lesion − control:\nDINOv2 {dino:+.2f}, MAE {mae:+.2f}",fontsize=6.5)
    ov=rgb[y:y+W,x:x+W].copy(); 
    ax[r,1].imshow(rgb[y:y+W,x:x+W]); 
    for k in cn: 
        kk=k[:,0,:]-[x,y]; ax[r,1].plot(kk[:,0],kk[:,1],color='#eda100',lw=0.6)
    ax[r,1].set_xlim(0,W); ax[r,1].set_ylim(W,0)
    ax[r,2].imshow(les[y:y+W,x:x+W]); ax[r,3].imshow(ctl[y2:y2+W,x2:x2+W])
    for a in ax[r]: a.set_xticks([]); a.set_yticks([])
for j,t in enumerate(['Full image','Lesion region\n(dilated masks outlined)','After lesion inpainting','Control region\nafter inpainting']): ax[0,j].set_title(t,fontsize=7)
fig.tight_layout(); fig.savefig(outp,dpi=600)
