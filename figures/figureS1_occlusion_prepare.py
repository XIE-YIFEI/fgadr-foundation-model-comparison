# Figure S1, step 1 - lesion and control occlusions for the example images.
#
# Repeats the occlusion of experiments/exp18_attribution_occlusion.py
# (same dilation, Telea inpainting radius, retina mask and seeded control
# shifts) for a few images at full resolution, and saves them for plotting.
#
# <workdir> must contain img/<image_id>.png (FGADR fundus image) and
# mask/<image_id>.png (FGADR microaneurysm mask). Writes <workdir>/<image_id>.npz.
#
# Usage (from the repository root; the images used in the paper):
#   python figures/figureS1_occlusion_prepare.py <workdir> 1165_1 0606_1 1740_3
import numpy as np, cv2, hashlib, sys, pandas as pd
from PIL import Image
D=sys.argv[1]
DILATE_PX, R, N_CONTROL, MAX_TRIES, MIN_COV, THRESH = 5,5,3,20000,0.95,127
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)); dil=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
def retina(rgb):
    cand=cv2.morphologyEx((rgb.max(axis=2)>20).astype(np.uint8),cv2.MORPH_CLOSE,kernel)
    cnts,_=cv2.findContours(cand,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    fg=np.zeros(rgb.shape[:2],np.uint8); cv2.drawContours(fg,[max(cnts,key=cv2.contourArea)],-1,1,thickness=-1); return fg
def control_shifts(occ,fg,image_id):
    ys,xs=np.nonzero(occ); H,W=occ.shape
    r=np.random.default_rng(int(hashlib.md5(f"exp18_{image_id}".encode()).hexdigest()[:8],16)); found=[]
    for _ in range(MAX_TRIES):
        dy=int(r.integers(-ys.min(),H-ys.max())); dx=int(r.integers(-xs.min(),W-xs.max()))
        if dy==0 and dx==0: continue
        ty,tx=ys+dy,xs+dx
        if fg[ty,tx].mean()>=MIN_COV and not occ[ty,tx].any():
            found.append((dy,dx))
            if len(found)==N_CONTROL: break
    return found
out={}
for iid in sys.argv[2:]:
    rgb=np.asarray(Image.open(f"{D}/img/{iid}.png").convert('RGB'))
    m=np.asarray(Image.open(f"{D}/mask/{iid}.png"))
    if m.ndim==3: m=m[...,0]
    m=m>THRESH; occ=cv2.dilate(m.astype(np.uint8),dil)
    les=cv2.inpaint(rgb,occ,R,cv2.INPAINT_TELEA)
    fg=retina(rgb); sh=control_shifts(occ,fg,iid); ys,xs=np.nonzero(occ)
    dy,dx=sh[0]; o2=np.zeros_like(occ); o2[ys+dy,xs+dx]=1; ctl=cv2.inpaint(rgb,o2,R,cv2.INPAINT_TELEA)
    print(iid,'mask_px',int(m.sum()),'occluded_px',int(occ.sum()),'shifts',sh)
    np.savez_compressed(f"{D}/{iid}.npz",rgb=rgb,occ=occ,les=les,ctl=ctl,o2=o2)
