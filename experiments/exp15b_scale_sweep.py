# Source: Exp11_MultiSeed.ipynb, cell 4 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp15b_scale_sweep
from common.paths import ROOT, PRETRAINED, load_index, load_master

# ============================================================
# Experiment 15B - Scale sweep for the frozen MA patch probe
#
# Question: is RETFound's weaker MA patch representation
# (Exp 15 Part B, native resolution) caused by a SCALE
# mismatch? RETFound was pretrained on whole fundus images
# resized to 224 - at that scale it is "in distribution".
#
# Same probe as Exp 15 Part B, repeated at 4 image scales:
#   S = 224  : whole image -> 224 (1 tile)    cell = 96 px of original
#   S = 448  : 2x2 tiles                      cell = 48 px
#   S = 896  : 4x4 tiles                      cell = 24 px
#   S = 1344 : 6x6 tiles (native, = Exp 15)   cell = 16 px
# Patch labels at each scale come from the original strict MA
# mask (positive = any MA pixel in the cell; negative = no MA
# in the 3x3 neighbourhood). Same patches for all encoders.
#
# Train split -> fit probe, Validation split -> evaluate.
# Test split is never read. Runtime on L4: ~30-40 minutes.
# ============================================================

import os, gc, json, hashlib, argparse, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm


OUT = ROOT / "experiment15_frozen_probe/scale_sweep"
OUT.mkdir(parents=True, exist_ok=True)
WEIGHTS = {
    "retfound": PRETRAINED / "RETFound_pretrained/RETFound_mae_natureCFP.pth",
    "meta_mae": PRETRAINED / "Meta_MAE_pretrained/mae_pretrain_vit_large.pth",
    "random": None,
}
SCALES = [224, 448, 896, 1344]
LAYERS = [5, 11, 17, 23]
LAYER_NAMES = ["block06", "block12", "block18", "block24"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CANVAS = 1344
DEVICE = torch.device("cuda")
assert torch.cuda.is_available()
SEED = 20261026

# ------------------------------------------------------------
# 0. Strict-mask cohort (train + val only)
# ------------------------------------------------------------
index = load_index()
master = load_master()
data = index.merge(master[["image_id", "mask_path", "mask_review_category"]], on="image_id", how="left")
data = data[data["split"].isin(["train", "val"])]
strict = data[data["mask_review_category"] == "strict_lesion_level_candidate"].reset_index(drop=True)
print(f"Strict-mask images: train {int((strict.split == 'train').sum())}, val {int((strict.split == 'val').sum())}")

def ensure_local(row):
    src, dst = Path(str(row.source_path)), Path(str(row.local_path))
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial"); shutil.copyfile(src, tmp); os.replace(tmp, dst)

with ThreadPoolExecutor(8) as ex:
    list(tqdm(ex.map(ensure_local, strict.itertuples(index=False)), total=len(strict), desc="Local image cache"))

def load_canvas(row):
    with Image.open(row.local_path) as im:
        img = np.asarray(im.convert("RGB"))
    with Image.open(row.mask_path) as mk:
        mask = np.asarray(mk.convert("L")) > 0
    if mask.shape != img.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (img.shape[1], img.shape[0]), Image.NEAREST)) > 0
    H, W = img.shape[:2]
    assert H <= CANVAS and W <= CANVAS
    c = np.zeros((CANVAS, CANVAS, 3), np.uint8); m = np.zeros((CANVAS, CANVAS), bool)
    t, l = (CANVAS - H) // 2, (CANVAS - W) // 2
    c[t:t + H, l:l + W] = img; m[t:t + H, l:l + W] = mask
    return c, m

print("Loading images and masks ...")
with ThreadPoolExecutor(8) as ex:
    canvases = list(tqdm(ex.map(load_canvas, strict.itertuples(index=False)), total=len(strict)))

# ------------------------------------------------------------
# 1. Patch sampling at each scale (shared by all encoders)
# ------------------------------------------------------------
def sample_cells(i, S):
    canvas, mask = canvases[i]
    n = S // 16; cell = CANVAS // n
    ma_cnt = mask.reshape(n, cell, n, cell).sum((1, 3))
    fg_frac = (canvas.mean(2) > 15).reshape(n, cell, n, cell).mean((1, 3))
    pos = (ma_cnt > 0) & (fg_frac >= 0.5)
    near = np.zeros_like(pos); padded = np.pad(ma_cnt > 0, 1)
    for dy in range(3):
        for dx in range(3):
            near |= padded[dy:dy + n, dx:dx + n]
    neg_pool = np.argwhere((~near) & (fg_frac >= 0.5)); pos_idx = np.argwhere(pos)
    r = np.random.default_rng(int(hashlib.md5(f"{strict.image_id[i]}_{S}".encode()).hexdigest()[:8], 16))
    k = min(len(neg_pool), max(20, 10 * len(pos_idx)))
    neg_idx = neg_pool[r.choice(len(neg_pool), size=k, replace=False)] if k > 0 else np.zeros((0, 2), int)
    return np.concatenate([pos_idx, neg_idx]).astype(np.int32), np.r_[np.ones(len(pos_idx)), np.zeros(len(neg_idx))]

META = {}
for S in SCALES:
    rows = []
    for i in range(len(strict)):
        cells, labels = sample_cells(i, S)
        for (cy, cx), lab in zip(cells, labels):
            rows.append((i, strict.split[i], strict.risk_group_id[i], int(cy), int(cx), int(lab)))
    META[S] = pd.DataFrame(rows, columns=["img", "split", "risk_group_id", "cy", "cx", "label"])
    m = META[S]
    print(f"S={S:>4}: train {int((m.split == 'train').sum()):>6} patches ({m[m.split == 'train'].label.mean():.3f} pos) | "
          f"val {int((m.split == 'val').sum()):>6} ({m[m.split == 'val'].label.mean():.3f} pos)")

# ------------------------------------------------------------
# 2. Encoders and feature extraction
# ------------------------------------------------------------
def build_encoder(name):
    torch.manual_seed(0)
    model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, global_pool="avg")
    if WEIGHTS[name] is not None:
        with torch.serialization.safe_globals([argparse.Namespace]):
            ckpt = torch.load(WEIGHTS[name], map_location="cpu", mmap=True, weights_only=True)
        enc = {k: v for k, v in ckpt["model"].items()
               if k.startswith(("patch_embed.", "blocks.")) or k in {"cls_token", "pos_embed"}}
        assert len(enc) == 292
        assert not model.load_state_dict(enc, strict=False).unexpected_keys
        del ckpt, enc
    return model.eval().requires_grad_(False).to(DEVICE)

def extract(model, S):
    out = {}
    hooks = [model.blocks[li].register_forward_hook(
        lambda m, i, o, li=li: out.__setitem__(li, o[:, 1:, :])) for li in LAYERS]
    meta = META[S]; feats = np.zeros((len(meta), len(LAYERS), 1024), np.float16)
    by_img = meta.groupby("img").indices; T = S // 224
    for i in tqdm(range(len(strict)), desc=f"S={S}", leave=False):
        rows = by_img.get(i, [])
        if len(rows) == 0: continue
        img = canvases[i][0]
        if S != CANVAS:
            img = np.asarray(Image.fromarray(img).resize((S, S), Image.BICUBIC))
        sub = meta.iloc[rows]
        tiles = sorted(set(zip(sub.cy // 14, sub.cx // 14)))
        batch = torch.stack([transforms.functional.normalize(
            torch.from_numpy(img[ty * 224:(ty + 1) * 224, tx * 224:(tx + 1) * 224]).permute(2, 0, 1).float() / 255,
            MEAN, STD) for ty, tx in tiles])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.forward_features(batch.to(DEVICE))
        tok = torch.stack([out[li].float() for li in LAYERS], 1)
        tpos = {t: j for j, t in enumerate(tiles)}
        for r, cy, cx in zip(rows, sub.cy.to_numpy(), sub.cx.to_numpy()):
            feats[r] = tok[tpos[(cy // 14, cx // 14)], :, (cy % 14) * 14 + (cx % 14), :].half().cpu().numpy()
    for h in hooks: h.remove()
    return feats

# ------------------------------------------------------------
# 3. Probe (identical to Exp 15 Part B)
# ------------------------------------------------------------
WD_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
def train_logreg(X, yy, wd, iters=300):
    Xt = torch.from_numpy(X).to(DEVICE); yt = torch.from_numpy(yy).to(DEVICE)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True) + 1e-6
    Xt = (Xt - mu) / sd
    lin = nn.Linear(X.shape[1], 1).to(DEVICE)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    pw = (1 - yt.mean()) / yt.mean()
    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(lin(Xt).squeeze(1), yt, pos_weight=pw) \
               + wd * lin.weight.pow(2).sum()
        loss.backward(); return loss
    opt.step(closure)
    def predict(Xn):
        with torch.no_grad():
            return torch.sigmoid(lin((torch.from_numpy(Xn).to(DEVICE) - mu) / sd).squeeze(1)).cpu().numpy()
    return predict

tr_groups = strict.loc[strict.split == "train", "risk_group_id"].unique()
HOLD = set(np.random.default_rng(SEED).choice(tr_groups, size=len(tr_groups) // 5, replace=False))

SCORES, rows_res = {}, []
for name in WEIGHTS:
    print(f"\n=== {name} ===")
    model = build_encoder(name)
    for S in SCALES:
        meta = META[S]; feats = extract(model, S)
        y = meta.label.to_numpy().astype(np.float32)
        ptr = (meta.split == "train").to_numpy(); pva = ~ptr
        inner_val = ptr & meta.risk_group_id.isin(HOLD).to_numpy(); inner_tr = ptr & ~inner_val
        for li, lname in enumerate(LAYER_NAMES):
            X = feats[:, li, :].astype(np.float32)
            wd = max(WD_GRID, key=lambda w: roc_auc_score(y[inner_val], train_logreg(X[inner_tr], y[inner_tr], w)(X[inner_val])))
            s = train_logreg(X[ptr], y[ptr], wd)(X[pva]); SCORES[(name, S, lname)] = s
            rows_res.append(dict(encoder=name, scale=S, cell_px=CANVAS // (S // 16), layer=lname,
                                 auroc=roc_auc_score(y[pva], s), auprc=average_precision_score(y[pva], s),
                                 prevalence=float(y[pva].mean())))
        last = [r for r in rows_res if r["encoder"] == name and r["scale"] == S]
        print(f"S={S:>4} | AUROC by layer: " + "  ".join(f"{r['layer']} {r['auroc']:.3f}" for r in last))
        del feats; gc.collect()
    del model; gc.collect(); torch.cuda.empty_cache()

res = pd.DataFrame(rows_res); res.to_csv(OUT / "scale_sweep_results.csv", index=False)

# ------------------------------------------------------------
# 4. Paired bootstrap RETFound - Meta MAE at each scale
# ------------------------------------------------------------
rng = np.random.default_rng(SEED); rows_bs = []
for S in SCALES:
    meta = META[S]; pva = (meta.split == "val").to_numpy()
    vm = meta[pva].reset_index(drop=True); y = vm.label.to_numpy()
    grp = vm.groupby("risk_group_id").indices; keys = list(grp)
    for lname in ["block12", "block24"]:
        a, b = SCORES[("retfound", S, lname)], SCORES[("meta_mae", S, lname)]
        d = []
        for _ in range(1000):
            idx = np.concatenate([grp[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
            if y[idx].min() == y[idx].max(): continue
            d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
        rows_bs.append(dict(scale=S, layer=lname, d_auroc=roc_auc_score(y, a) - roc_auc_score(y, b),
                            ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5)))
boot = pd.DataFrame(rows_bs); boot.to_csv(OUT / "scale_sweep_bootstrap.csv", index=False)

# ------------------------------------------------------------
# 5. Report
# ------------------------------------------------------------
fmt = lambda x: f"{x:.4f}"
print("\n" + "=" * 66 + "\nSCALE SWEEP - MA PATCH PROBE AUROC (validation)\n" + "=" * 66)
best = res.groupby(["scale", "encoder"]).auroc.max().unstack()
print("Best layer per encoder:"); print(best.to_string(float_format=fmt))
for lname in ["block12", "block24"]:
    print(f"\n{lname}:")
    print(res[res.layer == lname].pivot(index="scale", columns="encoder", values="auroc").to_string(float_format=fmt))
print("\nAUPRC, best layer (prevalence differs by scale - compare within a row):")
print(res.groupby(["scale", "encoder"]).auprc.max().unstack().to_string(float_format=fmt))
print("\nPaired AUROC difference RETFound - Meta MAE (risk-group bootstrap, 1000x):")
print(boot.to_string(index=False, float_format=fmt))
(OUT / "run_info.json").write_text(json.dumps(dict(scales=SCALES, strict_images=len(strict), test_images_used=0), indent=2))
print("\nTest images used: 0")
print("EXPERIMENT 15B DONE")
