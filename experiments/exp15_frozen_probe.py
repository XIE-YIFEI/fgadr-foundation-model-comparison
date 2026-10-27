# Source: Exp11_MultiSeed.ipynb, cell 3 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp15_frozen_probe
from common.paths import ROOT, PRETRAINED, CACHE, load_index, load_master

# ============================================================
# Experiment 15 - Frozen-feature probing (no fine-tuning)
#
# Question (RQ1, representation level): do RETFound's
# PRETRAINED features contain more DR / microaneurysm
# information than generic Meta MAE features?
#
# Encoders (all frozen, ViT-L/16):
#   retfound   - RETFound_mae_natureCFP (retina MAE)
#   meta_mae   - Meta MAE ImageNet-1k
#   random     - random init (floor: what can be read from
#                raw architecture + colour alone?)
#
# Part A  Image-level DR-grade linear probe (224 input)
#         + label-efficiency curve (1% ... 100% of labels)
# Part B  Patch-level MA probe at NATIVE resolution:
#         1280x1280 image tiled into 224x224 crops (no
#         downsampling), so each 16x16 ViT patch ~ one MA.
#         Labels from Exp 01 strict MA masks.
#
# Both parts probe 4 depths: blocks 6, 12, 18, 24.
# Train split -> fit probe, Validation split -> evaluate.
# Test split is never read (it is reserved for Exp 13).
# Runtime on L4: roughly 30-45 minutes.
# ============================================================

import os, gc, json, math, time, hashlib, argparse, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             roc_auc_score, average_precision_score)
from tqdm.auto import tqdm


OUT = ROOT / "experiment15_frozen_probe"
LOCAL = CACHE / "exp15_cache"            # large feature arrays stay local
OUT.mkdir(parents=True, exist_ok=True); LOCAL.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "retfound": PRETRAINED / "RETFound_pretrained/RETFound_mae_natureCFP.pth",
    "meta_mae": PRETRAINED / "Meta_MAE_pretrained/mae_pretrain_vit_large.pth",
    "random": None,
}
LAYERS = [5, 11, 17, 23]                         # block indices -> depth 6, 12, 18, 24
LAYER_NAMES = ["block06", "block12", "block18", "block24"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda")
assert torch.cuda.is_available()
SEED = 20261026
rng_global = np.random.default_rng(SEED)

# ------------------------------------------------------------
# 0. Cohort
# ------------------------------------------------------------
index = load_index()
master = load_master()
data = index.merge(master[["image_id", "mask_path", "mask_review_category"]], on="image_id", how="left")
data = data[data["split"].isin(["train", "val"])].reset_index(drop=True)   # never test
assert (data["split"] == "train").sum() == 1290 and (data["split"] == "val").sum() == 276

def ensure_local(row):
    src, dst = Path(str(row.source_path)), Path(str(row.local_path))
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial"); shutil.copyfile(src, tmp); os.replace(tmp, dst)

with ThreadPoolExecutor(8) as ex:
    list(tqdm(ex.map(ensure_local, data.itertuples(index=False)), total=len(data), desc="Local image cache"))

# ------------------------------------------------------------
# 1. Encoders with multi-depth token capture
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
        res = model.load_state_dict(enc, strict=False)
        assert not res.unexpected_keys
        del ckpt, enc
    model.eval().requires_grad_(False)
    return model.to(DEVICE)

class TokenTap:
    """Collects patch tokens (no CLS) from the chosen blocks."""
    def __init__(self, model):
        self.out, self.hooks = {}, []
        for li in LAYERS:
            self.hooks.append(model.blocks[li].register_forward_hook(self._make(li)))
    def _make(self, li):
        def f(m, i, o):
            self.out[li] = o[:, 1:, :]
        return f
    def remove(self):
        for h in self.hooks: h.remove()

@torch.inference_mode()
def tokens(model, tap, batch):
    tap.out.clear()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.forward_features(batch.to(DEVICE, non_blocking=True))
    return torch.stack([tap.out[li].float() for li in LAYERS], 1)   # (B, 4, 196, 1024)

# ------------------------------------------------------------
# 2. Part A data: whole images resized to 224
# ------------------------------------------------------------
tf224 = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
def load224(path):
    with Image.open(path) as im:
        return tf224(im.convert("RGB"))

print("Decoding 1566 images for Part A ...")
with ThreadPoolExecutor(8) as ex:
    imgs224 = torch.stack(list(tqdm(ex.map(load224, data["local_path"]), total=len(data))))

# ------------------------------------------------------------
# 3. Part B data: native-resolution tiles + MA patch labels
# ------------------------------------------------------------
TILE, GRID, PAD_TO = 224, 6, 1344                # 6x6 tiles of 224 on a 1344 canvas
strict = data[data["mask_review_category"] == "strict_lesion_level_candidate"].reset_index(drop=True)
print(f"Strict-mask images: train {int((strict.split == 'train').sum())}, val {int((strict.split == 'val').sum())}")

def prepare_image(row):
    """Returns tiles (uint8), and selected patch records with labels."""
    with Image.open(row.local_path) as im:
        img = np.asarray(im.convert("RGB"))
    with Image.open(row.mask_path) as mk:
        mask = np.asarray(mk.convert("L")) > 0
    if mask.shape != img.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (img.shape[1], img.shape[0]), Image.NEAREST)) > 0
    H, W = img.shape[:2]
    assert H <= PAD_TO and W <= PAD_TO, f"Unexpected image size {img.shape} for {row.image_id}"
    canvas = np.zeros((PAD_TO, PAD_TO, 3), np.uint8); mcanvas = np.zeros((PAD_TO, PAD_TO), bool)
    top, left = (PAD_TO - H) // 2, (PAD_TO - W) // 2
    canvas[top:top + H, left:left + W] = img
    mcanvas[top:top + H, left:left + W] = mask
    fg = canvas.mean(2) > 15                                     # retinal foreground
    n = PAD_TO // 16                                             # 84 x 84 patch cells
    ma_cnt = mcanvas.reshape(n, 16, n, 16).sum((1, 3))
    fg_frac = fg.reshape(n, 16, n, 16).mean((1, 3))
    pos = (ma_cnt > 0) & (fg_frac >= 0.5)
    # negatives: foreground cells with no MA in the 3x3 neighbourhood
    near = np.zeros_like(pos)
    padded = np.pad(ma_cnt > 0, 1)
    for dy in range(3):
        for dx in range(3):
            near |= padded[dy:dy + n, dx:dx + n]
    neg_pool = np.argwhere((~near) & (fg_frac >= 0.5))
    pos_idx = np.argwhere(pos)
    r = np.random.default_rng(int(hashlib.md5(row.image_id.encode()).hexdigest()[:8], 16))
    k = min(len(neg_pool), max(20, 10 * len(pos_idx)))
    neg_idx = neg_pool[r.choice(len(neg_pool), size=k, replace=False)] if k > 0 else np.zeros((0, 2), int)
    cells = np.concatenate([pos_idx, neg_idx]); labels = np.r_[np.ones(len(pos_idx)), np.zeros(len(neg_idx))]
    sizes = np.r_[ma_cnt[pos], np.zeros(len(neg_idx))]
    return canvas, cells.astype(np.int32), labels.astype(np.int8), sizes.astype(np.int32)

print("Preparing native-resolution tiles and MA patch labels ...")
with ThreadPoolExecutor(8) as ex:
    prepared = list(tqdm(ex.map(prepare_image, strict.itertuples(index=False)), total=len(strict)))

patch_meta = []
for i, (row, (_, cells, labels, sizes)) in enumerate(zip(strict.itertuples(index=False), prepared)):
    for (cy, cx), lab, sz in zip(cells, labels, sizes):
        patch_meta.append((i, row.image_id, row.split, row.risk_group_id, int(cy), int(cx), int(lab), int(sz)))
patch_meta = pd.DataFrame(patch_meta, columns=["img", "image_id", "split", "risk_group_id", "cy", "cx", "label", "ma_pixels"])
patch_meta.to_csv(OUT / "patch_sample_index.csv", index=False)
print(patch_meta.groupby(["split", "label"]).size().rename("patches").to_string())

def tile_tensor(canvas, ty, tx):
    t = torch.from_numpy(canvas[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE]).permute(2, 0, 1).float() / 255
    return transforms.functional.normalize(t, MEAN, STD)

# ------------------------------------------------------------
# 4. Feature extraction (per encoder)
# ------------------------------------------------------------
def extract(name):
    fA, fB = LOCAL / f"{name}_image.npy", LOCAL / f"{name}_patch.npy"
    if fA.exists() and fB.exists():
        return np.load(fA), np.load(fB)
    model = build_encoder(name); tap = TokenTap(model)
    # Part A: mean patch token per depth
    featsA = []
    for s in tqdm(range(0, len(imgs224), 32), desc=f"{name} image feats"):
        featsA.append(tokens(model, tap, imgs224[s:s + 32]).mean(2).half().cpu())
    featsA = torch.cat(featsA).numpy()
    # Part B: tokens at the sampled cells
    featsB = np.zeros((len(patch_meta), len(LAYERS), 1024), np.float16)
    by_img = patch_meta.groupby("img").indices
    for i in tqdm(range(len(strict)), desc=f"{name} patch feats"):
        canvas = prepared[i][0]; rows = by_img.get(i, [])
        if len(rows) == 0: continue
        sub = patch_meta.iloc[rows]
        tiles = sorted(set(zip(sub.cy // 14, sub.cx // 14)))
        batch = torch.stack([tile_tensor(canvas, ty, tx) for ty, tx in tiles])
        tok = tokens(model, tap, batch)                       # (T, 4, 196, 1024)
        tpos = {t: j for j, t in enumerate(tiles)}
        for r, cy, cx in zip(rows, sub.cy.to_numpy(), sub.cx.to_numpy()):
            j = tpos[(cy // 14, cx // 14)]; p = (cy % 14) * 14 + (cx % 14)
            featsB[r] = tok[j, :, p, :].half().cpu().numpy()
    tap.remove(); del model; gc.collect(); torch.cuda.empty_cache()
    np.save(fA, featsA); np.save(fB, featsB)
    return featsA, featsB

FEATS = {name: extract(name) for name in WEIGHTS}

# ------------------------------------------------------------
# 5. Part A: image-level DR-grade probe
# ------------------------------------------------------------
tr = (data["split"] == "train").to_numpy(); va = ~tr
y = data["dr_grade"].astype(int).to_numpy(); groups = data["risk_group_id"].to_numpy()
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]

def fit_lr(X, yy, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    clf.fit(sc.transform(X), yy)
    return sc, clf

def pick_C(X, yy, g):
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)
    scores = {C: [] for C in C_GRID}
    for a, b in cv.split(X, yy, g):
        for C in C_GRID:
            sc, clf = fit_lr(X[a], yy[a], C)
            scores[C].append(f1_score(yy[b], clf.predict(sc.transform(X[b])), average="macro"))
    return max(C_GRID, key=lambda C: np.mean(scores[C]))

def grade_metrics(yt, yp):
    return dict(macro_f1=f1_score(yt, yp, labels=range(5), average="macro", zero_division=0),
                qwk=cohen_kappa_score(yt, yp, weights="quadratic"), accuracy=accuracy_score(yt, yp),
                dr1_recall=float(((yp == 1) & (yt == 1)).sum() / max(1, (yt == 1).sum())))

rowsA, best_C = [], {}
for name, (fa, _) in FEATS.items():
    for li, lname in enumerate(LAYER_NAMES):
        X = fa[:, li, :].astype(np.float32)
        C = pick_C(X[tr], y[tr], groups[tr]); best_C[(name, lname)] = C
        sc, clf = fit_lr(X[tr], y[tr], C)
        rowsA.append(dict(encoder=name, layer=lname, C=C, **grade_metrics(y[va], clf.predict(sc.transform(X[va])))))
partA = pd.DataFrame(rowsA); partA.to_csv(OUT / "partA_image_probe.csv", index=False)

# Label-efficiency curve (last block, C fixed from full-data CV)
FRACS, REPEATS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00], 5
tr_idx = np.where(tr)[0]; rowsE = []
for name, (fa, _) in FEATS.items():
    X = fa[:, 3, :].astype(np.float32); C = best_C[(name, "block24")]
    for frac in FRACS:
        for rep in range(REPEATS if frac < 1 else 1):
            r = np.random.default_rng(1000 * rep + int(frac * 100))
            pick = []
            for g in range(5):                                   # stratified, >=1 per grade
                cls = tr_idx[y[tr_idx] == g]
                pick += list(r.choice(cls, size=max(1, round(frac * len(cls))), replace=False))
            sc, clf = fit_lr(X[pick], y[pick], C)
            rowsE.append(dict(encoder=name, fraction=frac, repeat=rep, n_train=len(pick),
                              **grade_metrics(y[va], clf.predict(sc.transform(X[va])))))
effic = pd.DataFrame(rowsE); effic.to_csv(OUT / "partA_label_efficiency.csv", index=False)

# ------------------------------------------------------------
# 6. Part B: patch-level MA probe (GPU logistic regression)
# ------------------------------------------------------------
ptr = (patch_meta["split"] == "train").to_numpy(); pva = ~ptr
yb = patch_meta["label"].to_numpy().astype(np.float32)
WD_GRID = [1e-4, 1e-3, 1e-2, 1e-1]

def train_logreg(X, yy, wd, epochs=300):
    Xt = torch.from_numpy(X).to(DEVICE); yt = torch.from_numpy(yy).to(DEVICE)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True) + 1e-6
    Xt = (Xt - mu) / sd
    lin = nn.Linear(X.shape[1], 1).to(DEVICE)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=epochs, line_search_fn="strong_wolfe")
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

# Hold out 20% of training risk groups to choose weight decay
tr_groups = patch_meta.loc[ptr, "risk_group_id"].unique()
hold = set(np.random.default_rng(SEED).choice(tr_groups, size=len(tr_groups) // 5, replace=False))
inner_val = ptr & patch_meta["risk_group_id"].isin(hold).to_numpy(); inner_tr = ptr & ~inner_val

rowsB, scoresB = [], {}
for name, (_, fb) in FEATS.items():
    for li, lname in enumerate(LAYER_NAMES):
        X = fb[:, li, :].astype(np.float32)
        wd = max(WD_GRID, key=lambda w: roc_auc_score(yb[inner_val], train_logreg(X[inner_tr], yb[inner_tr], w)(X[inner_val])))
        s = train_logreg(X[ptr], yb[ptr], wd)(X[pva]); scoresB[(name, lname)] = s
        rowsB.append(dict(encoder=name, layer=lname, weight_decay=wd,
                          auroc=roc_auc_score(yb[pva], s), auprc=average_precision_score(yb[pva], s),
                          prevalence=float(yb[pva].mean())))
partB = pd.DataFrame(rowsB); partB.to_csv(OUT / "partB_ma_patch_probe.csv", index=False)
pd.DataFrame({f"{n}_{l}": s for (n, l), s in scoresB.items()}).assign(
    image_id=patch_meta.loc[pva, "image_id"].to_numpy(), label=yb[pva],
    ma_pixels=patch_meta.loc[pva, "ma_pixels"].to_numpy()).to_csv(OUT / "partB_val_scores.csv", index=False)

# Paired cluster bootstrap: RETFound - Meta MAE (by source-risk group)
vmeta = patch_meta[pva].reset_index(drop=True)
grp_rows = vmeta.groupby("risk_group_id").indices; gkeys = list(grp_rows)
rng = np.random.default_rng(SEED); rowsBS = []
for lname in LAYER_NAMES:
    a, b = scoresB[("retfound", lname)], scoresB[("meta_mae", lname)]
    lab = yb[pva]; d_auc, d_ap = [], []
    for _ in range(1000):
        idx = np.concatenate([grp_rows[gkeys[k]] for k in rng.integers(0, len(gkeys), len(gkeys))])
        if lab[idx].min() == lab[idx].max(): continue
        d_auc.append(roc_auc_score(lab[idx], a[idx]) - roc_auc_score(lab[idx], b[idx]))
        d_ap.append(average_precision_score(lab[idx], a[idx]) - average_precision_score(lab[idx], b[idx]))
    rowsBS.append(dict(layer=lname,
                       d_auroc=roc_auc_score(lab, a) - roc_auc_score(lab, b),
                       d_auroc_ci=f"[{np.percentile(d_auc, 2.5):.4f}, {np.percentile(d_auc, 97.5):.4f}]",
                       d_auprc=average_precision_score(lab, a) - average_precision_score(lab, b),
                       d_auprc_ci=f"[{np.percentile(d_ap, 2.5):.4f}, {np.percentile(d_ap, 97.5):.4f}]"))
boot = pd.DataFrame(rowsBS); boot.to_csv(OUT / "partB_paired_bootstrap.csv", index=False)

# ------------------------------------------------------------
# 7. Report
# ------------------------------------------------------------
fmt = lambda x: f"{x:.4f}"
print("\n" + "=" * 66 + "\nPART A - IMAGE-LEVEL DR-GRADE LINEAR PROBE (frozen, validation)\n" + "=" * 66)
print(partA.pivot(index="layer", columns="encoder", values="macro_f1").to_string(float_format=fmt))
print("\nLast block (block24) full metrics:")
print(partA[partA.layer == "block24"].drop(columns="layer").to_string(index=False, float_format=fmt))
print("\nLABEL EFFICIENCY (block24, macro-F1 mean over repeats)")
print(effic.groupby(["fraction", "encoder"])["macro_f1"].mean().unstack().to_string(float_format=fmt))
print("\n" + "=" * 66 + "\nPART B - NATIVE-RESOLUTION MA PATCH PROBE (frozen, validation)\n" + "=" * 66)
print(f"Val patches: {int(pva.sum())}  (MA-positive {int(yb[pva].sum())}, prevalence {yb[pva].mean():.3f})")
print("AUROC:"); print(partB.pivot(index="layer", columns="encoder", values="auroc").to_string(float_format=fmt))
print("AUPRC:"); print(partB.pivot(index="layer", columns="encoder", values="auprc").to_string(float_format=fmt))
print("\nPaired difference RETFound - Meta MAE (source-risk-group bootstrap, 1000x):")
print(boot.to_string(index=False, float_format=fmt))
(OUT / "run_info.json").write_text(json.dumps(dict(
    images_train=int(tr.sum()), images_val=int(va.sum()), strict_mask_images=len(strict),
    patches=len(patch_meta), test_images_used=0), indent=2))
print("\nTest images used: 0")
print("EXPERIMENT 15 DONE")
