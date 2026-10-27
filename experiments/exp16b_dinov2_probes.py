# Source: Exp11_MultiSeed.ipynb, cell 7 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp16b_dinov2_probes
from common.paths import ROOT, PRETRAINED, load_index, load_master

# ============================================================
# Experiment 16B - Frozen-feature probes for the DINOv2 pair
#   dinov2_generic vs retfound_dinov2 (ViT-L/14, frozen)
#
# Part A  Image-level DR-grade linear probe at 224
#         + label-efficiency curve      (same as Exp 15 Part A)
# Part B  MA patch probe, scale sweep   (same as Exp 15B)
#         patch = 14 px, tile = 224 -> 16x16 tokens per tile
#   S = 224  : 1 tile     cell = 84 px of original
#   S = 448  : 2x2 tiles  cell = 42 px
#   S = 896  : 4x4 tiles  cell = 21 px
#   S = 1344 : 6x6 tiles  cell = 14 px (native)
#
# Train split -> fit probe, Validation -> evaluate.
# Test split is never read. Runtime on L4: ~40-60 minutes.
# ============================================================

import os, gc, json, hashlib, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from timm.layers import resample_abs_pos_embed
from PIL import Image
from torchvision import transforms
from safetensors.torch import load_file as load_safetensors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             roc_auc_score, average_precision_score)
from tqdm.auto import tqdm


OUT = ROOT / "experiment16_dinov2_pair/frozen_probes"; OUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = PRETRAINED / "DINOv2_pair"
SOURCES = {"dinov2_generic": CKPT_DIR / "model.safetensors",
           "retfound_dinov2": CKPT_DIR / "RETFound_dinov2_meh.pth"}
ARCH, PATCH, NT = "vit_large_patch14_dinov2", 14, 16      # 16x16 tokens per 224 tile
LAYERS = [5, 11, 17, 23]; LAYER_NAMES = ["block06", "block12", "block18", "block24"]
SCALES = [224, 448, 896, 1344]; CANVAS = 1344
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda"); assert torch.cuda.is_available()
SEED = 20261026
for p in SOURCES.values():
    assert p.is_file(), f"Missing {p} - run Exp 16 first"

# ------------------------------------------------------------
# 0. Data
# ------------------------------------------------------------
index = load_index()
master = load_master()
data = index.merge(master[["image_id", "mask_path", "mask_review_category"]], on="image_id", how="left")
data = data[data["split"].isin(["train", "val"])].reset_index(drop=True)
assert (data.split == "train").sum() == 1290 and (data.split == "val").sum() == 276

def ensure_local(row):
    src, dst = Path(str(row.source_path)), Path(str(row.local_path))
    if dst.is_file() and dst.stat().st_size == src.stat().st_size: return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial"); shutil.copyfile(src, tmp); os.replace(tmp, dst)
with ThreadPoolExecutor(8) as ex:
    list(tqdm(ex.map(ensure_local, data.itertuples(index=False)), total=len(data), desc="Local image cache"))

tf224 = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
def load224(p):
    with Image.open(p) as im: return tf224(im.convert("RGB"))
with ThreadPoolExecutor(8) as ex:
    imgs224 = torch.stack(list(tqdm(ex.map(load224, data.local_path), total=len(data), desc="Part A images")))

strict = data[data.mask_review_category == "strict_lesion_level_candidate"].reset_index(drop=True)
def load_canvas(row):
    with Image.open(row.local_path) as im: img = np.asarray(im.convert("RGB"))
    with Image.open(row.mask_path) as mk: mask = np.asarray(mk.convert("L")) > 0
    if mask.shape != img.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((img.shape[1], img.shape[0]), Image.NEAREST)) > 0
    H, W = img.shape[:2]; assert H <= CANVAS and W <= CANVAS
    c = np.zeros((CANVAS, CANVAS, 3), np.uint8); m = np.zeros((CANVAS, CANVAS), bool)
    t, l = (CANVAS - H) // 2, (CANVAS - W) // 2; c[t:t + H, l:l + W] = img; m[t:t + H, l:l + W] = mask
    return c, m
with ThreadPoolExecutor(8) as ex:
    canvases = list(tqdm(ex.map(load_canvas, strict.itertuples(index=False)), total=len(strict), desc="Part B images"))

def sample_cells(i, S):
    canvas, mask = canvases[i]; n = (S // 224) * NT; cell = CANVAS // n
    ma = mask.reshape(n, cell, n, cell).sum((1, 3)); fg = (canvas.mean(2) > 15).reshape(n, cell, n, cell).mean((1, 3))
    pos = (ma > 0) & (fg >= 0.5); near = np.zeros_like(pos); pad = np.pad(ma > 0, 1)
    for dy in range(3):
        for dx in range(3): near |= pad[dy:dy + n, dx:dx + n]
    pool = np.argwhere((~near) & (fg >= 0.5)); pidx = np.argwhere(pos)
    r = np.random.default_rng(int(hashlib.md5(f"{strict.image_id[i]}_{S}_p14".encode()).hexdigest()[:8], 16))
    k = min(len(pool), max(20, 10 * len(pidx)))
    nidx = pool[r.choice(len(pool), size=k, replace=False)] if k > 0 else np.zeros((0, 2), int)
    return np.concatenate([pidx, nidx]).astype(np.int32), np.r_[np.ones(len(pidx)), np.zeros(len(nidx))]

META = {}
for S in SCALES:
    rows = []
    for i in range(len(strict)):
        cells, lab = sample_cells(i, S)
        rows += [(i, strict.split[i], strict.risk_group_id[i], int(cy), int(cx), int(l)) for (cy, cx), l in zip(cells, lab)]
    META[S] = pd.DataFrame(rows, columns=["img", "split", "risk_group_id", "cy", "cx", "label"])
    print(f"S={S:>4}: cell {CANVAS // ((S // 224) * NT)} px | val patches {int((META[S].split == 'val').sum())} "
          f"({META[S][META[S].split == 'val'].label.mean():.3f} pos)")

# ------------------------------------------------------------
# 1. Encoders (same loader as Exp 16)
# ------------------------------------------------------------
def build_encoder(name):
    model = timm.create_model(ARCH, pretrained=False, img_size=224, num_classes=0, global_pool="avg")
    target = model.state_dict(); path = SOURCES[name]
    if path.suffix == ".safetensors":
        sd = load_safetensors(str(path))
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        for key in ["teacher", "model", "state_dict", "student"]:
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict): sd = sd[key]; break
    clean = {}
    for k, v in sd.items():
        for pre in ["module.", "backbone.", "encoder.", "model."]:
            if k.startswith(pre): k = k[len(pre):]
        clean[k.replace("mlp.w12.", "mlp.fc1.").replace("mlp.w3.", "mlp.fc2.")] = v
    if clean["pos_embed"].shape != target["pos_embed"].shape:
        clean["pos_embed"] = resample_abs_pos_embed(clean["pos_embed"], new_size=(NT, NT), num_prefix_tokens=1)
    keys = [k for k in target if not k.startswith(("head.", "fc_norm.", "norm."))]
    loaded = {k: clean[k] for k in keys if k in clean and clean[k].shape == target[k].shape}
    assert len(loaded) == len(keys), f"{name}: loaded {len(loaded)}/{len(keys)}"
    model.load_state_dict(loaded, strict=False); del sd, clean
    return model.eval().requires_grad_(False).to(DEVICE)

def run_tokens(model, batch, store):
    store.clear()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.forward_features(batch.to(DEVICE))
    return torch.stack([store[li].float() for li in LAYERS], 1)      # (B, 4, 256, 1024)

# ------------------------------------------------------------
# 2. Probes (identical to Exp 15 / 15B)
# ------------------------------------------------------------
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]
def fit_lr(X, y, C):
    sc = StandardScaler().fit(X); clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    return sc, clf.fit(sc.transform(X), y)
def pick_C(X, y, g):
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0); s = {C: [] for C in C_GRID}
    for a, b in cv.split(X, y, g):
        for C in C_GRID:
            sc, clf = fit_lr(X[a], y[a], C); s[C].append(f1_score(y[b], clf.predict(sc.transform(X[b])), average="macro"))
    return max(C_GRID, key=lambda C: np.mean(s[C]))
def grade_metrics(yt, yp):
    return dict(macro_f1=f1_score(yt, yp, labels=range(5), average="macro", zero_division=0),
                qwk=cohen_kappa_score(yt, yp, weights="quadratic"), accuracy=accuracy_score(yt, yp),
                dr1_recall=float(((yp == 1) & (yt == 1)).sum() / max(1, (yt == 1).sum())))

WD_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
def train_logreg(X, y, wd, iters=300):
    Xt = torch.from_numpy(X).to(DEVICE); yt = torch.from_numpy(y).to(DEVICE)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True) + 1e-6; Xt = (Xt - mu) / sd
    lin = nn.Linear(X.shape[1], 1).to(DEVICE)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    pw = (1 - yt.mean()) / yt.mean()
    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(lin(Xt).squeeze(1), yt, pos_weight=pw) + wd * lin.weight.pow(2).sum()
        loss.backward(); return loss
    opt.step(closure)
    def predict(Xn):
        with torch.no_grad(): return torch.sigmoid(lin((torch.from_numpy(Xn).to(DEVICE) - mu) / sd).squeeze(1)).cpu().numpy()
    return predict

tr = (data.split == "train").to_numpy(); va = ~tr
y = data.dr_grade.astype(int).to_numpy(); groups = data.risk_group_id.to_numpy()
tr_groups = strict.loc[strict.split == "train", "risk_group_id"].unique()
HOLD = set(np.random.default_rng(SEED).choice(tr_groups, size=len(tr_groups) // 5, replace=False))

# ------------------------------------------------------------
# 3. Run both encoders
# ------------------------------------------------------------
rowsA, rowsE, rowsB, SCORES = [], [], [], {}
FRACS, REPEATS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00], 5
for name in SOURCES:
    print(f"\n=== {name} ===")
    model = build_encoder(name); store = {}
    hooks = [model.blocks[li].register_forward_hook(lambda m, i, o, li=li: store.__setitem__(li, o[:, 1:, :])) for li in LAYERS]

    # Part A
    fa = torch.cat([run_tokens(model, imgs224[s:s + 32], store).mean(2).cpu()
                    for s in tqdm(range(0, len(imgs224), 32), desc="image feats", leave=False)]).numpy()
    Cs = {}
    for li, ln in enumerate(LAYER_NAMES):
        X = fa[:, li]; C = pick_C(X[tr], y[tr], groups[tr]); Cs[ln] = C
        sc, clf = fit_lr(X[tr], y[tr], C)
        rowsA.append(dict(encoder=name, layer=ln, C=C, **grade_metrics(y[va], clf.predict(sc.transform(X[va])))))
    X = fa[:, 3]; tr_idx = np.where(tr)[0]
    for frac in FRACS:
        for rep in range(REPEATS if frac < 1 else 1):
            r = np.random.default_rng(1000 * rep + int(frac * 100)); pick = []
            for g in range(5):
                cls = tr_idx[y[tr_idx] == g]; pick += list(r.choice(cls, size=max(1, round(frac * len(cls))), replace=False))
            sc, clf = fit_lr(X[pick], y[pick], Cs["block24"])
            rowsE.append(dict(encoder=name, fraction=frac, repeat=rep, **grade_metrics(y[va], clf.predict(sc.transform(X[va])))))
    print("Part A done:", {r["layer"]: round(r["macro_f1"], 3) for r in rowsA if r["encoder"] == name})

    # Part B
    for S in SCALES:
        meta = META[S]; feats = np.zeros((len(meta), 4, 1024), np.float16); by_img = meta.groupby("img").indices
        for i in tqdm(range(len(strict)), desc=f"S={S}", leave=False):
            rows = by_img.get(i, [])
            if len(rows) == 0: continue
            img = canvases[i][0]
            if S != CANVAS: img = np.asarray(Image.fromarray(img).resize((S, S), Image.BICUBIC))
            sub = meta.iloc[rows]; tiles = sorted(set(zip(sub.cy // NT, sub.cx // NT)))
            batch = torch.stack([transforms.functional.normalize(torch.from_numpy(np.ascontiguousarray(
                img[ty * 224:(ty + 1) * 224, tx * 224:(tx + 1) * 224])).permute(2, 0, 1).float() / 255, MEAN, STD) for ty, tx in tiles])
            tok = run_tokens(model, batch, store); tpos = {t: j for j, t in enumerate(tiles)}
            for r_, cy, cx in zip(rows, sub.cy.to_numpy(), sub.cx.to_numpy()):
                feats[r_] = tok[tpos[(cy // NT, cx // NT)], :, (cy % NT) * NT + (cx % NT)].half().cpu().numpy()
        yb = meta.label.to_numpy().astype(np.float32); ptr = (meta.split == "train").to_numpy(); pva = ~ptr
        iv = ptr & meta.risk_group_id.isin(HOLD).to_numpy(); it = ptr & ~iv
        for li, ln in enumerate(LAYER_NAMES):
            Xb = feats[:, li].astype(np.float32)
            wd = max(WD_GRID, key=lambda w: roc_auc_score(yb[iv], train_logreg(Xb[it], yb[it], w)(Xb[iv])))
            s = train_logreg(Xb[ptr], yb[ptr], wd)(Xb[pva]); SCORES[(name, S, ln)] = s
            rowsB.append(dict(encoder=name, scale=S, cell_px=CANVAS // ((S // 224) * NT), layer=ln,
                              auroc=roc_auc_score(yb[pva], s), auprc=average_precision_score(yb[pva], s)))
        print(f"S={S:>4} | AUROC: " + "  ".join(f"{r['layer']} {r['auroc']:.3f}" for r in rowsB if r["encoder"] == name and r["scale"] == S))
        del feats; gc.collect()
    for h in hooks: h.remove()
    del model; gc.collect(); torch.cuda.empty_cache()

partA, effic, partB = pd.DataFrame(rowsA), pd.DataFrame(rowsE), pd.DataFrame(rowsB)
partA.to_csv(OUT / "partA_image_probe.csv", index=False); effic.to_csv(OUT / "partA_label_efficiency.csv", index=False)
partB.to_csv(OUT / "partB_scale_sweep.csv", index=False)

rng = np.random.default_rng(SEED); rows_bs = []
for S in SCALES:
    vm = META[S][META[S].split == "val"].reset_index(drop=True); yv = vm.label.to_numpy()
    grp = vm.groupby("risk_group_id").indices; keys = list(grp)
    for ln in ["block12", "block24"]:
        a, b = SCORES[("retfound_dinov2", S, ln)], SCORES[("dinov2_generic", S, ln)]; d = []
        for _ in range(1000):
            idx = np.concatenate([grp[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
            if yv[idx].min() == yv[idx].max(): continue
            d.append(roc_auc_score(yv[idx], a[idx]) - roc_auc_score(yv[idx], b[idx]))
        rows_bs.append(dict(scale=S, layer=ln, d_auroc=roc_auc_score(yv, a) - roc_auc_score(yv, b),
                            ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5)))
boot = pd.DataFrame(rows_bs); boot.to_csv(OUT / "partB_bootstrap.csv", index=False)

fmt = lambda x: f"{x:.4f}"
print("\n" + "=" * 66 + "\nDINOv2 PAIR - PART A: IMAGE-LEVEL DR PROBE (frozen, validation)\n" + "=" * 66)
print(partA.pivot(index="layer", columns="encoder", values="macro_f1").to_string(float_format=fmt))
print("\nblock24 full metrics:"); print(partA[partA.layer == "block24"].to_string(index=False, float_format=fmt))
print("\nLabel efficiency (block24, macro-F1):")
print(effic.groupby(["fraction", "encoder"]).macro_f1.mean().unstack().to_string(float_format=fmt))
print("\n" + "=" * 66 + "\nDINOv2 PAIR - PART B: MA PATCH PROBE SCALE SWEEP (AUROC, best layer)\n" + "=" * 66)
print(partB.groupby(["scale", "encoder"]).auroc.max().unstack().to_string(float_format=fmt))
print("\nAUPRC, best layer:"); print(partB.groupby(["scale", "encoder"]).auprc.max().unstack().to_string(float_format=fmt))
print("\nPaired AUROC difference RETFound-DINOv2 - DINOv2 (risk-group bootstrap, 1000x):")
print(boot.to_string(index=False, float_format=fmt))
print("\nTest images used: 0")
print("EXPERIMENT 16B DONE")
