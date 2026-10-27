# Source: Exp13_Test.ipynb, cell 4 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp13b_test_frozen_probes
from common.paths import ROOT, PRETRAINED, load_index, load_master

# ============================================================
# Experiment 13B - TEST-SET REPLICATION OF THE FROZEN PROBES
#
# Same probes as Exp 15 / 15B (MAE pair) and Exp 16B (DINOv2
# pair), fitted on the TRAINING split exactly as before and
# evaluated ONCE on the sealed TEST split.
#
#   Part A  image-level DR-grade probe (224) + label efficiency
#   Part B  MA patch probe, scale sweep 224 / 448 / 896 / 1344
#
# The plan is written to Drive before any test image is read.
# Only small CSV/JSON files are written to Drive.
# Runtime on L4: roughly 1.5-2 hours (4 encoders).
# ============================================================

import os, gc, json, hashlib, argparse, shutil
from datetime import datetime, timezone
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


OUT = ROOT / "experiment13_test/frozen_probes"; OUT.mkdir(parents=True, exist_ok=True)
CK = PRETRAINED
V16, D14 = "vit_large_patch16_224", "vit_large_patch14_dinov2"
ENCODERS = {   # name: (pair, architecture, tokens per side of a 224 tile, weight file)
    "retfound":        ("mae",    V16, 14, CK / "RETFound_pretrained/RETFound_mae_natureCFP.pth"),
    "meta_mae":        ("mae",    V16, 14, CK / "Meta_MAE_pretrained/mae_pretrain_vit_large.pth"),
    "retfound_dinov2": ("dinov2", D14, 16, CK / "DINOv2_pair/RETFound_dinov2_meh.pth"),
    "dinov2_generic":  ("dinov2", D14, 16, CK / "DINOv2_pair/model.safetensors"),
}
PAIRS = {"mae": ("retfound", "meta_mae"), "dinov2": ("retfound_dinov2", "dinov2_generic")}
LAYERS = [5, 11, 17, 23]; LAYER_NAMES = ["block06", "block12", "block18", "block24"]
SCALES = [224, 448, 896, 1344]; CANVAS = 1344
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda"); assert torch.cuda.is_available(), "Switch the runtime to L4 GPU."
SEED = 20261026
for v in ENCODERS.values():
    assert v[3].is_file(), f"Missing weights: {v[3]}"

# ------------------------------------------------------------
# 0. Plan (written before the test split is read)
# ------------------------------------------------------------
PLAN = OUT / "Exp13B_Probe_Plan.json"
if not PLAN.exists():
    PLAN.write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        design="Probes identical to Exp 15/15B/16B; fitted on train split; evaluated once on test split",
        part_A="image-level multinomial logistic probe, blocks 6/12/18/24, C by grouped 4-fold CV on train; label efficiency 1-100% (block 24)",
        part_B="MA patch probe at scales 224/448/896/1344; positive = any MA pixel & >=50% foreground; negatives no MA in 3x3, 10:1, min 20/image; L2 logistic, penalty on 20% held-out train groups",
        primary="retinal minus generic AUROC at block 24, native scale (1344), per pair",
        secondary="all scales and blocks 12/24; AUPRC; Part A metrics",
        statistics="paired cluster bootstrap over test source-risk groups, 1000 resamples"), indent=2))
plan_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()

# ------------------------------------------------------------
# 1. Data: train (fit) and test (evaluate)
# ------------------------------------------------------------
index = load_index()
master = load_master()
data = index.merge(master[["image_id", "mask_path", "mask_review_category"]], on="image_id", how="left")
data = data[data.split.isin(["train", "test"])].reset_index(drop=True)
assert (data.split == "train").sum() == 1290 and (data.split == "test").sum() == 276

def ensure_local(row):
    src, dst = Path(str(row.source_path)), Path(str(row.local_path))
    if dst.is_file() and dst.stat().st_size == src.stat().st_size: return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial"); shutil.copyfile(src, tmp); os.replace(tmp, dst)
with ThreadPoolExecutor(8) as ex:
    list(tqdm(ex.map(ensure_local, data[data.split == "train"].itertuples(index=False)), total=1290, desc="Train image cache"))

def img_path(row):   # train from local cache, test straight from Drive
    return str(row.local_path) if row.split == "train" else str(row.source_path)

tf224 = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
def load224(row):
    with Image.open(img_path(row)) as im: return tf224(im.convert("RGB"))
with ThreadPoolExecutor(8) as ex:
    imgs224 = torch.stack(list(tqdm(ex.map(load224, data.itertuples(index=False)), total=len(data), desc="Part A images")))

strict = data[data.mask_review_category == "strict_lesion_level_candidate"].reset_index(drop=True)
print(f"Strict-mask images: train {int((strict.split == 'train').sum())}, test {int((strict.split == 'test').sum())}")
def load_canvas(row):
    with Image.open(img_path(row)) as im: img = np.asarray(im.convert("RGB"))
    with Image.open(row.mask_path) as mk: mask = np.asarray(mk.convert("L")) > 0
    if mask.shape != img.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((img.shape[1], img.shape[0]), Image.NEAREST)) > 0
    H, W = img.shape[:2]; assert H <= CANVAS and W <= CANVAS
    c = np.zeros((CANVAS, CANVAS, 3), np.uint8); m = np.zeros((CANVAS, CANVAS), bool)
    t, l = (CANVAS - H) // 2, (CANVAS - W) // 2; c[t:t + H, l:l + W] = img; m[t:t + H, l:l + W] = mask
    return c, m
with ThreadPoolExecutor(8) as ex:
    canvases = list(tqdm(ex.map(load_canvas, strict.itertuples(index=False)), total=len(strict), desc="Part B images"))

def sample_cells(i, S, nt):
    canvas, mask = canvases[i]; n = (S // 224) * nt; cell = CANVAS // n
    ma = mask.reshape(n, cell, n, cell).sum((1, 3)); fg = (canvas.mean(2) > 15).reshape(n, cell, n, cell).mean((1, 3))
    pos = (ma > 0) & (fg >= 0.5); near = np.zeros_like(pos); pad = np.pad(ma > 0, 1)
    for dy in range(3):
        for dx in range(3): near |= pad[dy:dy + n, dx:dx + n]
    pool = np.argwhere((~near) & (fg >= 0.5)); pidx = np.argwhere(pos)
    r = np.random.default_rng(int(hashlib.md5(f"{strict.image_id[i]}_{S}_{nt}".encode()).hexdigest()[:8], 16))
    k = min(len(pool), max(20, 10 * len(pidx)))
    nidx = pool[r.choice(len(pool), size=k, replace=False)] if k > 0 else np.zeros((0, 2), int)
    return np.concatenate([pidx, nidx]).astype(np.int32), np.r_[np.ones(len(pidx)), np.zeros(len(nidx))]

META = {}
for nt in [14, 16]:
    for S in SCALES:
        rows = []
        for i in range(len(strict)):
            cells, lab = sample_cells(i, S, nt)
            rows += [(i, strict.split[i], strict.risk_group_id[i], int(cy), int(cx), int(l)) for (cy, cx), l in zip(cells, lab)]
        META[(nt, S)] = pd.DataFrame(rows, columns=["img", "split", "risk_group_id", "cy", "cx", "label"])

# ------------------------------------------------------------
# 2. Encoders
# ------------------------------------------------------------
def build_encoder(name):
    pair, arch, nt, path = ENCODERS[name]
    if arch == V16:
        model = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="avg")
        with torch.serialization.safe_globals([argparse.Namespace]):
            ck = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        enc = {k: v for k, v in ck["model"].items() if k.startswith(("patch_embed.", "blocks.")) or k in {"cls_token", "pos_embed"}}
        assert len(enc) == 292; assert not model.load_state_dict(enc, strict=False).unexpected_keys
    else:
        model = timm.create_model(arch, pretrained=False, img_size=224, num_classes=0, global_pool="avg")
        target = model.state_dict()
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
            clean["pos_embed"] = resample_abs_pos_embed(clean["pos_embed"], new_size=(16, 16), num_prefix_tokens=1)
        keys = [k for k in target if not k.startswith(("head.", "fc_norm.", "norm."))]
        loaded = {k: clean[k] for k in keys if k in clean and clean[k].shape == target[k].shape}
        assert len(loaded) == len(keys), f"{name}: loaded {len(loaded)}/{len(keys)}"
        model.load_state_dict(loaded, strict=False)
    return model.eval().requires_grad_(False).to(DEVICE)

def run_tokens(model, batch, store):
    store.clear()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.forward_features(batch.to(DEVICE))
    return torch.stack([store[li].float() for li in LAYERS], 1)

# ------------------------------------------------------------
# 3. Probes (identical to Exp 15 / 15B / 16B)
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

tr = (data.split == "train").to_numpy(); te = ~tr
y = data.dr_grade.astype(int).to_numpy(); groups = data.risk_group_id.to_numpy()
tr_groups = strict.loc[strict.split == "train", "risk_group_id"].unique()
HOLD = set(np.random.default_rng(SEED).choice(tr_groups, size=len(tr_groups) // 5, replace=False))
FRACS, REPEATS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00], 5

# ------------------------------------------------------------
# 4. Run all four encoders
# ------------------------------------------------------------
rowsA, rowsE, rowsB, SCORES = [], [], [], {}
for name, (pair, arch, nt, _) in ENCODERS.items():
    print(f"\n=== {name} ({pair} pair) ===")
    model = build_encoder(name); store = {}
    hooks = [model.blocks[li].register_forward_hook(lambda m, i, o, li=li: store.__setitem__(li, o[:, 1:, :])) for li in LAYERS]
    fa = torch.cat([run_tokens(model, imgs224[s:s + 32], store).mean(2).cpu()
                    for s in tqdm(range(0, len(imgs224), 32), desc="image feats", leave=False)]).numpy()
    Cs = {}
    for li, ln in enumerate(LAYER_NAMES):
        X = fa[:, li]; C = pick_C(X[tr], y[tr], groups[tr]); Cs[ln] = C
        sc, clf = fit_lr(X[tr], y[tr], C)
        rowsA.append(dict(pair=pair, encoder=name, layer=ln, C=C, **grade_metrics(y[te], clf.predict(sc.transform(X[te])))))
    X = fa[:, 3]; tr_idx = np.where(tr)[0]
    for frac in FRACS:
        for rep in range(REPEATS if frac < 1 else 1):
            r = np.random.default_rng(1000 * rep + int(frac * 100)); pick = []
            for g in range(5):
                cls = tr_idx[y[tr_idx] == g]; pick += list(r.choice(cls, size=max(1, round(frac * len(cls))), replace=False))
            sc, clf = fit_lr(X[pick], y[pick], Cs["block24"])
            rowsE.append(dict(pair=pair, encoder=name, fraction=frac, repeat=rep, **grade_metrics(y[te], clf.predict(sc.transform(X[te])))))
    print("Part A (test macro-F1):", {r["layer"]: round(r["macro_f1"], 3) for r in rowsA if r["encoder"] == name})

    for S in SCALES:
        meta = META[(nt, S)]; feats = np.zeros((len(meta), 4, 1024), np.float16); by_img = meta.groupby("img").indices
        for i in tqdm(range(len(strict)), desc=f"S={S}", leave=False):
            rws = by_img.get(i, [])
            if len(rws) == 0: continue
            img = canvases[i][0]
            if S != CANVAS: img = np.asarray(Image.fromarray(img).resize((S, S), Image.BICUBIC))
            sub = meta.iloc[rws]; tiles = sorted(set(zip(sub.cy // nt, sub.cx // nt)))
            batch = torch.stack([transforms.functional.normalize(torch.from_numpy(np.ascontiguousarray(
                img[ty * 224:(ty + 1) * 224, tx * 224:(tx + 1) * 224])).permute(2, 0, 1).float() / 255, MEAN, STD) for ty, tx in tiles])
            tok = run_tokens(model, batch, store); tpos = {t: j for j, t in enumerate(tiles)}
            for r_, cy, cx in zip(rws, sub.cy.to_numpy(), sub.cx.to_numpy()):
                feats[r_] = tok[tpos[(cy // nt, cx // nt)], :, (cy % nt) * nt + (cx % nt)].half().cpu().numpy()
        yb = meta.label.to_numpy().astype(np.float32); ptr = (meta.split == "train").to_numpy(); pte = ~ptr
        iv = ptr & meta.risk_group_id.isin(HOLD).to_numpy(); it = ptr & ~iv
        for li, ln in enumerate(LAYER_NAMES):
            Xb = feats[:, li].astype(np.float32)
            wd = max(WD_GRID, key=lambda w: roc_auc_score(yb[iv], train_logreg(Xb[it], yb[it], w)(Xb[iv])))
            s = train_logreg(Xb[ptr], yb[ptr], wd)(Xb[pte]); SCORES[(name, S, ln)] = s
            rowsB.append(dict(pair=pair, encoder=name, scale=S, cell_px=CANVAS // ((S // 224) * nt), layer=ln,
                              auroc=roc_auc_score(yb[pte], s), auprc=average_precision_score(yb[pte], s),
                              prevalence=float(yb[pte].mean())))
        print(f"S={S:>4} | test AUROC: " + "  ".join(f"{r['layer']} {r['auroc']:.3f}" for r in rowsB if r["encoder"] == name and r["scale"] == S))
        del feats; gc.collect()
    for h in hooks: h.remove()
    del model; gc.collect(); torch.cuda.empty_cache()

partA, effic, partB = pd.DataFrame(rowsA), pd.DataFrame(rowsE), pd.DataFrame(rowsB)
partA.to_csv(OUT / "test_partA_image_probe.csv", index=False)
effic.to_csv(OUT / "test_partA_label_efficiency.csv", index=False)
partB.to_csv(OUT / "test_partB_scale_sweep.csv", index=False)

rng = np.random.default_rng(SEED); rows_bs = []
for pair, (ret, gen) in PAIRS.items():
    nt = ENCODERS[ret][2]
    for S in SCALES:
        vm = META[(nt, S)][META[(nt, S)].split == "test"].reset_index(drop=True); yv = vm.label.to_numpy()
        grp = vm.groupby("risk_group_id").indices; keys = list(grp)
        for ln in ["block12", "block24"]:
            a, b = SCORES[(ret, S, ln)], SCORES[(gen, S, ln)]; d = []
            for _ in range(1000):
                idx = np.concatenate([grp[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
                if yv[idx].min() == yv[idx].max(): continue
                d.append(roc_auc_score(yv[idx], a[idx]) - roc_auc_score(yv[idx], b[idx]))
            rows_bs.append(dict(pair=pair, scale=S, layer=ln, d_auroc=roc_auc_score(yv, a) - roc_auc_score(yv, b),
                                ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                                primary=(S == 1344 and ln == "block24")))
boot = pd.DataFrame(rows_bs); boot.to_csv(OUT / "test_partB_bootstrap.csv", index=False)

fmt = lambda x: f"{x:.4f}"
print("\n" + "=" * 70 + "\nEXP 13B - PART A: IMAGE-LEVEL PROBE ON TEST (macro-F1)\n" + "=" * 70)
print(partA.pivot_table(index="layer", columns="encoder", values="macro_f1").to_string(float_format=fmt))
print("\nblock24 full metrics:"); print(partA[partA.layer == "block24"].drop(columns=["layer"]).to_string(index=False, float_format=fmt))
print("\nLabel efficiency (block24, test macro-F1):")
print(effic.groupby(["fraction", "encoder"]).macro_f1.mean().unstack().to_string(float_format=fmt))
print("\n" + "=" * 70 + "\nEXP 13B - PART B: MA PATCH PROBE ON TEST (AUROC, best layer)\n" + "=" * 70)
print(partB.groupby(["pair", "scale", "encoder"]).auroc.max().unstack().to_string(float_format=fmt))
print("\nAUPRC, best layer:"); print(partB.groupby(["pair", "scale", "encoder"]).auprc.max().unstack().to_string(float_format=fmt))
print("\nPaired AUROC difference retinal - generic (test risk-group bootstrap, 1000x); primary = 1344 / block24:")
print(boot.to_string(index=False, float_format=fmt))
(OUT / "Exp13B_run_info.json").write_text(json.dumps(dict(plan_sha256=plan_sha, strict_train=int((strict.split == "train").sum()),
    strict_test=int((strict.split == "test").sum()), finished_utc=datetime.now(timezone.utc).isoformat()), indent=2))
print("\nPlan SHA256:", plan_sha)
print("EXPERIMENT 13B DONE")
