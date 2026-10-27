# Source: Exp13_Test_v2.ipynb, cell 8 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp14_external_idrid
from common.paths import ROOT, PRETRAINED, CACHE, IDRID_ROOT, load_index, load_master

# ============================================================
# Experiment 14 - EXTERNAL VALIDATION ON IDRiD
#
# No model is trained or tuned on IDRiD. Everything is fitted
# on FGADR (as before) and evaluated once on IDRiD.
#
# Part 1  Classification: the 15 fine-tuned FGADR models applied
#         to all 516 IDRiD grading images (DR grade 0-4).
# Part 2  Frozen probes (4 encoders), fitted on the FGADR
#         training split, evaluated on IDRiD:
#           A  image-level DR-grade probe (516 images)
#           B  MA patch probe, scales 224/448/896/1344
#              (81 IDRiD images with pixel-level MA masks)
#
# Geometry harmonisation: each IDRiD image is cropped to its
# fundus bounding box, padded to a square and resized to
# 1280 x 1280, matching FGADR. MA masks get the same transform
# (area resampling, any coverage = positive).
# No patient IDs in IDRiD -> bootstrap resamples images.
# Only small CSV/JSON files are written to Drive.
# Runtime on L4: about 2-2.5 hours.
# ============================================================

import os, gc, io, json, hashlib, argparse, shutil, zipfile
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
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
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, recall_score,
                             roc_auc_score, average_precision_score)
from tqdm.auto import tqdm


OUT = ROOT / "experiment14_external_idrid"; OUT.mkdir(parents=True, exist_ok=True)
IDRID_ZIPS = IDRID_ROOT
LOCAL = CACHE / "idrid"; LOCAL.mkdir(exist_ok=True)
DEVICE = torch.device("cuda"); assert torch.cuda.is_available(), "Switch the runtime to L4 GPU."
SEED = 20261027
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIDE, CANVAS = 1280, 1344
V16, D14 = "vit_large_patch16_224", "vit_large_patch14_dinov2"
CK = PRETRAINED
E11, E12, E16 = ROOT / "experiment11_multiseed", ROOT / "experiment12_official_recipe", ROOT / "experiment16_dinov2_pair"

FINETUNED = {  # id: (family, role, seed, arch, checkpoint)
    "mae_retfound_s42":   ("mae_matched", "retfound", 42,   V16, ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"),
    "mae_meta_s42":       ("mae_matched", "generic",  42,   V16, ROOT / "experiment07_results/meta_mae_vit_large_v1/best_model.pt"),
    "mae_retfound_s1042": ("mae_matched", "retfound", 1042, V16, E11 / "retfound_seed1042/best_model.pt"),
    "mae_meta_s1042":     ("mae_matched", "generic",  1042, V16, E11 / "meta_mae_seed1042/best_model.pt"),
    "mae_retfound_s2042": ("mae_matched", "retfound", 2042, V16, E11 / "retfound_seed2042/best_model.pt"),
    "mae_meta_s2042":     ("mae_matched", "generic",  2042, V16, E11 / "meta_mae_seed2042/best_model.pt"),
    "dino_retfound_s42":   ("dinov2_matched", "retfound", 42,   D14, E16 / "retfound_dinov2_seed42/best_model.pt"),
    "dino_generic_s42":    ("dinov2_matched", "generic",  42,   D14, E16 / "dinov2_generic_seed42/best_model.pt"),
    "dino_retfound_s1042": ("dinov2_matched", "retfound", 1042, D14, E16 / "retfound_dinov2_seed1042/best_model.pt"),
    "dino_generic_s1042":  ("dinov2_matched", "generic",  1042, D14, E16 / "dinov2_generic_seed1042/best_model.pt"),
    "dino_retfound_s2042": ("dinov2_matched", "retfound", 2042, D14, E16 / "retfound_dinov2_seed2042/best_model.pt"),
    "dino_generic_s2042":  ("dinov2_matched", "generic",  2042, D14, E16 / "dinov2_generic_seed2042/best_model.pt"),
    "official_retfound_s42": ("mae_official", "retfound", 42, V16, E12 / "retfound_seed42/best_model.pt"),
    "official_meta_s42":     ("mae_official", "generic",  42, V16, E12 / "meta_mae_seed42/best_model.pt"),
    "efficientnet_b0_s42":   ("baseline", "efficientnet", 42, "efficientnet_b0",
                              ROOT / "experiment03_results/efficientnet_b0_baseline_v1/best_model.pt"),
}
ENCODERS = {   # frozen pretrained encoders: (pair, arch, tokens per 224 tile, weights)
    "retfound":        ("mae",    V16, 14, CK / "RETFound_pretrained/RETFound_mae_natureCFP.pth"),
    "meta_mae":        ("mae",    V16, 14, CK / "Meta_MAE_pretrained/mae_pretrain_vit_large.pth"),
    "retfound_dinov2": ("dinov2", D14, 16, CK / "DINOv2_pair/RETFound_dinov2_meh.pth"),
    "dinov2_generic":  ("dinov2", D14, 16, CK / "DINOv2_pair/model.safetensors"),
}
PAIRS = {"mae": ("retfound", "meta_mae"), "dinov2": ("retfound_dinov2", "dinov2_generic")}
LAYERS = [5, 11, 17, 23]; LAYER_NAMES = ["block06", "block12", "block18", "block24"]
SCALES = [224, 448, 896, 1344]
for v in list(FINETUNED.values()):
    assert v[4].is_file(), f"Missing checkpoint: {v[4]}"
for v in ENCODERS.values():
    assert v[3].is_file(), f"Missing weights: {v[3]}"

PLAN = OUT / "Exp14_External_Plan.json"
if not PLAN.exists():
    PLAN.write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        dataset="IDRiD: 516 grading images (train+test), 81 images with pixel-level MA masks; no IDRiD data used for fitting",
        harmonisation="crop to fundus bounding box, pad square, resize to 1280x1280 (as FGADR)",
        primary=["classification: seed-mean macro-F1, retinal minus generic, per matched pair",
                 "MA patch probe: AUROC at native scale (1344), block 24, retinal minus generic, per pair"],
        secondary="QWK, accuracy, referable AUC, grade-1 recall, calibration; probe results at other scales/blocks; image-level probe",
        statistics="paired image-level bootstrap (no patient identifiers), 2000 (classification) / 1000 (probes) resamples"), indent=2))
plan_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()

# ------------------------------------------------------------
# 1. IDRiD: extract locally, harmonise geometry
# ------------------------------------------------------------
for z in sorted(IDRID_ZIPS.glob("*.zip")):
    flag = LOCAL / (z.stem + ".done")
    if not flag.exists():
        with zipfile.ZipFile(z) as zf: zf.extractall(LOCAL)
        flag.touch()
SEG = LOCAL / "A. Segmentation"; GRD = LOCAL / "B. Disease Grading"

def fundus_square(img, mask=None):
    g = img.mean(2) > 15
    ys, xs = np.where(g.any(1))[0], np.where(g.any(0))[0]
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img = img[y0:y1, x0:x1]; s = max(img.shape[:2])
    pad = lambda a, v: np.pad(a, [((s - a.shape[0]) // 2, s - a.shape[0] - (s - a.shape[0]) // 2),
                                  ((s - a.shape[1]) // 2, s - a.shape[1] - (s - a.shape[1]) // 2)] + [(0, 0)] * (a.ndim - 2),
                              constant_values=v)
    out = cv2.resize(pad(img, 0), (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    if mask is None:
        return out
    m = pad(mask[y0:y1, x0:x1].astype(np.float32), 0)
    return out, cv2.resize(m, (SIDE, SIDE), interpolation=cv2.INTER_AREA) > 0

# grading images
grd_rows = []
for split, csv_name, folder in [("train", "a. IDRiD_Disease Grading_Training Labels.csv", "a. Training Set"),
                                ("test", "b. IDRiD_Disease Grading_Testing Labels.csv", "b. Testing Set")]:
    lab = pd.read_csv(GRD / "2. Groundtruths" / csv_name).iloc[:, :2]
    lab.columns = ["name", "grade"]
    for r in lab.itertuples(index=False):
        grd_rows.append(dict(image_id=f"{split}_{r.name}", path=str(GRD / "1. Original Images" / folder / f"{r.name}.jpg"),
                             grade=int(r.grade)))
grd = pd.DataFrame(grd_rows)
assert len(grd) == 516 and all(Path(p).is_file() for p in grd.path)
print("IDRiD grading images:", len(grd), "| grade counts:", grd.grade.value_counts().sort_index().to_dict())

tf224 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
def load_grading(p):
    sq = fundus_square(np.asarray(Image.open(p).convert("RGB")))
    return tf224(Image.fromarray(sq).resize((224, 224), Image.BICUBIC))
with ThreadPoolExecutor(8) as ex:
    X_idrid = torch.stack(list(tqdm(ex.map(load_grading, grd.path), total=516, desc="IDRiD grading images")))

# segmentation images with MA masks
seg_rows = []
for folder in ["a. Training Set", "b. Testing Set"]:
    for p in sorted((SEG / "1. Original Images" / folder).glob("*.jpg")):
        m = SEG / "2. All Segmentation Groundtruths" / folder / "1. Microaneurysms" / f"{p.stem}_MA.tif"
        if m.is_file(): seg_rows.append(dict(image_id=p.stem, path=str(p), mask=str(m)))
seg = pd.DataFrame(seg_rows); print("IDRiD images with MA masks:", len(seg))

def load_seg(row):
    img = np.asarray(Image.open(row.path).convert("RGB"))
    mk = np.asarray(Image.open(row.mask)); mk = (mk.max(axis=2) if mk.ndim == 3 else mk) > 0
    sq, msq = fundus_square(img, mk)
    c = np.zeros((CANVAS, CANVAS, 3), np.uint8); m = np.zeros((CANVAS, CANVAS), bool)
    o = (CANVAS - SIDE) // 2; c[o:o + SIDE, o:o + SIDE] = sq; m[o:o + SIDE, o:o + SIDE] = msq
    return c, m
with ThreadPoolExecutor(8) as ex:
    seg_canvases = list(tqdm(ex.map(load_seg, seg.itertuples(index=False)), total=len(seg), desc="IDRiD MA images"))
print("IDRiD MA pixels per image (median, at 1280 px):", int(np.median([m.sum() for _, m in seg_canvases])))

# ------------------------------------------------------------
# 2. Part 1 - classification with the 15 fine-tuned models
# ------------------------------------------------------------
def build_finetuned(arch, ckpt):
    if arch == D14: m = timm.create_model(arch, pretrained=False, img_size=224, num_classes=5, global_pool="avg")
    elif arch == V16: m = timm.create_model(arch, pretrained=False, num_classes=5, global_pool="avg")
    else: m = timm.create_model(arch, pretrained=False, num_classes=5)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd, strict=True)
    return m.eval().to(DEVICE)

PRED_DIR = OUT / "predictions"; PRED_DIR.mkdir(exist_ok=True)
P = {}
for k, (fam, role, seed, arch, ckpt) in FINETUNED.items():
    f = PRED_DIR / f"{k}.csv"
    if f.exists():
        P[k] = pd.read_csv(f)[[f"prob_dr{g}" for g in range(5)]].to_numpy(); continue
    m = build_finetuned(arch, ckpt); probs = []
    with torch.inference_mode():
        for s in range(0, 516, 32):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                probs.append(torch.softmax(m(X_idrid[s:s + 32].to(DEVICE)).float(), 1).cpu())
    P[k] = torch.cat(probs).numpy()
    pd.DataFrame({"image_id": grd.image_id, "true_grade": grd.grade, **{f"prob_dr{g}": P[k][:, g] for g in range(5)}}).to_csv(f, index=False)
    del m; gc.collect(); torch.cuda.empty_cache()
y_idrid = grd.grade.to_numpy()

def ece(prob, yt, bins=15):
    conf, pred = prob.max(1), prob.argmax(1); e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any(): e += sel.mean() * abs((pred[sel] == yt[sel]).mean() - conf[sel].mean())
    return e
def cls_metrics(prob, yt):
    pred = prob.argmax(1); rec = recall_score(yt, pred, labels=range(5), average=None, zero_division=0)
    return dict(macro_f1=f1_score(yt, pred, labels=range(5), average="macro", zero_division=0),
                qwk=cohen_kappa_score(yt, pred, weights="quadratic"), accuracy=accuracy_score(yt, pred),
                referable_auc=roc_auc_score(yt >= 2, prob[:, 2:].sum(1)), dr1_recall=rec[1],
                brier=np.mean(((prob - np.eye(5)[yt]) ** 2).sum(1)), ece=ece(prob, yt))
KEY = ["macro_f1", "qwk", "accuracy", "referable_auc", "dr1_recall", "brier", "ece"]
per_model = pd.DataFrame([dict(model_id=k, family=v[0], role=v[1], seed=v[2], **cls_metrics(P[k], y_idrid)) for k, v in FINETUNED.items()])
per_model.to_csv(OUT / "idrid_classification_per_model.csv", index=False)

rng = np.random.default_rng(SEED)
BOOT = [rng.integers(0, 516, 516) for _ in range(2000)]
ids = lambda fam, role: [k for k, v in FINETUNED.items() if v[0] == fam and v[1] == role]
smean = lambda keys, idx: pd.DataFrame([cls_metrics(P[k][idx], y_idrid[idx]) for k in keys])[KEY].mean()
crow = []
for fam in ["mae_matched", "dinov2_matched", "mae_official"]:
    r, g = ids(fam, "retfound"), ids(fam, "generic"); full = np.arange(516)
    obs_r, obs_g = smean(r, full), smean(g, full)
    boots = pd.DataFrame([smean(r, b) - smean(g, b) for b in tqdm(BOOT, desc=fam, leave=False) if len(np.unique(y_idrid[b])) == 5])
    for mname in KEY:
        crow.append(dict(comparison=fam, metric=mname, retfound=obs_r[mname], generic=obs_g[mname],
                         diff=obs_r[mname] - obs_g[mname], ci_low=boots[mname].quantile(0.025), ci_high=boots[mname].quantile(0.975)))
ccomp = pd.DataFrame(crow); ccomp.to_csv(OUT / "idrid_classification_comparisons.csv", index=False)

# ------------------------------------------------------------
# 3. Part 2 - frozen probes: fit on FGADR train, test on IDRiD
# ------------------------------------------------------------
index = load_index()
master = load_master()
fg = index[index.split == "train"].merge(master[["image_id", "mask_path", "mask_review_category"]], on="image_id").reset_index(drop=True)
assert len(fg) == 1290

def ensure_local(row):
    src, dst = Path(str(row.source_path)), Path(str(row.local_path))
    if dst.is_file() and dst.stat().st_size == src.stat().st_size: return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial"); shutil.copyfile(src, tmp); os.replace(tmp, dst)
with ThreadPoolExecutor(8) as ex:
    list(tqdm(ex.map(ensure_local, fg.itertuples(index=False)), total=1290, desc="FGADR train cache"))
tfr = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                          transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
def load_fg(p):
    with Image.open(p) as im: return tfr(im.convert("RGB"))
with ThreadPoolExecutor(8) as ex:
    X_fg = torch.stack(list(tqdm(ex.map(load_fg, fg.local_path), total=1290, desc="FGADR train images")))
y_fg = fg.dr_grade.astype(int).to_numpy(); g_fg = fg.risk_group_id.to_numpy()

strict = fg[fg.mask_review_category == "strict_lesion_level_candidate"].reset_index(drop=True)
def load_fg_canvas(row):
    with Image.open(row.local_path) as im: img = np.asarray(im.convert("RGB"))
    with Image.open(row.mask_path) as mk: mask = np.asarray(mk.convert("L")) > 0
    if mask.shape != img.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((img.shape[1], img.shape[0]), Image.NEAREST)) > 0
    H, W = img.shape[:2]; c = np.zeros((CANVAS, CANVAS, 3), np.uint8); m = np.zeros((CANVAS, CANVAS), bool)
    t, l = (CANVAS - H) // 2, (CANVAS - W) // 2; c[t:t + H, l:l + W] = img; m[t:t + H, l:l + W] = mask
    return c, m
with ThreadPoolExecutor(8) as ex:
    fg_canvases = list(tqdm(ex.map(load_fg_canvas, strict.itertuples(index=False)), total=len(strict), desc="FGADR MA images"))
canvases = fg_canvases + seg_canvases
src = np.array(["fgadr"] * len(fg_canvases) + ["idrid"] * len(seg_canvases))
groups_all = np.concatenate([strict.risk_group_id.to_numpy(), seg.image_id.to_numpy()])
img_ids = np.concatenate([strict.image_id.to_numpy(), seg.image_id.to_numpy()])

def sample_cells(i, S, nt):
    canvas, mask = canvases[i]; n = (S // 224) * nt; cell = CANVAS // n
    ma = mask.reshape(n, cell, n, cell).sum((1, 3)); fgm = (canvas.mean(2) > 15).reshape(n, cell, n, cell).mean((1, 3))
    pos = (ma > 0) & (fgm >= 0.5); near = np.zeros_like(pos); pad = np.pad(ma > 0, 1)
    for dy in range(3):
        for dx in range(3): near |= pad[dy:dy + n, dx:dx + n]
    pool = np.argwhere((~near) & (fgm >= 0.5)); pidx = np.argwhere(pos)
    r = np.random.default_rng(int(hashlib.md5(f"{img_ids[i]}_{S}_{nt}".encode()).hexdigest()[:8], 16))
    k = min(len(pool), max(20, 10 * len(pidx)))
    nidx = pool[r.choice(len(pool), size=k, replace=False)] if k > 0 else np.zeros((0, 2), int)
    return np.concatenate([pidx, nidx]).astype(np.int32), np.r_[np.ones(len(pidx)), np.zeros(len(nidx))]
META = {}
for nt in [14, 16]:
    for S in SCALES:
        rows = []
        for i in range(len(canvases)):
            cells, lab = sample_cells(i, S, nt)
            rows += [(i, src[i], groups_all[i], int(cy), int(cx), int(l)) for (cy, cx), l in zip(cells, lab)]
        META[(nt, S)] = pd.DataFrame(rows, columns=["img", "source", "group", "cy", "cx", "label"])

def build_encoder(name):
    pair, arch, nt, path = ENCODERS[name]
    if arch == V16:
        model = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="avg")
        with torch.serialization.safe_globals([argparse.Namespace]):
            ck = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        enc = {k: v for k, v in ck["model"].items() if k.startswith(("patch_embed.", "blocks.")) or k in {"cls_token", "pos_embed"}}
        assert len(enc) == 292; model.load_state_dict(enc, strict=False)
    else:
        model = timm.create_model(arch, pretrained=False, img_size=224, num_classes=0, global_pool="avg"); target = model.state_dict()
        if path.suffix == ".safetensors": sd = load_safetensors(str(path))
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
        assert len(loaded) == len(keys); model.load_state_dict(loaded, strict=False)
    return model.eval().requires_grad_(False).to(DEVICE)

def run_tokens(model, batch, store):
    store.clear()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.forward_features(batch.to(DEVICE))
    return torch.stack([store[li].float() for li in LAYERS], 1)

C_GRID = [1e-3, 1e-2, 1e-1, 1.0]
def fit_lr(X, y, C):
    sc = StandardScaler().fit(X); return sc, LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
def pick_C(X, y, g):
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0); s = {C: [] for C in C_GRID}
    for a, b in cv.split(X, y, g):
        for C in C_GRID:
            sc, clf = fit_lr(X[a], y[a], C); s[C].append(f1_score(y[b], clf.predict(sc.transform(X[b])), average="macro"))
    return max(C_GRID, key=lambda C: np.mean(s[C]))
WD_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
def train_logreg(X, y, wd, iters=300):
    Xt = torch.from_numpy(X).to(DEVICE); yt = torch.from_numpy(y).to(DEVICE)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True) + 1e-6; Xt = (Xt - mu) / sd
    lin = nn.Linear(X.shape[1], 1).to(DEVICE)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=iters, line_search_fn="strong_wolfe"); pw = (1 - yt.mean()) / yt.mean()
    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(lin(Xt).squeeze(1), yt, pos_weight=pw) + wd * lin.weight.pow(2).sum()
        loss.backward(); return loss
    opt.step(closure)
    def predict(Xn):
        with torch.no_grad(): return torch.sigmoid(lin((torch.from_numpy(Xn).to(DEVICE) - mu) / sd).squeeze(1)).cpu().numpy()
    return predict
tr_groups = strict.risk_group_id.unique()
HOLD = set(np.random.default_rng(SEED).choice(tr_groups, size=len(tr_groups) // 5, replace=False))

rowsA, rowsB, SCORES = [], [], {}
for name, (pair, arch, nt, _) in ENCODERS.items():
    print(f"\n=== {name} ({pair} pair) ===")
    model = build_encoder(name); store = {}
    hooks = [model.blocks[li].register_forward_hook(lambda m, i, o, li=li: store.__setitem__(li, o[:, 1:, :])) for li in LAYERS]
    feat = lambda X: torch.cat([run_tokens(model, X[s:s + 32], store).mean(2).cpu() for s in range(0, len(X), 32)]).numpy()
    fa_fg, fa_id = feat(X_fg), feat(X_idrid)
    for li, ln in enumerate(LAYER_NAMES):
        C = pick_C(fa_fg[:, li], y_fg, g_fg); sc, clf = fit_lr(fa_fg[:, li], y_fg, C); pr = clf.predict(sc.transform(fa_id[:, li]))
        rowsA.append(dict(pair=pair, encoder=name, layer=ln, macro_f1=f1_score(y_idrid, pr, labels=range(5), average="macro", zero_division=0),
                          qwk=cohen_kappa_score(y_idrid, pr, weights="quadratic"), accuracy=accuracy_score(y_idrid, pr)))
    print("Part A (IDRiD macro-F1):", {r["layer"]: round(r["macro_f1"], 3) for r in rowsA if r["encoder"] == name})
    for S in SCALES:
        meta = META[(nt, S)]; feats = np.zeros((len(meta), 4, 1024), np.float16); by_img = meta.groupby("img").indices
        for i in tqdm(range(len(canvases)), desc=f"S={S}", leave=False):
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
        yb = meta.label.to_numpy().astype(np.float32); ptr = (meta.source == "fgadr").to_numpy(); pte = ~ptr
        iv = ptr & meta.group.isin(HOLD).to_numpy(); it = ptr & ~iv
        for li, ln in enumerate(LAYER_NAMES):
            Xb = feats[:, li].astype(np.float32)
            wd = max(WD_GRID, key=lambda w: roc_auc_score(yb[iv], train_logreg(Xb[it], yb[it], w)(Xb[iv])))
            s = train_logreg(Xb[ptr], yb[ptr], wd)(Xb[pte]); SCORES[(name, S, ln)] = s
            rowsB.append(dict(pair=pair, encoder=name, scale=S, layer=ln, auroc=roc_auc_score(yb[pte], s),
                              auprc=average_precision_score(yb[pte], s), prevalence=float(yb[pte].mean())))
        print(f"S={S:>4} | IDRiD AUROC: " + "  ".join(f"{r['layer']} {r['auroc']:.3f}" for r in rowsB if r["encoder"] == name and r["scale"] == S))
        del feats; gc.collect()
    for h in hooks: h.remove()
    del model; gc.collect(); torch.cuda.empty_cache()

partA, partB = pd.DataFrame(rowsA), pd.DataFrame(rowsB)
partA.to_csv(OUT / "idrid_partA_image_probe.csv", index=False); partB.to_csv(OUT / "idrid_partB_scale_sweep.csv", index=False)
rng = np.random.default_rng(SEED); bs = []
for pair, (ret, gen) in PAIRS.items():
    nt = ENCODERS[ret][2]
    for S in SCALES:
        vm = META[(nt, S)][META[(nt, S)].source == "idrid"].reset_index(drop=True); yv = vm.label.to_numpy()
        grp = vm.groupby("group").indices; keys = list(grp)
        for ln in ["block12", "block24"]:
            a, b = SCORES[(ret, S, ln)], SCORES[(gen, S, ln)]; d = []
            for _ in range(1000):
                idx = np.concatenate([grp[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
                if yv[idx].min() == yv[idx].max(): continue
                d.append(roc_auc_score(yv[idx], a[idx]) - roc_auc_score(yv[idx], b[idx]))
            bs.append(dict(pair=pair, scale=S, layer=ln, d_auroc=roc_auc_score(yv, a) - roc_auc_score(yv, b),
                           ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5), primary=(S == 1344 and ln == "block24")))
boot = pd.DataFrame(bs); boot.to_csv(OUT / "idrid_partB_bootstrap.csv", index=False)

# ------------------------------------------------------------
# 4. Report
# ------------------------------------------------------------
fmt = lambda x: f"{x:.4f}"
print("\n" + "=" * 70 + "\nEXP 14 - IDRiD PART 1: CLASSIFICATION (n=516, external)\n" + "=" * 70)
print(per_model[["model_id", "macro_f1", "qwk", "accuracy", "referable_auc", "dr1_recall", "brier", "ece"]].to_string(index=False, float_format=fmt))
print("\nComparisons, retinal - generic (seed mean; image bootstrap 2000x):")
print(ccomp.to_string(index=False, float_format=fmt))
print("\n" + "=" * 70 + "\nEXP 14 - IDRiD PART 2A: IMAGE-LEVEL PROBE (macro-F1)\n" + "=" * 70)
print(partA.pivot_table(index="layer", columns="encoder", values="macro_f1").to_string(float_format=fmt))
print("\n" + "=" * 70 + "\nEXP 14 - IDRiD PART 2B: MA PATCH PROBE (AUROC, best layer)\n" + "=" * 70)
print(partB.groupby(["pair", "scale", "encoder"]).auroc.max().unstack().to_string(float_format=fmt))
print("\nPaired AUROC difference retinal - generic (image bootstrap 1000x); primary = 1344 / block24:")
print(boot.to_string(index=False, float_format=fmt))
(OUT / "Exp14_run_info.json").write_text(json.dumps(dict(plan_sha256=plan_sha, idrid_grading=516, idrid_ma=len(seg),
    finished_utc=datetime.now(timezone.utc).isoformat()), indent=2))
print("\nPlan SHA256:", plan_sha)
print("EXPERIMENT 14 DONE")
