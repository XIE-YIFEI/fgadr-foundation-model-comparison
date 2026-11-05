# Experiment 17 - REVIEWER SENSITIVITY ANALYSES FOR THE MA PATCH PROBE
#
# Run from the repository root (Colab, L4 GPU):
#     python -m experiments.exp17_probe_sensitivity
# Smoke test on CPU with random data and a tiny ViT (no real data needed):
#     EXP17_SMOKE=1 python -m experiments.exp17_probe_sensitivity
#
# Same encoders, cell definition, training sample and logistic probe as
# Exp 13B (test-set frozen probes). Probes are fitted on the TRAINING split
# and evaluated on the TEST split. Four analyses:
#
#   A  baseline        Exp 13B sampling, re-run as a sanity check
#   B  evaluation sets probe from A, scored on
#                        all_cells       every foreground cell of each test image
#                        hard_negatives  positives + MA-free cells next to a lesion
#                      and per-image AUROC (mean over images with both classes)
#   C  mask sets       probe refitted and evaluated on
#                        lenient   strict + masks excluded only for grey-level
#                                  encoding (binarised at 127)
#                        stricter  strict masks whose largest component is at
#                                  most the strict-set 75th percentile (758 px)
#   D  centred tiles   tiles overlap by half (stride 112 px in the scaled image);
#                      each cell is read from the tile whose centre is nearest,
#                      so no cell sits on a tile edge (scales 448/896/1344)
#
# Output: CSV files in <results_root>/experiment17_probe_sensitivity
# Paired retinal-minus-generic AUROC differences use a cluster bootstrap over
# test source-risk groups (1000 resamples), as in Exp 13B.
# Runtime on L4: roughly 3-4 hours for all four encoders. Each finished encoder
# is saved to exp17_checkpoint_<encoder>.pkl; a rerun skips finished encoders.

import os, gc, json, hashlib, pickle
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

SMOKE = os.environ.get("EXP17_SMOKE") == "1"
LAYERS = [5, 11, 17, 23]; LAYER_NAMES = ["block06", "block12", "block18", "block24"]
SCALES = [224, 448, 896, 1344]; CANVAS = 1344; STRIDE = 112
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SEED = 20261105
N_BOOT = 1000
STRICTER_MAX_LARGEST_PX = 758          # 75th percentile of the strict set (Data S1)

if SMOKE:
    from experiments._smoke import make_smoke_inputs
    OUT, data, ENCODERS, build_encoder, DEVICE = make_smoke_inputs("exp17")
    SCALES = [224, 448]; N_BOOT = 50
else:
    from common.paths import ROOT, MASK_AUDIT_CSV, load_index, load_master
    from experiments.exp17_encoders import ENCODERS, build_encoder
    OUT = ROOT / "experiment17_probe_sensitivity"; OUT.mkdir(parents=True, exist_ok=True)
    DEVICE = torch.device("cuda"); assert torch.cuda.is_available(), "Switch the runtime to L4 GPU."
    index = load_index(); master = load_master()
    audit = pd.read_csv(MASK_AUDIT_CSV, dtype={"image_id": str})
    data = (index.merge(master[["image_id", "mask_path", "mask_review_category"]], on="image_id", how="left")
                 .merge(audit[["image_id", "review_reason", "largest_component_area_px"]], on="image_id", how="left"))
    data = data[data.split.isin(["train", "test"])].reset_index(drop=True)
AMP = dict(device_type="cuda", dtype=torch.bfloat16) if DEVICE.type == "cuda" else dict(device_type="cpu", enabled=False)
PAIRS = {"mae": ("retfound", "meta_mae"), "dinov2": ("retfound_dinov2", "dinov2_generic")}

# ------------------------------------------------------------
# 1. Mask sets
# ------------------------------------------------------------
strict_rule = data.mask_review_category.eq("strict_lesion_level_candidate")
grey_only = data.mask_review_category.eq("requires_manual_review") & data.review_reason.eq("intermediate_grayscale_encoding")
MASK_SETS = {
    "strict": strict_rule,
    "lenient": strict_rule | grey_only,
    "stricter": strict_rule & data.largest_component_area_px.le(STRICTER_MAX_LARGEST_PX),
}
for k, v in MASK_SETS.items():
    print(f"mask set {k:9s}: train {int((v & data.split.eq('train')).sum())}, test {int((v & data.split.eq('test')).sum())}")
img_rows = data[MASK_SETS["lenient"]].reset_index(drop=True)      # superset of all three sets
ROW = {iid: i for i, iid in enumerate(img_rows.image_id)}

def img_path(row):
    return str(row.source_path)

def load_canvas(row):
    with Image.open(img_path(row)) as im: img = np.asarray(im.convert("RGB"))
    with Image.open(row.mask_path) as mk: m = np.asarray(mk.convert("L"))
    thr = 127 if row.mask_review_category != "strict_lesion_level_candidate" else 0   # strict masks: as Exp 13B
    mask = m > thr
    if mask.shape != img.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((img.shape[1], img.shape[0]), Image.NEAREST)) > 0
    H, W = img.shape[:2]; assert H <= CANVAS and W <= CANVAS
    c = np.zeros((CANVAS, CANVAS, 3), np.uint8); mm = np.zeros((CANVAS, CANVAS), bool)
    t, l = (CANVAS - H) // 2, (CANVAS - W) // 2; c[t:t + H, l:l + W] = img; mm[t:t + H, l:l + W] = mask
    return c, mm
with ThreadPoolExecutor(8) as ex:
    canvases = list(tqdm(ex.map(load_canvas, img_rows.itertuples(index=False)), total=len(img_rows), desc="images"))

# ------------------------------------------------------------
# 2. Cells
# ------------------------------------------------------------
def cell_maps(i, S, nt):
    canvas, mask = canvases[i]; n = (S // 224) * nt; cell = CANVAS // n
    ma = mask.reshape(n, cell, n, cell).sum((1, 3)) > 0
    fg = (canvas.mean(2) > 15).reshape(n, cell, n, cell).mean((1, 3)) >= 0.5
    near = np.zeros_like(ma); pad = np.pad(ma, 1)
    for dy in range(3):
        for dx in range(3): near |= pad[dy:dy + n, dx:dx + n]
    return ma, fg, near

def sample_cells(i, S, nt):          # identical rule and seed to Exp 13B
    ma, fg, near = cell_maps(i, S, nt)
    pos = ma & fg; pool = np.argwhere((~near) & fg); pidx = np.argwhere(pos)
    r = np.random.default_rng(int(hashlib.md5(f"{img_rows.image_id[i]}_{S}_{nt}".encode()).hexdigest()[:8], 16))
    k = min(len(pool), max(20, 10 * len(pidx)))
    nidx = pool[r.choice(len(pool), size=k, replace=False)] if k > 0 else np.zeros((0, 2), int)
    return np.concatenate([pidx, nidx]).astype(np.int32), np.r_[np.ones(len(pidx)), np.zeros(len(nidx))]

def all_cells(i, S, nt):
    """Every foreground cell. kind: 0 far negative, 1 near negative (hard), 2 positive."""
    ma, fg, near = cell_maps(i, S, nt)
    idx = np.argwhere(fg)
    kind = np.where(ma[fg], 2, np.where(near[fg], 1, 0))
    return idx.astype(np.int32), kind.astype(np.int8)

# ------------------------------------------------------------
# 3. Features
# ------------------------------------------------------------
def make_store(model):
    store = {}
    hooks = [model.blocks[li].register_forward_hook(lambda m, i, o, li=li: store.__setitem__(li, o[:, 1:, :])) for li in LAYERS]
    return store, hooks

def run_tokens(model, batch, store):
    store.clear()
    with torch.inference_mode(), torch.autocast(**AMP):
        model.forward_features(batch.to(DEVICE))
    return torch.stack([store[li].float() for li in LAYERS], 1)      # (B, 4, tokens, D)

def to_tensor(tile):
    return transforms.functional.normalize(torch.from_numpy(np.ascontiguousarray(tile)).permute(2, 0, 1).float() / 255, MEAN, STD)

def scaled(i, S):
    img = canvases[i][0]
    return img if S == CANVAS else np.asarray(Image.fromarray(img).resize((S, S), Image.BICUBIC))

def features(model, store, i, S, nt, cells, centred=False, batch=48):
    """Token features (4 layers) for the given cells of image i at scale S."""
    img = scaled(i, S); p = 224 // nt; D = None
    if not centred:     # Exp 13B tiling: non-overlapping 224 tiles
        tile_of = [(int(cy) // nt * 224, int(cx) // nt * 224) for cy, cx in cells]
    else:               # tile whose centre is nearest to the cell, origins on a 112-px grid
        K = (S - 224) // STRIDE
        def org(c):
            centre = c * p + p / 2
            return int(np.clip(np.round((centre - 112) / STRIDE), 0, K)) * STRIDE
        tile_of = [(org(int(cy)), org(int(cx))) for cy, cx in cells]
    tiles = sorted(set(tile_of)); tpos = {t: j for j, t in enumerate(tiles)}
    out = None
    for s in range(0, len(tiles), batch):
        chunk = tiles[s:s + batch]
        tok = run_tokens(model, torch.stack([to_tensor(img[oy:oy + 224, ox:ox + 224]) for oy, ox in chunk]), store)
        if out is None:
            D = tok.shape[-1]; out = np.zeros((len(cells), 4, D), np.float16)
        for r_, ((cy, cx), t) in enumerate(zip(cells, tile_of)):
            j = tpos[t] - s
            if 0 <= j < len(chunk):
                ty, tx = (int(cy) * p - t[0]) // p, (int(cx) * p - t[1]) // p
                out[r_] = tok[j, :, ty * nt + tx].half().cpu().numpy()
    return out

# ------------------------------------------------------------
# 4. Probe (identical to Exp 13B)
# ------------------------------------------------------------
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

def fit_probe(Xtr, ytr, gtr, hold):
    iv = np.isin(gtr, list(hold)); it = ~iv
    wd = max(WD_GRID, key=lambda w: roc_auc_score(ytr[iv], train_logreg(Xtr[it], ytr[it], w)(Xtr[iv])))
    return train_logreg(Xtr, ytr, wd), wd

def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan

# ------------------------------------------------------------
# 5. Run
# ------------------------------------------------------------
rows, SC = [], {}      # SC[(variant, encoder, S, layer)] = (scores, labels, groups, image_ids)
def record(variant, name, pair, S, nt, ln, s, y, g, iid):
    SC[(variant, name, S, ln)] = (s, y, g, iid)
    rows.append(dict(variant=variant, pair=pair, encoder=name, scale=S, cell_px=CANVAS // ((S // 224) * nt), layer=ln,
                     n_cells=len(y), n_pos=int(y.sum()), prevalence=float(y.mean()), auroc=safe_auc(y, s)))

for name, (pair, nt) in {k: (v[0], v[2]) for k, v in ENCODERS.items()}.items():
    ck = OUT / f"exp17_checkpoint_{name}.pkl"
    if ck.exists():      # finished in an earlier run: reuse, do not recompute
        saved = pickle.loads(ck.read_bytes()); rows += saved["rows"]; SC.update(saved["SC"])
        print(f"\n=== {name} === loaded from {ck.name}"); continue
    n_rows0 = len(rows)
    print(f"\n=== {name} ===")
    model = build_encoder(name); store, hooks = make_store(model)
    for S in SCALES:
        for mset in ["strict", "lenient", "stricter"]:
            sel = data[MASK_SETS[mset]]
            tr_ids = sel.image_id[sel.split == "train"].tolist(); te_ids = sel.image_id[sel.split == "test"].tolist()
            tr_groups = sel.risk_group_id[sel.split == "train"].unique()
            hold = set(np.random.default_rng(SEED).choice(tr_groups, size=max(1, len(tr_groups) // 5), replace=False))
            centred_opts = [False, True] if (mset == "strict" and S > 224) else [False]
            for centred in centred_opts:
                Xs, ys, gs = [], [], []
                for iid in tqdm(tr_ids, desc=f"{name} S={S} {mset}{' centred' if centred else ''} train", leave=False):
                    i = ROW[iid]; cells, lab = sample_cells(i, S, nt)
                    if len(cells) == 0: continue
                    Xs.append(features(model, store, i, S, nt, cells, centred)); ys.append(lab)
                    gs.append(np.full(len(lab), img_rows.risk_group_id[i]))
                Xtr = np.concatenate(Xs); ytr = np.concatenate(ys).astype(np.float32); gtr = np.concatenate(gs)
                probes = {ln: fit_probe(Xtr[:, li].astype(np.float32), ytr, gtr, hold)[0] for li, ln in enumerate(LAYER_NAMES)}
                del Xtr, Xs; gc.collect()
                base_tag = f"{mset}_centred" if centred else ("baseline" if mset == "strict" else mset)
                acc = {ln: dict(base=[[], [], [], []], allc=[[], [], [], []], hard=[[], [], [], []]) for ln in LAYER_NAMES}
                for iid in tqdm(te_ids, desc=f"{name} S={S} {base_tag} test", leave=False):
                    i = ROW[iid]; g = img_rows.risk_group_id[i]
                    cells, lab = sample_cells(i, S, nt)
                    F = features(model, store, i, S, nt, cells, centred) if len(cells) else None
                    if base_tag == "baseline":
                        acells, kind = all_cells(i, S, nt); FA = features(model, store, i, S, nt, acells)
                    for li, ln in enumerate(LAYER_NAMES):
                        if F is not None:
                            a = acc[ln]["base"]; a[0].append(probes[ln](F[:, li].astype(np.float32))); a[1].append(lab)
                            a[2].append(np.full(len(lab), g)); a[3].append(np.full(len(lab), iid))
                        if base_tag == "baseline":
                            s_all = probes[ln](FA[:, li].astype(np.float32)); y_all = (kind == 2).astype(float)
                            a = acc[ln]["allc"]; a[0].append(s_all); a[1].append(y_all); a[2].append(np.full(len(y_all), g)); a[3].append(np.full(len(y_all), iid))
                            h = kind >= 1
                            a = acc[ln]["hard"]; a[0].append(s_all[h]); a[1].append(y_all[h]); a[2].append(np.full(int(h.sum()), g)); a[3].append(np.full(int(h.sum()), iid))
                for ln in LAYER_NAMES:
                    for key, tag in [("base", base_tag), ("allc", "all_cells"), ("hard", "hard_negatives")]:
                        a = acc[ln][key]
                        if not a[0]: continue
                        record(tag, name, pair, S, nt, ln, *[np.concatenate(v) for v in a])
                b24 = [r for r in rows if r["variant"] == base_tag and r["encoder"] == name and r["scale"] == S and r["layer"] == "block24"]
                print(f"S={S:>4} {base_tag:16s} block24 AUROC {b24[-1]['auroc']:.3f}" if b24 else f"S={S:>4} {base_tag:16s} done")
    for h in hooks: h.remove()
    ck.write_bytes(pickle.dumps(dict(rows=rows[n_rows0:], SC={k: v for k, v in SC.items() if k[1] == name})))
    del model; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

res = pd.DataFrame(rows); res.to_csv(OUT / "exp17_auroc_by_variant.csv", index=False)

# ------------------------------------------------------------
# 6. Per-image AUROC and paired bootstrap (retinal - generic)
# ------------------------------------------------------------
def per_image(s, y, iid):
    d = pd.DataFrame(dict(s=s, y=y, iid=iid)); out = {}
    for k, v in d.groupby("iid"):
        if 0 < v.y.sum() < len(v): out[k] = roc_auc_score(v.y, v.s)
    return pd.Series(out)

rng = np.random.default_rng(SEED); brow = []
for (variant, name, S, ln) in list(SC):
    pair = ENCODERS[name][0]; ret, gen = PAIRS[pair]
    if name != ret or (variant, gen, S, ln) not in SC: continue
    sa, ya, ga, ia = SC[(variant, ret, S, ln)]; sb, yb, gb, ib = SC[(variant, gen, S, ln)]
    assert np.array_equal(ya, yb) and np.array_equal(ia, ib), "cells must match within a pair"
    grp = pd.Series(np.arange(len(ya))).groupby(ga).indices; keys = list(grp)
    d = []
    for _ in range(N_BOOT):
        idx = np.concatenate([grp[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        if ya[idx].min() == ya[idx].max(): continue
        d.append(roc_auc_score(ya[idx], sa[idx]) - roc_auc_score(ya[idx], sb[idx]))
    pa, pb = per_image(sa, ya, ia), per_image(sb, yb, ib); common = pa.index.intersection(pb.index)
    pdiff = (pa[common] - pb[common]); g_img = pd.Series(ga, index=ia).groupby(level=0).first()[common]
    gi = pd.Series(np.arange(len(common))).groupby(g_img.to_numpy()).indices; gk = list(gi); dd = []
    for _ in range(N_BOOT):
        idx = np.concatenate([gi[gk[k]] for k in rng.integers(0, len(gk), len(gk))]); dd.append(pdiff.to_numpy()[idx].mean())
    brow.append(dict(variant=variant, pair=pair, scale=S, layer=ln,
                     pooled_auroc_retinal=roc_auc_score(ya, sa), pooled_auroc_generic=roc_auc_score(yb, sb),
                     pooled_diff=roc_auc_score(ya, sa) - roc_auc_score(yb, sb),
                     pooled_ci_low=np.percentile(d, 2.5), pooled_ci_high=np.percentile(d, 97.5),
                     per_image_n=len(common), per_image_auroc_retinal=float(pa[common].mean()),
                     per_image_auroc_generic=float(pb[common].mean()), per_image_diff=float(pdiff.mean()),
                     per_image_ci_low=np.percentile(dd, 2.5), per_image_ci_high=np.percentile(dd, 97.5)))
boot = pd.DataFrame(brow).sort_values(["variant", "pair", "scale", "layer"])
boot.to_csv(OUT / "exp17_paired_differences.csv", index=False)

fmt = lambda x: f"{x:.4f}"
print("\n" + "=" * 70 + "\nEXP 17 - PAIRED DIFFERENCES, block 24 (retinal - generic)\n" + "=" * 70)
print(boot[boot.layer == "block24"].to_string(index=False, float_format=fmt))
(OUT / "Exp17_run_info.json").write_text(json.dumps(dict(
    finished_utc=datetime.now(timezone.utc).isoformat(), smoke=SMOKE, scales=SCALES, n_boot=N_BOOT,
    stricter_max_largest_px=STRICTER_MAX_LARGEST_PX,
    mask_sets={k: dict(train=int((v & data.split.eq("train")).sum()), test=int((v & data.split.eq("test")).sum()))
               for k, v in MASK_SETS.items()}), indent=2))
print("EXPERIMENT 17 DONE")
