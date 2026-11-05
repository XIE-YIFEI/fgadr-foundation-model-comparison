# Experiment 18 - REVIEWER ANALYSES FOR THE FINE-TUNED ATTRIBUTION RESULT
#
# Run from the repository root (Colab, L4 GPU):
#     python -m experiments.exp18_attribution_occlusion
# Smoke test on CPU with random data and a tiny ViT (no real data needed):
#     EXP18_SMOKE=1 python -m experiments.exp18_attribution_occlusion
#
# Same test MA cohort, 12 fine-tuned models, Grad-CAM and retina-constrained
# spatial null as Exp 13C. Three additions:
#
#   A  Grad-CAM for the PREDICTED class, next to the reference-grade map
#   B  results split by correctly classified vs misclassified images
#   C  lesion occlusion: the MA mask is dilated by 5 px at full resolution
#      and inpainted (Telea, radius 5); a control occlusion inpaints the same
#      dilated mask translated to lesion-free retina (>=95% of pixels in the
#      retinal foreground, no overlap with the dilated lesions; 3 shifts,
#      averaged). Effects on p(reference grade) and on the expected grade.
#      Lesion-specific effect = lesion effect - control effect.
#
# Output: CSV files in <results_root>/experiment18_attribution_occlusion
# Paired retinal-minus-generic differences: cluster bootstrap over test
# source-risk groups (2000 resamples), as in Exp 13C.
# Runtime on L4: about 30-40 minutes.

import os, gc, json, hashlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

SMOKE = os.environ.get("EXP18_SMOKE") == "1"
SIZE, MIN_COVERAGE, MIN_NULL, THRESH = 224, 0.95, 100, 127
DILATE_PX, INPAINT_RADIUS, N_CONTROL, MAX_TRIES = 5, 5, 3, 20000
N_BOOT, SEED = 2000, 20261105

if SMOKE:
    from experiments._smoke import make_smoke_inputs
    OUT, cohort, MODELS, build, DEVICE = make_smoke_inputs("exp18")
    N_BOOT, MIN_NULL = 200, 10
else:
    from common.paths import ROOT, MASK_AUDIT_CSV, load_index, load_master
    OUT = ROOT / "experiment18_attribution_occlusion"; OUT.mkdir(parents=True, exist_ok=True)
    DEVICE = torch.device("cuda"); assert torch.cuda.is_available(), "Switch the runtime to L4 GPU."
    V16, D14 = "vit_large_patch16_224", "vit_large_patch14_dinov2"
    E11, E16 = ROOT / "experiment11_multiseed", ROOT / "experiment16_dinov2_pair"
    MODELS = {  # id: (pair, role, seed, arch, checkpoint) - identical to Exp 13C
        "mae_retfound_s42":    ("mae", "retinal", 42,   V16, ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"),
        "mae_generic_s42":     ("mae", "generic", 42,   V16, ROOT / "experiment07_results/meta_mae_vit_large_v1/best_model.pt"),
        "mae_retfound_s1042":  ("mae", "retinal", 1042, V16, E11 / "retfound_seed1042/best_model.pt"),
        "mae_generic_s1042":   ("mae", "generic", 1042, V16, E11 / "meta_mae_seed1042/best_model.pt"),
        "mae_retfound_s2042":  ("mae", "retinal", 2042, V16, E11 / "retfound_seed2042/best_model.pt"),
        "mae_generic_s2042":   ("mae", "generic", 2042, V16, E11 / "meta_mae_seed2042/best_model.pt"),
        "dino_retfound_s42":   ("dinov2", "retinal", 42,   D14, E16 / "retfound_dinov2_seed42/best_model.pt"),
        "dino_generic_s42":    ("dinov2", "generic", 42,   D14, E16 / "dinov2_generic_seed42/best_model.pt"),
        "dino_retfound_s1042": ("dinov2", "retinal", 1042, D14, E16 / "retfound_dinov2_seed1042/best_model.pt"),
        "dino_generic_s1042":  ("dinov2", "generic", 1042, D14, E16 / "dinov2_generic_seed1042/best_model.pt"),
        "dino_retfound_s2042": ("dinov2", "retinal", 2042, D14, E16 / "retfound_dinov2_seed2042/best_model.pt"),
        "dino_generic_s2042":  ("dinov2", "generic", 2042, D14, E16 / "dinov2_generic_seed2042/best_model.pt"),
    }
    for k, v in MODELS.items():
        assert v[4].is_file(), f"Missing checkpoint {k}: {v[4]}"

    def build(arch, ckpt):
        if arch == D14:
            m = timm.create_model(arch, pretrained=False, img_size=224, num_classes=5, global_pool="avg")
        else:
            m = timm.create_model(arch, pretrained=False, num_classes=5, global_pool="avg")
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        m.load_state_dict(sd, strict=True)
        return m.eval().requires_grad_(False)

    # Test MA cohort: same rule as Exp 13C
    index = load_index(); master = load_master()
    qual = pd.read_csv(MASK_AUDIT_CSV, dtype={"image_id": str})
    t = index[index.split == "test"].merge(master[["image_id", "mask_path"]], on="image_id").merge(
        qual, on="image_id", how="left", suffixes=("", "_q"))
    flag = lambda c: t[c].astype(str).str.lower().isin(["true", "1"])
    primary = (t.review_category.eq("strict_lesion_level_candidate") & ~flag("is_empty_mask") & flag("dimensions_match")
               & ~flag("threshold_sensitive_flag") & ~flag("grayscale_encoding_flag") & ~flag("broad_region_flag")
               & t.total_mask_area_px.gt(0) & t.mask_path.map(lambda p: Path(str(p)).is_file()))
    cohort = t[primary].sort_values("image_id").reset_index(drop=True)
AMP = dict(device_type="cuda", dtype=torch.bfloat16) if DEVICE.type == "cuda" else dict(device_type="cpu", enabled=False)
print(f"Test MA cohort: {len(cohort)} images, {cohort.risk_group_id.nunique()} source-risk groups")
assert len(cohort) > 0, "Empty cohort: check fgadr_image_dir and fgadr_ma_mask_dir in config.yaml"

# ------------------------------------------------------------
# 1. Images, masks, foreground, occluded copies
# ------------------------------------------------------------
pre = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                          transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_PX + 1, 2 * DILATE_PX + 1))

def retina(rgb):
    cand = cv2.morphologyEx((rgb.max(axis=2) > 20).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg = np.zeros(rgb.shape[:2], np.uint8); cv2.drawContours(fg, [max(cnts, key=cv2.contourArea)], -1, 1, thickness=-1)
    return fg

def control_shifts(occ, fg, image_id):
    """Translations of the dilated lesion mask into lesion-free retina."""
    ys, xs = np.nonzero(occ); H, W = occ.shape
    r = np.random.default_rng(int(hashlib.md5(f"exp18_{image_id}".encode()).hexdigest()[:8], 16))
    found = []
    for _ in range(MAX_TRIES):
        dy = int(r.integers(-ys.min(), H - ys.max())); dx = int(r.integers(-xs.min(), W - xs.max()))
        if dy == 0 and dx == 0: continue
        ty, tx = ys + dy, xs + dx
        if fg[ty, tx].mean() >= MIN_COVERAGE and not occ[ty, tx].any():
            found.append((dy, dx))
            if len(found) == N_CONTROL: break
    return found

X, X_les, X_ctl, masks, fgs, occ_info = [], [], [], [], [], []
for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="Images, masks, occlusions"):
    with Image.open(row.source_path) as im:
        rgb_full = np.asarray(im.convert("RGB"))
    with Image.open(row.mask_path) as mk:
        m = np.asarray(mk)
    if m.ndim == 3: m = m[..., 0]
    m = m > THRESH
    if m.shape != rgb_full.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (rgb_full.shape[1], rgb_full.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    X.append(pre(Image.fromarray(rgb_full)).unsqueeze(0))
    # 224-px mask and eroded foreground for Grad-CAM (as Exp 13C)
    rgb224 = np.asarray(Image.fromarray(rgb_full).resize((SIZE, SIZE), Image.Resampling.BICUBIC))
    masks.append(cv2.resize(m.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float64))
    fgs.append(cv2.erode(retina(rgb224), kernel).astype(np.float64))
    # occlusions at full resolution
    occ = cv2.dilate(m.astype(np.uint8), dil_kernel)
    X_les.append(pre(Image.fromarray(cv2.inpaint(rgb_full, occ, INPAINT_RADIUS, cv2.INPAINT_TELEA))).unsqueeze(0))
    fg_full = retina(rgb_full); shifts = control_shifts(occ, fg_full, row.image_id); ctl = []
    ys, xs = np.nonzero(occ)
    for dy, dx in shifts:
        o2 = np.zeros_like(occ); o2[ys + dy, xs + dx] = 1
        ctl.append(pre(Image.fromarray(cv2.inpaint(rgb_full, o2, INPAINT_RADIUS, cv2.INPAINT_TELEA))).unsqueeze(0))
    X_ctl.append(ctl)
    occ_info.append(dict(image_id=row.image_id, occluded_px=int(occ.sum()), mask_px=int(m.sum()),
                         occluded_fraction_of_retina=float(occ.sum() / max(1, fg_full.sum())), n_control=len(shifts)))
occ_info = pd.DataFrame(occ_info); occ_info.to_csv(OUT / "exp18_occlusion_geometry.csv", index=False)
print(f"Images with {N_CONTROL} control shifts: {(occ_info.n_control == N_CONTROL).sum()} / {len(occ_info)}")

# ------------------------------------------------------------
# 2. Spatial null (identical to Exp 13C, eroded foreground)
# ------------------------------------------------------------
def fft_shift(values, mask):
    return np.fft.ifft2(np.fft.fft2(values.astype(np.float64)) * np.conj(np.fft.fft2(mask))).real

def geometry(mask, fg):
    ys, xs = np.nonzero(mask > 0)
    dx = np.arange(-int(xs.min()), SIZE - int(xs.max())); dy = np.arange(-int(ys.min()), SIZE - int(ys.max()))
    sy, sx = np.meshgrid(dy % SIZE, dx % SIZE, indexing="ij")
    cov = fft_shift(fg, mask) / mask.sum()
    ok = (cov[sy, sx] >= MIN_COVERAGE) & ~((sy == 0) & (sx == 0))
    obs_cov = float((fg * mask).sum() / mask.sum())
    return dict(eligible=obs_cov >= MIN_COVERAGE and ok.sum() >= MIN_NULL, ys=sy[ok], xs=sx[ok])

def spatial(cam, mask, g):
    s = fft_shift(cam, mask) / (float(mask.sum()) * float(cam.mean()))
    obs = float(s[0, 0]); null = s[g["ys"], g["xs"]]
    return dict(raw=obs, adjusted=float(obs - null.mean()), percentile=float(100 * np.mean(null < obs)))

GEO = [geometry(m, f) for m, f in zip(masks, fgs)]
print(f"Eligible for the spatial null: {sum(g['eligible'] for g in GEO)} / {len(GEO)}")

# ------------------------------------------------------------
# 3. Grad-CAM (reference grade and predicted class) and probabilities
# ------------------------------------------------------------
def to_cam(a, grad):
    n = a.shape[1] - 1; side = int(round(n ** 0.5)); assert side * side == n
    f = a[0, 1:, :].detach().float().T.reshape(-1, side, side)
    w = grad[0, 1:, :].detach().float().T.reshape(-1, side, side).mean(dim=(1, 2), keepdim=True)
    raw = torch.relu((w * f).sum(0)).cpu().numpy()
    if raw.max() <= 1e-12: return None
    cam = cv2.resize(raw, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return cam / cam.max()

def probs(model, xs):
    with torch.inference_mode(), torch.autocast(**AMP):
        return torch.softmax(model(torch.cat(xs).to(DEVICE)).float(), 1).cpu().numpy()

GRADES = np.arange(5)
records = []
for mid, (pair, role, seed, arch, ckpt) in MODELS.items():
    model = build(arch, ckpt).to(DEVICE)
    cap = {}
    def hook(mod, inp, out):
        a = out.detach().requires_grad_(True); cap["a"] = a; return a
    h = model.blocks[23].register_forward_hook(hook)
    try:
        for i, x in enumerate(tqdm(X, desc=mid, leave=False)):
            g_true = int(cohort.dr_grade[i]); cap.clear()
            with torch.enable_grad():
                with torch.autocast(**AMP):
                    logits = model(x.to(DEVICE))
                p0 = torch.softmax(logits.float(), 1)[0].detach().cpu().numpy(); pred = int(p0.argmax())
                a = cap["a"]
                cam_true = to_cam(a, torch.autograd.grad(logits[0, g_true], a, retain_graph=True)[0])
                cam_pred = cam_true if pred == g_true else to_cam(a, torch.autograd.grad(logits[0, pred], a)[0])
            h.remove()
            p_les = probs(model, [X_les[i]])[0]
            p_ctl = probs(model, X_ctl[i]).mean(0) if X_ctl[i] else np.full(5, np.nan)
            h = model.blocks[23].register_forward_hook(hook)
            E = lambda p: float((GRADES * p).sum())
            rec = dict(model_id=mid, pair=pair, role=role, seed=seed, image_id=cohort.image_id[i],
                       risk_group_id=cohort.risk_group_id[i], true_grade=g_true, pred_grade=pred,
                       correct=bool(pred == g_true), p_true=float(p0[g_true]), expected_grade=E(p0),
                       dp_lesion=float(p0[g_true] - p_les[g_true]), dp_control=float(p0[g_true] - p_ctl[g_true]),
                       dE_lesion=E(p0) - E(p_les), dE_control=E(p0) - E(p_ctl),
                       pred_changed_lesion=bool(int(p_les.argmax()) != pred))
            rec["dp_specific"] = rec["dp_lesion"] - rec["dp_control"]
            rec["dE_specific"] = rec["dE_lesion"] - rec["dE_control"]
            for tag, cam in [("true", cam_true), ("pred", cam_pred)]:
                ok = GEO[i]["eligible"] and cam is not None
                for k, v in (spatial(cam, masks[i], GEO[i]) if ok else dict(raw=np.nan, adjusted=np.nan, percentile=np.nan)).items():
                    rec[f"{k}_{tag}"] = v
            records.append(rec)
    finally:
        h.remove()
    del model; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    sub = pd.DataFrame([r for r in records if r["model_id"] == mid])
    print(f"{mid}: accuracy {sub.correct.mean():.3f}, adjusted(true) {sub.adjusted_true.mean():.3f}, "
          f"adjusted(pred) {sub.adjusted_pred.mean():.3f}, dp_specific {sub.dp_specific.mean():.4f}")

res = pd.DataFrame(records); res.to_csv(OUT / "exp18_per_image.csv", index=False)

# ------------------------------------------------------------
# 4. Summaries with cluster bootstrap over source-risk groups
# ------------------------------------------------------------
rng = np.random.default_rng(SEED)
def boot_mean(df, col):
    d = df[["risk_group_id", col]].dropna()
    if len(d) == 0: return np.nan, np.nan, np.nan, 0
    g = d.groupby("risk_group_id")[col].agg(["sum", "count"]); s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g))); b = s[idx].sum(1) / c[idx].sum(1)
    return float(d[col].mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(d)

def boot_group_diff(df, col, by):
    """mean(col | by) - mean(col | not by), resampling source-risk groups."""
    d = df[["risk_group_id", col, by]].dropna(); d[by] = d[by].astype(bool)
    if d[by].sum() == 0 or (~d[by]).sum() == 0: return np.nan, np.nan, np.nan
    gs = {k: v for k, v in d.groupby("risk_group_id")}; keys = list(gs); b = []
    for _ in range(N_BOOT):
        s = pd.concat([gs[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        if s[by].all() or (~s[by]).all(): continue
        b.append(s.loc[s[by], col].mean() - s.loc[~s[by], col].mean())
    return float(d.loc[d[by], col].mean() - d.loc[~d[by], col].mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

METRICS = ["adjusted_true", "adjusted_pred", "raw_pred", "percentile_pred", "dp_lesion", "dp_control", "dp_specific",
           "dE_lesion", "dE_control", "dE_specific"]

# 4a. per model, overall and by correctness
mrows = []
for mid, sub in res.groupby("model_id", sort=False):
    for subset, s in [("all", sub), ("correct", sub[sub.correct]), ("misclassified", sub[~sub.correct])]:
        for m in METRICS:
            mean, lo, hi, n = boot_mean(s, m)
            mrows.append(dict(model_id=mid, subset=subset, metric=m, n=n, mean=mean, ci_low=lo, ci_high=hi))
    for m in ["adjusted_true", "adjusted_pred", "dp_specific", "dE_specific"]:
        diff, lo, hi = boot_group_diff(sub, m, "correct")
        mrows.append(dict(model_id=mid, subset="correct_minus_misclassified", metric=m, n=len(sub), mean=diff, ci_low=lo, ci_high=hi))
per_model = pd.DataFrame(mrows); per_model.to_csv(OUT / "exp18_per_model.csv", index=False)

# 4b. paired retinal - generic, overall and on images both models classify correctly / both misclassify
prow = []
for pair in ["mae", "dinov2"]:
    for scope, seeds in [("seed42", [42]), ("seed mean", sorted(res.seed.unique()))]:
        sub = res[(res.pair == pair) & res.seed.isin(seeds)]
        per = sub.groupby(["image_id", "risk_group_id", "seed", "role"])[METRICS + ["correct"]].mean().unstack("role")
        both_c = (per[("correct", "retinal")] == 1) & (per[("correct", "generic")] == 1)
        both_w = (per[("correct", "retinal")] == 0) & (per[("correct", "generic")] == 0)
        for subset, keep in [("all", slice(None)), ("both_correct", both_c), ("both_misclassified", both_w)]:
            p = per[keep] if subset != "all" else per
            diff = pd.DataFrame({m: p[(m, "retinal")] - p[(m, "generic")] for m in METRICS}).reset_index()
            diff = diff.groupby(["image_id", "risk_group_id"])[METRICS].mean().reset_index()   # average over seeds
            for m in METRICS:
                mean, lo, hi, n = boot_mean(diff, m)
                prow.append(dict(pair=pair, scope=scope, subset=subset, metric=m, images=n,
                                 retinal=float(p[(m, "retinal")].mean()) if len(p) else np.nan,
                                 generic=float(p[(m, "generic")].mean()) if len(p) else np.nan,
                                 diff_retinal_minus_generic=mean, ci_low=lo, ci_high=hi))
paired = pd.DataFrame(prow); paired.to_csv(OUT / "exp18_paired.csv", index=False)

fmt = lambda x: f"{x:.4f}"
print("\n" + "=" * 70 + "\nEXP 18 - PER MODEL (all images)\n" + "=" * 70)
print(per_model[per_model.subset == "all"].pivot(index="model_id", columns="metric", values="mean")[METRICS].to_string(float_format=fmt))
print("\nPAIRED retinal - generic (all images)")
print(paired[paired.subset == "all"].to_string(index=False, float_format=fmt))
(OUT / "Exp18_run_info.json").write_text(json.dumps(dict(
    finished_utc=datetime.now(timezone.utc).isoformat(), smoke=SMOKE, cohort=len(cohort),
    eligible_spatial_null=int(sum(g["eligible"] for g in GEO)), with_controls=int((occ_info.n_control == N_CONTROL).sum()),
    dilate_px=DILATE_PX, inpaint_radius=INPAINT_RADIUS, n_control=N_CONTROL, n_boot=N_BOOT), indent=2))
print("EXPERIMENT 18 DONE")
