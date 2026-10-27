# Source: Exp13_Test_v2.ipynb, cell 5 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp13c_test_attribution
from common.paths import ROOT, MASK_AUDIT_CSV, load_index, load_master

# ============================================================
# Experiment 13C - TEST-SET REPLICATION OF THE ATTRIBUTION
# ANALYSIS (Grad-CAM vs microaneurysm masks, spatial null)
#
# Same Grad-CAM and retina-constrained spatial null as
# Exp 05 / 07.33 (validation), applied once to the sealed test
# split, for BOTH pairs and all three seeds:
#   MAE pair:    RETFound-MAE vs ImageNet MAE      (14x14 grid)
#   DINOv2 pair: RETFound-DINOv2 vs DINOv2         (16x16 grid)
# Primary (pre-specified): seed-42 models of each pair,
#   retinal minus generic null-adjusted enrichment.
# Secondary: mean over seeds 42/1042/2042; raw enrichment;
#   spatial percentile; no-erosion foreground sensitivity.
# Only small CSV/JSON files are written to Drive.
# Runtime on L4: about 15-20 minutes.
# ============================================================

import gc, json, hashlib
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


OUT = ROOT / "experiment13_test/attribution"; OUT.mkdir(parents=True, exist_ok=True)
QUALITY_PATH = MASK_AUDIT_CSV
DEVICE = torch.device("cuda"); assert torch.cuda.is_available(), "Switch the runtime to L4 GPU."

V16, D14 = "vit_large_patch16_224", "vit_large_patch14_dinov2"
E11, E16 = ROOT / "experiment11_multiseed", ROOT / "experiment16_dinov2_pair"
MODELS = {  # id: (pair, role, seed, arch, checkpoint)
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

SIZE, MIN_COVERAGE, MIN_NULL, THRESH = 224, 0.95, 100, 127

PLAN = OUT / "Exp13C_Attribution_Plan.json"
if not PLAN.exists():
    PLAN.write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        cohort="test images meeting the validation primary_ma_candidate rule (strict mask, non-empty, no quality flags)",
        cam="Grad-CAM on final-block patch tokens, true-grade logit, ReLU, bilinear to 224, max-normalised (as Exp 05/07)",
        null="all mask translations keeping >=95% weight in eroded retinal foreground; >=100 shifts; images failing excluded",
        primary="seed-42 models: retinal minus generic null-adjusted enrichment, per pair",
        secondary="seed-mean differences; raw enrichment; spatial percentile; per-model means; no-erosion sensitivity",
        statistics="paired cluster bootstrap over test source-risk groups, 2000 resamples"), indent=2))
plan_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()

# ------------------------------------------------------------
# 1. Test MA cohort (same rule as the validation cohort)
# ------------------------------------------------------------
index = load_index()
master = load_master()
qual = pd.read_csv(QUALITY_PATH, dtype={"image_id": str})
t = index[index.split == "test"].merge(master[["image_id", "mask_path"]], on="image_id").merge(qual, on="image_id", how="left",
                                                                                               suffixes=("", "_q"))
flag = lambda c: t[c].astype(str).str.lower().isin(["true", "1"])
primary = (t.review_category.eq("strict_lesion_level_candidate") & ~flag("is_empty_mask") & flag("dimensions_match")
           & ~flag("threshold_sensitive_flag") & ~flag("grayscale_encoding_flag") & ~flag("broad_region_flag")
           & t.total_mask_area_px.gt(0) & t.mask_path.map(lambda p: Path(str(p)).is_file()))
cohort = t[primary].sort_values("image_id").reset_index(drop=True)
print(f"Test MA cohort: {len(cohort)} images, {cohort.risk_group_id.nunique()} source-risk groups")

pre = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                          transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
inputs, masks, fgs_eroded, fgs_raw = [], [], [], []
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="Images, masks, foreground"):
    with Image.open(row.source_path) as im:
        rgb_full = im.convert("RGB")
        inputs.append(pre(rgb_full).unsqueeze(0))
        rgb = np.asarray(rgb_full.resize((SIZE, SIZE), Image.Resampling.BICUBIC))
    with Image.open(row.mask_path) as mk:
        m = np.asarray(mk)
    if m.ndim == 3: m = m[..., 0]
    masks.append(cv2.resize((m > THRESH).astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float64))
    cand = cv2.morphologyEx((rgb.max(axis=2) > 20).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg = np.zeros((SIZE, SIZE), np.uint8); cv2.drawContours(fg, [max(cnts, key=cv2.contourArea)], -1, 1, thickness=-1)
    fgs_raw.append(fg.astype(np.float64)); fgs_eroded.append(cv2.erode(fg, kernel).astype(np.float64))

# ------------------------------------------------------------
# 2. Spatial null (identical to Exp 07.33)
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
    return dict(eligible=obs_cov >= MIN_COVERAGE and ok.sum() >= MIN_NULL, ys=sy[ok], xs=sx[ok], n=int(ok.sum()))

def spatial(cam, mask, g):
    s = fft_shift(cam, mask) / (float(mask.sum()) * float(cam.mean()))
    obs = float(s[0, 0]); null = s[g["ys"], g["xs"]]
    return dict(raw=obs, null_mean=float(null.mean()), adjusted=float(obs - null.mean()),
                percentile=float(100 * np.mean(null < obs)))

GEO = {"eroded": [geometry(m, f) for m, f in zip(masks, fgs_eroded)],
       "no_erosion": [geometry(m, f) for m, f in zip(masks, fgs_raw)]}
for k, gl in GEO.items():
    print(f"Eligible ({k}): {sum(g['eligible'] for g in gl)} / {len(gl)}")

# ------------------------------------------------------------
# 3. Grad-CAM (same computation as Exp 05 / 07 / 10)
# ------------------------------------------------------------
def build(arch, ckpt):
    if arch == D14:
        m = timm.create_model(arch, pretrained=False, img_size=224, num_classes=5, global_pool="avg")
    else:
        m = timm.create_model(arch, pretrained=False, num_classes=5, global_pool="avg")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    m.load_state_dict(sd, strict=True)
    return m.eval().requires_grad_(False).to(DEVICE)

def cams_for(model):
    cap = {}
    def hook(mod, inp, out):
        a = out.detach().requires_grad_(True); cap["a"] = a; return a
    h = model.blocks[23].register_forward_hook(hook)
    out = []
    try:
        for x, g in zip(inputs, cohort.dr_grade.astype(int)):
            cap.clear()
            with torch.enable_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x.to(DEVICE))
                a = cap["a"]; grad = torch.autograd.grad(logits[0, g], a)[0]
            n = a.shape[1] - 1; side = int(round(n ** 0.5)); assert side * side == n
            f = a[0, 1:, :].detach().float().T.reshape(-1, side, side)
            w = grad[0, 1:, :].detach().float().T.reshape(-1, side, side).mean(dim=(1, 2), keepdim=True)
            raw = torch.relu((w * f).sum(0)).cpu().numpy()
            if raw.max() <= 1e-12:
                out.append(None); continue
            cam = cv2.resize(raw, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            out.append(cam / cam.max())
    finally:
        h.remove()
    return out

records = []
for mid, (pair, role, seed, arch, ckpt) in MODELS.items():
    model = build(arch, ckpt); cams = cams_for(model)
    del model; gc.collect(); torch.cuda.empty_cache()
    for i, cam in enumerate(cams):
        for fgk, gl in GEO.items():
            g = gl[i]
            if not g["eligible"] or cam is None: continue
            records.append(dict(model_id=mid, pair=pair, role=role, seed=seed, foreground=fgk,
                                image_id=cohort.image_id[i], risk_group_id=cohort.risk_group_id[i],
                                true_grade=int(cohort.dr_grade[i]), **spatial(cam, masks[i], g)))
    print(f"Grad-CAM done: {mid} (degenerate maps: {sum(c is None for c in cams)})")

res = pd.DataFrame(records); res.to_csv(OUT / "test_attribution_per_image.csv", index=False)

# ------------------------------------------------------------
# 4. Paired comparisons with cluster bootstrap
# ------------------------------------------------------------
rng = np.random.default_rng(20261027)
def boot_mean(df_diff, col, n=2000):
    g = df_diff.groupby("risk_group_id")[col].agg(["sum", "count"]); s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(n, len(g))); b = s[idx].sum(1) / c[idx].sum(1)
    return float(df_diff[col].mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

rows = []
for fgk in ["eroded", "no_erosion"]:
    for pair in ["mae", "dinov2"]:
        for scope, seeds in [("seed42 (primary)" if fgk == "eroded" else "seed42", [42]), ("seed mean", [42, 1042, 2042])]:
            sub = res[(res.foreground == fgk) & (res.pair == pair) & res.seed.isin(seeds)]
            per = sub.groupby(["image_id", "risk_group_id", "role"])[["raw", "adjusted", "percentile"]].mean().unstack("role")
            per = per.dropna()
            diff = pd.DataFrame({m: per[(m, "retinal")] - per[(m, "generic")] for m in ["raw", "adjusted", "percentile"]}).reset_index()
            for m in ["raw", "adjusted", "percentile"]:
                mean, lo, hi = boot_mean(diff, m)
                rows.append(dict(foreground=fgk, pair=pair, scope=scope, metric=m, images=len(diff),
                                 retinal=float(per[(m, "retinal")].mean()), generic=float(per[(m, "generic")].mean()),
                                 diff_retinal_minus_generic=mean, ci_low=lo, ci_high=hi))
comp = pd.DataFrame(rows); comp.to_csv(OUT / "test_attribution_comparisons.csv", index=False)

model_rows = []
for mid, sub in res[res.foreground == "eroded"].groupby("model_id"):
    mean, lo, hi = boot_mean(sub, "adjusted")
    model_rows.append(dict(model_id=mid, images=len(sub), raw_mean=sub.raw.mean(), raw_gt1=int((sub.raw > 1).sum()),
                           adjusted_mean=mean, adjusted_ci_low=lo, adjusted_ci_high=hi,
                           above_95th=int((sub.percentile >= 95).sum())))
per_model = pd.DataFrame(model_rows); per_model.to_csv(OUT / "test_attribution_per_model.csv", index=False)

fmt = lambda x: f"{x:.4f}"
print("\n" + "=" * 70 + "\nEXP 13C - TEST-SET MA ATTRIBUTION\n" + "=" * 70)
print(f"Cohort: {len(cohort)} test images; eligible (eroded) = {sum(g['eligible'] for g in GEO['eroded'])}")
print("\nPER MODEL (eroded foreground)")
print(per_model.to_string(index=False, float_format=fmt))
print("\nPAIRED COMPARISONS: retinal - generic (2000x risk-group bootstrap)")
print(comp.to_string(index=False, float_format=fmt))
(OUT / "Exp13C_run_info.json").write_text(json.dumps(dict(plan_sha256=plan_sha, cohort=len(cohort),
    groups=int(cohort.risk_group_id.nunique()), finished_utc=datetime.now(timezone.utc).isoformat()), indent=2))
print("\nPlan SHA256:", plan_sha)
print("EXPERIMENT 13C DONE")
