# Source: Exp11_MultiSeed.ipynb, cell 2 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp10_gradcam_randomization
from common.paths import ROOT, load_exp10_manifest

# ============================================================
# Experiment 10 (final, streamlined) - Cascading parameter
# randomization sanity check for Grad-CAM
#
# Same Grad-CAM + same R1-R6 re-initialization rules as the
# audited Exp 10 pilot / production code. Differences:
#   - one script, both models, all stages, 156 images
#   - one randomization seed (1042), as decided
# Validation MA cohort only. Test split is never read.
# Runtime: about 5-10 minutes on L4.
# ============================================================

import gc, json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from timm.layers import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm


EXP10 = ROOT / "experiment10_results"
OUT_DIR = EXP10 / "final_cascade_seed1042"

MODELS = {
    "RETFound": dict(
        checkpoint=ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt",
        reference=ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz",
        reference_key="retfound_cam"),
    "Meta_MAE": dict(
        checkpoint=ROOT / "experiment07_results/meta_mae_vit_large_v1/best_model.pt",
        reference=ROOT / "experiment07_results/gradcam/Meta_MAE_MA_CAM_Maps_V1.npz",
        reference_key="meta_mae_cam"),
}
SEED = 1042
STAGES = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
STAGE_LABELS = {
    "R0": "trained model", "R1": "+ head", "R2": "+ fc_norm & block 23",
    "R3": "+ blocks 18-22", "R4": "+ blocks 12-17", "R5": "+ blocks 0-11",
    "R6": "+ patch/pos embed & cls (fully random)"}
DEVICE = torch.device("cuda")
assert torch.cuda.is_available()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Cohort and images (156 validation MA images)
# ------------------------------------------------------------
manifest = load_exp10_manifest()
assert len(manifest) == 156 and manifest["split"].eq("val").all()

preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

inputs = []
for _, row in tqdm(manifest.iterrows(), total=156, desc="Decoding images"):
    with Image.open(row["resolved_image_path"]) as im:
        inputs.append(preprocess(im.convert("RGB")).unsqueeze(0))
grades = manifest["true_grade"].astype(int).tolist()
image_ids = manifest["image_id"].tolist()

# ------------------------------------------------------------
# 2. Grad-CAM (identical to audited Exp 10 implementation)
# ------------------------------------------------------------
def compare_cams(first, second):
    if first is None or second is None:
        return dict(pearson=np.nan, spearman=np.nan, ssim=np.nan, mae=np.nan)
    a, b = first.astype(np.float64).ravel(), second.astype(np.float64).ravel()
    ac, bc = a - a.mean(), b - b.mean()
    den = np.linalg.norm(ac) * np.linalg.norm(bc)
    return dict(pearson=float(np.dot(ac, bc) / den) if den > 1e-12 else np.nan,
                spearman=float(spearmanr(a, b).statistic),
                ssim=float(structural_similarity(first, second, data_range=1.0)),
                mae=float(np.mean(np.abs(a - b))))

def make_cam_fn(model):
    captured = {}
    def hook_fn(module, inp, out):
        act = out.detach().requires_grad_(True)
        captured["a"] = act
        return act
    hook = model.blocks[23].register_forward_hook(hook_fn)

    def calc(x, grade):
        captured.clear()
        with torch.enable_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.to(DEVICE))
            act = captured["a"]
            grad = torch.autograd.grad(logits[0, grade], act)[0]
        feats = act[0, 1:, :].detach().float().transpose(0, 1).reshape(1024, 14, 14)
        grads = grad[0, 1:, :].detach().float().transpose(0, 1).reshape(1024, 14, 14)
        raw = torch.relu((grads.mean(dim=(1, 2), keepdim=True) * feats).sum(0)).cpu().numpy()
        raw_max = float(raw.max())
        cam = None
        if raw_max > 1e-12:
            cam = cv2.resize(raw, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            cam /= float(cam.max())
        return logits[0].detach().float().cpu().numpy(), cam, raw_max
    return calc, hook

def reinitialize_block(block):
    with torch.no_grad():
        for m in block.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                m.reset_parameters()

def apply_stage(model, stage):
    # Cumulative: called in order R1 -> R6 on the same model.
    with torch.no_grad():
        if stage == "R1":
            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            trunc_normal_(model.head.weight, std=0.02); nn.init.zeros_(model.head.bias)
        elif stage == "R2":
            model.fc_norm.reset_parameters(); reinitialize_block(model.blocks[23])
        elif stage == "R3":
            for i in range(18, 23): reinitialize_block(model.blocks[i])
        elif stage == "R4":
            for i in range(12, 18): reinitialize_block(model.blocks[i])
        elif stage == "R5":
            for i in range(12): reinitialize_block(model.blocks[i])
        elif stage == "R6":
            model.patch_embed.proj.reset_parameters()
            nn.init.normal_(model.cls_token, std=1e-6)
            trunc_normal_(model.pos_embed, std=0.02)

# ------------------------------------------------------------
# 3. Run cascade for both models
# ------------------------------------------------------------
records, cam_store = [], {}
for model_name, spec in MODELS.items():
    print(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}")
    ckpt = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    source_state = ckpt["model_state_dict"]
    model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=5,
                              global_pool="avg", device="meta", weight_init="skip").to_empty(device=DEVICE)
    model.load_state_dict(source_state, strict=True)
    model.eval(); model.requires_grad_(False)
    calc, hook = make_cam_fn(model)

    with np.load(spec["reference"], allow_pickle=False) as arc:
        pos = {str(v): i for i, v in enumerate(arc["image_ids"])}
        refs = np.stack([arc[spec["reference_key"]][pos[i]] for i in image_ids]).astype(np.float32)

    base_logits, base_cams = None, None
    try:
        for stage in STAGES:
            if stage != "R0":
                apply_stage(model, stage)
            n_changed = sum(not torch.equal(t.detach().cpu(), source_state[k])
                            for k, t in model.state_dict().items())
            outs = [calc(x, g) for x, g in tqdm(zip(inputs, grades), total=156, desc=stage, leave=False)]
            logits = np.stack([o[0] for o in outs]); cams = [o[1] for o in outs]
            cam_store[f"{model_name}_{stage}"] = np.stack(
                [c if c is not None else np.zeros((224, 224), np.float32) for c in cams])
            if stage == "R0":
                repro = [compare_cams(c, r)["pearson"] for c, r in zip(cams, refs)]
                print(f"R0 reproduces saved CAMs: min Pearson = {min(repro):.6f}")
                assert min(repro) > 0.999, "R0 does not reproduce the saved CAMs - stop and report."
                base_logits, base_cams = logits, cams
            for i in range(156):
                m = compare_cams(base_cams[i], cams[i])
                records.append(dict(model=model_name, seed=SEED, stage=stage, image_id=image_ids[i],
                                    true_grade=grades[i], risk_group_id=manifest["risk_group_id"][i],
                                    tensors_changed=n_changed, degenerate_cam=cams[i] is None,
                                    raw_max=outs[i][2], logit_l2=float(np.linalg.norm(logits[i] - base_logits[i])),
                                    **{f"vs_r0_{k}": v for k, v in m.items()}))
            print(f"{stage:>3} {STAGE_LABELS[stage]:<42} tensors changed {n_changed:>3} | "
                  f"mean Pearson vs R0 {np.nanmean([r['vs_r0_pearson'] for r in records[-156:]]):.3f}")
    finally:
        hook.remove()
        del model, ckpt, source_state; gc.collect(); torch.cuda.empty_cache()

results = pd.DataFrame(records)
results.to_csv(OUT_DIR / "cascade_per_image.csv", index=False)
np.savez_compressed(OUT_DIR / "cascade_cams.npz", image_ids=np.array(image_ids), **cam_store)

# R6 is a fully random network: both models should be identical
r6_same = np.allclose(cam_store["RETFound_R6"], cam_store["Meta_MAE_R6"], atol=1e-5)

# ------------------------------------------------------------
# 4. Summary with source-risk-group cluster bootstrap CIs
# ------------------------------------------------------------
rng = np.random.default_rng(20261025)
def cluster_ci(df, col, n_boot=2000):
    g = df.groupby("risk_group_id")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(boots, [2.5, 97.5])

rows = []
for (model_name, stage), df in results.groupby(["model", "stage"], sort=False):
    row = dict(model=model_name, stage=stage, description=STAGE_LABELS[stage],
               degenerate=int(df["degenerate_cam"].sum()))
    for col in ["vs_r0_pearson", "vs_r0_spearman", "vs_r0_ssim", "logit_l2"]:
        d = df.dropna(subset=[col])
        lo, hi = cluster_ci(d, col) if stage != "R0" else (np.nan, np.nan)
        row[col] = d[col].mean(); row[col + "_ci"] = f"[{lo:.3f}, {hi:.3f}]" if stage != "R0" else ""
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "cascade_summary.csv", index=False)
(OUT_DIR / "run_info.json").write_text(json.dumps(dict(
    seed=SEED, images=156, risk_groups=int(manifest["risk_group_id"].nunique()),
    r6_identical_across_models=bool(r6_same), test_images_used=0), indent=2))

print("\n" + "=" * 60 + "\nCASCADING RANDOMIZATION SUMMARY (seed 1042, 156 val MA images)\n" + "=" * 60)
print(summary[["model", "stage", "description", "vs_r0_pearson", "vs_r0_pearson_ci",
               "vs_r0_ssim", "logit_l2", "degenerate"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nR6 CAMs identical across models:", r6_same)
print("Test images used: 0")
print("EXPERIMENT 10 FINAL DONE")
