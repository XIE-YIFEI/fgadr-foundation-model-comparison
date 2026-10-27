# Source: Exp13_Test.ipynb, cell 3 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp13a_test_classification
from common.paths import ROOT, load_index

# ============================================================
# Experiment 13A - FINAL TEST-SET CLASSIFICATION EVALUATION
#
# This is the first and only time the frozen FGADR test split
# (276 images) is used for classification.
#
# 1. Writes the pre-specified analysis plan to Drive BEFORE
#    any test image is read.
# 2. Runs inference once per model and saves test predictions.
#    If predictions already exist they are re-used, never
#    recomputed (re-running the cell only re-does statistics).
# 3. Pre-specified comparisons with paired source-risk-group
#    cluster bootstrap (2000 resamples).
# Runtime on L4: about 15-20 minutes.
# ============================================================

import os, gc, json, hashlib, shutil
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, recall_score
from tqdm.auto import tqdm


OUT = ROOT / "experiment13_test"; PRED_DIR = OUT / "test_predictions"
OUT.mkdir(parents=True, exist_ok=True); PRED_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda"); assert torch.cuda.is_available()

V16, D14 = "vit_large_patch16_224", "vit_large_patch14_dinov2"
E11, E12, E16 = ROOT / "experiment11_multiseed", ROOT / "experiment12_official_recipe", ROOT / "experiment16_dinov2_pair"
# model_id: (family, encoder, seed, architecture, checkpoint)
MODELS = {
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
for k, v in MODELS.items():
    assert v[4].is_file(), f"Missing checkpoint for {k}: {v[4]}"

# ------------------------------------------------------------
# 1. Pre-specified analysis plan (written before test access)
# ------------------------------------------------------------
PLAN_PATH = OUT / "Exp13_Analysis_Plan.json"
plan = dict(
    created_utc=datetime.now(timezone.utc).isoformat(),
    test_split="FGADR frozen test split (276 images), first use for classification",
    models={k: dict(family=v[0], encoder=v[1], seed=v[2], arch=v[3], checkpoint=str(v[4])) for k, v in MODELS.items()},
    checkpoint_rule="best validation macro-F1 epoch (fixed before test access)",
    preprocessing="Resize 224x224 bicubic, ImageNet normalisation, no TTA",
    primary_metric="macro-F1 (5-class)",
    secondary_metrics=["QWK", "accuracy", "referable-DR AUC (P(grade>=2))", "DR1 recall",
                       "per-grade recall", "Brier score", "ECE (15 bins)", "NLL"],
    primary_comparisons=["mae_matched: retfound - generic, mean over seeds 42/1042/2042",
                         "dinov2_matched: retfound - generic, mean over seeds 42/1042/2042"],
    secondary_comparisons=["mae_official: retfound - generic (seed 42)",
                           "seed-ensemble (mean probabilities) within each matched family"],
    statistics="paired cluster bootstrap over source-risk groups, 2000 resamples, percentile 95% CI; no multiplicity correction (2 primary comparisons reported with CIs)",
)
if not PLAN_PATH.exists():
    PLAN_PATH.write_text(json.dumps(plan, indent=2))
    print("Analysis plan written:", PLAN_PATH)
else:
    print("Analysis plan already exists (created", json.loads(PLAN_PATH.read_text())["created_utc"], ") - kept unchanged")
plan_sha = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()

# ------------------------------------------------------------
# 2. Test data
# ------------------------------------------------------------
index = load_index()
test = index[index.split == "test"].reset_index(drop=True)
assert len(test) == 276 and test.image_id.is_unique
assert set(test.risk_group_id).isdisjoint(index.loc[index.split != "test", "risk_group_id"])

need_inference = [k for k in MODELS if not (PRED_DIR / f"{k}.csv").exists()]
if need_inference:
    tf = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                             transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def load(p):
        with Image.open(p) as im: return tf(im.convert("RGB"))
    with ThreadPoolExecutor(8) as ex:
        X = torch.stack(list(tqdm(ex.map(load, test.source_path), total=276, desc="Decoding test images")))

    def build(arch, ckpt):
        if arch == D14:
            m = timm.create_model(arch, pretrained=False, img_size=224, num_classes=5, global_pool="avg")
        elif arch == V16:
            m = timm.create_model(arch, pretrained=False, num_classes=5, global_pool="avg")
        else:
            m = timm.create_model(arch, pretrained=False, num_classes=5)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        m.load_state_dict(sd, strict=True)
        return m.eval().to(DEVICE)

    for k in need_inference:
        fam, enc, seed, arch, ckpt = MODELS[k]
        m = build(arch, ckpt); probs = []
        with torch.inference_mode():
            for s in range(0, 276, 32):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    probs.append(torch.softmax(m(X[s:s + 32].to(DEVICE)).float(), 1).cpu())
        P = torch.cat(probs).numpy()
        pd.DataFrame({"image_id": test.image_id, "true_grade": test.dr_grade.astype(int),
                      **{f"prob_dr{g}": P[:, g] for g in range(5)}}).to_csv(PRED_DIR / f"{k}.csv", index=False)
        print(f"Inference done: {k}")
        del m; gc.collect(); torch.cuda.empty_cache()

PRED = {k: pd.read_csv(PRED_DIR / f"{k}.csv", dtype={"image_id": str}) for k in MODELS}
for k, df in PRED.items():
    assert df.image_id.tolist() == test.image_id.tolist(), k
y = test.dr_grade.astype(int).to_numpy(); groups = test.risk_group_id.to_numpy()
P = {k: df[[f"prob_dr{g}" for g in range(5)]].to_numpy() for k, df in PRED.items()}

# ------------------------------------------------------------
# 3. Metrics
# ------------------------------------------------------------
def ece(prob, yt, bins=15):
    conf, pred = prob.max(1), prob.argmax(1); edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any(): e += sel.mean() * abs((pred[sel] == yt[sel]).mean() - conf[sel].mean())
    return e

def metrics(prob, yt):
    pred = prob.argmax(1); ref_t = yt >= 2
    rec = recall_score(yt, pred, labels=range(5), average=None, zero_division=0)
    out = dict(macro_f1=f1_score(yt, pred, labels=range(5), average="macro", zero_division=0),
               qwk=cohen_kappa_score(yt, pred, weights="quadratic"), accuracy=accuracy_score(yt, pred),
               referable_auc=roc_auc_score(ref_t, prob[:, 2:].sum(1)) if 0 < ref_t.mean() < 1 else np.nan,
               dr1_recall=rec[1], brier=np.mean(((prob - np.eye(5)[yt]) ** 2).sum(1)),
               ece=ece(prob, yt), nll=-np.mean(np.log(np.clip(prob[np.arange(len(yt)), yt], 1e-12, 1))))
    out.update({f"recall_dr{g}": rec[g] for g in range(5)})
    return out

per_model = pd.DataFrame([dict(model_id=k, family=v[0], encoder=v[1], seed=v[2], **metrics(P[k], y)) for k, v in MODELS.items()])
per_model.to_csv(OUT / "test_metrics_per_model.csv", index=False)

KEY = ["macro_f1", "qwk", "accuracy", "referable_auc", "dr1_recall", "brier", "ece"]
def family_ids(fam, enc): return [k for k, v in MODELS.items() if v[0] == fam and v[1] == enc]
COMPARISONS = {"mae_matched (primary)": "mae_matched", "dinov2_matched (primary)": "dinov2_matched",
               "mae_official (secondary)": "mae_official"}

def seed_mean_metrics(ids, idx):
    return pd.DataFrame([metrics(P[k][idx], y[idx]) for k in ids])[KEY].mean()

rng = np.random.default_rng(20261026)
gidx = pd.Series(range(276)).groupby(groups).indices; gkeys = list(gidx)
BOOT = [np.concatenate([gidx[gkeys[j]] for j in rng.integers(0, len(gkeys), len(gkeys))]) for _ in range(2000)]

rows = []
for label, fam in COMPARISONS.items():
    r_ids, g_ids = family_ids(fam, "retfound"), family_ids(fam, "generic")
    full = np.arange(276)
    obs = seed_mean_metrics(r_ids, full) - seed_mean_metrics(g_ids, full)
    boots = []
    for idx in tqdm(BOOT, desc=label, leave=False):
        if len(np.unique(y[idx])) < 5: continue
        boots.append(seed_mean_metrics(r_ids, idx) - seed_mean_metrics(g_ids, idx))
    boots = pd.DataFrame(boots)
    for mname in KEY:
        rows.append(dict(comparison=label, metric=mname,
                         retfound=seed_mean_metrics(r_ids, full)[mname], generic=seed_mean_metrics(g_ids, full)[mname],
                         diff_retfound_minus_generic=obs[mname],
                         ci_low=boots[mname].quantile(0.025), ci_high=boots[mname].quantile(0.975),
                         n_seeds=len(r_ids), n_boot=len(boots)))
comp = pd.DataFrame(rows); comp.to_csv(OUT / "test_comparisons_bootstrap.csv", index=False)

ens_rows = []
for fam in ["mae_matched", "dinov2_matched"]:
    for enc in ["retfound", "generic"]:
        ens_rows.append(dict(family=fam, encoder=enc, **metrics(np.mean([P[k] for k in family_ids(fam, enc)], 0), y)))
ens = pd.DataFrame(ens_rows); ens.to_csv(OUT / "test_metrics_seed_ensemble.csv", index=False)

# ------------------------------------------------------------
# 4. Report
# ------------------------------------------------------------
fmt = lambda x: f"{x:.4f}"
print("\n" + "=" * 70 + "\nEXPERIMENT 13A - FGADR TEST SET (n=276) - PER MODEL\n" + "=" * 70)
print(per_model[["model_id", "macro_f1", "qwk", "accuracy", "referable_auc", "dr1_recall", "brier", "ece"]]
      .to_string(index=False, float_format=fmt))
print("\nMEAN ± SD OVER SEEDS")
print(per_model[per_model.family.str.endswith("matched")].groupby(["family", "encoder"])[KEY]
      .agg(["mean", "std"]).round(4).to_string())
print("\n" + "=" * 70 + "\nPRE-SPECIFIED COMPARISONS: RETFound - generic (paired risk-group bootstrap)\n" + "=" * 70)
for label in COMPARISONS:
    print(f"\n{label}")
    print(comp[comp.comparison == label][["metric", "retfound", "generic", "diff_retfound_minus_generic", "ci_low", "ci_high"]]
          .to_string(index=False, float_format=fmt))
print("\nSEED ENSEMBLE (mean probabilities)")
print(ens[["family", "encoder"] + KEY].to_string(index=False, float_format=fmt))
print("\nPer-grade recall (seed mean):")
print(per_model[per_model.family.str.endswith("matched")].groupby(["family", "encoder"])[[f"recall_dr{g}" for g in range(5)]]
      .mean().round(3).to_string())
(OUT / "Exp13A_run_info.json").write_text(json.dumps(dict(
    analysis_plan_sha256=plan_sha, test_images=276, test_risk_groups=int(test.risk_group_id.nunique()),
    models=len(MODELS), finished_utc=datetime.now(timezone.utc).isoformat()), indent=2))
print("\nAnalysis plan SHA256:", plan_sha)
print("EXPERIMENT 13A DONE")
