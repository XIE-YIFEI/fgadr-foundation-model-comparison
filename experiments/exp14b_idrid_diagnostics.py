# Source: Exp13_Test_v2.ipynb, cell 9 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp14b_idrid_diagnostics
from common.paths import ROOT, CACHE, IDRID_ROOT, load_index

# ============================================================
# Experiment 14B - diagnostics for the IDRiD classification drop
# No GPU needed (CPU runtime is fine). ~3-5 minutes.
# Uses the saved IDRiD predictions; does not re-run any model.
#   1. True vs predicted grade distribution per model
#   2. Confusion matrices (best and worst matched MAE model)
#   3. Prior-insensitive metrics: one-vs-rest AUC per grade
#   4. Side-by-side picture: FGADR vs harmonised IDRiD inputs
# ============================================================
import zipfile
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

OUT = ROOT / "experiment14_external_idrid"; PRED = OUT / "predictions"
index = load_index()

# 1. grade distributions
fg_prior = index[index.split == "train"].dr_grade.value_counts(normalize=True).sort_index()
rows = []
for f in sorted(PRED.glob("*.csv")):
    d = pd.read_csv(f); prob = d[[f"prob_dr{g}" for g in range(5)]].to_numpy(); y = d.true_grade.to_numpy()
    pred = prob.argmax(1)
    r = dict(model=f.stem, **{f"pred_dr{g}": (pred == g).mean() for g in range(5)})
    for g in range(5):
        r[f"auc_dr{g}"] = roc_auc_score(y == g, prob[:, g])
    r["mean_ovr_auc"] = np.mean([r[f"auc_dr{g}"] for g in range(5)])
    rows.append(r)
diag = pd.DataFrame(rows); diag.to_csv(OUT / "idrid_diagnostics.csv", index=False)
print("Grade share - FGADR train:", fg_prior.round(3).to_dict())
print("Grade share - IDRiD true: ", (pd.Series(y).value_counts(normalize=True).sort_index()).round(3).to_dict())
print("\nPredicted grade share on IDRiD:")
print(diag[["model"] + [f"pred_dr{g}" for g in range(5)]].to_string(index=False, float_format="%.3f"))
print("\nOne-vs-rest AUC per grade on IDRiD (not affected by label-prior shift):")
print(diag[["model"] + [f"auc_dr{g}" for g in range(5)] + ["mean_ovr_auc"]].to_string(index=False, float_format="%.3f"))

# 2. confusion matrices
for m in ["mae_meta_s1042", "mae_retfound_s1042", "dino_retfound_s2042", "dino_generic_s2042"]:
    d = pd.read_csv(PRED / f"{m}.csv")
    cm = confusion_matrix(d.true_grade, d[[f"prob_dr{g}" for g in range(5)]].to_numpy().argmax(1), labels=range(5))
    print(f"\nConfusion matrix {m} (rows = true grade 0-4, columns = predicted 0-4)\n{cm}")

# 4. picture: 4 FGADR training images vs 4 harmonised IDRiD images
LOCAL = CACHE / "idrid"
if not (LOCAL / "B. Disease Grading").exists():
    with zipfile.ZipFile(IDRID_ROOT / "B. Disease Grading.zip") as zf:
        zf.extractall(LOCAL, [n for n in zf.namelist() if "a. Training Set/IDRiD_00" in n])
def fundus_square(img):
    g = img.mean(2) > 15; ys, xs = np.where(g.any(1))[0], np.where(g.any(0))[0]
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; s = max(img.shape[:2])
    t, l = (s - img.shape[0]) // 2, (s - img.shape[1]) // 2
    c = np.zeros((s, s, 3), np.uint8); c[t:t + img.shape[0], l:l + img.shape[1]] = img
    return cv2.resize(c, (1280, 1280), interpolation=cv2.INTER_AREA)
fg = index[index.split == "train"].groupby("dr_grade").head(1).sort_values("dr_grade").head(4)
idr = sorted((LOCAL / "B. Disease Grading/1. Original Images/a. Training Set").glob("IDRiD_00*.jpg"))[:4]
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for k, r in enumerate(fg.itertuples()):
    ax[0, k].imshow(Image.open(r.source_path).convert("RGB").resize((224, 224), Image.BICUBIC)); ax[0, k].set_title(f"FGADR grade {r.dr_grade}")
for k, p in enumerate(idr):
    ax[1, k].imshow(Image.fromarray(fundus_square(np.asarray(Image.open(p).convert("RGB")))).resize((224, 224), Image.BICUBIC)); ax[1, k].set_title(f"IDRiD {p.stem}")
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "idrid_vs_fgadr_inputs.png", dpi=80); plt.show()
print("EXP 14B DONE")
