# Source: Exp13_Test_v2.ipynb, cell 10 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp14c_colour_normalisation
from common.paths import ROOT, CACHE, IDRID_ROOT, load_index

# ============================================================
# Experiment 14C - IDRiD colour-normalisation sensitivity
# Question: is the IDRiD grading collapse driven by colour/
# acquisition appearance? Test-time Reinhard colour transfer
# (LAB mean/std inside the fundus) maps every IDRiD image onto
# the average FGADR-train colour statistics. No IDRiD labels
# are used to build the transform. No model is retrained.
# L4 GPU, ~15-20 minutes. Only small CSVs are written to Drive.
# ============================================================
import gc, json, zipfile, hashlib
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pandas as pd, torch, timm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, roc_auc_score
from tqdm.auto import tqdm

OUT = ROOT / "experiment14_external_idrid/colour_sensitivity"; OUT.mkdir(parents=True, exist_ok=True)
LOCAL = CACHE / "idrid"; LOCAL.mkdir(exist_ok=True)
DEVICE = torch.device("cuda"); assert torch.cuda.is_available(), "Switch the runtime to L4 GPU."
SEED = 20261027
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
V16, D14 = "vit_large_patch16_224", "vit_large_patch14_dinov2"
E11, E12, E16 = ROOT / "experiment11_multiseed", ROOT / "experiment12_official_recipe", ROOT / "experiment16_dinov2_pair"
FINETUNED = {
    "mae_retfound_s42":   ("mae_matched", "retfound", V16, ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"),
    "mae_meta_s42":       ("mae_matched", "generic",  V16, ROOT / "experiment07_results/meta_mae_vit_large_v1/best_model.pt"),
    "mae_retfound_s1042": ("mae_matched", "retfound", V16, E11 / "retfound_seed1042/best_model.pt"),
    "mae_meta_s1042":     ("mae_matched", "generic",  V16, E11 / "meta_mae_seed1042/best_model.pt"),
    "mae_retfound_s2042": ("mae_matched", "retfound", V16, E11 / "retfound_seed2042/best_model.pt"),
    "mae_meta_s2042":     ("mae_matched", "generic",  V16, E11 / "meta_mae_seed2042/best_model.pt"),
    "dino_retfound_s42":   ("dinov2_matched", "retfound", D14, E16 / "retfound_dinov2_seed42/best_model.pt"),
    "dino_generic_s42":    ("dinov2_matched", "generic",  D14, E16 / "dinov2_generic_seed42/best_model.pt"),
    "dino_retfound_s1042": ("dinov2_matched", "retfound", D14, E16 / "retfound_dinov2_seed1042/best_model.pt"),
    "dino_generic_s1042":  ("dinov2_matched", "generic",  D14, E16 / "dinov2_generic_seed1042/best_model.pt"),
    "dino_retfound_s2042": ("dinov2_matched", "retfound", D14, E16 / "retfound_dinov2_seed2042/best_model.pt"),
    "dino_generic_s2042":  ("dinov2_matched", "generic",  D14, E16 / "dinov2_generic_seed2042/best_model.pt"),
    "official_retfound_s42": ("mae_official", "retfound", V16, E12 / "retfound_seed42/best_model.pt"),
    "official_meta_s42":     ("mae_official", "generic",  V16, E12 / "meta_mae_seed42/best_model.pt"),
    "efficientnet_b0_s42":   ("baseline", "efficientnet", "efficientnet_b0",
                              ROOT / "experiment03_results/efficientnet_b0_baseline_v1/best_model.pt"),
}
(OUT / "Exp14C_Plan.json").write_text(json.dumps(dict(
    created_utc=datetime.now(timezone.utc).isoformat(),
    purpose="post-hoc sensitivity analysis, added after Exp14 showed grade over-calling on IDRiD",
    transform="Reinhard LAB colour transfer inside fundus mask to mean FGADR-train statistics; no IDRiD labels used",
    reported="same metrics as Exp14 Part 1 + predicted-grade share + mean one-vs-rest AUC"), indent=2))

# --- FGADR target colour statistics (training split only) ---
index = load_index()
tr = index[index.split == "train"].sample(300, random_state=0)
def lab_stats(img):
    m = img.mean(2) > 15; lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)[m]
    return lab.mean(0), lab.std(0)
def fg_stats(p):
    with Image.open(p) as im: return lab_stats(np.asarray(im.convert("RGB").resize((512, 512), Image.BICUBIC)))
with ThreadPoolExecutor(8) as ex:
    st = list(tqdm(ex.map(fg_stats, tr.source_path), total=300, desc="FGADR colour stats"))
T_MU, T_SD = np.mean([s[0] for s in st], 0), np.mean([s[1] for s in st], 0)
print("FGADR target LAB mean/std:", T_MU.round(1), T_SD.round(1))

# --- IDRiD grading images ---
z = IDRID_ROOT / "B. Disease Grading.zip"
if not (LOCAL / "B. Disease Grading/2. Groundtruths").exists():
    with zipfile.ZipFile(z) as zf: zf.extractall(LOCAL)
GRD = LOCAL / "B. Disease Grading"
rows = []
for split, csv_name, folder in [("train", "a. IDRiD_Disease Grading_Training Labels.csv", "a. Training Set"),
                                ("test", "b. IDRiD_Disease Grading_Testing Labels.csv", "b. Testing Set")]:
    lab = pd.read_csv(GRD / "2. Groundtruths" / csv_name).iloc[:, :2]; lab.columns = ["name", "grade"]
    rows += [dict(image_id=f"{split}_{r.name}", path=str(GRD / "1. Original Images" / folder / f"{r.name}.jpg"), grade=int(r.grade))
             for r in lab.itertuples(index=False)]
grd = pd.DataFrame(rows); assert len(grd) == 516; y = grd.grade.to_numpy()

def fundus_square(img):
    g = img.mean(2) > 15; ys, xs = np.where(g.any(1))[0], np.where(g.any(0))[0]
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; s = max(img.shape[:2])
    t, l = (s - img.shape[0]) // 2, (s - img.shape[1]) // 2
    c = np.zeros((s, s, 3), np.uint8); c[t:t + img.shape[0], l:l + img.shape[1]] = img
    return cv2.resize(c, (1280, 1280), interpolation=cv2.INTER_AREA)
def reinhard(img):
    m = img.mean(2) > 15; lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    mu, sd = lab[m].mean(0), lab[m].std(0) + 1e-6
    lab[m] = (lab[m] - mu) / sd * T_SD + T_MU
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB); out[~m] = 0
    return out
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
def load(p):
    sq = reinhard(fundus_square(np.asarray(Image.open(p).convert("RGB"))))
    return tf(Image.fromarray(sq).resize((224, 224), Image.BICUBIC)), sq
with ThreadPoolExecutor(8) as ex:
    res = list(tqdm(ex.map(load, grd.path), total=516, desc="IDRiD colour-normalised"))
X = torch.stack([r[0] for r in res])
Image.fromarray(np.hstack([cv2.resize(r[1], (256, 256)) for r in res[:4]])).save(OUT / "idrid_colour_normalised_examples.png")
del res; gc.collect()

# --- run the 15 models ---
def build(arch, ckpt):
    if arch == D14: m = timm.create_model(arch, pretrained=False, img_size=224, num_classes=5, global_pool="avg")
    elif arch == V16: m = timm.create_model(arch, pretrained=False, num_classes=5, global_pool="avg")
    else: m = timm.create_model(arch, pretrained=False, num_classes=5)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd, strict=True)
    return m.eval().to(DEVICE)
P = {}
for k, (fam, role, arch, ckpt) in tqdm(FINETUNED.items(), desc="models"):
    m = build(arch, ckpt); out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, 516, 32): out.append(torch.softmax(m(X[s:s + 32].to(DEVICE)).float(), 1).cpu())
    P[k] = torch.cat(out).numpy(); del m; gc.collect(); torch.cuda.empty_cache()
    pd.DataFrame({"image_id": grd.image_id, "true_grade": y, **{f"prob_dr{g}": P[k][:, g] for g in range(5)}}).to_csv(OUT / f"pred_{k}.csv", index=False)

def metrics(prob, yt):
    pred = prob.argmax(1)
    return dict(macro_f1=f1_score(yt, pred, labels=range(5), average="macro", zero_division=0),
                qwk=cohen_kappa_score(yt, pred, weights="quadratic"), accuracy=accuracy_score(yt, pred),
                referable_auc=roc_auc_score(yt >= 2, prob[:, 2:].sum(1)),
                mean_ovr_auc=np.mean([roc_auc_score(yt == g, prob[:, g]) for g in range(5)]))
KEY = ["macro_f1", "qwk", "accuracy", "referable_auc", "mean_ovr_auc"]
per = pd.DataFrame([dict(model_id=k, family=v[0], role=v[1], **metrics(P[k], y),
                         **{f"pred_dr{g}": (P[k].argmax(1) == g).mean() for g in range(5)}) for k, v in FINETUNED.items()])
orig = pd.read_csv(ROOT / "experiment14_external_idrid/idrid_classification_per_model.csv").set_index("model_id")
per["macro_f1_before"] = per.model_id.map(orig.macro_f1); per["qwk_before"] = per.model_id.map(orig.qwk)
per.to_csv(OUT / "idrid_colournorm_per_model.csv", index=False)

rng = np.random.default_rng(SEED); BOOT = [rng.integers(0, 516, 516) for _ in range(1000)]
ids = lambda fam, role: [k for k, v in FINETUNED.items() if v[0] == fam and v[1] == role]
smean = lambda keys, idx: pd.DataFrame([metrics(P[k][idx], y[idx]) for k in keys])[KEY].mean()
crow = []
for fam in ["mae_matched", "dinov2_matched", "mae_official"]:
    r, g = ids(fam, "retfound"), ids(fam, "generic"); full = np.arange(516)
    o_r, o_g = smean(r, full), smean(g, full)
    bt = pd.DataFrame([smean(r, b) - smean(g, b) for b in tqdm(BOOT, desc=fam, leave=False) if len(np.unique(y[b])) == 5])
    for mname in KEY:
        crow.append(dict(comparison=fam, metric=mname, retfound=o_r[mname], generic=o_g[mname], diff=o_r[mname] - o_g[mname],
                         ci_low=bt[mname].quantile(0.025), ci_high=bt[mname].quantile(0.975)))
comp = pd.DataFrame(crow); comp.to_csv(OUT / "idrid_colournorm_comparisons.csv", index=False)

f = lambda x: f"{x:.3f}"
print("\n" + "=" * 70 + "\nEXP 14C - IDRiD WITH COLOUR NORMALISATION (n=516)\n" + "=" * 70)
print(per[["model_id", "macro_f1_before", "macro_f1", "qwk_before", "qwk", "accuracy", "referable_auc", "mean_ovr_auc"]].to_string(index=False, float_format=f))
print("\nPredicted grade share:")
print(per[["model_id"] + [f"pred_dr{g}" for g in range(5)]].to_string(index=False, float_format=f))
print("\nRetinal - generic (seed mean, image bootstrap 1000x):")
print(comp.to_string(index=False, float_format=f))
print("EXP 14C DONE")
