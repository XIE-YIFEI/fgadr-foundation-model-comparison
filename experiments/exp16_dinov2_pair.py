# Source: Exp13_Test.ipynb, cell 1 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp16_dinov2_pair
from common.paths import ROOT, PRETRAINED, CACHE, load_index

# ============================================================
# Experiment 16 - Second controlled pair: DINOv2 objective
#
#   dinov2_generic  : DINOv2 ViT-L/14 (LVD-142M natural images)
#   retfound_dinov2 : RETFound-DINOv2 (MEH retinal images)
#
# Same question as the MAE pair, different self-supervised
# objective. Fine-tuning protocol IDENTICAL to Exp 11
# (matched low-LR protocol), 3 seeds each = 6 runs.
# Input 224 -> 16x16 patch grid (patch 14); position
# embeddings of both checkpoints are resampled the same way.
#
# Train + Validation only. Test split is never read.
# Finished runs are skipped; interrupted runs resume.
# Runtime on L4: roughly 3-4 hours for all 6 runs.
# v2: resume files are kept on local disk; only final files go to Drive.
# ============================================================

import os, gc, json, time, random, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from timm.layers import resample_abs_pos_embed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors


OUT_ROOT = ROOT / "experiment16_dinov2_pair"
CKPT_DIR = PRETRAINED / "DINOv2_pair"
OUT_ROOT.mkdir(parents=True, exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)

ARCH = "vit_large_patch14_dinov2"      # timm architecture (weights loaded manually below)
RUNS = [("dinov2_generic", 42), ("retfound_dinov2", 42),
        ("dinov2_generic", 1042), ("retfound_dinov2", 1042),
        ("dinov2_generic", 2042), ("retfound_dinov2", 2042)]

CFG = dict(num_classes=5, micro_batch_size=4, accumulation_steps=4, learning_rate=1e-5,
           weight_decay=0.05, gradient_clip_norm=1.0, scheduler_eta_min=1e-7,
           maximum_epochs=25, minimum_epochs=8, patience=6,
           class_weights=[1.6152555557576007, 1.1187667747158858, 0.6665034324838375,
                          0.6394716814936021, 0.9600025555490744])
DEVICE = torch.device("cuda")
assert torch.cuda.is_available()

# ------------------------------------------------------------
# 1. Download + cache the two pretrained checkpoints on Drive
# ------------------------------------------------------------
token = os.environ.get("HF_TOKEN")
def fetch(repo, filename):
    target = CKPT_DIR / filename
    if not target.exists():
        p = hf_hub_download(repo, filename, token=token)
        shutil.copyfile(p, str(target) + ".partial"); os.replace(str(target) + ".partial", target)
    return target

SOURCES = {
    "dinov2_generic": fetch("timm/vit_large_patch14_dinov2.lvd142m", "model.safetensors"),
    "retfound_dinov2": fetch("YukunZhou/RETFound_dinov2_meh", "RETFound_dinov2_meh.pth"),
}

def raw_state(name):
    path = SOURCES[name]
    if path.suffix == ".safetensors":
        return load_safetensors(str(path))
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ["teacher", "model", "state_dict", "student"]:
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            print(f"  [{name}] using checkpoint['{key}']"); obj = obj[key]; break
    return obj

def build_model(name, seed):
    torch.manual_seed(seed)
    model = timm.create_model(ARCH, pretrained=False, img_size=224, num_classes=5, global_pool="avg")
    target = model.state_dict()
    sd = raw_state(name)
    clean = {}
    for k, v in sd.items():
        for pre in ["module.", "backbone.", "encoder.", "model."]:
            if k.startswith(pre): k = k[len(pre):]
        k = k.replace("mlp.w12.", "mlp.fc1.").replace("mlp.w3.", "mlp.fc2.")
        clean[k] = v
    if "pos_embed" in clean and clean["pos_embed"].shape != target["pos_embed"].shape:
        clean["pos_embed"] = resample_abs_pos_embed(clean["pos_embed"], new_size=(16, 16), num_prefix_tokens=1)
    encoder_keys = [k for k in target if not k.startswith(("head.", "fc_norm.", "norm."))]
    loaded = {k: clean[k] for k in encoder_keys if k in clean and clean[k].shape == target[k].shape}
    missing = sorted(set(encoder_keys) - set(loaded))
    if missing:
        print(f"[{name}] could not match {len(missing)} encoder tensors, e.g. {missing[:5]}")
        print("Checkpoint key examples:", list(clean)[:15])
        raise RuntimeError("Checkpoint format not recognised - paste this output to Claude.")
    model.load_state_dict(loaded, strict=False)
    print(f"  [{name}] loaded {len(loaded)}/{len(encoder_keys)} encoder tensors")
    del sd, clean; gc.collect()
    model.set_grad_checkpointing(enable=True)
    return model.to(DEVICE)

# Fail fast: check both checkpoints load before any training
for n in SOURCES:
    m = build_model(n, 0); del m; gc.collect(); torch.cuda.empty_cache()

# ------------------------------------------------------------
# 2. Data (identical to Exp 11)
# ------------------------------------------------------------
data = load_index()
train_frame, val_frame = data[data.split == "train"].copy(), data[data.split == "val"].copy()
assert len(train_frame) == 1290 and len(val_frame) == 276

def ensure_local(row):
    src, dst = Path(str(row.source_path)), Path(str(row.local_path))
    if dst.is_file() and dst.stat().st_size == src.stat().st_size: return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial"); shutil.copyfile(src, tmp); os.replace(tmp, dst)
with ThreadPoolExecutor(8) as ex:
    list(tqdm(ex.map(ensure_local, pd.concat([train_frame, val_frame]).itertuples(index=False)),
              total=1566, desc="Local image cache"))

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BIC = transforms.InterpolationMode.BICUBIC
train_tf = transforms.Compose([transforms.Resize((224, 224), interpolation=BIC), transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([transforms.Resize((224, 224), interpolation=BIC), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])

class FGADRDataset(Dataset):
    def __init__(self, frame, tf): self.frame, self.tf = frame.reset_index(drop=True), tf
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        row = self.frame.iloc[i]
        with Image.open(str(row["local_path"])) as im: img = im.convert("RGB")
        return self.tf(img), torch.tensor(int(row["dr_grade"])), str(row["image_id"])

train_dataset, val_dataset = FGADRDataset(train_frame, train_tf), FGADRDataset(val_frame, eval_tf)

# ------------------------------------------------------------
# 3. Training loop (identical logic to Exp 11)
# ------------------------------------------------------------
def evaluate(model, loader, criterion, cw):
    model.eval(); y, p, ids, probs = [], [], [], []; ln = ld = 0.0
    with torch.inference_mode():
        for images, labels, image_ids in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(images)
            ln += float(criterion(logits.float(), labels).sum()); ld += float(cw[labels].sum())
            pr = torch.softmax(logits.float(), 1)
            y += labels.cpu().tolist(); p += pr.argmax(1).cpu().tolist(); ids += list(image_ids); probs += pr.cpu().tolist()
    ya, pa = np.array(y), np.array(p); t, q = ya >= 2, pa >= 2
    m = dict(val_loss=ln / ld, val_accuracy=accuracy_score(y, p),
             val_macro_f1=f1_score(y, p, labels=list(range(5)), average="macro", zero_division=0),
             val_qwk=float(cohen_kappa_score(y, p, labels=list(range(5)), weights="quadratic")),
             referable_sensitivity=float((t & q).sum() / t.sum()), referable_specificity=float((~t & ~q).sum() / (~t).sum()))
    return m, pd.DataFrame({"image_id": ids, "true_grade": y, "predicted_grade": p,
                            **{f"prob_dr{g}": [r[g] for r in probs] for g in range(5)}})

def atomic_save(obj, path):
    tmp = Path(str(path) + ".partial"); torch.save(obj, tmp); os.replace(tmp, path)

def train_run(name, seed):
    run_dir = OUT_ROOT / f"{name}_seed{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    summary_path, best_path = run_dir / "run_summary.json", run_dir / "best_model.pt"
    # v2: resume state stays on the LOCAL Colab disk (not Drive) to avoid a huge Drive upload backlog
    local_dir = CACHE / "exp16_resume" / f"{name}_seed{seed}"; local_dir.mkdir(parents=True, exist_ok=True)
    state_path = local_dir / "last_state.pt"
    (run_dir / "last_state.pt").unlink(missing_ok=True)
    if summary_path.exists():
        print(f"\n[{name} seed {seed}] already finished - skipped."); return json.loads(summary_path.read_text())
    print(f"\n{'=' * 60}\nTRAINING {name} | seed {seed}\n{'=' * 60}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    tl_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True,
                           generator=gen, persistent_workers=True)
    vl_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    model = build_model(name, seed)
    cw = torch.tensor(CFG["class_weights"], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"], foreach=False)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["maximum_epochs"], eta_min=CFG["scheduler_eta_min"])
    start, best_f1, best_ep, no_imp, hist = 1, -1.0, 0, 0, []
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); sch.load_state_dict(st["scheduler"])
        torch.set_rng_state(st["torch_rng"]); torch.cuda.set_rng_state_all(st["cuda_rng"])
        random.setstate(st["py_rng"]); np.random.set_state(st["np_rng"]); gen.set_state(st["gen_rng"])
        start, best_f1, best_ep, no_imp, hist = st["epoch"] + 1, st["best_f1"], st["best_epoch"], st["no_improve"], st["history"]
        del st; gc.collect(); print(f"Resumed after epoch {start - 1}")
    for epoch in range(start, CFG["maximum_epochs"] + 1):
        t0 = time.perf_counter(); model.train(); opt.zero_grad(set_to_none=True)
        gw = torch.zeros((), device=DEVICE); tn = td = 0.0; ty, tp = [], []
        for b, (images, labels, _) in enumerate(tqdm(tl_loader, desc=f"Epoch {epoch}", leave=False)):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images); ls = criterion(logits.float(), labels).sum()
            assert torch.isfinite(ls); ls.backward()
            w = cw[labels].sum().detach(); gw += w; tn += float(ls.detach()); td += float(w)
            ty += labels.cpu().tolist(); tp += logits.detach().argmax(1).cpu().tolist()
            if (b + 1) % CFG["accumulation_steps"] == 0 or (b + 1) == len(tl_loader):
                for prm in model.parameters():
                    if prm.grad is not None: prm.grad.div_(gw)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["gradient_clip_norm"], error_if_nonfinite=True)
                opt.step(); opt.zero_grad(set_to_none=True); gw.zero_()
        m, pred = evaluate(model, vl_loader, criterion, cw); sch.step()
        row = dict(epoch=epoch, train_loss=tn / td, minutes=(time.perf_counter() - t0) / 60,
                   train_macro_f1=f1_score(ty, tp, labels=list(range(5)), average="macro", zero_division=0), **m)
        hist.append(row); improved = m["val_macro_f1"] > best_f1
        if improved:
            best_f1, best_ep, no_imp = m["val_macro_f1"], epoch, 0
            atomic_save(model.state_dict(), local_dir / "best_model.pt"); pred.to_csv(run_dir / "best_validation_predictions.csv", index=False)
        else:
            no_imp += 1
        pd.DataFrame(hist).to_csv(run_dir / "training_history.csv", index=False)
        atomic_save(dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sch.state_dict(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), py_rng=random.getstate(),
                         np_rng=np.random.get_state(), gen_rng=gen.get_state(), epoch=epoch, best_f1=best_f1,
                         best_epoch=best_ep, no_improve=no_imp, history=hist), state_path)
        print(f"Epoch {epoch:02d} | train F1 {row['train_macro_f1']:.3f} | val F1 {m['val_macro_f1']:.4f} | QWK {m['val_qwk']:.4f} "
              f"| acc {m['val_accuracy']:.4f} | {'BEST' if improved else f'no improve {no_imp}'} | {row['minutes']:.1f} min")
        if epoch >= CFG["minimum_epochs"] and no_imp >= CFG["patience"]:
            print(f"Early stopping at epoch {epoch}"); break
    br = next(r for r in hist if r["epoch"] == best_ep)
    summary = dict(model=name, seed=seed, best_epoch=best_ep, epochs_run=len(hist), test_images_used=0,
                   **{k: br[k] for k in ["val_accuracy", "val_macro_f1", "val_qwk", "referable_sensitivity", "referable_specificity"]})
    (run_dir / "run_config.json").write_text(json.dumps(dict(CFG, model=name, seed=seed, arch=ARCH, img_size=224), indent=2))
    shutil.copyfile(local_dir / "best_model.pt", str(best_path) + ".partial"); os.replace(str(best_path) + ".partial", best_path)
    summary_path.write_text(json.dumps(summary, indent=2)); state_path.unlink(missing_ok=True)
    del model, opt, sch; gc.collect(); torch.cuda.empty_cache()
    return summary

results = [train_run(n, s) for n, s in RUNS]
table = pd.DataFrame(results).sort_values(["model", "seed"])
table.to_csv(OUT_ROOT / "dinov2_pair_summary.csv", index=False)
print("\n" + "=" * 60 + "\nDINOv2 PAIR SUMMARY (validation, best epoch)\n" + "=" * 60)
print(table[["model", "seed", "best_epoch", "val_accuracy", "val_macro_f1", "val_qwk"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nMean ± SD across seeds:")
print(table.groupby("model")[["val_accuracy", "val_macro_f1", "val_qwk"]].agg(["mean", "std"]).round(4).to_string())
try:
    mae = pd.read_csv(ROOT / "experiment11_multiseed/multiseed_summary.csv")
    print("\nFor reference - MAE pair (Exp 11), mean across seeds:")
    print(mae.groupby("model")[["val_accuracy", "val_macro_f1", "val_qwk"]].mean().round(4).to_string())
except Exception as e:
    print("(MAE pair summary not found:", e, ")")
print("\nTest images used: 0")
print("EXPERIMENT 16 DONE")
