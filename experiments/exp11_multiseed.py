# Source: Exp11_MultiSeed.ipynb, cell 0 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp11_multiseed
from common.paths import ROOT, PRETRAINED, load_index

# ============================================================
# Experiment 11 - Multi-seed retraining (RETFound vs Meta MAE)
#
# Purpose: check whether "Meta MAE > RETFound" holds across
# training seeds. Protocol is IDENTICAL to Exp 04 / Exp 07
# (seed 42). Only the seed changes.
#
# - Train + Validation only. Test split is never read.
# - Runs one after another; finished runs are skipped.
# - If Colab disconnects, just run this cell again: it resumes
#   from the last finished epoch.
# ============================================================

import os, gc, json, time, random, argparse, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm.auto import tqdm


# ------------------------------------------------------------
# 0. Settings (only edit RUNS if you want fewer runs)
# ------------------------------------------------------------

OUT_ROOT = ROOT / "experiment11_multiseed"

WEIGHTS = {
    "retfound": PRETRAINED / "RETFound_pretrained/RETFound_mae_natureCFP.pth",
    "meta_mae": PRETRAINED / "Meta_MAE_pretrained/mae_pretrain_vit_large.pth",
}

# Seed 42 already exists (Exp 04 / Exp 07). These are the new seeds.
RUNS = [
    ("retfound", 1042),
    ("meta_mae", 1042),
    ("retfound", 2042),
    ("meta_mae", 2042),
]

# Protocol copied from Exp 07.16 locked config (matched to RETFound).
CFG = dict(
    architecture="vit_large_patch16_224",
    num_classes=5,
    micro_batch_size=4,
    accumulation_steps=4,          # effective batch 16
    learning_rate=1e-5,
    weight_decay=0.05,
    gradient_clip_norm=1.0,
    scheduler_eta_min=1e-7,
    maximum_epochs=25,
    minimum_epochs=8,
    patience=6,
    class_weights=[1.6152555557576007, 1.1187667747158858,
                   0.6665034324838375, 0.6394716814936021,
                   0.9600025555490744],
)

DEVICE = torch.device("cuda")
assert torch.cuda.is_available(), "Turn on the GPU runtime (L4)."
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Data (same transforms as Exp 07.20F)
# ------------------------------------------------------------

data = load_index()
train_frame = data[data["split"] == "train"].copy()
val_frame = data[data["split"] == "val"].copy()
assert len(train_frame) == 1290 and len(val_frame) == 276
assert set(train_frame["risk_group_id"]).isdisjoint(val_frame["risk_group_id"])

def ensure_local(row):
    src, dst = Path(str(row.source_path)), Path(str(row.local_path))
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)

active = pd.concat([train_frame, val_frame])
with ThreadPoolExecutor(max_workers=8) as ex:
    list(tqdm(ex.map(ensure_local, active.itertuples(index=False)),
              total=len(active), desc="Local image cache"))

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BICUBIC = transforms.InterpolationMode.BICUBIC
train_tf = transforms.Compose([
    transforms.Resize((224, 224), interpolation=BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224), interpolation=BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class FGADRDataset(Dataset):
    def __init__(self, frame, tf):
        self.frame = frame.reset_index(drop=True)
        self.tf = tf
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, i):
        row = self.frame.iloc[i]
        with Image.open(str(row["local_path"])) as im:
            img = im.convert("RGB")
        return self.tf(img), torch.tensor(int(row["dr_grade"])), str(row["image_id"])

train_dataset = FGADRDataset(train_frame, train_tf)
val_dataset = FGADRDataset(val_frame, eval_tf)

# ------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------

def build_model(model_name, seed):
    torch.manual_seed(seed)
    model = timm.create_model(CFG["architecture"], pretrained=False,
                              num_classes=5, global_pool="avg")
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(WEIGHTS[model_name], map_location="cpu",
                          mmap=True, weights_only=True)
    src = ckpt["model"]
    enc = {k: v for k, v in src.items()
           if k.startswith(("patch_embed.", "blocks.")) or k in {"cls_token", "pos_embed"}}
    assert len(enc) == 292
    res = model.load_state_dict(enc, strict=False)
    assert set(res.missing_keys) == {"fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"}
    assert not res.unexpected_keys
    del ckpt, src, enc
    gc.collect()
    model.set_grad_checkpointing(enable=True)
    return model.to(DEVICE)

def evaluate(model, loader, criterion, class_weights):
    model.eval()
    y, p, ids, probs = [], [], [], []
    loss_num = loss_den = 0.0
    with torch.inference_mode():
        for images, labels, image_ids in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            loss_num += float(criterion(logits.float(), labels).sum())
            loss_den += float(class_weights[labels].sum())
            pr = torch.softmax(logits.float(), 1)
            y += labels.cpu().tolist(); p += pr.argmax(1).cpu().tolist()
            ids += list(image_ids); probs += pr.cpu().tolist()
    y_a, p_a = np.array(y), np.array(p)
    tr, pr_ = y_a >= 2, p_a >= 2
    metrics = dict(
        val_loss=loss_num / loss_den,
        val_accuracy=accuracy_score(y, p),
        val_macro_f1=f1_score(y, p, labels=list(range(5)), average="macro", zero_division=0),
        val_qwk=float(cohen_kappa_score(y, p, labels=list(range(5)), weights="quadratic")),
        referable_sensitivity=float((tr & pr_).sum() / tr.sum()),
        referable_specificity=float((~tr & ~pr_).sum() / (~tr).sum()),
    )
    pred_df = pd.DataFrame({"image_id": ids, "true_grade": y, "predicted_grade": p,
                            **{f"prob_dr{g}": [r[g] for r in probs] for g in range(5)}})
    return metrics, pred_df

def atomic_save(obj, path):
    tmp = Path(str(path) + ".partial")
    torch.save(obj, tmp)
    os.replace(tmp, path)

# ------------------------------------------------------------
# 3. One training run
# ------------------------------------------------------------

def train_run(model_name, seed):
    run_dir = OUT_ROOT / f"{model_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        print(f"\n[{model_name} seed {seed}] already finished - skipped.")
        return json.loads(summary_path.read_text())

    state_path = run_dir / "last_state.pt"   # resume file, deleted at the end
    best_path = run_dir / "best_model.pt"
    print(f"\n{'=' * 60}\nTRAINING {model_name} | seed {seed}\n{'=' * 60}")

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    gen = torch.Generator(); gen.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0,
                              pin_memory=True, generator=gen)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0,
                            pin_memory=True)

    model = build_model(model_name, seed)
    class_weights = torch.tensor(CFG["class_weights"], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"],
                                  weight_decay=CFG["weight_decay"], foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["maximum_epochs"], eta_min=CFG["scheduler_eta_min"])

    start_epoch, best_f1, best_epoch, no_improve, history = 1, -1.0, 0, 0, []
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        torch.set_rng_state(st["torch_rng"]); torch.cuda.set_rng_state_all(st["cuda_rng"])
        random.setstate(st["py_rng"]); np.random.set_state(st["np_rng"]); gen.set_state(st["gen_rng"])
        start_epoch, best_f1, best_epoch = st["epoch"] + 1, st["best_f1"], st["best_epoch"]
        no_improve, history = st["no_improve"], st["history"]
        del st; gc.collect()
        print(f"Resumed after epoch {start_epoch - 1} (best F1 {best_f1:.4f} @ epoch {best_epoch})")

    for epoch in range(start_epoch, CFG["maximum_epochs"] + 1):
        t0 = time.perf_counter()
        model.train(); optimizer.zero_grad(set_to_none=True)
        group_w = torch.zeros((), device=DEVICE)
        tl_num = tl_den = 0.0; ty, tp = [], []
        for b, (images, labels, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss_sum = criterion(logits.float(), labels).sum()
            assert torch.isfinite(loss_sum)
            loss_sum.backward()
            w = class_weights[labels].sum().detach()
            group_w += w; tl_num += float(loss_sum.detach()); tl_den += float(w)
            ty += labels.cpu().tolist(); tp += logits.detach().argmax(1).cpu().tolist()
            if (b + 1) % CFG["accumulation_steps"] == 0 or (b + 1) == len(train_loader):
                for prm in model.parameters():
                    if prm.grad is not None:
                        prm.grad.div_(group_w)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["gradient_clip_norm"],
                                               error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); group_w.zero_()

        metrics, pred_df = evaluate(model, val_loader, criterion, class_weights)
        lr_used = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = dict(epoch=epoch, train_loss=tl_num / tl_den,
                   train_macro_f1=f1_score(ty, tp, labels=list(range(5)), average="macro", zero_division=0),
                   learning_rate=lr_used, minutes=(time.perf_counter() - t0) / 60, **metrics)
        history.append(row)

        improved = metrics["val_macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, no_improve = metrics["val_macro_f1"], epoch, 0
            atomic_save(model.state_dict(), best_path)
            pred_df.to_csv(run_dir / "best_validation_predictions.csv", index=False)
        else:
            no_improve += 1
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all(), py_rng=random.getstate(),
                         np_rng=np.random.get_state(), gen_rng=gen.get_state(), epoch=epoch,
                         best_f1=best_f1, best_epoch=best_epoch, no_improve=no_improve,
                         history=history), state_path)

        print(f"Epoch {epoch:02d} | train F1 {row['train_macro_f1']:.3f} | val F1 {metrics['val_macro_f1']:.4f} "
              f"| QWK {metrics['val_qwk']:.4f} | acc {metrics['val_accuracy']:.4f} "
              f"| {'BEST' if improved else f'no improve {no_improve}'} | {row['minutes']:.1f} min")

        if epoch >= CFG["minimum_epochs"] and no_improve >= CFG["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    best_row = next(r for r in history if r["epoch"] == best_epoch)
    summary = dict(model=model_name, seed=seed, best_epoch=best_epoch,
                   epochs_run=len(history), test_images_used=0,
                   **{k: best_row[k] for k in ["val_accuracy", "val_macro_f1", "val_qwk",
                                               "referable_sensitivity", "referable_specificity"]})
    (run_dir / "run_config.json").write_text(json.dumps(dict(CFG, model=model_name, seed=seed), indent=2))
    summary_path.write_text(json.dumps(summary, indent=2))
    state_path.unlink(missing_ok=True)   # free ~4 GB of Drive space

    del model, optimizer, scheduler
    gc.collect(); torch.cuda.empty_cache()
    return summary

# ------------------------------------------------------------
# 4. Run everything, then print the comparison table
# ------------------------------------------------------------

results = [train_run(m, s) for m, s in RUNS]

# Add the original seed-42 runs (best epoch by validation macro-F1)
for model_name, path in [
    ("retfound", ROOT / "experiment04_results/retfound_vit_large_v1/training_history.csv"),
    ("meta_mae", ROOT / "experiment07_results/meta_mae_vit_large_v1/training_history.csv"),
]:
    try:
        h = pd.read_csv(path)
        b = h.loc[h["val_macro_f1"].idxmax()]
        results.append(dict(model=model_name, seed=42, best_epoch=int(b["epoch"]),
                            val_accuracy=b.get("val_accuracy", np.nan),
                            val_macro_f1=b["val_macro_f1"], val_qwk=b.get("val_qwk", np.nan)))
    except Exception as e:
        print(f"(Could not read seed-42 history for {model_name}: {e})")

table = pd.DataFrame(results).sort_values(["model", "seed"])
table.to_csv(OUT_ROOT / "multiseed_summary.csv", index=False)

print("\n" + "=" * 60 + "\nMULTI-SEED SUMMARY (validation, best epoch)\n" + "=" * 60)
print(table[["model", "seed", "best_epoch", "val_accuracy", "val_macro_f1", "val_qwk"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nMean ± SD across seeds:")
print(table.groupby("model")[["val_accuracy", "val_macro_f1", "val_qwk"]]
      .agg(["mean", "std"]).round(4).to_string())
print("\nTest images used: 0")
print("EXPERIMENT 11 DONE")
