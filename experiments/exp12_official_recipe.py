# Source: Exp11_MultiSeed.ipynb, cell 1 (Google Colab, L4 GPU).
# Only Colab/Google Drive specifics were changed; paths now come from config.yaml
# via common/paths.py. Run from the repository root:
#     python -m experiments.exp12_official_recipe
from common.paths import ROOT, PRETRAINED, load_index

# ============================================================
# Experiment 12 - Sensitivity analysis: RETFound official
# fine-tuning recipe, applied to BOTH encoders.
#
# Purpose: answer the reviewer question "Was RETFound
# disadvantaged by the matched low-LR protocol?"
#
# Recipe (RETFound main_finetune.py defaults for CFP grading):
#   batch 16, blr 5e-3 -> lr = 5e-3 * 16 / 256 = 3.125e-4
#   layer-wise LR decay 0.65, weight decay 0.05, drop path 0.2
#   50 epochs, 10 warm-up epochs, cosine to 1e-6
#   label smoothing 0.1, RandAugment (rand-m9-mstd0.5-inc1),
#   random erasing 0.25, no class weights, no gradient clipping
# Deviations (kept identical to our main protocol on purpose):
#   eval transform = Resize(224,224); best epoch = val macro-F1;
#   early stopping (min 15 epochs, patience 10).
#
# Train + Validation only. Test split is never read.
# If Colab disconnects, run the cell again: it resumes.
# ============================================================

import os, gc, json, time, math, random, argparse, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from timm.data import create_transform
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm.auto import tqdm


OUT_ROOT = ROOT / "experiment12_official_recipe"
WEIGHTS = {
    "retfound": PRETRAINED / "RETFound_pretrained/RETFound_mae_natureCFP.pth",
    "meta_mae": PRETRAINED / "Meta_MAE_pretrained/mae_pretrain_vit_large.pth",
}
RUNS = [("retfound", 42), ("meta_mae", 42)]

CFG = dict(
    architecture="vit_large_patch16_224", num_classes=5,
    micro_batch_size=4, accumulation_steps=4,      # effective batch 16
    blr=5e-3, layer_decay=0.65, weight_decay=0.05, drop_path=0.2,
    min_lr=1e-6, warmup_epochs=10, maximum_epochs=50,
    minimum_epochs=15, patience=10, label_smoothing=0.1,
    auto_augment="rand-m9-mstd0.5-inc1", re_prob=0.25,
)
CFG["lr"] = CFG["blr"] * CFG["micro_batch_size"] * CFG["accumulation_steps"] / 256

DEVICE = torch.device("cuda")
assert torch.cuda.is_available(), "Turn on the GPU runtime (L4)."
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------
data = load_index()
train_frame = data[data["split"] == "train"].copy()
val_frame = data[data["split"] == "val"].copy()
assert len(train_frame) == 1290 and len(val_frame) == 276

def ensure_local(row):
    src, dst = Path(str(row.source_path)), Path(str(row.local_path))
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copyfile(src, tmp); os.replace(tmp, dst)

active = pd.concat([train_frame, val_frame])
with ThreadPoolExecutor(max_workers=8) as ex:
    list(tqdm(ex.map(ensure_local, active.itertuples(index=False)),
              total=len(active), desc="Local image cache"))

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = create_transform(
    input_size=224, is_training=True, color_jitter=None,
    auto_augment=CFG["auto_augment"], interpolation="bicubic",
    re_prob=CFG["re_prob"], re_mode="pixel", re_count=1, mean=MEAN, std=STD)
eval_tf = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class FGADRDataset(Dataset):
    def __init__(self, frame, tf):
        self.frame, self.tf = frame.reset_index(drop=True), tf
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
# 2. Model + layer-wise LR decay parameter groups
# ------------------------------------------------------------
def build_model(model_name, seed):
    torch.manual_seed(seed)
    model = timm.create_model(CFG["architecture"], pretrained=False, num_classes=5,
                              global_pool="avg", drop_path_rate=CFG["drop_path"])
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(WEIGHTS[model_name], map_location="cpu", mmap=True, weights_only=True)
    enc = {k: v for k, v in ckpt["model"].items()
           if k.startswith(("patch_embed.", "blocks.")) or k in {"cls_token", "pos_embed"}}
    assert len(enc) == 292
    res = model.load_state_dict(enc, strict=False)
    assert set(res.missing_keys) == {"fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"}
    assert not res.unexpected_keys
    # MAE fine-tuning convention: small init for the new head
    nn.init.trunc_normal_(model.head.weight, std=2e-5)
    del ckpt, enc; gc.collect()
    model.set_grad_checkpointing(enable=True)
    return model.to(DEVICE)

def param_groups_lrd(model):
    num_layers = len(model.blocks) + 1          # 25
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim == 1 or name in {"pos_embed", "cls_token"}
        if name in {"cls_token", "pos_embed"} or name.startswith("patch_embed"):
            layer_id = 0
        elif name.startswith("blocks."):
            layer_id = int(name.split(".")[1]) + 1
        else:
            layer_id = num_layers
        key = (layer_id, no_decay)
        if key not in groups:
            groups[key] = dict(params=[], weight_decay=0.0 if no_decay else CFG["weight_decay"],
                               lr_scale=CFG["layer_decay"] ** (num_layers - layer_id))
        groups[key]["params"].append(p)
    return list(groups.values())

def lr_at(epoch_float):
    if epoch_float < CFG["warmup_epochs"]:
        return CFG["lr"] * epoch_float / CFG["warmup_epochs"]
    t = (epoch_float - CFG["warmup_epochs"]) / (CFG["maximum_epochs"] - CFG["warmup_epochs"])
    return CFG["min_lr"] + (CFG["lr"] - CFG["min_lr"]) * 0.5 * (1 + math.cos(math.pi * t))

def evaluate(model, loader):
    model.eval()
    y, p, ids, probs = [], [], [], []
    with torch.inference_mode():
        for images, labels, image_ids in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images.to(DEVICE))
            pr = torch.softmax(logits.float(), 1).cpu()
            y += labels.tolist(); p += pr.argmax(1).tolist(); ids += list(image_ids); probs += pr.tolist()
    y_a, p_a = np.array(y), np.array(p)
    tr, pr_ = y_a >= 2, p_a >= 2
    metrics = dict(
        val_accuracy=accuracy_score(y, p),
        val_macro_f1=f1_score(y, p, labels=list(range(5)), average="macro", zero_division=0),
        val_qwk=float(cohen_kappa_score(y, p, labels=list(range(5)), weights="quadratic")),
        referable_sensitivity=float((tr & pr_).sum() / tr.sum()),
        referable_specificity=float((~tr & ~pr_).sum() / (~tr).sum()))
    pred_df = pd.DataFrame({"image_id": ids, "true_grade": y, "predicted_grade": p,
                            **{f"prob_dr{g}": [r[g] for r in probs] for g in range(5)}})
    return metrics, pred_df

def atomic_save(obj, path):
    tmp = Path(str(path) + ".partial"); torch.save(obj, tmp); os.replace(tmp, path)

# ------------------------------------------------------------
# 3. One run
# ------------------------------------------------------------
def train_run(model_name, seed):
    run_dir = OUT_ROOT / f"{model_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        print(f"\n[{model_name} seed {seed}] already finished - skipped.")
        return json.loads(summary_path.read_text())
    state_path, best_path = run_dir / "last_state.pt", run_dir / "best_model.pt"
    print(f"\n{'=' * 60}\nOFFICIAL RECIPE: {model_name} | seed {seed} | lr {CFG['lr']:.3e}\n{'=' * 60}")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2,
                              pin_memory=True, generator=gen, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(model_name, seed)
    optimizer = torch.optim.AdamW(param_groups_lrd(model), lr=CFG["lr"], betas=(0.9, 0.999), foreach=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

    start_epoch, best_f1, best_epoch, no_improve, history = 1, -1.0, 0, 0, []
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["torch_rng"]); torch.cuda.set_rng_state_all(st["cuda_rng"])
        random.setstate(st["py_rng"]); np.random.set_state(st["np_rng"]); gen.set_state(st["gen_rng"])
        start_epoch, best_f1, best_epoch = st["epoch"] + 1, st["best_f1"], st["best_epoch"]
        no_improve, history = st["no_improve"], st["history"]
        del st; gc.collect()
        print(f"Resumed after epoch {start_epoch - 1} (best F1 {best_f1:.4f} @ epoch {best_epoch})")

    steps_per_epoch = math.ceil(len(train_loader) / CFG["accumulation_steps"])
    for epoch in range(start_epoch, CFG["maximum_epochs"] + 1):
        t0 = time.perf_counter()
        model.train(); optimizer.zero_grad(set_to_none=True)
        tl, n, step, ty, tp = 0.0, 0, 0, [], []
        for b, (images, labels, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            loss = criterion(logits.float(), labels)
            assert torch.isfinite(loss)
            (loss / CFG["accumulation_steps"]).backward()
            tl += float(loss.detach()) * len(labels); n += len(labels)
            ty += labels.cpu().tolist(); tp += logits.detach().argmax(1).cpu().tolist()
            if (b + 1) % CFG["accumulation_steps"] == 0 or (b + 1) == len(train_loader):
                lr = lr_at(epoch - 1 + step / steps_per_epoch)
                for g in optimizer.param_groups:
                    g["lr"] = lr * g["lr_scale"]
                optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1

        metrics, pred_df = evaluate(model, val_loader)
        row = dict(epoch=epoch, train_loss=tl / n,
                   train_macro_f1=f1_score(ty, tp, labels=list(range(5)), average="macro", zero_division=0),
                   base_lr_end=lr, minutes=(time.perf_counter() - t0) / 60, **metrics)
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
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                         py_rng=random.getstate(), np_rng=np.random.get_state(), gen_rng=gen.get_state(),
                         epoch=epoch, best_f1=best_f1, best_epoch=best_epoch,
                         no_improve=no_improve, history=history), state_path)
        print(f"Epoch {epoch:02d} | lr {lr:.2e} | train F1 {row['train_macro_f1']:.3f} | val F1 {metrics['val_macro_f1']:.4f} "
              f"| QWK {metrics['val_qwk']:.4f} | acc {metrics['val_accuracy']:.4f} "
              f"| {'BEST' if improved else f'no improve {no_improve}'} | {row['minutes']:.1f} min")
        if epoch >= CFG["minimum_epochs"] and no_improve >= CFG["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    best_row = next(r for r in history if r["epoch"] == best_epoch)
    summary = dict(model=model_name, seed=seed, recipe="retfound_official", best_epoch=best_epoch,
                   epochs_run=len(history), test_images_used=0,
                   **{k: best_row[k] for k in ["val_accuracy", "val_macro_f1", "val_qwk",
                                               "referable_sensitivity", "referable_specificity"]})
    (run_dir / "run_config.json").write_text(json.dumps(dict(CFG, model=model_name, seed=seed), indent=2))
    summary_path.write_text(json.dumps(summary, indent=2))
    state_path.unlink(missing_ok=True)
    del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    return summary

# ------------------------------------------------------------
# 4. Run + compare with the matched protocol (seed 42)
# ------------------------------------------------------------
results = [train_run(m, s) for m, s in RUNS]
table = pd.DataFrame(results)
try:
    ms = pd.read_csv(ROOT / "experiment11_multiseed/multiseed_summary.csv")
    ms = ms[ms["seed"] == 42].assign(recipe="matched_low_lr")
    table = pd.concat([ms, table], ignore_index=True)
except Exception as e:
    print("(Could not read Exp 11 summary:", e, ")")
table.to_csv(OUT_ROOT / "official_recipe_summary.csv", index=False)

print("\n" + "=" * 60 + "\nRECIPE SENSITIVITY (validation, seed 42, best epoch)\n" + "=" * 60)
print(table[["recipe", "model", "best_epoch", "val_accuracy", "val_macro_f1", "val_qwk"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nTest images used: 0")
print("EXPERIMENT 12 DONE")
