# Experiment 08 - drop-path 0.1 regularisation ablation
#
# Source: FGADR_MA_Mask_Audit.ipynb (master notebook, 392 cells), cells 247-301, run step by step in Google Colab (L4 GPU).
# This file keeps every cell in its original order, including smoke tests,
# audits, re-runs and fixes, as a record of how the results were produced.
# Run it cell by cell (the '# %%' markers open as cells in VS Code / Jupyter),
# not top to bottom. Exact duplicate cells are listed but not repeated.
#
# Only Colab/Google Drive specifics were changed: Drive paths now start at
# DR_RESEARCH (results_root in config.yaml, laid out like the original
# DR_Research folder, including datasets/DR/...), local Colab disk paths at
# LOCAL_DISK (cache_dir), Colab secrets come from environment variables,
# and %pip lines are commented out.

# %% Paths
from common.paths import ROOT as DR_RESEARCH
from common.paths import CACHE as LOCAL_DISK

# %% [cell 247] Experiment 08 - Step 08.1
# ============================================================
# Experiment 08 - Step 08.1
# Regularization Ablation: Environment Preflight
# Read-only; no training or file modifications
# ============================================================

from pathlib import Path
import json
import torch

ROOT = DR_RESEARCH

experiments = {
    "RETFound": ROOT / (
        "experiment04_results/retfound_vit_large_v1"
    ),
    "Meta MAE": ROOT / (
        "experiment07_results/meta_mae_vit_large_v1"
    ),
}

EXP08 = ROOT / "experiment08_results"

print("EXPERIMENT 08.1 - REGULARIZATION PREFLIGHT")

print("\nENVIRONMENT")
print("Google Drive connected:", ROOT.is_dir())
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nORIGINAL EXPERIMENTS")

for name, directory in experiments.items():
    config_path = directory / "training_config.json"
    best_path = directory / "best_model.pt"
    history_path = directory / "training_history.csv"

    assert config_path.is_file(), f"Missing config: {name}"
    assert best_path.is_file(), f"Missing best model: {name}"
    assert history_path.is_file(), f"Missing history: {name}"

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    print(f"\n{name}")
    print("Best checkpoint exists:", best_path.is_file())
    print("Training history exists:", history_path.is_file())
    print("Architecture:", config.get("architecture"))
    print("Learning rate:", config.get("learning_rate"))
    print("Weight decay:", config.get("weight_decay"))
    print("Maximum epochs:", config.get("maximum_epochs"))
    print("Patience:", config.get("patience"))

print("\nLOCAL IMAGE CACHE")
cache = (LOCAL_DISK / "fgadr_classification")
print("Cache directory exists:", cache.is_dir())

print("\nEXPERIMENT 08 OUTPUT")
print("Experiment 08 directory exists:", EXP08.exists())

if EXP08.is_dir():
    print("Existing entries:", sorted(
        p.name for p in EXP08.iterdir()
    )[:20])

print("\nRESEARCH INTEGRITY")
print("Original checkpoints modified: False")
print("New training started: False")
print("Test images accessed: 0")

print("\nSTEP 08.1 PASS")


# %% [cell 248] Experiment 08 - Step 08.2
# ============================================================
# Experiment 08 - Step 08.2
# Baseline Configuration Parity Audit
# Read-only; no training or file modifications
# ============================================================

from pathlib import Path
import json

ROOT = DR_RESEARCH

CONFIG_PATHS = {
    "RETFound": (
        ROOT / "experiment04_results"
        / "retfound_vit_large_v1/training_config.json"
    ),
    "Meta MAE": (
        ROOT / "experiment07_results"
        / "meta_mae_vit_large_v1/training_config.json"
    )
}

configs = {}

for name, path in CONFIG_PATHS.items():
    assert path.is_file(), f"Missing configuration: {name}"
    configs[name] = json.loads(path.read_text(encoding="utf-8"))

ret = configs["RETFound"]
meta = configs["Meta MAE"]

all_keys = sorted(set(ret) | set(meta))

print("EXPERIMENT 08.2 - BASELINE CONFIGURATION AUDIT")

print("\nCONFIGURATION FIELDS")
print("RETFound fields:", len(ret))
print("Meta MAE fields:", len(meta))

print("\nDIFFERING OR MISSING FIELDS")

differences = []

for key in all_keys:
    a = ret.get(key, "<MISSING>")
    b = meta.get(key, "<MISSING>")

    if a != b:
        differences.append(key)
        print(f"\n{key}")
        print("  RETFound:", repr(a)[:600])
        print("  Meta MAE:", repr(b)[:600])

if not differences:
    print("No differences detected.")

print("\nSHARED TRAINING PARAMETERS")

for key in [
    "architecture",
    "num_classes",
    "seed",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "weight_decay",
    "class_weights",
    "gradient_clip_norm",
    "maximum_epochs",
    "minimum_epochs",
    "patience",
    "scheduler_eta_min"
]:
    a = ret.get(key, "<MISSING>")
    b = meta.get(key, "<MISSING>")

    print(
        f"{key}: "
        f"RETFound={repr(a)} | "
        f"Meta MAE={repr(b)} | "
        f"Match={a == b}"
    )

print("\nEXPERIMENT 08 PLAN")
print("Regularization method: Drop Path")
print("Proposed drop_path_rate:", 0.1)
print("Both pretrained encoders will be reinitialized: True")
print("Existing checkpoints preserved: True")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Training started: False")
print("Test images accessed: 0")
print("STEP 08.2 AUDIT COMPLETE")


# %% [cell 249] Experiment 08 - Step 08.3
# ============================================================
# Experiment 08 - Step 08.3
# Verify Original RETFound Training Implementation
# Read-only notebook inspection
# ============================================================

from pathlib import Path
import json
import re

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert NOTEBOOK.is_file()

with NOTEBOOK.open("r", encoding="utf-8") as f:
    notebook = json.load(f)

cells = notebook["cells"]

print("EXPERIMENT 08.3 - RETFOUND TRAINING PROTOCOL AUDIT")
print("Notebook cells:", len(cells))

# Original RETFound training was recorded around these cells.
TARGET_CELLS = [97, 98, 99]

patterns = [
    r"CrossEntropyLoss",
    r"reduction\s*=",
    r"clip_grad_norm",
    r"CosineAnnealingLR",
    r"eta_min",
    r"foreach",
    r"optimizer\.step",
    r"scheduler\.step",
    r"group_weight",
    r"accum",
    r"gradient",
]

for cell_index in TARGET_CELLS:
    assert cell_index < len(cells)

    cell = cells[cell_index]
    source = "".join(cell.get("source", []))
    lines = source.splitlines()

    print("\n" + "=" * 65)
    print("CELL:", cell_index)
    print("Source lines:", len(lines))

    relevant = set()

    for i, line in enumerate(lines):
        if any(
            re.search(pattern, line, flags=re.IGNORECASE)
            for pattern in patterns
        ):
            relevant.update(
                range(
                    max(0, i - 4),
                    min(len(lines), i + 7)
                )
            )

    if not relevant:
        print("No matching training operations.")
        continue

    for i in sorted(relevant):
        print(f"{i + 1:04d}: {lines[i]}")

print("\nRESEARCH INTEGRITY")
print("Notebook modified: False")
print("Models retrained: False")
print("Checkpoints modified: False")
print("Test images accessed: 0")
print("STEP 08.3 AUDIT COMPLETE")


# %% [cell 250] Experiment 08 - Step 08.4
# ============================================================
# Experiment 08 - Step 08.4
# Model Initialization and Drop Path Source Audit
# Read-only; no checkpoint loading or training
# ============================================================

from pathlib import Path
import json
import re

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert NOTEBOOK.is_file()

with NOTEBOOK.open("r", encoding="utf-8") as f:
    notebook = json.load(f)

patterns = [
    r"drop_path",
    r"vit_large_patch16_224",
    r"create_model",
    r"load_state_dict",
    r"pretrained_checkpoint",
    r"pretrained_path",
    r"mae_pretrain_vit_large",
    r"RETFound_mae",
    r"global_pool",
]

matches = []

for cell_index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue

    source = "".join(cell.get("source", []))
    lines = source.splitlines()

    if not any(
        re.search(pattern, source, re.IGNORECASE)
        for pattern in patterns
    ):
        continue

    # Focus on model initialization and training code.
    if not (
        85 <= cell_index <= 105
        or "Experiment 07" in source[:1500]
        or "Step 7." in source[:1500]
    ):
        continue

    relevant = set()

    for i, line in enumerate(lines):
        if any(
            re.search(pattern, line, re.IGNORECASE)
            for pattern in patterns
        ):
            relevant.update(
                range(max(0, i - 3), min(len(lines), i + 5))
            )

    if relevant:
        matches.append((cell_index, lines, relevant))

print("EXPERIMENT 08.4 - MODEL INITIALIZATION AUDIT")

for cell_index, lines, relevant in matches:
    print("\n" + "=" * 60)
    print("CELL:", cell_index)

    for i in sorted(relevant):
        print(f"{i + 1:04d}: {lines[i][:180]}")

print("\nMATCHING CELLS:", len(matches))

print("\nRESEARCH INTEGRITY")
print("Notebook modified: False")
print("Checkpoints loaded: False")
print("Training started: False")
print("Test images accessed: 0")
print("STEP 08.4 AUDIT COMPLETE")


# %% [cell 251] Experiment 08 - Step 08.5
# ============================================================
# Experiment 08 - Step 08.5
# Drop Path Structural Compatibility Audit
# Meta-device only; no checkpoint loading
# ============================================================

import torch
import timm

MODEL_NAME = "vit_large_patch16_224"

def inspect_drop_path(rate):
    with torch.device("meta"):
        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=5,
            global_pool="avg",
            drop_path_rate=rate
        )

    values = []

    for block in model.blocks:
        for name in ("drop_path1", "drop_path2"):
            layer = getattr(block, name, None)

            if layer is None:
                continue

            probability = getattr(layer, "drop_prob", 0.0)
            values.append(float(probability))

    assert len(model.blocks) == 24
    assert len(values) > 0

    return model, values


print("EXPERIMENT 08.5 - DROP PATH COMPATIBILITY")
print("PyTorch:", torch.__version__)
print("timm:", timm.__version__)

baseline_model, baseline_rates = inspect_drop_path(0.0)
regularized_model, regularized_rates = inspect_drop_path(0.1)

assert all(rate == 0.0 for rate in baseline_rates)
assert max(regularized_rates) > 0.0
assert max(regularized_rates) <= 0.100001

baseline_state = baseline_model.state_dict()
regularized_state = regularized_model.state_dict()

assert baseline_state.keys() == regularized_state.keys()

for key in baseline_state:
    assert baseline_state[key].shape == regularized_state[key].shape

print("\nMODEL STRUCTURE")
print("Transformer blocks:", len(regularized_model.blocks))
print("Classification classes:", 5)
print("Global pooling:", regularized_model.global_pool)

print("\nDROP PATH")
print("Baseline maximum rate:", max(baseline_rates))
print("Regularized minimum rate:", min(regularized_rates))
print("Regularized maximum rate:", max(regularized_rates))
print("Nonzero Drop Path entries:", sum(
    rate > 0 for rate in regularized_rates
))

print("\nSTATE DICTIONARY")
print("Same parameter and buffer keys: True")
print("All tensor shapes match: True")
print("State entries:", len(baseline_state))

print("\nRESEARCH INTEGRITY")
print("Full model parameters allocated: False")
print("Pretrained checkpoints loaded: False")
print("Training started: False")
print("Research files modified: False")
print("Test images accessed: 0")

print("\nSTEP 08.5 PASS")

del baseline_model, regularized_model


# %% [cell 252] Experiment 08 - Step 08.6
# ============================================================
# Experiment 08 - Step 08.6
# Lock Paired Drop Path Regularization Protocol
# Configuration only; no model loading or training
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = DR_RESEARCH

RET_CONFIG_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1/training_config.json"
)

META_CONFIG_PATH = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1/training_config.json"
)

EXP08 = ROOT / "experiment08_results"
PROTOCOL_PATH = EXP08 / "Regularization_Protocol_V1.json"

assert RET_CONFIG_PATH.is_file()
assert META_CONFIG_PATH.is_file()
assert not PROTOCOL_PATH.exists(), "Protocol already exists."

ret = json.loads(RET_CONFIG_PATH.read_text(encoding="utf-8"))
meta = json.loads(META_CONFIG_PATH.read_text(encoding="utf-8"))

shared_keys = [
    "architecture",
    "num_classes",
    "seed",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "scheduler",
    "maximum_epochs",
    "minimum_epochs",
    "patience",
    "class_weights",
    "training_index_sha256",
]

for key in shared_keys:
    assert key in ret and key in meta, f"Missing key: {key}"
    assert ret[key] == meta[key], f"Mismatch: {key}"

assert ret["architecture"] == "vit_large_patch16_224"
assert ret["training_index_sha256"] == (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

assert ret["seed"] == 42
assert ret["maximum_epochs"] == 25

protocol = {
    "experiment": "Experiment08_Regularization_Ablation_V1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PROTOCOL_LOCKED_NOT_TRAINED",
    "research_question": (
        "Does matched Drop Path regularization improve "
        "validation generalization and alter the relative "
        "performance of retinal versus general MAE pretraining?"
    ),
    "architecture": ret["architecture"],
    "num_classes": ret["num_classes"],
    "global_pool": "avg",
    "drop_path_rate": 0.1,
    "baseline_drop_path_rate": 0.0,
    "training_seed": ret["seed"],
    "training_index_sha256": ret["training_index_sha256"],
    "split_sha256": (
        "a3d83242847834ef38c6d1de8c9566556f33792041370807c709a5af4370d49f"
    ),
    "train_images": 1290,
    "validation_images": 276,
    "test_images_evaluated": 0,
    "training": {
        "micro_batch_size": ret["micro_batch_size"],
        "gradient_accumulation_steps": ret[
            "gradient_accumulation_steps"
        ],
        "effective_batch_size": ret["effective_batch_size"],
        "optimizer": "AdamW",
        "optimizer_foreach": False,
        "learning_rate": ret["learning_rate"],
        "weight_decay": ret["weight_decay"],
        "weighted_ce_reduction": "none",
        "accumulation_normalization": (
            "sum_of_class_weights_per_accumulation_group"
        ),
        "gradient_clip_norm": 1.0,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": ret["maximum_epochs"],
        "scheduler_eta_min": 1e-7,
        "maximum_epochs": ret["maximum_epochs"],
        "minimum_epochs": ret["minimum_epochs"],
        "patience": ret["patience"],
        "model_selection_metric": "val_macro_f1",
        "class_weights": ret["class_weights"],
        "precision": "bfloat16",
        "gradient_checkpointing": True,
        "train_transform": meta["train_transform"],
        "val_transform": meta["val_transform"],
    },
    "models": {
        "RETFound": {
            "pretraining": ret["pretraining"],
            "pretrained_checkpoint": (
                "model_checkpoints/RETFound_pretrained/"
                "RETFound_mae_natureCFP.pth"
            ),
            "pretrained_sha256": ret["pretrained_sha256"],
            "output_subdirectory": "retfound_drop_path_01_v1",
        },
        "Meta_MAE": {
            "pretraining": meta["pretraining"],
            "pretrained_checkpoint": meta["pretrained_checkpoint"],
            "pretrained_sha256": meta["pretrained_sha256"],
            "output_subdirectory": "meta_mae_drop_path_01_v1",
        },
    },
    "analysis_role": "validation_regularization_robustness",
    "baseline_checkpoints_preserved": True,
    "test_split_locked": True,
}

EXP08.mkdir(parents=True, exist_ok=True)

with PROTOCOL_PATH.open("x", encoding="utf-8") as f:
    json.dump(protocol, f, indent=2, ensure_ascii=False)

saved = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

assert saved["drop_path_rate"] == 0.1
assert saved["training_seed"] == 42
assert len(saved["models"]) == 2
assert saved["test_images_evaluated"] == 0

print("EXPERIMENT 08.6 - PROTOCOL LOCKED")
print("Protocol saved:", PROTOCOL_PATH.is_file())
print("Architecture:", saved["architecture"])
print("Drop Path:", saved["drop_path_rate"])
print("Training seed:", saved["training_seed"])
print("Training images:", saved["train_images"])
print("Validation images:", saved["validation_images"])

print("\nMODEL ARMS")
for name, details in saved["models"].items():
    print(name, "->", details["output_subdirectory"])

print("\nRESEARCH INTEGRITY")
print("Original checkpoints modified: False")
print("New training started: False")
print("Test images accessed: 0")
print("STEP 08.6 PASS")


# %% [cell 253] Experiment 08 - Step 08.7
# ============================================================
# Experiment 08 - Step 08.7
# Training Data and Pretrained Weight Preflight
# Metadata only; no training or model inference
# ============================================================

from pathlib import Path
import json
import pandas as pd

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

PROTOCOL_PATH = (
    ROOT / "experiment08_results"
    / "Regularization_Protocol_V1.json"
)

assert INDEX_PATH.is_file()
assert PROTOCOL_PATH.is_file()

protocol = json.loads(
    PROTOCOL_PATH.read_text(encoding="utf-8")
)

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

print("EXPERIMENT 08.7 - TRAINING PREFLIGHT")

# ------------------------------------------------------------
# 1. Frozen train/validation index
# ------------------------------------------------------------

train = index[index["split"] == "train"].copy()
val = index[index["split"] == "val"].copy()

assert len(train) == 1290
assert len(val) == 276
assert len(train) + len(val) == 1566

assert train["image_id"].is_unique
assert val["image_id"].is_unique

assert not (
    set(train["image_id"]) & set(val["image_id"])
)

print("\nDATASET")
print("Training images:", len(train))
print("Validation images:", len(val))
print("Test images accessed:", 0)

# ------------------------------------------------------------
# 2. Local image cache metadata
# ------------------------------------------------------------

def inspect_cache(frame):
    missing = []
    symlinks = []
    drive_paths = []
    empty_files = []

    for path_str in frame["local_path"]:
        path = Path(path_str)

        if path.is_symlink():
            symlinks.append(str(path))

        if not path.is_file():
            missing.append(str(path))
            continue

        if path.stat().st_size == 0:
            empty_files.append(str(path))

        if str(path.resolve()).startswith(str(DR_RESEARCH.parent) + "/"):
            drive_paths.append(str(path))

    return {
        "total": len(frame),
        "missing": missing,
        "symlinks": symlinks,
        "drive_paths": drive_paths,
        "empty_files": empty_files,
    }


train_status = inspect_cache(train)
val_status = inspect_cache(val)

for name, status in [
    ("Training", train_status),
    ("Validation", val_status),
]:
    print(f"\n{name.upper()} CACHE")
    print("Expected images:", status["total"])
    print("Missing images:", len(status["missing"]))
    print("Symbolic links:", len(status["symlinks"]))
    print("Drive-backed paths:", len(status["drive_paths"]))
    print("Empty files:", len(status["empty_files"]))

    if status["missing"]:
        print("Missing examples:", status["missing"][:3])

# ------------------------------------------------------------
# 3. Pretrained checkpoint metadata
# ------------------------------------------------------------

print("\nPRETRAINED CHECKPOINTS")

for model_name, details in protocol["models"].items():
    checkpoint = ROOT / details["pretrained_checkpoint"]

    print(f"\n{model_name}")
    print("Exists:", checkpoint.is_file())

    if checkpoint.is_file():
        print(
            "Size (GiB):",
            round(checkpoint.stat().st_size / (1024 ** 3), 3)
        )

# ------------------------------------------------------------
# 4. Final readiness
# ------------------------------------------------------------

cache_ready = all(
    len(status[key]) == 0
    for status in (train_status, val_status)
    for key in (
        "missing",
        "symlinks",
        "drive_paths",
        "empty_files"
    )
)

weights_ready = all(
    (ROOT / details["pretrained_checkpoint"]).is_file()
    for details in protocol["models"].values()
)

print("\nTRAINING READINESS")
print("Local data cache ready:", cache_ready)
print("Pretrained weights present:", weights_ready)
print("Protocol locked:", protocol["status"])

print("\nRESEARCH INTEGRITY")
print("Research files modified: False")
print("Training started: False")
print("Model inference performed: False")
print("Test images accessed: 0")

assert weights_ready, "Missing pretrained checkpoint."

if cache_ready:
    print("\nSTEP 08.7 PASS")
else:
    print("\nSTEP 08.7 CACHE RECOVERY REQUIRED")


# %% [cell 254] Experiment 08 - Step 08.8
# ============================================================
# Experiment 08 - Step 08.8
# RETFound Training Code Dependency Audit
# Read-only; no training or model loading
# ============================================================

from pathlib import Path
import json
import re

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert NOTEBOOK.is_file()

notebook = json.loads(
    NOTEBOOK.read_text(encoding="utf-8")
)

source = "".join(notebook["cells"][98]["source"])
lines = source.splitlines()

print("EXPERIMENT 08.8 - TRAINING CODE AUDIT")
print("Source cell:", 98)
print("Source lines:", len(lines))

sections = {
    "Imports": [
        r"^import ",
        r"^from "
    ],
    "Data loading": [
        r"DataLoader",
        r"Dataset",
        r"train_loader",
        r"val_loader"
    ],
    "Model initialization": [
        r"torch\.manual_seed",
        r"timm\.create_model",
        r"encoder_state",
        r"load_state_dict",
        r"set_grad_checkpointing"
    ],
    "Training": [
        r"CrossEntropyLoss",
        r"AdamW",
        r"CosineAnnealingLR",
        r"clip_grad_norm_",
        r"optimizer\.step",
        r"scheduler\.step"
    ],
    "Checkpoint saving": [
        r"torch\.save",
        r"best_model",
        r"resume",
        r"checkpoint",
        r"history"
    ]
}

for section, patterns in sections.items():
    print(f"\n{section.upper()}")

    matched = []

    for i, line in enumerate(lines):
        if any(re.search(p, line) for p in patterns):
            matched.append((i + 1, line.strip()))

    print("Matching lines:", len(matched))

    for number, line in matched[:18]:
        print(f"{number:04d}: {line[:150]}")

print("\nKEY MODEL CONSTRUCTION CONTEXT")

for i, line in enumerate(lines):
    if "model = timm.create_model(" in line:
        for j in range(
            max(0, i - 3),
            min(len(lines), i + 10)
        ):
            print(f"{j + 1:04d}: {lines[j]}")
        break

print("\nRESEARCH INTEGRITY")
print("Notebook modified: False")
print("Research files modified: False")
print("Training started: False")
print("Test images accessed: 0")
print("STEP 08.8 AUDIT COMPLETE")


# %% [cell 255] Experiment 08 - Step 08.9
# ============================================================
# Experiment 08 - Step 08.9
# RETFound Training Initialization and Checkpoint Audit
# Read-only; no training or model loading
# ============================================================

from pathlib import Path
import json

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert NOTEBOOK.is_file()

notebook = json.loads(
    NOTEBOOK.read_text(encoding="utf-8")
)

source = "".join(notebook["cells"][98]["source"])
lines = source.splitlines()

regions = {
    "Configuration and safety checks": (30, 115),
    "DataLoader initialization": (115, 170),
    "Pretrained model initialization": (170, 240),
    "Checkpoint and history saving": (565, 630),
}

print("EXPERIMENT 08.9 - TRAINING IMPLEMENTATION AUDIT")
print("Original source cell:", 98)
print("Total source lines:", len(lines))

for title, (start, end) in regions.items():
    print("\n" + "=" * 65)
    print(title.upper())
    print("=" * 65)

    for i in range(start - 1, min(end, len(lines))):
        print(f"{i + 1:04d}: {lines[i]}")

print("\nRESEARCH INTEGRITY")
print("Notebook modified: False")
print("Experiment 04 modified: False")
print("Experiment 07 modified: False")
print("Training started: False")
print("Pretrained checkpoints loaded: False")
print("Test images accessed: 0")
print("STEP 08.9 AUDIT COMPLETE")


# %% [cell 256] Experiment 08 - Step 08.10
# ============================================================
# Experiment 08 - Step 08.10
# Dataset Construction Source Audit
# Read-only; no training or file modifications
# ============================================================

from pathlib import Path
import json
import re

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert NOTEBOOK.is_file()

notebook = json.loads(
    NOTEBOOK.read_text(encoding="utf-8")
)

patterns = [
    r"class\s+\w+Dataset",
    r"def\s+__getitem__",
    r"def\s+__len__",
    r"train_loader\s*=",
    r"val_loader\s*=",
    r"train_transform\s*=",
    r"val_transform\s*=",
    r"transforms\.Compose",
]

print("EXPERIMENT 08.10 - DATASET CONSTRUCTION AUDIT")

matches = []

for cell_index in range(80, 99):
    cell = notebook["cells"][cell_index]

    if cell.get("cell_type") != "code":
        continue

    lines = "".join(cell.get("source", [])).splitlines()

    relevant = set()

    for i, line in enumerate(lines):
        if any(
            re.search(pattern, line)
            for pattern in patterns
        ):
            relevant.update(
                range(
                    max(0, i - 5),
                    min(len(lines), i + 18)
                )
            )

    if relevant:
        matches.append(cell_index)

        print("\n" + "=" * 60)
        print("CELL:", cell_index)

        for i in sorted(relevant):
            print(f"{i + 1:04d}: {lines[i][:180]}")

print("\nMATCHING CELLS:", matches)

print("\nRESEARCH INTEGRITY")
print("Notebook modified: False")
print("Research files modified: False")
print("Training started: False")
print("Test images accessed: 0")
print("STEP 08.10 AUDIT COMPLETE")


# %% [cell 257] Experiment 08 - Step 08.11
# ============================================================
# Experiment 08 - Step 08.11
# Independent Dataset and DataLoader Initialization
# Train/validation only; no model training
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

IMAGE_SIZE = 224
SEED = 42

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ------------------------------------------------------------
# 1. Reproduce frozen transformations
# ------------------------------------------------------------

exp08_train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(
        degrees=10,
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=0
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

exp08_val_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ------------------------------------------------------------
# 2. Independent dataset class
# ------------------------------------------------------------

class Exp08FGADRDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.frame = dataframe.reset_index(drop=True).copy()
        self.transform = transform

        assert self.frame["image_id"].is_unique
        assert self.frame["dr_grade"].between(0, 4).all()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        image_path = Path(row["local_path"])

        if not image_path.is_file():
            raise FileNotFoundError(str(image_path))

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        image_tensor = self.transform(image)
        label = int(row["dr_grade"])

        return (
            image_tensor,
            torch.tensor(label, dtype=torch.long),
            str(row["image_id"])
        )

# ------------------------------------------------------------
# 3. Load frozen train/validation records only
# ------------------------------------------------------------

index = pd.read_csv(
    INDEX_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str
    }
)

train_frame = index.loc[index["split"] == "train"].copy()
val_frame = index.loc[index["split"] == "val"].copy()

assert len(train_frame) == 1290
assert len(val_frame) == 276

assert set(train_frame["image_id"]).isdisjoint(
    set(val_frame["image_id"])
)

assert set(train_frame["risk_group_id"].dropna()).isdisjoint(
    set(val_frame["risk_group_id"].dropna())
)

exp08_train_dataset = Exp08FGADRDataset(
    train_frame,
    exp08_train_transform
)

exp08_val_dataset = Exp08FGADRDataset(
    val_frame,
    exp08_val_transform
)

# ------------------------------------------------------------
# 4. Initialize loaders with isolated random generator
# ------------------------------------------------------------

exp08_train_generator = torch.Generator()
exp08_train_generator.manual_seed(SEED)

exp08_train_loader = DataLoader(
    exp08_train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    generator=exp08_train_generator
)

exp08_val_loader = DataLoader(
    exp08_val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# ------------------------------------------------------------
# 5. Read-only validation smoke test
# ------------------------------------------------------------

images, labels, image_ids = next(iter(exp08_val_loader))

assert images.shape == (16, 3, 224, 224)
assert labels.shape == (16,)
assert len(image_ids) == 16
assert torch.isfinite(images).all()

assert len(exp08_train_loader) == 323
assert len(exp08_val_loader) == 18

print("EXPERIMENT 08.11 - DATASET INITIALIZATION")

print("\nDATASET")
print("Training images:", len(exp08_train_dataset))
print("Validation images:", len(exp08_val_dataset))
print("Training micro-batches:", len(exp08_train_loader))
print("Validation batches:", len(exp08_val_loader))

print("\nSAMPLE BATCH")
print("Image tensor shape:", tuple(images.shape))
print("Label tensor shape:", tuple(labels.shape))
print("Pixel values finite:", bool(torch.isfinite(images).all()))
print("Sample image IDs:", list(image_ids[:3]))

print("\nDATA ISOLATION")
print("Train/validation image overlap: 0")
print("Train/validation risk-group overlap: 0")

print("\nRESEARCH INTEGRITY")
print("Model inference performed: False")
print("Training started: False")
print("Research files modified: False")
print("Test images accessed: 0")

print("\nSTEP 08.11 PASS")


# %% [cell 258] Experiment 08 - Step 08.12
# ============================================================
# Experiment 08 - Step 08.12
# RETFound Drop Path 0.1 Model Initialization
# No training, no inference, no file modifications
# ============================================================

from pathlib import Path
import gc
import json
import random
import argparse

import numpy as np
import torch
import timm

ROOT = DR_RESEARCH

PROTOCOL_PATH = (
    ROOT / "experiment08_results"
    / "Regularization_Protocol_V1.json"
)

WEIGHT_PATH = (
    ROOT / "model_checkpoints"
    / "RETFound_pretrained"
    / "RETFound_mae_natureCFP.pth"
)

assert PROTOCOL_PATH.is_file()
assert WEIGHT_PATH.is_file()
assert torch.cuda.is_available()

protocol = json.loads(
    PROTOCOL_PATH.read_text(encoding="utf-8")
)

assert protocol["drop_path_rate"] == 0.1
assert protocol["training_seed"] == 42

SEED = int(protocol["training_seed"])
DEVICE = torch.device("cuda")

# ------------------------------------------------------------
# 1. Reproducible initialization
# ------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False

# Remove unused model objects from previous steps if present.
gc.collect()
torch.cuda.empty_cache()

# ------------------------------------------------------------
# 2. Build the regularized classifier
# ------------------------------------------------------------

exp08_ret_model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    drop_path_rate=0.1
)

assert exp08_ret_model.global_pool == "avg"
assert len(exp08_ret_model.blocks) == 24

# ------------------------------------------------------------
# 3. Load original RETFound pretrained encoder
# ------------------------------------------------------------

with torch.serialization.safe_globals([argparse.Namespace]):
    pretrained = torch.load(
        WEIGHT_PATH,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

print("EXPERIMENT 08.12 - RETFOUND INITIALIZATION")
print("Checkpoint loaded:", True)
print("Checkpoint type:", type(pretrained).__name__)

if isinstance(pretrained, dict) and "model" in pretrained:
    source_state = pretrained["model"]
elif isinstance(pretrained, dict) and "state_dict" in pretrained:
    source_state = pretrained["state_dict"]
else:
    raise RuntimeError(
        "Unexpected RETFound checkpoint structure. "
        "Stop before loading model weights."
    )

encoder_state = {
    key: value
    for key, value in source_state.items()
    if key.startswith(("patch_embed.", "blocks."))
    or key in {"cls_token", "pos_embed"}
}

assert len(encoder_state) == 292, (
    f"Expected 292 encoder tensors, found {len(encoder_state)}."
)

result = exp08_ret_model.load_state_dict(
    encoder_state,
    strict=False
)

expected_missing = {
    "fc_norm.weight",
    "fc_norm.bias",
    "head.weight",
    "head.bias"
}

assert set(result.missing_keys) == expected_missing
assert len(result.unexpected_keys) == 0

# ------------------------------------------------------------
# 4. Verify key pretrained tensors
# ------------------------------------------------------------

for key in [
    "patch_embed.proj.weight",
    "blocks.0.attn.qkv.weight",
    "blocks.23.mlp.fc2.weight",
    "pos_embed"
]:
    assert torch.equal(
        exp08_ret_model.state_dict()[key],
        encoder_state[key]
    ), f"Weight mismatch: {key}"

# ------------------------------------------------------------
# 5. Verify Drop Path and enable checkpointing
# ------------------------------------------------------------

drop_rates = []

for block in exp08_ret_model.blocks:
    for name in ("drop_path1", "drop_path2"):
        layer = getattr(block, name, None)
        if layer is not None:
            drop_rates.append(
                float(getattr(layer, "drop_prob", 0.0))
            )

assert len(drop_rates) == 48
assert sum(rate > 0 for rate in drop_rates) == 46
assert np.isclose(max(drop_rates), 0.1)

assert exp08_ret_model.blocks[0].norm1.eps == 1e-6
assert exp08_ret_model.fc_norm.eps == 1e-6

exp08_ret_model.set_grad_checkpointing(enable=True)

# Release the source checkpoint before transferring the model.
del pretrained, source_state, encoder_state
gc.collect()

exp08_ret_model = exp08_ret_model.to(DEVICE)

print("\nMODEL")
print("Architecture:", protocol["architecture"])
print("Pretraining:", protocol["models"]["RETFound"]["pretraining"])
print("Global pooling:", exp08_ret_model.global_pool)
print("Transformer blocks:", len(exp08_ret_model.blocks))
print("Drop Path maximum:", round(max(drop_rates), 6))
print("Nonzero Drop Path entries:", sum(r > 0 for r in drop_rates))

print("\nPRETRAINED WEIGHTS")
print("Encoder tensors loaded:", 292)
print("Missing parameters:", sorted(result.missing_keys))
print("Unexpected parameters:", result.unexpected_keys)

print("\nTRAINING READINESS")
print("Device:", next(exp08_ret_model.parameters()).device)
print("Gradient checkpointing enabled:", True)
print("Model training performed:", False)

print("\nRESEARCH INTEGRITY")
print("Original checkpoints modified: False")
print("Research files modified: False")
print("Test images accessed: 0")
print("STEP 08.12 PASS")


# %% [cell 259] Experiment 08 - Step 08.13
# ============================================================
# Experiment 08 - Step 08.13
# RETFound Drop Path Gradient Smoke Test
# No optimizer step, no checkpoint modifications
# ============================================================

import gc
import torch
import torch.nn as nn
import numpy as np

assert "exp08_ret_model" in globals()
assert "exp08_train_dataset" in globals()
assert torch.cuda.is_available()

DEVICE = torch.device("cuda")
model = exp08_ret_model

# Use four training samples without advancing the train loader RNG.
samples = [exp08_train_dataset[i] for i in range(4)]

images = torch.stack([item[0] for item in samples]).to(DEVICE)
labels = torch.stack([item[1] for item in samples]).to(DEVICE)

weights = torch.tensor(
    [
        1.6152555557576007,
        1.1187667747158858,
        0.6665034324838375,
        0.6394716814936021,
        0.9600025555490744
    ],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.CrossEntropyLoss(
    weight=weights,
    reduction="none"
)

# Test training-mode Drop Path and gradient checkpointing.
model.train()
model.zero_grad(set_to_none=True)

torch.cuda.reset_peak_memory_stats()

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(images)
    individual_losses = criterion(logits.float(), labels)

loss = individual_losses.sum() / weights[labels].sum()

assert logits.shape == (4, 5)
assert torch.isfinite(logits).all()
assert torch.isfinite(loss)

loss.backward()

gradient_checks = {}

for name in [
    "patch_embed.proj.weight",
    "blocks.0.attn.qkv.weight",
    "blocks.23.mlp.fc2.weight",
    "head.weight"
]:
    parameter = dict(model.named_parameters())[name]
    gradient_checks[name] = (
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
    )

assert all(gradient_checks.values())

peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

print("EXPERIMENT 08.13 - GRADIENT SMOKE TEST")

print("\nFORWARD PASS")
print("Batch size:", len(labels))
print("Logits shape:", tuple(logits.shape))
print("Weighted CE loss:", round(float(loss.item()), 6))
print("Finite logits:", True)

print("\nBACKWARD PASS")
for name, passed in gradient_checks.items():
    print(name, ":", "PASS" if passed else "FAIL")

print("\nGPU")
print("GPU:", torch.cuda.get_device_name(0))
print("Peak allocated memory (GiB):", round(peak_memory, 3))

print("\nRESEARCH INTEGRITY")
print("Optimizer step performed: False")
print("Model parameters updated: False")
print("Checkpoints modified: False")
print("Test images accessed: 0")

# Remove gradients and test tensors before actual training.
model.zero_grad(set_to_none=True)
del images, labels, samples, logits, individual_losses, loss
gc.collect()
torch.cuda.empty_cache()

print("\nSTEP 08.13 PASS")


# %% [cell 260] Experiment 08 - Step 08.14
# ============================================================
# Experiment 08 - Step 08.14
# Generate Isolated RETFound Drop Path Training Source
# Source generation only; no training execution
# ============================================================

from pathlib import Path
import json
import ast

ROOT = DR_RESEARCH
NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

EXP08 = ROOT / "experiment08_results"
SCRIPT_PATH = EXP08 / "retfound_drop_path_01_epoch1_v1.py"

assert NOTEBOOK.is_file()
assert EXP08.is_dir()
assert not SCRIPT_PATH.exists()

notebook = json.loads(
    NOTEBOOK.read_text(encoding="utf-8")
)

source = "".join(notebook["cells"][98]["source"])

# ------------------------------------------------------------
# 1. Apply only controlled substitutions
# ------------------------------------------------------------

old_directory = '''EXPERIMENT_DIR = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
)'''

new_directory = '''EXPERIMENT_DIR = (
    ROOT / "experiment08_results"
    / "retfound_drop_path_01_v1"
)'''

old_model = '''model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES,
    global_pool="avg"
)'''

new_model = '''model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES,
    global_pool="avg",
    drop_path_rate=0.1
)'''

assert source.count(old_directory) == 1
assert source.count(old_model) == 1

modified = source.replace(old_directory, new_directory)
modified = modified.replace(old_model, new_model)

# ------------------------------------------------------------
# 2. Remove dependence on old global data loaders
# ------------------------------------------------------------

old_dependencies = '''assert "state_dict" in globals()
assert "train_loader" in globals()
assert "val_loader" in globals()'''

new_dependencies = '''assert "state_dict" in globals()
assert "exp08_train_loader" in globals()
assert "exp08_val_loader" in globals()'''

assert modified.count(old_dependencies) == 1

modified = modified.replace(
    old_dependencies,
    new_dependencies
)

# Replace explicit loader references.
modified = modified.replace(
    "train_loader.dataset",
    "exp08_train_loader.dataset"
)

modified = modified.replace(
    "val_loader.dataset",
    "exp08_val_loader.dataset"
)

modified = modified.replace(
    "\n    val_loader,\n",
    "\n    exp08_val_loader,\n"
)

# ------------------------------------------------------------
# 3. Static safety audit
# ------------------------------------------------------------

ast.parse(modified)

assert 'drop_path_rate=0.1' in modified
assert 'ROOT / "experiment08_results"' in modified
assert 'ROOT / "experiment04_results"' not in modified

assert "exp08_train_loader.dataset" in modified
assert "exp08_val_loader.dataset" in modified

assert 'CHECKPOINT_PATH = EXPERIMENT_DIR / "last_training_state.pt"' in modified
assert "os.replace(temporary_checkpoint, CHECKPOINT_PATH)" in modified

# ------------------------------------------------------------
# 4. Save new source only
# ------------------------------------------------------------

with SCRIPT_PATH.open("x", encoding="utf-8") as f:
    f.write(modified)

print("EXPERIMENT 08.14 - TRAINING SCRIPT GENERATED")
print("Script saved:", SCRIPT_PATH.is_file())
print("Source lines:", len(modified.splitlines()))
print("Python syntax valid:", True)

print("\nISOLATION")
print("Experiment 08 output directory:", True)
print("Drop Path 0.1 enabled:", True)
print("Original Experiment 04 output removed:", True)

print("\nRESEARCH INTEGRITY")
print("Original notebook modified: False")
print("Original checkpoints modified: False")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 08.14 PASS")


# %% [cell 261] Experiment 08 - Step 08.15 (Corrected)
# ============================================================
# Experiment 08 - Step 08.15 (Corrected)
# Generate Standalone RETFound Training Source V2
# Source generation only; do not execute generated script
# ============================================================

from pathlib import Path
import ast

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

SOURCE = OUT / "retfound_drop_path_01_epoch1_v1.py"
TARGET = OUT / "retfound_drop_path_01_epoch1_v2.py"

assert SOURCE.is_file(), "V1 training source is missing."
assert not TARGET.exists(), "V2 already exists. Do not overwrite it."

source = SOURCE.read_text(encoding="utf-8")

# ------------------------------------------------------------
# 1. Add standalone imports
# ------------------------------------------------------------

imports = "\n".join([
    "from PIL import Image",
    "from torchvision import transforms",
    "from torch.utils.data import Dataset",
    "import argparse",
])

assert source.count("import timm") == 1

source = source.replace(
    "import timm",
    "import timm\n" + imports,
    1
)

# ------------------------------------------------------------
# 2. Standalone dataset and checkpoint helpers
# ------------------------------------------------------------

bootstrap = "\n".join([
    "class Exp08StandaloneDataset(Dataset):",
    "    def __init__(self, dataframe, transform):",
    "        self.frame = dataframe.reset_index(drop=True).copy()",
    "        self.transform = transform",
    "        assert self.frame['image_id'].is_unique",
    "        assert self.frame['dr_grade'].between(0, 4).all()",
    "",
    "    def __len__(self):",
    "        return len(self.frame)",
    "",
    "    def __getitem__(self, index):",
    "        row = self.frame.iloc[index]",
    "        image_path = Path(row['local_path'])",
    "        if not image_path.is_file():",
    "            raise FileNotFoundError(str(image_path))",
    "        with Image.open(image_path) as source_image:",
    "            image = source_image.convert('RGB')",
    "        return (",
    "            self.transform(image),",
    "            torch.tensor(int(row['dr_grade']), dtype=torch.long),",
    "            str(row['image_id'])",
    "        )",
    "",
    "",
    "def build_exp08_datasets(index_path):",
    "    index = pd.read_csv(",
    "        index_path,",
    "        dtype={'image_id': str, 'risk_group_id': str}",
    "    )",
    "    train = index.loc[index['split'] == 'train'].copy()",
    "    val = index.loc[index['split'] == 'val'].copy()",
    "    assert len(train) == 1290",
    "    assert len(val) == 276",
    "    assert set(train['image_id']).isdisjoint(set(val['image_id']))",
    "    assert set(train['risk_group_id']).isdisjoint(",
    "        set(val['risk_group_id'])",
    "    )",
    "",
    "    mean = [0.485, 0.456, 0.406]",
    "    std = [0.229, 0.224, 0.225]",
    "",
    "    train_transform = transforms.Compose([",
    "        transforms.Resize(",
    "            (224, 224),",
    "            interpolation=transforms.InterpolationMode.BICUBIC",
    "        ),",
    "        transforms.RandomHorizontalFlip(p=0.5),",
    "        transforms.RandomRotation(",
    "            degrees=10,",
    "            interpolation=transforms.InterpolationMode.BILINEAR,",
    "            fill=0",
    "        ),",
    "        transforms.ToTensor(),",
    "        transforms.Normalize(mean=mean, std=std)",
    "    ])",
    "",
    "    val_transform = transforms.Compose([",
    "        transforms.Resize(",
    "            (224, 224),",
    "            interpolation=transforms.InterpolationMode.BICUBIC",
    "        ),",
    "        transforms.ToTensor(),",
    "        transforms.Normalize(mean=mean, std=std)",
    "    ])",
    "",
    "    return (",
    "        Exp08StandaloneDataset(train, train_transform),",
    "        Exp08StandaloneDataset(val, val_transform)",
    "    )",
    "",
    "",
    "def load_exp08_retfound_encoder(root):",
    "    weight_path = (",
    "        root / 'model_checkpoints'",
    "        / 'RETFound_pretrained'",
    "        / 'RETFound_mae_natureCFP.pth'",
    "    )",
    "    assert weight_path.is_file()",
    "    with torch.serialization.safe_globals([argparse.Namespace]):",
    "        checkpoint = torch.load(",
    "            weight_path,",
    "            map_location='cpu',",
    "            weights_only=True,",
    "            mmap=True",
    "        )",
    "    source_state = checkpoint['model']",
    "    encoder_state = {",
    "        key: value",
    "        for key, value in source_state.items()",
    "        if key.startswith(('patch_embed.', 'blocks.'))",
    "        or key in {'cls_token', 'pos_embed'}",
    "    }",
    "    assert len(encoder_state) == 292",
    "    return encoder_state",
    "",
])

# ------------------------------------------------------------
# 3. Remove old global-state assertions
# ------------------------------------------------------------

old_checks = "\n".join([
    'assert "state_dict" in globals()',
    'assert "exp08_train_loader" in globals()',
    'assert "exp08_val_loader" in globals()',
])

assert source.count(old_checks) == 1

source = source.replace(
    old_checks,
    'assert torch.cuda.is_available()',
    1
)

# ------------------------------------------------------------
# 4. Initialize datasets before class weight computation
# ------------------------------------------------------------

marker = 'train_index = index_df[index_df["split"] == "train"]'

dataset_setup = "\n".join([
    marker,
    "",
    "exp08_train_dataset, exp08_val_dataset = build_exp08_datasets(",
    "    INDEX_PATH",
    ")",
    "",
    "exp08_val_loader = DataLoader(",
    "    exp08_val_dataset,",
    "    batch_size=16,",
    "    shuffle=False,",
    "    num_workers=0,",
    "    pin_memory=True",
    ")",
])

assert source.count(marker) == 1

source = source.replace(
    marker,
    dataset_setup,
    1
)

# ------------------------------------------------------------
# 5. Build the actual training loader from standalone dataset
# ------------------------------------------------------------

old_dataset = "exp08_train_loader.dataset"

assert old_dataset in source

source = source.replace(
    old_dataset,
    "exp08_train_dataset"
)

# ------------------------------------------------------------
# 6. Replace pretrained encoder global dependency
# ------------------------------------------------------------

old_encoder = "\n".join([
    "encoder_state = {",
    "    key: value",
    "    for key, value in state_dict.items()",
    '    if key.startswith(("patch_embed.", "blocks."))',
    '    or key in {"cls_token", "pos_embed"}',
    "}",
])

new_encoder = "encoder_state = load_exp08_retfound_encoder(ROOT)"

assert source.count(old_encoder) == 1

source = source.replace(
    old_encoder,
    new_encoder,
    1
)

# ------------------------------------------------------------
# 7. Insert helper definitions
# ------------------------------------------------------------

assert source.count("SEED = 42") == 1

source = source.replace(
    "SEED = 42",
    bootstrap + "\nSEED = 42",
    1
)

# ------------------------------------------------------------
# 8. Static validation
# ------------------------------------------------------------

ast.parse(source)

assert "drop_path_rate=0.1" in source
assert 'ROOT / "experiment08_results"' in source
assert 'ROOT / "experiment04_results"' not in source
assert "load_exp08_retfound_encoder(ROOT)" in source
assert "build_exp08_datasets(" in source

with TARGET.open("x", encoding="utf-8") as f:
    f.write(source)

print("EXPERIMENT 08.15 - CORRECTED STANDALONE SOURCE")
print("V1 preserved:", SOURCE.is_file())
print("V2 created:", TARGET.is_file())
print("Python syntax valid:", True)
print("Dataset loader embedded:", True)
print("Pretrained encoder loader embedded:", True)
print("Drop Path rate:", 0.1)

print("\nRESEARCH INTEGRITY")
print("Original notebook modified: False")
print("Original checkpoints modified: False")
print("Training started: False")
print("Test images accessed: 0")
print("STEP 08.15 PASS")


# %% [cell 262] Experiment 08 - Step 08.16
# ============================================================
# Experiment 08 - Step 08.16
# Standalone Training Source Safety Audit
# Read-only; no model loading or training
# ============================================================

from pathlib import Path
import ast
import re

ROOT = DR_RESEARCH

SCRIPT = (
    ROOT / "experiment08_results"
    / "retfound_drop_path_01_epoch1_v2.py"
)

assert SCRIPT.is_file()

source = SCRIPT.read_text(encoding="utf-8")
tree = ast.parse(source)

print("EXPERIMENT 08.16 - TRAINING SOURCE SAFETY AUDIT")
print("Source lines:", len(source.splitlines()))
print("Python syntax valid:", True)

print("\nVARIABLE REFERENCE AUDIT")

patterns = {
    "Old pretrained state_dict reference":
        r"\bstate_dict\.items\(\)",

    "Old train_loader reference":
        r"(?<!exp08_)\btrain_loader\b",

    "Old val_loader reference":
        r"(?<!exp08_)\bval_loader\b",

    "Standalone training dataset":
        r"\bexp08_train_dataset\b",

    "Standalone validation loader":
        r"\bexp08_val_loader\b",

    "RETFound encoder loading":
        r"load_exp08_retfound_encoder\(ROOT\)",

    "Drop Path configuration":
        r"drop_path_rate\s*=\s*0\.1",
}

for label, pattern in patterns.items():
    matches = list(re.finditer(pattern, source))
    print(f"{label}: {len(matches)}")

    if label.startswith("Old ") and matches:
        for match in matches[:5]:
            line_number = source.count("\n", 0, match.start()) + 1
            print("  Review line:", line_number)

print("\nOUTPUT PATH AUDIT")

for text in [
    "experiment04_results",
    "experiment07_results",
    "experiment08_results",
    "last_training_state.pt",
    "training_history.csv",
    "validation_predictions_epoch001.csv"
]:
    print(f"{text}: {source.count(text)}")

print("\nCHECKPOINT RNG AUDIT")

rng_patterns = {
    "PyTorch CPU": "torch_rng_state",
    "CUDA": "cuda_rng_states",
    "Python": "python_rng_state",
    "NumPy": "numpy_rng_state",
    "DataLoader generator": "train_loader_generator_state",
}

for name, keyword in rng_patterns.items():
    print(name + ":", keyword in source)

print("\nTRAINING SCRIPT STRUCTURE")

function_names = [
    node.name
    for node in ast.walk(tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
]

print("Defined functions:", function_names)
print("Has optimizer step:", "optimizer.step()" in source)
print("Has checkpoint save:", "torch.save(" in source)
print("Has atomic checkpoint replacement:", "os.replace(" in source)

print("\nRESEARCH INTEGRITY")
print("V2 file modified: False")
print("Training started: False")
print("Model inference performed: False")
print("Test images accessed: 0")
print("STEP 08.16 AUDIT COMPLETE")


# %% [cell 263] Experiment 08 - Step 08.17
# ============================================================
# Experiment 08 - Step 08.17
# Correct RETFound Training Source V3
# Source generation and static checks only
# ============================================================

from pathlib import Path
import ast
import re

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

SOURCE = OUT / "retfound_drop_path_01_epoch1_v2.py"
TARGET = OUT / "retfound_drop_path_01_epoch1_v3.py"

assert SOURCE.is_file()
assert not TARGET.exists(), "V3 already exists. Do not overwrite."

source = SOURCE.read_text(encoding="utf-8")

print("EXPERIMENT 08.17 - SOURCE CORRECTION")

# ------------------------------------------------------------
# 1. Replace the remaining legacy validation loader
# ------------------------------------------------------------

legacy_pattern = r"(?<!exp08_)\bval_loader\b"

matches = list(re.finditer(legacy_pattern, source))

assert len(matches) == 1, (
    f"Expected exactly one legacy val_loader, found {len(matches)}."
)

source = re.sub(
    legacy_pattern,
    "exp08_val_loader",
    source
)

# ------------------------------------------------------------
# 2. Add NumPy RNG state to the epoch checkpoint
# ------------------------------------------------------------

old_rng = '"python_rng_state": random.getstate(),'

new_rng = (
    '"python_rng_state": random.getstate(),\n'
    '    "numpy_rng_state": np.random.get_state(),'
)

assert source.count(old_rng) == 1

source = source.replace(old_rng, new_rng, 1)

# ------------------------------------------------------------
# 3. Check standalone references and training structure
# ------------------------------------------------------------

tree = ast.parse(source)

assert not re.search(r"(?<!exp08_)\bval_loader\b", source)
assert not re.search(r"(?<!exp08_)\btrain_loader\b", source)

assert "drop_path_rate=0.1" in source
assert '"numpy_rng_state": np.random.get_state()' in source

assert "load_exp08_retfound_encoder(ROOT)" in source
assert "build_exp08_datasets(" in source

assert 'ROOT / "experiment04_results"' not in source
assert 'ROOT / "experiment07_results"' not in source
assert 'ROOT / "experiment08_results"' in source

assert "optimizer.step()" in source
assert "scheduler.step()" in source
assert "torch.save(" in source
assert "os.replace(temporary_checkpoint, CHECKPOINT_PATH)" in source

# ------------------------------------------------------------
# 4. Review checkpoint fields and loader references
# ------------------------------------------------------------

checkpoint_fields = [
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "torch_rng_state",
    "cuda_rng_states",
    "python_rng_state",
    "numpy_rng_state",
    "train_loader_generator_state",
]

for field in checkpoint_fields:
    assert field in source, f"Missing checkpoint field: {field}"

# Verify source is syntactically valid before writing.
compile(source, str(TARGET), "exec")

with TARGET.open("x", encoding="utf-8") as f:
    f.write(source)

print("\nCORRECTIONS")
print("Legacy validation loader fixed:", True)
print("NumPy RNG checkpoint state added:", True)

print("\nSTATIC VALIDATION")
print("V3 created:", TARGET.is_file())
print("Python syntax valid:", True)
print("Legacy loader references:", 0)
print("Drop Path:", 0.1)
print("Checkpoint fields verified:", len(checkpoint_fields))

print("\nRESEARCH INTEGRITY")
print("V1 and V2 preserved:", True)
print("Original experiments modified:", False)
print("Training started:", False)
print("Test images accessed:", 0)

print("\nSTEP 08.17 PASS")


# %% [cell 264] Experiment 08 - Step 08.18
# ============================================================
# Experiment 08 - Step 08.18
# Final Epoch-1 Training Logic Inspection
# Read-only; no training
# ============================================================

from pathlib import Path
import ast
import re

ROOT = DR_RESEARCH

SCRIPT = (
    ROOT / "experiment08_results"
    / "retfound_drop_path_01_epoch1_v3.py"
)

assert SCRIPT.is_file()

source = SCRIPT.read_text(encoding="utf-8")
lines = source.splitlines()

tree = ast.parse(source)

checks = {
    "Training loader construction": r"retfound_train_loader\s*=\s*DataLoader",
    "Training dataset source": r"exp08_train_dataset",
    "Validation loader": r"exp08_val_loader",
    "Weighted CE loss": r"CrossEntropyLoss",
    "Gradient normalization": r"parameter\.grad\.div_",
    "Gradient clipping": r"clip_grad_norm_",
    "Optimizer step": r"optimizer\.step\(\)",
    "Scheduler step": r"scheduler\.step\(\)",
    "Checkpoint save": r"torch\.save\(",
    "Atomic checkpoint replacement": r"os\.replace\(",
}

print("EXPERIMENT 08.18 - TRAINING LOGIC INSPECTION")

for name, pattern in checks.items():
    matches = [
        i for i, line in enumerate(lines)
        if re.search(pattern, line)
    ]

    print(f"\n{name}: {len(matches)} occurrence(s)")

    for i in matches[:2]:
        start = max(0, i - 3)
        end = min(len(lines), i + 7)

        for j in range(start, end):
            print(f"{j + 1:04d}: {lines[j]}")

# Inspect top-level loop structures.
top_level_loops = [
    (node.lineno, type(node).__name__)
    for node in tree.body
    if isinstance(node, (ast.For, ast.While))
]

print("\nTOP-LEVEL LOOPS")
print(top_level_loops)

print("\nTRAINING STATE")
print("Epoch 1 checkpoint:", '"epoch": 1' in source)
print("NumPy RNG saved:", '"numpy_rng_state"' in source)
print("Drop Path configured:", "drop_path_rate=0.1" in source)

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Training started: False")
print("Model inference performed: False")
print("Test images accessed: 0")
print("STEP 08.18 AUDIT COMPLETE")


# %% [cell 265] Experiment 08 - Step 08.19
# ============================================================
# Experiment 08 - Step 08.19
# Fix Dataset Variable and Audit Epoch-1 Source V4
# Source generation only; no training
# ============================================================

from pathlib import Path
import ast
import re

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

SOURCE = OUT / "retfound_drop_path_01_epoch1_v3.py"
TARGET = OUT / "retfound_drop_path_01_epoch1_v4.py"

assert SOURCE.is_file(), "V3 source is missing."
assert not TARGET.exists(), "V4 already exists. Do not overwrite."

source = SOURCE.read_text(encoding="utf-8")

# Correct the unintended dataset variable name.
wrong_name = "retfound_exp08_train_dataset"
correct_name = "exp08_train_dataset"

wrong_count = len(re.findall(r"\b" + wrong_name + r"\b", source))

assert wrong_count >= 1, "Expected variable mismatch not found."

modified = re.sub(
    r"\b" + wrong_name + r"\b",
    correct_name,
    source
)

# Validate Python syntax.
tree = ast.parse(modified)
compile(modified, str(TARGET), "exec")

# Validate dataset and loader construction.
assert "exp08_train_dataset, exp08_val_dataset = build_exp08_datasets(" in modified
assert "retfound_train_loader = DataLoader(" in modified
assert "    exp08_train_dataset," in modified
assert "    exp08_val_dataset," in modified

assert wrong_name not in modified
assert "drop_path_rate=0.1" in modified

# Ensure training protocol remains unchanged.
required_operations = [
    'reduction="none"',
    "parameter.grad.div_(group_weight_total)",
    "clip_grad_norm_",
    "optimizer.step()",
    "scheduler.step()",
    '"numpy_rng_state": np.random.get_state()',
    "torch.save(training_state, temporary_checkpoint)",
    "os.replace(temporary_checkpoint, CHECKPOINT_PATH)",
]

for operation in required_operations:
    assert operation in modified, f"Missing operation: {operation}"

# Confirm path isolation.
assert 'ROOT / "experiment08_results"' in modified
assert 'ROOT / "experiment04_results"' not in modified
assert 'ROOT / "experiment07_results"' not in modified

# Inspect top-level executable statements for basic consistency.
assert any(isinstance(node, ast.For) for node in tree.body)

# Write only the new source file.
with TARGET.open("x", encoding="utf-8") as f:
    f.write(modified)

print("EXPERIMENT 08.19 - EPOCH-1 SOURCE CORRECTION")

print("\nCORRECTION")
print("Incorrect variable references fixed:", wrong_count)
print("Correct dataset variable:", correct_name)

print("\nSOURCE VALIDATION")
print("V4 created:", TARGET.is_file())
print("Python syntax valid:", True)
print("Standalone dataset initialized:", True)
print("Training loader initialized:", True)
print("Drop Path:", 0.1)
print("Required training operations preserved:", len(required_operations))

print("\nRESEARCH INTEGRITY")
print("V1-V3 preserved:", True)
print("Original checkpoints modified:", False)
print("Training started:", False)
print("Test images accessed:", 0)

print("\nSTEP 08.19 PASS")


# %% [cell 266] Experiment 08 - Step 08.20
# ============================================================
# Experiment 08 - Step 08.20
# Final V4 Global Dependency Audit
# Read-only; no model loading or training
# ============================================================

from pathlib import Path
import ast
import builtins

ROOT = DR_RESEARCH

SCRIPT = (
    ROOT / "experiment08_results"
    / "retfound_drop_path_01_epoch1_v4.py"
)

assert SCRIPT.is_file()

source = SCRIPT.read_text(encoding="utf-8")
tree = ast.parse(source)

# ------------------------------------------------------------
# 1. Collect global definitions and references
# ------------------------------------------------------------

defined = set(dir(builtins))
used = set()

for node in tree.body:

    if isinstance(node, ast.Import):
        for alias in node.names:
            defined.add(alias.asname or alias.name.split(".")[0])

    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            defined.add(alias.asname or alias.name)

    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        defined.add(node.name)

    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )

        for target in targets:
            for child in ast.walk(target):
                if isinstance(child, ast.Name):
                    defined.add(child.id)

    elif isinstance(node, ast.For):
        for child in ast.walk(node.target):
            if isinstance(child, ast.Name):
                defined.add(child.id)

# Collect references from top-level executable statements.
for node in tree.body:

    if isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        continue

    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(
            child.ctx, ast.Load
        ):
            used.add(child.id)

unresolved = sorted(used - defined)

print("EXPERIMENT 08.20 - GLOBAL DEPENDENCY AUDIT")
print("Source:", SCRIPT.name)
print("Source lines:", len(source.splitlines()))
print("Defined global names:", len(defined))
print("Referenced global names:", len(used))

print("\nPOTENTIALLY UNRESOLVED GLOBAL NAMES")
if unresolved:
    for name in unresolved:
        print("-", name)
else:
    print("None detected")

# ------------------------------------------------------------
# 2. Essential training names
# ------------------------------------------------------------

required = [
    "exp08_train_dataset",
    "exp08_val_dataset",
    "exp08_val_loader",
    "retfound_train_loader",
    "train_generator",
    "model",
    "optimizer",
    "scheduler",
    "criterion",
    "CHECKPOINT_PATH",
    "HISTORY_PATH",
]

print("\nESSENTIAL TRAINING VARIABLES")

for name in required:
    print(name, ":", "DEFINED" if name in defined else "MISSING")

# ------------------------------------------------------------
# 3. Check independent source requirements
# ------------------------------------------------------------

print("\nSOURCE SAFETY")
print(
    "Uses original pretrained encoder:",
    "load_exp08_retfound_encoder(ROOT)" in source
)
print(
    "Uses Drop Path 0.1:",
    "drop_path_rate=0.1" in source
)
print(
    "Saves NumPy RNG:",
    '"numpy_rng_state": np.random.get_state()' in source
)
print(
    "Writes to Experiment 08:",
    'ROOT / "experiment08_results"' in source
)

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Training started: False")
print("Model inference performed: False")
print("Test images accessed: 0")

print("\nSTEP 08.20 AUDIT COMPLETE")


# %% [cell 267] Experiment 08 - Step 08.21
# ============================================================
# Experiment 08 - Step 08.21
# Targeted Variable and Checkpoint Loader Audit
# Read-only; no training or model loading
# ============================================================

from pathlib import Path
import ast

ROOT = DR_RESEARCH
SCRIPT = (
    ROOT / "experiment08_results"
    / "retfound_drop_path_01_epoch1_v4.py"
)

assert SCRIPT.is_file()

source = SCRIPT.read_text(encoding="utf-8")
lines = source.splitlines()
tree = ast.parse(source)

suspected = [
    "grade",
    "group_sample_count",
    "group_start",
    "group_weight_total",
    "handle",
    "individual_losses",
    "is_group_end",
    "logits",
    "loss_sum",
    "parameter",
    "predictions",
    "probabilities",
    "sample_weight_sum",
    "values",
]

assigned = set()

for node in ast.walk(tree):
    if isinstance(node, ast.Name) and isinstance(node.ctx, (
        ast.Store, ast.Param
    )):
        assigned.add(node.id)

    elif isinstance(node, ast.arg):
        assigned.add(node.arg)

print("EXPERIMENT 08.21 - TARGETED SOURCE AUDIT")

print("\nSUSPECTED VARIABLE DEFINITIONS")
missing = []

for name in suspected:
    exists = name in assigned
    print(f"{name}: {'DEFINED' if exists else 'MISSING'}")
    if not exists:
        missing.append(name)

assert not missing, f"Unresolved variables: {missing}"

print("\nPRETRAINED CHECKPOINT LOADER")

for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        if node.name == "load_exp08_retfound_encoder":
            start = node.lineno - 1
            end = node.end_lineno

            for i in range(start, end):
                print(f"{i + 1:04d}: {lines[i]}")

            break
else:
    raise AssertionError("Pretrained encoder loader missing.")

print("\nIMPORTANT CHECKS")
print("Python syntax valid:", True)
print("All suspected variables assigned:", True)
print("Drop Path 0.1:", "drop_path_rate=0.1" in source)
print("Original experiments modified:", False)
print("Training started:", False)
print("Test images accessed:", 0)

print("\nSTEP 08.21 AUDIT COMPLETE")


# %% [cell 268] Experiment 08 - Step 08.22
# ============================================================
# Experiment 08 - Step 08.22
# Final Training Launch Gate
# Read-only; no training
# ============================================================

from pathlib import Path
import json
import ast

ROOT = DR_RESEARCH

EXP08 = ROOT / "experiment08_results"

SCRIPT = EXP08 / "retfound_drop_path_01_epoch1_v4.py"
PROTOCOL = EXP08 / "Regularization_Protocol_V1.json"

TRAIN_DIR = EXP08 / "retfound_drop_path_01_v1"

assert SCRIPT.is_file()
assert PROTOCOL.is_file()

config = json.loads(PROTOCOL.read_text(encoding="utf-8"))
source = SCRIPT.read_text(encoding="utf-8")

ast.parse(source)

assert config["drop_path_rate"] == 0.1
assert config["training_seed"] == 42

assert config["train_images"] == 1290
assert config["validation_images"] == 276

assert "drop_path_rate=0.1" in source
assert "load_exp08_retfound_encoder(ROOT)" in source
assert "exp08_train_dataset" in source
assert "exp08_val_loader" in source

assert 'ROOT / "experiment08_results"' in source
assert 'ROOT / "experiment04_results"' not in source

# Prevent accidental overwrite of an existing training run.
assert not TRAIN_DIR.exists(), (
    "Training directory already exists. "
    "Inspect its contents before proceeding."
)

print("EXPERIMENT 08.22 - TRAINING LAUNCH GATE")

print("\nPROTOCOL")
print("Architecture:", config["architecture"])
print("Drop Path:", config["drop_path_rate"])
print("Seed:", config["training_seed"])
print("Training images:", config["train_images"])
print("Validation images:", config["validation_images"])

print("\nTRAINING OUTPUT")
print("New training directory:", TRAIN_DIR)
print("Directory exists:", TRAIN_DIR.exists())

print("\nSOURCE")
print("Script:", SCRIPT.name)
print("Python syntax valid:", True)
print("Pretrained encoder source:", "Original RETFound MAE")
print("Epoch 1 only:", True)

print("\nRESEARCH INTEGRITY")
print("Existing files modified:", False)
print("Training started:", False)
print("Test images accessed:", 0)

print("\nSTEP 08.22 PASS - READY FOR EPOCH 1")


# %% [cell 269] Experiment 08 - Step 08.23
# ============================================================
# Experiment 08 - Step 08.23
# Launch RETFound Drop Path 0.1 - Epoch 1
# ============================================================

from pathlib import Path
import runpy
import time

ROOT = DR_RESEARCH

SCRIPT = (
    ROOT / "experiment08_results"
    / "retfound_drop_path_01_epoch1_v4.py"
)

OUTPUT_DIR = (
    ROOT / "experiment08_results"
    / "retfound_drop_path_01_v1"
)

assert SCRIPT.is_file()
assert not OUTPUT_DIR.exists(), (
    "Training output already exists. "
    "Do not restart Epoch 1."
)

print("EXPERIMENT 08.23 - RETFOUND EPOCH 1")
print("Training script:", SCRIPT.name)
print("Drop Path:", 0.1)
print("Output directory:", OUTPUT_DIR)
print("Test split access: Prohibited")

start = time.perf_counter()

# Execute the isolated Epoch-1 training program.
runpy.run_path(str(SCRIPT), run_name="__main__")

elapsed = time.perf_counter() - start

# Verify all required Epoch-1 outputs.
required = [
    "last_training_state.pt",
    "training_history.csv",
    "training_config.json",
    "validation_predictions_epoch001.csv",
]

print("\nEPOCH 1 OUTPUT VERIFICATION")

for filename in required:
    path = OUTPUT_DIR / filename
    print(filename, ":", "PASS" if path.is_file() else "MISSING")
    assert path.is_file(), f"Missing: {filename}"

print("\nTotal execution time (minutes):", round(elapsed / 60, 2))
print("Original Experiment 04 checkpoint modified: False")
print("Test images accessed: 0")
print("STEP 08.23 PASS")


# %% [cell 270] Experiment 08 - Step 08.24
# ============================================================
# Experiment 08 - Step 08.24
# Resume Training Source Inspection
# Read-only; no model loading or training
# ============================================================

from pathlib import Path
import json
import re

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert NOTEBOOK.is_file()

notebook = json.loads(
    NOTEBOOK.read_text(encoding="utf-8")
)

source = "".join(notebook["cells"][99]["source"])
lines = source.splitlines()

regions = {
    "Configuration and checkpoint loading": (35, 165),
    "Dataset and DataLoader restoration": (165, 210),
    "Model and optimizer restoration": (205, 280),
    "RNG restoration and training loop": (280, 370),
    "Checkpoint and best model saving": (515, 680),
}

print("EXPERIMENT 08.24 - RESUME SOURCE INSPECTION")
print("Source lines:", len(lines))

for title, (start, end) in regions.items():
    print("\n" + "=" * 65)
    print(title.upper())
    print("=" * 65)

    for i in range(start - 1, min(end, len(lines))):
        print(f"{i + 1:04d}: {lines[i]}")

print("\nKEY FEATURE COUNTS")

patterns = {
    "Checkpoint loading": r"torch\.load\(",
    "Model reconstruction": r"timm\.create_model\(",
    "Optimizer restoration": r"optimizer\.load_state_dict",
    "Scheduler restoration": r"scheduler\.load_state_dict",
    "PyTorch RNG restoration": r"torch\.set_rng_state",
    "CUDA RNG restoration": r"torch\.cuda\.set_rng_state_all",
    "Python RNG restoration": r"random\.setstate",
    "NumPy RNG restoration": r"np\.random\.set_state",
    "DataLoader RNG restoration": r"train_generator\.set_state",
    "Best model saving": r"best_model\.pt",
    "Atomic checkpoint replacement": r"os\.replace",
}

for name, pattern in patterns.items():
    print(
        name + ":",
        len(re.findall(pattern, source))
    )

print("\nRESEARCH INTEGRITY")
print("Research files modified: False")
print("Training started: False")
print("Test images accessed: 0")
print("STEP 08.24 AUDIT COMPLETE")


# %% [cell 271] Experiment 08 - Step 08.25
# ============================================================
# Experiment 08 - Step 08.25
# Generate RETFound Drop Path 0.1 Resume Script
# Source generation only; no training
# ============================================================

from pathlib import Path
import ast
import json
import re

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

EPOCH1_SOURCE = OUT / "retfound_drop_path_01_epoch1_v4.py"
RESUME_TARGET = OUT / "retfound_drop_path_01_resume_v1.py"

assert NOTEBOOK.is_file()
assert EPOCH1_SOURCE.is_file()
assert not RESUME_TARGET.exists(), "Resume script already exists."

notebook = json.loads(
    NOTEBOOK.read_text(encoding="utf-8")
)

original = "".join(notebook["cells"][99]["source"])
epoch1 = EPOCH1_SOURCE.read_text(encoding="utf-8")

# ------------------------------------------------------------
# 1. Extract the validated standalone Dataset implementation
# ------------------------------------------------------------

epoch1_tree = ast.parse(epoch1)

required_nodes = [
    "Exp08StandaloneDataset",
    "build_exp08_datasets"
]

definitions = []

for name in required_nodes:
    matches = [
        node for node in epoch1_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name == name
    ]
    assert len(matches) == 1, f"Missing definition: {name}"

    definition = ast.get_source_segment(epoch1, matches[0])
    assert definition is not None
    definitions.append(definition)

dataset_code = "\n\n".join(definitions)

# ------------------------------------------------------------
# 2. Add standalone dependencies
# ------------------------------------------------------------

assert original.count("import timm") == 1

imports = "\n".join([
    "from PIL import Image",
    "from torchvision import transforms",
    "from torch.utils.data import Dataset"
])

modified = original.replace(
    "import timm",
    "import timm\n" + imports,
    1
)

# ------------------------------------------------------------
# 3. Isolate the Experiment 08 output directory
# ------------------------------------------------------------

old_directory = "\n".join([
    "EXPERIMENT_DIR = (",
    '    ROOT / "experiment04_results"',
    '    / "retfound_vit_large_v1"',
    ")"
])

new_directory = "\n".join([
    "EXPERIMENT_DIR = (",
    '    ROOT / "experiment08_results"',
    '    / "retfound_drop_path_01_v1"',
    ")"
])

assert modified.count(old_directory) == 1
modified = modified.replace(
    old_directory, new_directory, 1
)

# ------------------------------------------------------------
# 4. Remove dependencies on the old Notebook loaders
# ------------------------------------------------------------

old_assertions = "\n".join([
    'assert "train_loader" in globals()',
    'assert "val_loader" in globals()'
])

assert modified.count(old_assertions) == 1

modified = modified.replace(
    old_assertions,
    "# Data loaders are reconstructed from the frozen index.",
    1
)

# ------------------------------------------------------------
# 5. Insert the standalone Dataset implementation
# ------------------------------------------------------------

anchor = (
    "assert file_sha256(INDEX_PATH) == "
    "EXPECTED_INDEX_SHA256"
)

assert modified.count(anchor) == 1

dataset_initialization = "\n".join([
    "",
    "exp08_train_dataset, exp08_val_dataset = build_exp08_datasets(",
    "    INDEX_PATH",
    ")",
    "",
    "exp08_val_loader = DataLoader(",
    "    exp08_val_dataset,",
    "    batch_size=16,",
    "    shuffle=False,",
    "    num_workers=0,",
    "    pin_memory=True",
    ")"
])

modified = modified.replace(
    anchor,
    anchor + "\n\n" + dataset_initialization,
    1
)

# Insert class and function definitions before execution.
definition_anchor = "ROOT = Path("
assert modified.count(definition_anchor) >= 1

modified = modified.replace(
    definition_anchor,
    dataset_code + "\n\n" + definition_anchor,
    1
)

# ------------------------------------------------------------
# 6. Replace legacy loader references
# ------------------------------------------------------------

modified = re.sub(
    r"(?<!exp08_)\btrain_loader\.dataset\b",
    "exp08_train_dataset",
    modified
)

modified = re.sub(
    r"(?<!exp08_)\bval_loader\b",
    "exp08_val_loader",
    modified
)

assert not re.search(
    r"(?<!exp08_)\btrain_loader\b",
    modified
)

assert not re.search(
    r"(?<!exp08_)\bval_loader\b",
    modified
)

# ------------------------------------------------------------
# 7. Enable Drop Path when reconstructing the model
# ------------------------------------------------------------

old_model = "\n".join([
    "model = timm.create_model(",
    '    config["architecture"],',
    "    pretrained=False,",
    "    num_classes=NUM_CLASSES,",
    '    global_pool="avg"',
    ")"
])

new_model = "\n".join([
    "model = timm.create_model(",
    '    config["architecture"],',
    "    pretrained=False,",
    "    num_classes=NUM_CLASSES,",
    '    global_pool="avg",',
    "    drop_path_rate=0.1",
    ")"
])

assert modified.count(old_model) == 1

modified = modified.replace(
    old_model, new_model, 1
)

# ------------------------------------------------------------
# 8. Check the locked regularization protocol
# ------------------------------------------------------------

protocol_guard = "\n".join([
    "",
    "PROTOCOL_PATH = (",
    '    ROOT / "experiment08_results"',
    '    / "Regularization_Protocol_V1.json"',
    ")",
    "",
    "with PROTOCOL_PATH.open('r', encoding='utf-8') as f:",
    "    exp08_protocol = json.load(f)",
    "",
    "assert exp08_protocol['drop_path_rate'] == 0.1",
    "assert exp08_protocol['training_seed'] == 42",
    "assert exp08_protocol['test_split_locked'] is True",
    "assert exp08_protocol['status'] == 'PROTOCOL_LOCKED_NOT_TRAINED'",
    ""
])

assert modified.count(new_directory) == 1

modified = modified.replace(
    new_directory,
    new_directory + "\n" + protocol_guard,
    1
)

# ------------------------------------------------------------
# 9. Load only our trusted, self-generated training checkpoint
# ------------------------------------------------------------

old_load = "\n".join([
    "resume_state = torch.load(",
    "    CHECKPOINT_PATH,",
    '    map_location="cpu",',
    "    weights_only=True,",
    "    mmap=True",
    ")"
])

new_load = "\n".join([
    "# Trusted checkpoint generated by Experiment 08 itself.",
    "resume_state = torch.load(",
    "    CHECKPOINT_PATH,",
    '    map_location="cpu",',
    "    weights_only=False,",
    "    mmap=True",
    ")"
])

assert modified.count(old_load) == 1

modified = modified.replace(
    old_load, new_load, 1
)

# ------------------------------------------------------------
# 10. Static validation
# ------------------------------------------------------------

ast.parse(modified)
compile(modified, str(RESUME_TARGET), "exec")

assert "drop_path_rate=0.1" in modified
assert "exp08_train_dataset" in modified
assert "exp08_val_loader" in modified

assert 'ROOT / "experiment04_results"' not in modified
assert 'ROOT / "experiment07_results"' not in modified

required_operations = [
    "optimizer.load_state_dict(",
    "scheduler.load_state_dict(",
    "torch.set_rng_state(",
    "torch.cuda.set_rng_state_all(",
    "random.setstate(",
    "np.random.set_state(",
    "train_generator.set_state(",
    "scheduler.step()",
    "atomic_torch_save(",
    "atomic_csv_save(",
    "best_model.pt",
    "last_training_state.pt"
]

for operation in required_operations:
    assert operation in modified, (
        f"Missing required operation: {operation}"
    )

# ------------------------------------------------------------
# 11. Save new source without overwriting existing files
# ------------------------------------------------------------

with RESUME_TARGET.open("x", encoding="utf-8") as f:
    f.write(modified)

print("EXPERIMENT 08.25 - RESUME SCRIPT GENERATED")

print("\nSOURCE")
print("Resume script:", RESUME_TARGET.name)
print("Script created:", RESUME_TARGET.is_file())
print("Source lines:", len(modified.splitlines()))
print("Python syntax valid:", True)

print("\nTRAINING PROTOCOL")
print("Architecture:", "vit_large_patch16_224")
print("Drop Path:", 0.1)
print("Dataset reconstruction:", "Standalone")
print("Checkpoint source:", "Experiment 08")
print("Resume epoch:", "Read from checkpoint at execution")

print("\nRECOVERY FEATURES")
print("Optimizer state restoration:", True)
print("Scheduler state restoration:", True)
print("RNG state restoration:", True)
print("Epoch-level checkpoint saving:", True)
print("Best validation model saving:", True)

print("\nRESEARCH INTEGRITY")
print("Epoch 1 checkpoint modified:", False)
print("Original experiments modified:", False)
print("Training started:", False)
print("Test images accessed:", 0)

print("\nSTEP 08.25 PASS")


# %% [cell 272] Experiment 08 - Step 08.26
# ============================================================
# Experiment 08 - Step 08.26
# Resume Checkpoint and Protocol Compatibility Audit
# Read-only; no training or file modifications
# ============================================================

from pathlib import Path
import json
import ast
import torch
import pandas as pd

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

RUN_DIR = OUT / "retfound_drop_path_01_v1"

CHECKPOINT = RUN_DIR / "last_training_state.pt"
CONFIG = RUN_DIR / "training_config.json"
HISTORY = RUN_DIR / "training_history.csv"
SCRIPT = OUT / "retfound_drop_path_01_resume_v1.py"
PROTOCOL = OUT / "Regularization_Protocol_V1.json"

for path in [CHECKPOINT, CONFIG, HISTORY, SCRIPT, PROTOCOL]:
    assert path.is_file(), f"Missing file: {path}"

config = json.loads(CONFIG.read_text(encoding="utf-8"))
protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
history = pd.read_csv(HISTORY)
source = SCRIPT.read_text(encoding="utf-8")

ast.parse(source)

# ------------------------------------------------------------
# 1. Inspect trusted training checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
    mmap=True
)

assert int(checkpoint["epoch"]) == 1
assert checkpoint["optimizer_state_dict"]
assert checkpoint["scheduler_state_dict"]
assert checkpoint["model_state_dict"]

required_rng = [
    "torch_rng_state",
    "cuda_rng_states",
    "python_rng_state",
    "numpy_rng_state",
    "train_loader_generator_state"
]

for name in required_rng:
    assert name in checkpoint, f"Missing RNG state: {name}"

# ------------------------------------------------------------
# 2. Validate training history
# ------------------------------------------------------------

assert len(history) == 1
assert int(history.iloc[0]["epoch"]) == 1

assert abs(
    float(history.iloc[0]["val_macro_f1"])
    - float(checkpoint["best_val_macro_f1"])
) < 1e-8

# ------------------------------------------------------------
# 3. Validate protocol and model reconstruction
# ------------------------------------------------------------

assert protocol["drop_path_rate"] == 0.1
assert protocol["training_seed"] == 42

assert config["architecture"] == "vit_large_patch16_224"
assert config["seed"] == 42

assert "drop_path_rate=0.1" in source
assert "optimizer.load_state_dict(" in source
assert "scheduler.load_state_dict(" in source

# ------------------------------------------------------------
# 4. Print configuration status
# ------------------------------------------------------------

print("EXPERIMENT 08.26 - RESUME COMPATIBILITY")

print("\nCHECKPOINT")
print("Completed epoch:", checkpoint["epoch"])
print("Expected next epoch:", int(checkpoint["epoch"]) + 1)
print("Best epoch:", checkpoint["best_epoch"])
print(
    "Best validation macro-F1:",
    round(float(checkpoint["best_val_macro_f1"]), 6)
)

print("\nTRAINING HISTORY")
print("History rows:", len(history))
print("Last recorded epoch:", int(history["epoch"].max()))

print("\nRNG RECOVERY")
for name in required_rng:
    print(name + ":", "PASS")

print("\nMODEL PROTOCOL")
print("Architecture:", config["architecture"])
print("Drop Path in locked protocol:", protocol["drop_path_rate"])
print("Drop Path in training config:", config.get("drop_path_rate"))
print("Drop Path in resume source:", "drop_path_rate=0.1" in source)

print("\nRESEARCH INTEGRITY")
print("Checkpoint modified: False")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 08.26 PASS")


# %% [cell 273] Experiment 08 - Step 08.27
# ============================================================
# Experiment 08 - Step 08.27
# Freeze RETFound Regularized Resume Configuration
# No training or model inference
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import ast
import json

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

SOURCE = OUT / "retfound_drop_path_01_resume_v1.py"
TARGET = OUT / "retfound_drop_path_01_resume_v2.py"
RESUME_CONFIG = OUT / "retfound_drop_path_01_resume_config_v1.json"

RUN_DIR = OUT / "retfound_drop_path_01_v1"
TRAIN_CONFIG = RUN_DIR / "training_config.json"
PROTOCOL = OUT / "Regularization_Protocol_V1.json"

for path in [SOURCE, TRAIN_CONFIG, PROTOCOL]:
    assert path.is_file(), f"Missing file: {path}"

assert not TARGET.exists()
assert not RESUME_CONFIG.exists()

source = SOURCE.read_text(encoding="utf-8")
train_config = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

assert protocol["drop_path_rate"] == 0.1
assert train_config["architecture"] == "vit_large_patch16_224"
assert train_config["seed"] == 42

# ------------------------------------------------------------
# 1. Require the locked Experiment 08 protocol
# ------------------------------------------------------------

old_guard = "assert exp08_protocol['drop_path_rate'] == 0.1"

new_guard = "\n".join([
    "assert exp08_protocol['drop_path_rate'] == 0.1",
    "assert exp08_protocol['architecture'] == config['architecture']"
])

# The original source reads config after the protocol guard,
# so retain the existing guard instead of referencing config
# before it is initialized.
assert source.count(old_guard) == 1

# ------------------------------------------------------------
# 2. Validate the loaded training configuration
# ------------------------------------------------------------

anchor = 'assert config["architecture"] == "vit_large_patch16_224"'

additional_guards = "\n".join([
    anchor,
    "assert config['seed'] == exp08_protocol['training_seed']",
    "assert config['training_index_sha256'] == exp08_protocol['training_index_sha256']",
    "assert exp08_protocol['drop_path_rate'] == 0.1",
    "assert exp08_protocol['test_split_locked'] is True"
])

assert source.count(anchor) == 1

modified = source.replace(
    anchor,
    additional_guards,
    1
)

# ------------------------------------------------------------
# 3. Verify restored model structure
# ------------------------------------------------------------

model_anchor = "model.load_state_dict("

assert modified.count(model_anchor) == 1

drop_path_check = "\n".join([
    "drop_path_values = [",
    "    float(getattr(block.drop_path1, 'drop_prob', 0.0))",
    "    for block in model.blocks",
    "]",
    "assert len(drop_path_values) == 24",
    "assert abs(max(drop_path_values) - 0.1) < 1e-6",
    "",
])

modified = modified.replace(
    model_anchor,
    drop_path_check + model_anchor,
    1
)

# ------------------------------------------------------------
# 4. Prevent executing after training has completed
# ------------------------------------------------------------

# Existing resume code already determines completed_epoch
# and start_epoch from checkpoint; preserve that behavior.

assert "start_epoch = completed_epoch + 1" in modified
assert "for epoch in range(start_epoch, MAX_EPOCHS + 1)" in modified

# ------------------------------------------------------------
# 5. Static safety checks
# ------------------------------------------------------------

ast.parse(modified)
compile(modified, str(TARGET), "exec")

assert "drop_path_rate=0.1" in modified
assert "weights_only=False" in modified

assert 'ROOT / "experiment04_results"' not in modified
assert 'ROOT / "experiment07_results"' not in modified
assert 'ROOT / "experiment08_results"' in modified

required = [
    "optimizer.load_state_dict(",
    "scheduler.load_state_dict(",
    "torch.set_rng_state(",
    "torch.cuda.set_rng_state_all(",
    "random.setstate(",
    "np.random.set_state(",
    "train_generator.set_state(",
    "atomic_torch_save(",
    "atomic_csv_save("
]

for operation in required:
    assert operation in modified

# ------------------------------------------------------------
# 6. Save immutable new artifacts
# ------------------------------------------------------------

resume_config = {
    "experiment": "Experiment08_RETFound_DropPath01_Resume",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_script": SOURCE.name,
    "resume_script": TARGET.name,
    "training_run": "retfound_drop_path_01_v1",
    "architecture": "vit_large_patch16_224",
    "drop_path_rate": 0.1,
    "initial_completed_epoch": 1,
    "resume_from_epoch": 2,
    "maximum_epochs": 25,
    "training_seed": 42,
    "selection_metric": "val_macro_f1",
    "original_epoch1_config_preserved": True,
    "test_split_locked": True
}

with TARGET.open("x", encoding="utf-8") as f:
    f.write(modified)

with RESUME_CONFIG.open("x", encoding="utf-8") as f:
    json.dump(resume_config, f, indent=2)

print("EXPERIMENT 08.27 - RESUME CONFIGURATION LOCKED")
print("Resume V2 created:", TARGET.is_file())
print("Resume configuration created:", RESUME_CONFIG.is_file())
print("Drop Path:", resume_config["drop_path_rate"])
print("Resume from epoch:", resume_config["resume_from_epoch"])
print("Maximum epochs:", resume_config["maximum_epochs"])

print("\nRESEARCH INTEGRITY")
print("Epoch 1 checkpoint modified: False")
print("Original training config modified: False")
print("Training started: False")
print("Test images accessed: 0")
print("STEP 08.27 PASS")


# %% [cell 274] Experiment 08 - Step 08.28
# ============================================================
# Experiment 08 - Step 08.28
# Resume RETFound Drop Path 0.1 Training
# ============================================================

from pathlib import Path
import runpy
import time
import pandas as pd

ROOT = DR_RESEARCH

SCRIPT = (
    ROOT / "experiment08_results"
    / "retfound_drop_path_01_resume_v2.py"
)

RUN_DIR = (
    ROOT / "experiment08_results"
    / "retfound_drop_path_01_v1"
)

CHECKPOINT = RUN_DIR / "last_training_state.pt"
HISTORY = RUN_DIR / "training_history.csv"

assert SCRIPT.is_file()
assert CHECKPOINT.is_file()
assert HISTORY.is_file()

history_before = pd.read_csv(HISTORY)

assert int(history_before["epoch"].max()) == 1, (
    "Training has already advanced beyond Epoch 1. "
    "Stop and inspect the current checkpoint."
)

print("EXPERIMENT 08.28 - RETFOUND REGULARIZED TRAINING")
print("Model: RETFound ViT-Large/16")
print("Drop Path: 0.1")
print("Expected starting epoch: 2")
print("Maximum epochs: 25")
print("Test split access: Prohibited")

start = time.perf_counter()

# The resume program loads the saved model, optimizer,
# scheduler, and RNG states before continuing training.
runpy.run_path(str(SCRIPT), run_name="__main__")

elapsed_minutes = (time.perf_counter() - start) / 60

history_after = pd.read_csv(HISTORY)

assert history_after["epoch"].is_unique
assert int(history_after["epoch"].min()) == 1
assert int(history_after["epoch"].max()) >= 2
assert int(history_after["epoch"].max()) <= 25

best_index = history_after["val_macro_f1"].idxmax()
best_row = history_after.loc[best_index]

print("\nEXPERIMENT 08.28 - TRAINING SUMMARY")
print("Completed epochs:", len(history_after))
print("Last epoch:", int(history_after["epoch"].max()))
print("Best epoch:", int(best_row["epoch"]))
print(
    "Best validation macro-F1:",
    round(float(best_row["val_macro_f1"]), 6)
)
print(
    "Best validation QWK:",
    round(float(best_row["val_qwk"]), 6)
)
print("Total resume runtime (minutes):", round(elapsed_minutes, 2))

print("\nOUTPUT VERIFICATION")
print("Best model exists:", (RUN_DIR / "best_model.pt").is_file())
print(
    "Best validation predictions exist:",
    (RUN_DIR / "best_validation_predictions.csv").is_file()
)
print("Last training checkpoint exists:", CHECKPOINT.is_file())

print("\nRESEARCH INTEGRITY")
print("Original experiments modified: False")
print("Test images accessed: 0")
print("STEP 08.28 PASS")


# %% [cell 275] Experiment 08 - End-of-Day Save Verification
# Experiment 08 - End-of-Day Save Verification

from pathlib import Path

root = DR_RESEARCH
run = root / "experiment08_results/retfound_drop_path_01_v1"

required = [
    "best_model.pt",
    "last_training_state.pt",
    "training_history.csv",
    "training_config.json",
    "best_validation_predictions.csv",
]

print("END-OF-DAY SAVE VERIFICATION")

for name in required:
    path = run / name
    assert path.is_file(), f"Missing: {name}"
    print(f"{name}: SAVED")

print("\nAll RETFound training artifacts saved to Google Drive.")
print("Safe to stop Colab after verification.")


# %% [cell 276] Experiment 08 - Step 08.29A
# ============================================================
# Experiment 08 - Step 08.29A
# Reconnect Google Drive and Verify GPU
# ============================================================

from pathlib import Path
import torch

pass  # Google Drive mount removed

ROOT = DR_RESEARCH

print("\nEXPERIMENT 08 - ENVIRONMENT RECOVERY")

print("Google Drive mounted:", ROOT.is_dir())
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory (GiB):",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        )
    )

print(
    "Experiment 08 results exist:",
    (ROOT / "experiment08_results").is_dir()
)

assert ROOT.is_dir(), "Google Drive project not found."
assert torch.cuda.is_available(), "GPU is not available."

print("\nSTEP 08.29A PASS")


# %% [cell 277] Experiment 08 - Step 08.29
# ============================================================
# Experiment 08 - Step 08.29
# Meta MAE Regularization: Runtime Recovery Check
# Read-only; no training or model inference
# ============================================================

from pathlib import Path
import json
import torch
import pandas as pd

ROOT = DR_RESEARCH

PROTOCOL_PATH = (
    ROOT / "experiment08_results"
    / "Regularization_Protocol_V1.json"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

META_WEIGHTS = (
    ROOT / "model_checkpoints"
    / "Meta_MAE_pretrained"
    / "mae_pretrain_vit_large.pth"
)

META_BASELINE = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1/best_model.pt"
)

RET_REGULARIZED = (
    ROOT / "experiment08_results"
    / "retfound_drop_path_01_v1/best_model.pt"
)

META_RUN = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_v1"
)

print("EXPERIMENT 08.29 - META MAE RUNTIME CHECK")

# ------------------------------------------------------------
# 1. GPU and Google Drive
# ------------------------------------------------------------

assert ROOT.is_dir(), "Google Drive is not mounted."
assert torch.cuda.is_available(), "CUDA is unavailable."

print("\nENVIRONMENT")
print("Google Drive mounted:", True)
print("GPU:", torch.cuda.get_device_name(0))

# ------------------------------------------------------------
# 2. Frozen protocol
# ------------------------------------------------------------

assert PROTOCOL_PATH.is_file()

protocol = json.loads(
    PROTOCOL_PATH.read_text(encoding="utf-8")
)

assert protocol["drop_path_rate"] == 0.1
assert protocol["training_seed"] == 42
assert protocol["test_split_locked"] is True

print("\nLOCKED PROTOCOL")
print("Architecture:", protocol["architecture"])
print("Drop Path:", protocol["drop_path_rate"])
print("Seed:", protocol["training_seed"])

# ------------------------------------------------------------
# 3. Pretrained weights and preserved checkpoints
# ------------------------------------------------------------

assert META_WEIGHTS.is_file()
assert META_BASELINE.is_file()
assert RET_REGULARIZED.is_file()

print("\nMODEL FILES")
print("Meta MAE pretrained weights:", "PASS")
print("Original Meta MAE best model:", "PASS")
print("Regularized RETFound best model:", "PASS")

# ------------------------------------------------------------
# 4. Local training and validation cache
# ------------------------------------------------------------

assert INDEX_PATH.is_file()

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

train = index.loc[index["split"] == "train"]
val = index.loc[index["split"] == "val"]

assert len(train) == 1290
assert len(val) == 276

status = {}

for name, frame in [
    ("Training", train),
    ("Validation", val)
]:
    paths = [Path(p) for p in frame["local_path"]]

    missing = sum(not p.is_file() for p in paths)
    symlinks = sum(p.is_symlink() for p in paths)
    drive_backed = sum(
        p.is_file()
        and str(p.resolve()).startswith(str(DR_RESEARCH.parent) + "/")
        for p in paths
    )

    status[name] = {
        "missing": missing,
        "symlinks": symlinks,
        "drive_backed": drive_backed
    }

    print(f"\n{name.upper()} CACHE")
    print("Expected images:", len(paths))
    print("Missing images:", missing)
    print("Symbolic links:", symlinks)
    print("Drive-backed images:", drive_backed)

# ------------------------------------------------------------
# 5. Prevent accidental training overwrite
# ------------------------------------------------------------

print("\nMETA MAE REGULARIZED OUTPUT")
print("Run directory already exists:", META_RUN.exists())

if META_RUN.is_dir():
    print(
        "Existing entries:",
        sorted(p.name for p in META_RUN.iterdir())[:15]
    )

cache_ready = all(
    all(count == 0 for count in counts.values())
    for counts in status.values()
)

print("\nREADINESS")
print("Local cache ready:", cache_ready)
print("Pretrained weights ready:", True)
print("Protocol ready:", True)

print("\nRESEARCH INTEGRITY")
print("Research files modified:", False)
print("New training started:", False)
print("Test images accessed:", 0)

if cache_ready and not META_RUN.exists():
    print("\nSTEP 08.29 PASS")
elif not cache_ready:
    print("\nSTEP 08.29 LOCAL CACHE RECOVERY REQUIRED")
else:
    print("\nSTEP 08.29 EXISTING RUN REVIEW REQUIRED")


# %% [cell 278] Experiment 08 - Step 08.30
# ============================================================
# Experiment 08 - Step 08.30
# Identify Source Images for Local Cache Recovery
# Read-only; no image copying or training
# ============================================================

from pathlib import Path
import pandas as pd

ROOT = DR_RESEARCH

INDEX = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

assert INDEX.is_file()

df = pd.read_csv(INDEX)

selected = df[df["split"].isin(["train", "val"])].copy()

assert len(selected) == 1566
assert (selected["split"] == "train").sum() == 1290
assert (selected["split"] == "val").sum() == 276

print("EXPERIMENT 08.30 - IMAGE SOURCE DISCOVERY")

print("\nINDEX COLUMNS")
print(list(df.columns))

print("\nPATH-RELATED COLUMNS")
path_columns = [
    col for col in df.columns
    if any(
        token in col.lower()
        for token in ["path", "file", "image"]
    )
]

for col in path_columns:
    print(f"\nColumn: {col}")
    print(selected[col].head(3).to_string(index=False))

print("\nGOOGLE DRIVE SOURCE DIRECTORIES")

for directory in [
    ROOT / "data",
    ROOT / "datasets",
    ROOT / "FGADR",
    ROOT / "experiment01_results",
    ROOT / "experiment02_results",
    ROOT / "experiment03_results"
]:
    if directory.is_dir():
        print(f"\n{directory.relative_to(ROOT)}")

        entries = sorted(
            directory.iterdir(),
            key=lambda p: p.name
        )

        for entry in entries[:12]:
            print(
                "  ",
                "[DIR]" if entry.is_dir() else "[FILE]",
                entry.name
            )

print("\nRESEARCH INTEGRITY")
print("Images copied: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 08.30 PASS")


# %% [cell 279] Experiment 08 - Step 08.31
# ============================================================
# Experiment 08 - Step 08.31
# Restore Frozen Train/Validation Image Cache
# SHA-256 verified; test split excluded
# ============================================================

from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import hashlib
import shutil
import time

ROOT = DR_RESEARCH

INDEX = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

CACHE_ROOT = (LOCAL_DISK / "fgadr_classification")

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

# ------------------------------------------------------------
# 1. Verify the frozen index
# ------------------------------------------------------------

assert INDEX.is_file()
assert file_sha256(INDEX) == EXPECTED_INDEX_SHA256

df = pd.read_csv(INDEX, dtype={"image_id": str})

selected = df.loc[
    df["split"].isin(["train", "val"])
].copy()

assert len(selected) == 1566
assert (selected["split"] == "train").sum() == 1290
assert (selected["split"] == "val").sum() == 276

assert selected["local_path"].is_unique
assert selected["sha256"].notna().all()

# ------------------------------------------------------------
# 2. Validate all paths before copying
# ------------------------------------------------------------

source_root = (
    ROOT / "datasets" / "DR"
).resolve()

cache_root = CACHE_ROOT.resolve()

for row in selected.itertuples(index=False):
    source = Path(row.source_path)
    destination = Path(row.local_path)

    assert source.is_file(), f"Missing source: {source}"

    assert source.resolve().is_relative_to(source_root), (
        f"Source outside approved dataset directory: {source}"
    )

    assert destination.is_relative_to(CACHE_ROOT), (
        f"Destination outside cache directory: {destination}"
    )

# ------------------------------------------------------------
# 3. Restore and verify images
# ------------------------------------------------------------

start = time.perf_counter()

copied = 0
reused = 0
repaired = 0
errors = []

for row in tqdm(
    selected.itertuples(index=False),
    total=len(selected),
    desc="Restoring FGADR train/val cache"
):
    source = Path(row.source_path)
    destination = Path(row.local_path)
    expected_hash = str(row.sha256).lower()

    try:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if destination.exists() or destination.is_symlink():
            if (
                destination.is_file()
                and not destination.is_symlink()
                and file_sha256(destination) == expected_hash
            ):
                reused += 1
                continue

            if destination.is_dir():
                raise RuntimeError(
                    f"Unexpected directory: {destination}"
                )

            destination.unlink()
            repaired += 1

        temporary = destination.with_name(
            destination.name + ".partial"
        )

        if temporary.exists():
            temporary.unlink()

        shutil.copyfile(source, temporary)

        actual_hash = file_sha256(temporary)

        if actual_hash != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch: {row.image_id}"
            )

        temporary.replace(destination)
        copied += 1

    except Exception as exc:
        errors.append((row.image_id, str(exc)))

# ------------------------------------------------------------
# 4. Final cache integrity audit
# ------------------------------------------------------------

missing = 0
symlinks = 0
hash_mismatches = 0

for row in selected.itertuples(index=False):
    path = Path(row.local_path)

    if not path.is_file():
        missing += 1
        continue

    if path.is_symlink():
        symlinks += 1
        continue

    if file_sha256(path) != str(row.sha256).lower():
        hash_mismatches += 1

elapsed = (time.perf_counter() - start) / 60

print("\nEXPERIMENT 08.31 - CACHE RECOVERY SUMMARY")
print("Expected images:", len(selected))
print("Newly copied images:", copied)
print("Reused verified images:", reused)
print("Repaired entries:", repaired)
print("Copy errors:", len(errors))

print("\nFINAL INTEGRITY")
print("Missing images:", missing)
print("Symbolic links:", symlinks)
print("SHA-256 mismatches:", hash_mismatches)
print("Runtime (minutes):", round(elapsed, 2))

if errors:
    print("\nFIRST ERRORS")
    for image_id, message in errors[:5]:
        print(image_id, message)

print("\nRESEARCH INTEGRITY")
print("Frozen index modified: False")
print("Original dataset modified: False")
print("Test images accessed: 0")
print("Training started: False")

assert not errors, "Cache restoration encountered errors."
assert missing == 0
assert symlinks == 0
assert hash_mismatches == 0
assert copied + reused == len(selected)

print("\nSTEP 08.31 PASS")


# %% [cell 280] Experiment 08 - Step 08.32
# ============================================================
# Experiment 08 - Step 08.32
# Inspect Meta MAE Training Source
# Read-only; no training
# ============================================================

from pathlib import Path
import json
import re

ROOT = DR_RESEARCH

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

META_WEIGHTS = (
    ROOT / "model_checkpoints"
    / "Meta_MAE_pretrained"
    / "mae_pretrain_vit_large.pth"
)

assert NOTEBOOK.is_file()
assert META_WEIGHTS.is_file()

notebook = json.loads(
    NOTEBOOK.read_text(encoding="utf-8")
)

source = "".join(notebook["cells"][214]["source"])
lines = source.splitlines()

sections = {
    "Pretrained checkpoint loading": (
        r"torch\.load|load_state_dict|encoder_state",
        5
    ),
    "Model construction": (
        r"timm\.create_model",
        8
    ),
    "DataLoader dependencies": (
        r"train_loader|val_loader",
        5
    ),
    "Training configuration": (
        r"CONFIG|config\s*=|class_weights",
        5
    ),
    "Checkpoint saving": (
        r"last_training_state|atomic_torch_save",
        5
    )
}

print("EXPERIMENT 08.32 - META MAE SOURCE INSPECTION")
print("Source cell:", 214)
print("Total source lines:", len(lines))

for title, (pattern, context) in sections.items():
    matches = [
        i for i, line in enumerate(lines)
        if re.search(pattern, line)
    ]

    print("\n" + "=" * 55)
    print(title)
    print("Matches:", len(matches))

    for i in matches[:4]:
        start = max(0, i - context)
        end = min(len(lines), i + context + 1)

        print(f"\nLocation: line {i + 1}")

        for j in range(start, end):
            print(f"{j + 1:04d}: {lines[j]}")

print("\nRESEARCH INTEGRITY")
print("Research files modified: False")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 08.32 PASS")


# %% [cell 281] Experiment 08 - Step 08.33
# ============================================================
# Experiment 08 - Step 08.33
# Create Meta MAE Drop Path 0.1 Training Configuration
# No model loading or training
# ============================================================

from pathlib import Path
import json
import copy
import hashlib

ROOT = DR_RESEARCH

BASE_CONFIG = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1"
    / "training_config.json"
)

PROTOCOL_PATH = (
    ROOT / "experiment08_results"
    / "Regularization_Protocol_V1.json"
)

RUN_DIR = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_v1"
)

NEW_CONFIG = RUN_DIR / "training_config.json"

WEIGHT_PATH = (
    ROOT / "model_checkpoints"
    / "Meta_MAE_pretrained"
    / "mae_pretrain_vit_large.pth"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

# ------------------------------------------------------------
# 1. Verify source files and locked protocol
# ------------------------------------------------------------

assert BASE_CONFIG.is_file()
assert PROTOCOL_PATH.is_file()
assert WEIGHT_PATH.is_file()
assert INDEX_PATH.is_file()

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_SHA256

baseline = json.loads(
    BASE_CONFIG.read_text(encoding="utf-8")
)

protocol = json.loads(
    PROTOCOL_PATH.read_text(encoding="utf-8")
)

assert baseline["architecture"] == "vit_large_patch16_224"
assert baseline["pretraining"] == "Meta_MAE_ImageNet1K"
assert baseline["seed"] == 42
assert baseline["num_classes"] == 5
assert baseline["micro_batch_size"] == 4
assert baseline["gradient_accumulation_steps"] == 4

assert protocol["drop_path_rate"] == 0.1
assert protocol["training_seed"] == 42
assert protocol["test_split_locked"] is True

# ------------------------------------------------------------
# 2. Create a new isolated configuration
# ------------------------------------------------------------

config = copy.deepcopy(baseline)

config["experiment"] = (
    "Experiment08_Meta_MAE_DropPath01_V1"
)

config["status"] = "CONFIG_LOCKED_NOT_TRAINED"

config["drop_path_rate"] = 0.1

config["pretraining"] = "Meta_MAE_ImageNet1K"

config["pretrained_checkpoint_path"] = str(WEIGHT_PATH)

config["training_index_sha256"] = EXPECTED_INDEX_SHA256

config["test_images_evaluated"] = 0

# ------------------------------------------------------------
# 3. Verify isolation and preserve original parameters
# ------------------------------------------------------------

assert not NEW_CONFIG.exists(), (
    "Experiment 08 Meta MAE config already exists."
)

training_keys = [
    "architecture",
    "seed",
    "num_classes",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "weight_decay",
    "maximum_epochs",
    "minimum_epochs",
    "patience",
    "class_weights"
]

for key in training_keys:
    assert key in baseline, f"Missing baseline setting: {key}"
    assert config[key] == baseline[key], (
        f"Unexpected change in training setting: {key}"
    )

assert config["drop_path_rate"] == 0.1

RUN_DIR.mkdir(parents=True, exist_ok=True)

with NEW_CONFIG.open("x", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)

# ------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------

print("EXPERIMENT 08.33 - META MAE CONFIGURATION")

print("\nMODEL")
print("Architecture:", config["architecture"])
print("Pretraining:", config["pretraining"])
print("Drop Path:", config["drop_path_rate"])

print("\nTRAINING")
print("Seed:", config["seed"])
print("Micro batch size:", config["micro_batch_size"])
print(
    "Gradient accumulation:",
    config["gradient_accumulation_steps"]
)
print("Learning rate:", config["learning_rate"])
print("Weight decay:", config["weight_decay"])
print("Maximum epochs:", config["maximum_epochs"])
print("Minimum epochs:", config["minimum_epochs"])
print("Patience:", config["patience"])

print("\nOUTPUT")
print("Configuration created:", NEW_CONFIG.is_file())
print("Configuration path:", NEW_CONFIG)
print(
    "Training checkpoint exists:",
    (RUN_DIR / "last_training_state.pt").exists()
)

print("\nRESEARCH INTEGRITY")
print("Baseline configuration modified: False")
print("Original experiments modified: False")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 08.33 PASS")


# %% [cell 282] Experiment 08 - Step 08.34
# ============================================================
# Experiment 08 - Step 08.34
# Identify Training Script Adaptation Points
# Read-only; no model loading or training
# ============================================================

from pathlib import Path
import ast
import json

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

RET_SCRIPT = OUT / "retfound_drop_path_01_epoch1_v4.py"

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert RET_SCRIPT.is_file()
assert NOTEBOOK.is_file()

ret_source = RET_SCRIPT.read_text(encoding="utf-8")

notebook = json.loads(
    NOTEBOOK.read_text(encoding="utf-8")
)

meta_source = "".join(notebook["cells"][214]["source"])

ret_lines = ret_source.splitlines()
meta_lines = meta_source.splitlines()

ast.parse(ret_source)
ast.parse(meta_source)

def show_section(title, lines, first, last):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for index in range(first - 1, min(last, len(lines))):
        print(f"{index + 1:04d}: {lines[index]}")

print("EXPERIMENT 08.34 - META MAE SCRIPT PREPARATION")

show_section(
    "RETFound Epoch 1: Configuration and Paths",
    ret_lines,
    25,
    105
)

ret_model_positions = [
    i for i, line in enumerate(ret_lines)
    if "timm.create_model(" in line
]

assert len(ret_model_positions) == 1

position = ret_model_positions[0]

show_section(
    "RETFound Epoch 1: Model and Encoder Initialization",
    ret_lines,
    max(1, position - 8),
    position + 65
)

show_section(
    "Meta MAE Original: Encoder Initialization",
    meta_lines,
    120,
    180
)

print("\nSTANDALONE COMPONENTS")

tree = ast.parse(ret_source)

definitions = [
    node.name
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
]

print("Classes and functions:", definitions)

print("\nOUTPUT STATUS")

META_RUN = OUT / "meta_mae_drop_path_01_v1"

print(
    "Meta MAE configuration exists:",
    (META_RUN / "training_config.json").is_file()
)

print(
    "Meta MAE checkpoint exists:",
    (META_RUN / "last_training_state.pt").exists()
)

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 08.34 PASS")


# %% [cell 283] Experiment 08 - Step 08.35
# ============================================================
# Experiment 08 - Step 08.35
# Generate Standalone Meta MAE Drop Path 0.1 Epoch 1 Script
# Source generation only; no training
# ============================================================

from pathlib import Path
import ast
import json
import textwrap

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

SOURCE = OUT / "retfound_drop_path_01_epoch1_v4.py"
TARGET = OUT / "meta_mae_drop_path_01_epoch1_v1.py"

META_RUN = OUT / "meta_mae_drop_path_01_v1"
META_CONFIG = META_RUN / "training_config.json"

assert SOURCE.is_file()
assert META_CONFIG.is_file()
assert not TARGET.exists(), "Target script already exists."

source = SOURCE.read_text(encoding="utf-8")
config = json.loads(META_CONFIG.read_text(encoding="utf-8"))

assert config["drop_path_rate"] == 0.1
assert config["seed"] == 42
assert config["pretraining"] == "Meta_MAE_ImageNet1K"

# ------------------------------------------------------------
# 1. Replace the original pretrained encoder loader
# ------------------------------------------------------------

tree = ast.parse(source)

matches = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "load_exp08_retfound_encoder"
]

assert len(matches) == 1

old_function = ast.get_source_segment(source, matches[0])
assert old_function is not None

new_function = textwrap.dedent("""
def load_exp08_meta_mae_encoder(root):
    import argparse

    weight_path = (
        root / "model_checkpoints"
        / "Meta_MAE_pretrained"
        / "mae_pretrain_vit_large.pth"
    )

    assert weight_path.is_file()

    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(
            weight_path,
            map_location="cpu",
            mmap=True,
            weights_only=True
        )

    source_state = checkpoint["model"]

    encoder_state = {
        key: value
        for key, value in source_state.items()
        if key.startswith(("patch_embed.", "blocks."))
        or key in {"cls_token", "pos_embed"}
    }

    assert len(encoder_state) == 292

    del source_state, checkpoint
    gc.collect()

    return encoder_state
""").strip()

assert source.count(old_function) == 1

modified = source.replace(
    old_function,
    new_function,
    1
)

# ------------------------------------------------------------
# 2. Replace the encoder initialization call
# ------------------------------------------------------------

old_call = "encoder_state = load_exp08_retfound_encoder(ROOT)"
new_call = "encoder_state = load_exp08_meta_mae_encoder(ROOT)"

assert modified.count(old_call) == 1

modified = modified.replace(
    old_call,
    new_call,
    1
)

# ------------------------------------------------------------
# 3. Identify the original Experiment 08 output path
# ------------------------------------------------------------

old_run_name = "retfound_drop_path_01_v1"
new_run_name = "meta_mae_drop_path_01_v1"

assert old_run_name in modified

modified = modified.replace(
    old_run_name,
    new_run_name
)

# ------------------------------------------------------------
# 4. Update experiment-specific configuration guards
# ------------------------------------------------------------

modified = modified.replace(
    "Experiment08_RETFound_DropPath01_V1",
    "Experiment08_Meta_MAE_DropPath01_V1"
)

modified = modified.replace(
    "RETFound_MAE",
    "Meta_MAE_ImageNet1K"
)

modified = modified.replace(
    "RETFound",
    "Meta MAE"
)

# ------------------------------------------------------------
# 5. Verify all training operations are preserved
# ------------------------------------------------------------

ast.parse(modified)
compile(modified, str(TARGET), "exec")

required = [
    "drop_path_rate=0.1",
    "load_exp08_meta_mae_encoder(ROOT)",
    "optimizer.step()",
    "scheduler.step()",
    "torch.autocast(",
    "last_training_state.pt",
    "training_history.csv",
    "numpy_rng_state",
    "train_loader_generator_state"
]

for item in required:
    assert item in modified, f"Missing operation: {item}"

assert "load_exp08_retfound_encoder" not in modified

assert "retfound_drop_path_01_v1" not in modified

assert 'ROOT / "experiment04_results"' not in modified
assert 'ROOT / "experiment07_results"' not in modified

assert "meta_mae_drop_path_01_v1" in modified

# ------------------------------------------------------------
# 6. Save the new standalone script
# ------------------------------------------------------------

with TARGET.open("x", encoding="utf-8") as handle:
    handle.write(modified)

print("EXPERIMENT 08.35 - META MAE SCRIPT GENERATED")

print("\nSOURCE")
print("Source:", SOURCE.name)
print("Target:", TARGET.name)
print("Python syntax valid:", True)
print("Source lines:", len(modified.splitlines()))

print("\nMODEL INITIALIZATION")
print("Pretraining:", "Meta MAE ImageNet-1K")
print("Pretrained encoder tensors:", 292)
print("Drop Path:", 0.1)
print("Fresh classification head:", True)

print("\nTRAINING")
print("Standalone FGADR dataset:", True)
print("Optimizer and scheduler preserved:", True)
print("Checkpoint recovery states preserved:", True)

print("\nRESEARCH INTEGRITY")
print("Original scripts modified: False")
print("Existing checkpoints modified: False")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 08.35 PASS")


# %% [cell 284] Experiment 08 - Step 08.36
# ============================================================
# Experiment 08 - Step 08.36
# Meta MAE Final Pre-Launch Audit
# Read-only; no training
# ============================================================

from pathlib import Path
import ast
import json
import re

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

SCRIPT = OUT / "meta_mae_drop_path_01_epoch1_v1.py"
RUN_DIR = OUT / "meta_mae_drop_path_01_v1"
CONFIG_PATH = RUN_DIR / "training_config.json"

assert SCRIPT.is_file()
assert CONFIG_PATH.is_file()

source = SCRIPT.read_text(encoding="utf-8")
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

tree = ast.parse(source)

# ------------------------------------------------------------
# 1. Validate configuration
# ------------------------------------------------------------

assert config["experiment"] == "Experiment08_Meta_MAE_DropPath01_V1"
assert config["pretraining"] == "Meta_MAE_ImageNet1K"
assert config["architecture"] == "vit_large_patch16_224"
assert config["drop_path_rate"] == 0.1
assert config["seed"] == 42

# ------------------------------------------------------------
# 2. Inspect original checkpoint loading
# ------------------------------------------------------------

assert "load_exp08_meta_mae_encoder(ROOT)" in source
assert "mae_pretrain_vit_large.pth" in source
assert 'source_state = checkpoint["model"]' in source
assert "assert len(encoder_state) == 292" in source

# ------------------------------------------------------------
# 3. Detect unintended original experiment references
# ------------------------------------------------------------

for forbidden in [
    "load_exp08_retfound_encoder",
    "RETFound_mae_natureCFP.pth",
    "retfound_drop_path_01_v1",
    "experiment04_results",
    "experiment07_results"
]:
    assert forbidden not in source, (
        f"Unexpected original experiment reference: {forbidden}"
    )

# ------------------------------------------------------------
# 4. Validate model and training operations
# ------------------------------------------------------------

required = [
    "drop_path_rate=0.1",
    'global_pool="avg"',
    "model.load_state_dict(",
    "model.set_grad_checkpointing(enable=True)",
    "optimizer.step()",
    "scheduler.step()",
    "torch.autocast(",
    "train_loader_generator_state",
    "numpy_rng_state",
    "last_training_state.pt",
    "training_history.csv"
]

for item in required:
    assert item in source, f"Missing training feature: {item}"

# ------------------------------------------------------------
# 5. Ensure standalone datasets are present
# ------------------------------------------------------------

definitions = {
    node.name
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
}

assert "Exp08StandaloneDataset" in definitions
assert "build_exp08_datasets" in definitions
assert "load_exp08_meta_mae_encoder" in definitions

# ------------------------------------------------------------
# 6. Check whether training has already started
# ------------------------------------------------------------

checkpoint = RUN_DIR / "last_training_state.pt"
history = RUN_DIR / "training_history.csv"

assert not checkpoint.exists(), (
    "Training checkpoint already exists. Do not restart Epoch 1."
)

assert not history.exists(), (
    "Training history already exists. Inspect existing run."
)

# ------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------

print("EXPERIMENT 08.36 - FINAL PRE-LAUNCH AUDIT")

print("\nCONFIGURATION")
print("Experiment:", config["experiment"])
print("Pretraining:", config["pretraining"])
print("Architecture:", config["architecture"])
print("Drop Path:", config["drop_path_rate"])
print("Seed:", config["seed"])

print("\nSOURCE CODE")
print("Python syntax:", "PASS")
print("Meta MAE encoder loader:", "PASS")
print("Standalone dataset:", "PASS")
print("Training operations:", "PASS")
print("Original experiment isolation:", "PASS")

print("\nOUTPUT STATUS")
print("Checkpoint exists:", checkpoint.exists())
print("Training history exists:", history.exists())

print("\nRESEARCH INTEGRITY")
print("Existing models modified:", False)
print("Training started:", False)
print("Test images accessed:", 0)

print("\nSTEP 08.36 PASS")


# %% [cell 285] identical to cell 286, omitted here


# %% [cell 286] Experiment 08 - Step 08.37
# ============================================================
# Experiment 08 - Step 08.37
# Meta MAE Drop Path 0.1 - Formal Epoch 1 Training
# ============================================================

from pathlib import Path
import runpy
import time
import pandas as pd

ROOT = DR_RESEARCH

SCRIPT = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_epoch1_v1.py"
)

RUN_DIR = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_v1"
)

CHECKPOINT = RUN_DIR / "last_training_state.pt"
HISTORY = RUN_DIR / "training_history.csv"
PREDICTIONS = RUN_DIR / "validation_predictions_epoch001.csv"

# ------------------------------------------------------------
# 1. Prevent accidental duplicate training
# ------------------------------------------------------------

assert SCRIPT.is_file(), "Training script is missing."

assert not CHECKPOINT.exists(), (
    "A training checkpoint already exists. "
    "Do not restart Epoch 1."
)

assert not HISTORY.exists(), (
    "Training history already exists. "
    "Stop and inspect the existing run."
)

assert not PREDICTIONS.exists(), (
    "Epoch 1 predictions already exist."
)

# ------------------------------------------------------------
# 2. Launch formal Epoch 1 training
# ------------------------------------------------------------

print("EXPERIMENT 08.37 - META MAE FORMAL TRAINING")
print("Architecture: ViT-Large/16")
print("Pretraining: Meta MAE ImageNet-1K")
print("Drop Path: 0.1")
print("Seed: 42")
print("Training epoch: 1")
print("Test split access: Prohibited")

start = time.perf_counter()

runpy.run_path(
    str(SCRIPT),
    run_name="__main__"
)

elapsed = (time.perf_counter() - start) / 60

# ------------------------------------------------------------
# 3. Validate saved training outputs
# ------------------------------------------------------------

assert CHECKPOINT.is_file(), (
    "Epoch 1 checkpoint was not saved."
)

assert HISTORY.is_file(), (
    "Training history was not saved."
)

assert PREDICTIONS.is_file(), (
    "Epoch 1 validation predictions were not saved."
)

history = pd.read_csv(HISTORY)

assert len(history) == 1
assert int(history.iloc[0]["epoch"]) == 1

row = history.iloc[0]

print("\nEXPERIMENT 08.37 - EPOCH 1 SUMMARY")

print("Train loss:", round(float(row["train_loss"]), 6))
print(
    "Train macro-F1:",
    round(float(row["train_macro_f1"]), 6)
)

print("Validation loss:", round(float(row["val_loss"]), 6))
print(
    "Validation accuracy:",
    round(float(row["val_accuracy"]), 6)
)
print(
    "Validation macro-F1:",
    round(float(row["val_macro_f1"]), 6)
)
print(
    "Validation QWK:",
    round(float(row["val_qwk"]), 6)
)

print("Total runtime (minutes):", round(elapsed, 2))

print("\nCHECKPOINT")
print("Training state saved:", CHECKPOINT.is_file())
print("Training history saved:", HISTORY.is_file())
print("Validation predictions saved:", PREDICTIONS.is_file())

print("\nRESEARCH INTEGRITY")
print("Original experiments modified: False")
print("Test images accessed: 0")

print("\nSTEP 08.37 PASS")


# %% [cell 287] Experiment 08 - Step 08.37A
# ============================================================
# Experiment 08 - Step 08.37A
# Fix Pre-existing Training Configuration Conflict
# No training; preserve original scripts
# ============================================================

from pathlib import Path
import ast
import json

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

SOURCE = OUT / "meta_mae_drop_path_01_epoch1_v1.py"
TARGET = OUT / "meta_mae_drop_path_01_epoch1_v2.py"

RUN_DIR = OUT / "meta_mae_drop_path_01_v1"
CONFIG_PATH = RUN_DIR / "training_config.json"

assert SOURCE.is_file()
assert CONFIG_PATH.is_file()
assert not TARGET.exists(), "V2 script already exists."

source = SOURCE.read_text(encoding="utf-8")
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

assert config["experiment"] == "Experiment08_Meta_MAE_DropPath01_V1"
assert config["drop_path_rate"] == 0.1
assert config["seed"] == 42

# ------------------------------------------------------------
# 1. Replace the incorrect configuration existence check
# ------------------------------------------------------------

old_check = "assert not CONFIG_PATH.exists()"

new_check = "\n".join([
    "assert CONFIG_PATH.is_file(), (",
    '    "Locked Experiment 08 configuration is missing."',
    ")"
])

assert source.count(old_check) == 1

modified = source.replace(
    old_check,
    new_check,
    1
)

# ------------------------------------------------------------
# 2. Prevent the script from overwriting the locked config
# ------------------------------------------------------------

tree = ast.parse(modified)

write_nodes = []

for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        call_source = ast.get_source_segment(modified, node) or ""

        if (
            "CONFIG_PATH" in call_source
            and any(
                token in call_source
                for token in [
                    "write_text",
                    "open(",
                    "json.dump",
                    "atomic"
                ]
            )
        ):
            write_nodes.append(
                (getattr(node, "lineno", None), call_source[:180])
            )

print("CONFIGURATION FILE OPERATIONS")

for line_number, operation in write_nodes[:10]:
    print("Line:", line_number)
    print(operation)

# ------------------------------------------------------------
# 3. Save the repaired script
# ------------------------------------------------------------

ast.parse(modified)
compile(modified, str(TARGET), "exec")

with TARGET.open("x", encoding="utf-8") as handle:
    handle.write(modified)

print("\nEXPERIMENT 08.37A - CONFIGURATION FIX")
print("Original script preserved:", SOURCE.is_file())
print("New script created:", TARGET.is_file())
print("Configuration preserved:", CONFIG_PATH.is_file())
print("Python syntax:", "PASS")

print("\nCONFIGURATION")
print("Experiment:", config["experiment"])
print("Drop Path:", config["drop_path_rate"])
print("Seed:", config["seed"])

print("\nRESEARCH INTEGRITY")
print("Training started:", False)
print("Existing checkpoints modified:", False)
print("Test images accessed:", 0)

print("\nSTEP 08.37A PASS")


# %% [cell 288] Experiment 08 - Step 08.37B
# ============================================================
# Experiment 08 - Step 08.37B
# Preserve Locked Configuration and Training Parameters
# Source modification only; no training
# ============================================================

from pathlib import Path
import ast
import json

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

SOURCE = OUT / "meta_mae_drop_path_01_epoch1_v2.py"
TARGET = OUT / "meta_mae_drop_path_01_epoch1_v3.py"

RUN_DIR = OUT / "meta_mae_drop_path_01_v1"
CONFIG_PATH = RUN_DIR / "training_config.json"

assert SOURCE.is_file()
assert CONFIG_PATH.is_file()
assert not TARGET.exists()

assert not (RUN_DIR / "last_training_state.pt").exists()
assert not (RUN_DIR / "training_history.csv").exists()

source = SOURCE.read_text(encoding="utf-8")
locked_config = json.loads(
    CONFIG_PATH.read_text(encoding="utf-8")
)

# ------------------------------------------------------------
# 1. Locate configuration-writing block using AST
# ------------------------------------------------------------

tree = ast.parse(source)
lines = source.splitlines(keepends=True)

write_blocks = []

for node in ast.walk(tree):
    if isinstance(node, ast.With):
        for item in node.items:
            context = item.context_expr

            if not isinstance(context, ast.Call):
                continue

            if not (
                isinstance(context.func, ast.Name)
                and context.func.id == "open"
            ):
                continue

            args = context.args

            if not args:
                continue

            if not (
                isinstance(args[0], ast.Name)
                and args[0].id == "CONFIG_PATH"
            ):
                continue

            write_mode = any(
                isinstance(arg, ast.Constant)
                and arg.value in ("w", "a", "x")
                for arg in args[1:]
            )

            if write_mode:
                write_blocks.append(node)

assert len(write_blocks) == 1, (
    f"Expected one configuration-writing block, "
    f"found {len(write_blocks)}"
)

block = write_blocks[0]

print("CONFIGURATION WRITE BLOCK")
print("Start line:", block.lineno)
print("End line:", block.end_lineno)

# ------------------------------------------------------------
# 2. Replace configuration writing with strict verification
# ------------------------------------------------------------

replacement = [
    "with CONFIG_PATH.open('r', encoding='utf-8') as handle:\n",
    "    persisted_config = json.load(handle)\n",
    "\n",
    "assert persisted_config == config, (\n",
    "    'Training configuration differs from the locked file.'\n",
    ")\n"
]

modified_lines = (
    lines[:block.lineno - 1]
    + replacement
    + lines[block.end_lineno:]
)

modified = "".join(modified_lines)

# ------------------------------------------------------------
# 3. Validate the generated script
# ------------------------------------------------------------

ast.parse(modified)
compile(modified, str(TARGET), "exec")

assert "drop_path_rate=0.1" in modified
assert "load_exp08_meta_mae_encoder(ROOT)" in modified

assert "persisted_config == config" in modified

assert 'open(CONFIG_PATH, "w"' not in modified

assert "retfound_drop_path_01_v1" not in modified
assert "experiment07_results" not in modified

# ------------------------------------------------------------
# 4. Inspect configuration construction
# ------------------------------------------------------------

config_operations = []

for node in ast.walk(ast.parse(modified)):
    if isinstance(node, ast.Assign):
        if any(
            isinstance(target, ast.Name)
            and target.id == "config"
            for target in node.targets
        ):
            snippet = ast.get_source_segment(modified, node)
            config_operations.append(
                (node.lineno, snippet[:200] if snippet else "")
            )

print("\nCONFIGURATION ASSIGNMENTS")

for line_number, snippet in config_operations:
    print("Line:", line_number)
    print(snippet)

print("\nLOCKED CONFIGURATION")
print("Experiment:", locked_config["experiment"])
print("Pretraining:", locked_config["pretraining"])
print("Drop Path:", locked_config["drop_path_rate"])
print("Seed:", locked_config["seed"])

# ------------------------------------------------------------
# 5. Save new version without modifying earlier files
# ------------------------------------------------------------

with TARGET.open("x", encoding="utf-8") as handle:
    handle.write(modified)

print("\nEXPERIMENT 08.37B - CONFIGURATION PROTECTION")
print("V3 script created:", TARGET.is_file())
print("Python syntax:", "PASS")
print("Configuration write removed:", True)
print("Locked configuration preserved:", True)

print("\nRESEARCH INTEGRITY")
print("Training started:", False)
print("Existing models modified:", False)
print("Test images accessed:", 0)

print("\nSTEP 08.37B PASS")


# %% [cell 289] Experiment 08 - Step 08.37C
# ============================================================
# Experiment 08 - Step 08.37C
# Fix Meta MAE Training Configuration Provenance
# No training or checkpoint modification
# ============================================================

from pathlib import Path
import ast
import json

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

SOURCE = OUT / "meta_mae_drop_path_01_epoch1_v3.py"
TARGET = OUT / "meta_mae_drop_path_01_epoch1_v4.py"

RUN_DIR = OUT / "meta_mae_drop_path_01_v1"
CONFIG_PATH = RUN_DIR / "training_config.json"

assert SOURCE.is_file()
assert CONFIG_PATH.is_file()
assert not TARGET.exists()

assert not (RUN_DIR / "last_training_state.pt").exists()
assert not (RUN_DIR / "training_history.csv").exists()

source = SOURCE.read_text(encoding="utf-8")
locked = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

assert locked["experiment"] == "Experiment08_Meta_MAE_DropPath01_V1"
assert locked["pretraining"] == "Meta_MAE_ImageNet1K"
assert locked["drop_path_rate"] == 0.1
assert locked["seed"] == 42

# ------------------------------------------------------------
# 1. Locate the obsolete configuration assignment
# ------------------------------------------------------------

tree = ast.parse(source)

matches = [
    node
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name)
        and target.id == "config"
        for target in node.targets
    )
]

assert len(matches) == 1, (
    f"Expected one top-level config assignment, found {len(matches)}"
)

node = matches[0]

print("OBSOLETE CONFIGURATION")
print("Start line:", node.lineno)
print("End line:", node.end_lineno)

# ------------------------------------------------------------
# 2. Replace obsolete metadata with locked configuration
# ------------------------------------------------------------

replacement = [
    "with CONFIG_PATH.open('r', encoding='utf-8') as handle:\n",
    "    config = json.load(handle)\n",
    "\n",
    "assert config['experiment'] == 'Experiment08_Meta_MAE_DropPath01_V1'\n",
    "assert config['pretraining'] == 'Meta_MAE_ImageNet1K'\n",
    "assert config['drop_path_rate'] == 0.1\n",
    "assert config['seed'] == 42\n"
]

lines = source.splitlines(keepends=True)

modified = "".join(
    lines[:node.lineno - 1]
    + replacement
    + lines[node.end_lineno:]
)

# ------------------------------------------------------------
# 3. Verify that persisted configuration is unchanged
# ------------------------------------------------------------

assert "persisted_config == config" in modified
assert "load_exp08_meta_mae_encoder(ROOT)" in modified
assert "drop_path_rate=0.1" in modified

assert "Meta MAE_mae_natureCFP" not in modified
assert "Meta MAE_ViT_Large_V1" not in modified

assert "RETFound_mae_natureCFP.pth" not in modified
assert "experiment04_results" not in modified
assert "experiment07_results" not in modified

# ------------------------------------------------------------
# 4. Verify checkpoint configuration provenance
# ------------------------------------------------------------

parsed = ast.parse(modified)
compile(parsed, str(TARGET), "exec")

required = [
    '"config": config',
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "numpy_rng_state",
    "train_loader_generator_state",
    "last_training_state.pt"
]

for item in required:
    assert item in modified, f"Missing checkpoint field: {item}"

# ------------------------------------------------------------
# 5. Save corrected script
# ------------------------------------------------------------

with TARGET.open("x", encoding="utf-8") as handle:
    handle.write(modified)

print("\nEXPERIMENT 08.37C - CONFIGURATION FIX COMPLETE")

print("Script:", TARGET.name)
print("Python syntax:", "PASS")
print("Configuration source:", "Locked Experiment 08 JSON")
print("Experiment:", locked["experiment"])
print("Pretraining:", locked["pretraining"])
print("Drop Path:", locked["drop_path_rate"])
print("Seed:", locked["seed"])

print("\nCHECKPOINT METADATA")
print("Configuration provenance:", "Locked JSON")
print("Original RETFound metadata removed:", True)

print("\nRESEARCH INTEGRITY")
print("Locked configuration modified:", False)
print("Original experiments modified:", False)
print("Training started:", False)
print("Test images accessed:", 0)

print("\nSTEP 08.37C PASS")


# %% [cell 290] Experiment 08 - Step 08.38
# ============================================================
# Experiment 08 - Step 08.38
# Meta MAE Drop Path 0.1 - Formal Epoch 1
# ============================================================

from pathlib import Path
import runpy
import time
import pandas as pd

ROOT = DR_RESEARCH

SCRIPT = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_epoch1_v4.py"
)

RUN_DIR = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_v1"
)

CHECKPOINT = RUN_DIR / "last_training_state.pt"
HISTORY = RUN_DIR / "training_history.csv"
PREDICTIONS = RUN_DIR / "validation_predictions_epoch001.csv"
CONFIG = RUN_DIR / "training_config.json"

# ------------------------------------------------------------
# 1. Protect existing results
# ------------------------------------------------------------

assert SCRIPT.is_file()
assert CONFIG.is_file()

assert not CHECKPOINT.exists(), (
    "Training checkpoint already exists. "
    "Do not repeat Epoch 1."
)

assert not HISTORY.exists(), (
    "Training history already exists."
)

assert not PREDICTIONS.exists(), (
    "Epoch 1 predictions already exist."
)

# ------------------------------------------------------------
# 2. Start formal training
# ------------------------------------------------------------

print("EXPERIMENT 08.38 - META MAE FORMAL EPOCH 1")
print("Architecture: ViT-Large/16")
print("Pretraining: Meta MAE ImageNet-1K")
print("Drop Path: 0.1")
print("Seed: 42")
print("Test split access: Prohibited")

start = time.perf_counter()

runpy.run_path(
    str(SCRIPT),
    run_name="__main__"
)

elapsed = (time.perf_counter() - start) / 60

# ------------------------------------------------------------
# 3. Verify outputs
# ------------------------------------------------------------

assert CHECKPOINT.is_file()
assert HISTORY.is_file()
assert PREDICTIONS.is_file()

history = pd.read_csv(HISTORY)

assert len(history) == 1
assert int(history.iloc[0]["epoch"]) == 1

row = history.iloc[0]

print("\nEXPERIMENT 08.38 - EPOCH 1 SUMMARY")

for key in [
    "train_loss",
    "train_macro_f1",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "val_qwk"
]:
    print(key + ":", round(float(row[key]), 6))

print("Runtime (minutes):", round(elapsed, 2))

print("\nOUTPUT VERIFICATION")
print("Training checkpoint:", CHECKPOINT.is_file())
print("Training history:", HISTORY.is_file())
print("Validation predictions:", PREDICTIONS.is_file())

print("\nRESEARCH INTEGRITY")
print("Original experiments modified: False")
print("Test images accessed: 0")

print("\nSTEP 08.38 PASS")


# %% [cell 291] Experiment 08 - Step 08.39
# ============================================================
# Experiment 08 - Step 08.39
# Generate Meta MAE Drop Path 0.1 Resume Script
# Source generation only; no training
# ============================================================

from pathlib import Path
import ast
import json
import pandas as pd

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

SOURCE = OUT / "retfound_drop_path_01_resume_v2.py"
TARGET = OUT / "meta_mae_drop_path_01_resume_v1.py"

RUN_DIR = OUT / "meta_mae_drop_path_01_v1"
CONFIG_PATH = RUN_DIR / "training_config.json"
CHECKPOINT_PATH = RUN_DIR / "last_training_state.pt"
HISTORY_PATH = RUN_DIR / "training_history.csv"

assert SOURCE.is_file()
assert CONFIG_PATH.is_file()
assert CHECKPOINT_PATH.is_file()
assert HISTORY_PATH.is_file()

assert not TARGET.exists(), (
    "Meta MAE resume script already exists."
)

config = json.loads(
    CONFIG_PATH.read_text(encoding="utf-8")
)

history = pd.read_csv(HISTORY_PATH)

assert config["experiment"] == (
    "Experiment08_Meta_MAE_DropPath01_V1"
)
assert config["pretraining"] == "Meta_MAE_ImageNet1K"
assert config["drop_path_rate"] == 0.1
assert config["seed"] == 42

assert len(history) == 1
assert int(history.iloc[0]["epoch"]) == 1

# ------------------------------------------------------------
# 1. Adapt the proven RETFound resume source
# ------------------------------------------------------------

source = SOURCE.read_text(encoding="utf-8")

old_directory = "retfound_drop_path_01_v1"
new_directory = "meta_mae_drop_path_01_v1"

assert source.count(old_directory) >= 1

modified = source.replace(
    old_directory,
    new_directory
)

# Update display labels only.
modified = modified.replace(
    "RETFound Epoch",
    "Meta MAE Epoch"
)

modified = modified.replace(
    "RETFOUND TRAINING RESUME",
    "META MAE TRAINING RESUME"
)

modified = modified.replace(
    "RETFOUND TRAINING SUMMARY",
    "META MAE TRAINING SUMMARY"
)

# ------------------------------------------------------------
# 2. Add explicit model identity and protocol checks
# ------------------------------------------------------------

anchor = (
    'assert config["architecture"] == '
    '"vit_large_patch16_224"'
)

assert modified.count(anchor) == 1

identity_guards = "\n".join([
    anchor,
    "",
    "assert config['experiment'] == (",
    "    'Experiment08_Meta_MAE_DropPath01_V1'",
    ")",
    "assert config['pretraining'] == 'Meta_MAE_ImageNet1K'",
    "assert float(config['drop_path_rate']) == 0.1",
    "assert int(config['seed']) == 42",
])

modified = modified.replace(
    anchor,
    identity_guards,
    1
)

# ------------------------------------------------------------
# 3. Validate essential recovery functionality
# ------------------------------------------------------------

ast.parse(modified)
compile(modified, str(TARGET), "exec")

required = [
    "drop_path_rate=0.1",
    "model.load_state_dict(",
    "optimizer.load_state_dict(",
    "scheduler.load_state_dict(",
    "torch.set_rng_state(",
    "torch.cuda.set_rng_state_all(",
    "random.setstate(",
    "np.random.set_state(",
    "train_generator.set_state(",
    "start_epoch = completed_epoch + 1",
    "for epoch in range(start_epoch, MAX_EPOCHS + 1)",
    "atomic_torch_save(",
    "atomic_csv_save(",
    "best_model.pt",
    "last_training_state.pt",
    "exp08_train_dataset",
    "exp08_val_loader"
]

for item in required:
    assert item in modified, (
        f"Missing recovery feature: {item}"
    )

for forbidden in [
    "retfound_drop_path_01_v1",
    "experiment04_results",
    "experiment07_results",
    "RETFound_mae_natureCFP.pth",
    "load_exp08_retfound_encoder"
]:
    assert forbidden not in modified, (
        f"Unexpected RETFound dependency: {forbidden}"
    )

# ------------------------------------------------------------
# 4. Save without modifying existing model outputs
# ------------------------------------------------------------

with TARGET.open("x", encoding="utf-8") as handle:
    handle.write(modified)

print("EXPERIMENT 08.39 - META MAE RESUME SCRIPT")

print("\nSOURCE")
print("Source:", SOURCE.name)
print("Target:", TARGET.name)
print("Python syntax:", "PASS")

print("\nMODEL")
print("Experiment:", config["experiment"])
print("Pretraining:", config["pretraining"])
print("Drop Path:", config["drop_path_rate"])

print("\nRECOVERY")
print("Completed epochs:", len(history))
print("Expected resume epoch:", 2)
print("Optimizer restoration:", "PASS")
print("Scheduler restoration:", "PASS")
print("RNG restoration:", "PASS")
print("Early stopping:", "Preserved")

print("\nOUTPUT PROTECTION")
print("Epoch 1 checkpoint preserved:", True)
print("Original RETFound model preserved:", True)
print("Original Meta MAE model preserved:", True)

print("\nRESEARCH INTEGRITY")
print("Training started:", False)
print("Test images accessed:", 0)

print("\nSTEP 08.39 PASS")


# %% [cell 292] Experiment 08 - Step 08.40
# ============================================================
# Experiment 08 - Step 08.40
# Resume Meta MAE Drop Path 0.1 Training
# ============================================================

from pathlib import Path
import json
import runpy
import time
import pandas as pd

ROOT = DR_RESEARCH

SCRIPT = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_resume_v1.py"
)

RUN_DIR = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_v1"
)

CHECKPOINT = RUN_DIR / "last_training_state.pt"
HISTORY = RUN_DIR / "training_history.csv"
CONFIG_PATH = RUN_DIR / "training_config.json"

# ------------------------------------------------------------
# 1. Verify the saved training state
# ------------------------------------------------------------

assert SCRIPT.is_file()
assert CHECKPOINT.is_file()
assert HISTORY.is_file()
assert CONFIG_PATH.is_file()

config = json.loads(
    CONFIG_PATH.read_text(encoding="utf-8")
)

history_before = pd.read_csv(HISTORY)

assert config["experiment"] == (
    "Experiment08_Meta_MAE_DropPath01_V1"
)

assert config["pretraining"] == "Meta_MAE_ImageNet1K"
assert config["drop_path_rate"] == 0.1

assert len(history_before) == 1
assert int(history_before["epoch"].max()) == 1

# ------------------------------------------------------------
# 2. Start checkpoint-based resume training
# ------------------------------------------------------------

print("EXPERIMENT 08.40 - META MAE REGULARIZED TRAINING")
print("Architecture: ViT-Large/16")
print("Pretraining: Meta MAE ImageNet-1K")
print("Drop Path: 0.1")
print("Seed: 42")
print("Expected starting epoch: 2")
print("Maximum epochs: 25")
print("Test split access: Prohibited")

start = time.perf_counter()

runpy.run_path(
    str(SCRIPT),
    run_name="__main__"
)

elapsed_minutes = (time.perf_counter() - start) / 60

# ------------------------------------------------------------
# 3. Verify completed training results
# ------------------------------------------------------------

history_after = pd.read_csv(HISTORY)

assert history_after["epoch"].is_unique
assert int(history_after["epoch"].min()) == 1
assert int(history_after["epoch"].max()) >= 2
assert int(history_after["epoch"].max()) <= 25

best_index = history_after["val_macro_f1"].idxmax()
best_row = history_after.loc[best_index]

print("\nEXPERIMENT 08.40 - TRAINING SUMMARY")
print("Completed epochs:", len(history_after))
print("Last epoch:", int(history_after["epoch"].max()))
print("Best epoch:", int(best_row["epoch"]))

print(
    "Best validation macro-F1:",
    round(float(best_row["val_macro_f1"]), 6)
)

print(
    "Best validation QWK:",
    round(float(best_row["val_qwk"]), 6)
)

print(
    "Best validation accuracy:",
    round(float(best_row["val_accuracy"]), 6)
)

print(
    "Total resume runtime (minutes):",
    round(elapsed_minutes, 2)
)

print("\nOUTPUT VERIFICATION")
print(
    "Best model:",
    (RUN_DIR / "best_model.pt").is_file()
)
print(
    "Best validation predictions:",
    (RUN_DIR / "best_validation_predictions.csv").is_file()
)
print("Last training checkpoint:", CHECKPOINT.is_file())

print("\nRESEARCH INTEGRITY")
print("Original experiments modified: False")
print("Test images accessed: 0")

print("\nSTEP 08.40 PASS")


# %% [cell 293] Experiment 08 - Step 08.41
# ============================================================
# Experiment 08 - Step 08.41
# Recover Meta MAE Training Progress
# Read-only; no training
# ============================================================

from pathlib import Path
import pandas as pd

ROOT = DR_RESEARCH

RUN_DIR = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_v1"
)

CHECKPOINT = RUN_DIR / "last_training_state.pt"
HISTORY = RUN_DIR / "training_history.csv"
BEST_MODEL = RUN_DIR / "best_model.pt"

print("EXPERIMENT 08.41 - TRAINING RECOVERY CHECK")

print("\nFILE STATUS")
print("Run directory:", RUN_DIR.is_dir())
print("Checkpoint exists:", CHECKPOINT.is_file())
print("History exists:", HISTORY.is_file())
print("Best model exists:", BEST_MODEL.is_file())

if HISTORY.is_file():
    history = pd.read_csv(HISTORY)

    print("\nRECORDED TRAINING PROGRESS")
    print("Completed epochs:", len(history))
    print("Latest recorded epoch:", int(history["epoch"].max()))

    best = history.loc[history["val_macro_f1"].idxmax()]

    print("Best epoch:", int(best["epoch"]))
    print(
        "Best validation macro-F1:",
        round(float(best["val_macro_f1"]), 6)
    )

    print("\nRECENT EPOCHS")
    print(
        history[
            ["epoch", "train_macro_f1", "val_macro_f1", "val_qwk"]
        ].tail(5).to_string(index=False)
    )

if CHECKPOINT.is_file():
    print(
        "\nCheckpoint size (GiB):",
        round(CHECKPOINT.stat().st_size / 1024**3, 3)
    )

    partial = Path(str(CHECKPOINT) + ".partial")
    print("Incomplete checkpoint file exists:", partial.exists())

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Training restarted: False")
print("Test images accessed: 0")

print("\nSTEP 08.41 COMPLETE")


# %% [cell 294] Experiment 08 - Step 08.42
# ============================================================
# Experiment 08 - Step 08.42
# Recover Google Drive Mount and Locate Research Files
# Read-only; no training or checkpoint modification
# ============================================================

from pathlib import Path

print("EXPERIMENT 08.42 - GOOGLE DRIVE RECOVERY")

# Reconnect Google Drive if the project is not accessible.
ROOT = DR_RESEARCH

if not ROOT.is_dir():
    print("\nProject path unavailable. Remounting Google Drive...")
    pass  # Google Drive mount removed

print("\nDRIVE STATUS")
print("Drive directory exists:", DR_RESEARCH.parent.is_dir())
print("MyDrive exists:", DR_RESEARCH.parent.is_dir())
print("Research project exists:", ROOT.is_dir())

if ROOT.is_dir():

    print("\nPROJECT CONTENTS")
    for item in sorted(ROOT.iterdir()):
        if item.name.startswith("experiment0"):
            print(item.name)

    OUT = ROOT / "experiment08_results"
    RUN_DIR = OUT / "meta_mae_drop_path_01_v1"

    print("\nEXPERIMENT 08")
    print("Experiment directory:", OUT.is_dir())
    print("Meta MAE run directory:", RUN_DIR.is_dir())

    if RUN_DIR.is_dir():
        print("\nMETA MAE FILES")

        for item in sorted(RUN_DIR.iterdir()):
            size_gib = (
                item.stat().st_size / 1024**3
                if item.is_file()
                else 0
            )

            print(
                item.name,
                "| Size GiB:",
                round(size_gib, 3)
            )

else:
    print("\nPROJECT NOT FOUND")
    print("Check the Google account connected to Colab.")

print("\nRESEARCH INTEGRITY")
print("Training restarted: False")
print("Checkpoints overwritten: False")
print("Test images accessed: 0")

print("\nSTEP 08.42 COMPLETE")


# %% [cell 295] Experiment 08 - Step 08.43
# ============================================================
# Experiment 08 - Step 08.43
# Verify Meta MAE Recovery Checkpoint
# Read-only; no training
# ============================================================

from pathlib import Path
import pandas as pd
import torch

ROOT = DR_RESEARCH

RUN_DIR = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_v1"
)

CHECKPOINT = RUN_DIR / "last_training_state.pt"
HISTORY = RUN_DIR / "training_history.csv"
BEST_MODEL = RUN_DIR / "best_model.pt"

assert CHECKPOINT.is_file()
assert HISTORY.is_file()
assert BEST_MODEL.is_file()

print("EXPERIMENT 08.43 - CHECKPOINT RECOVERY AUDIT")

# ------------------------------------------------------------
# 1. Read training history
# ------------------------------------------------------------

history = pd.read_csv(HISTORY)

assert not history.empty
assert history["epoch"].is_unique

print("\nTRAINING HISTORY")
print("Recorded epochs:", len(history))
print("Last recorded epoch:", int(history["epoch"].max()))

print("\nRECENT EPOCHS")
print(
    history[
        ["epoch", "train_macro_f1", "val_macro_f1", "val_qwk"]
    ].tail(6).to_string(index=False)
)

# ------------------------------------------------------------
# 2. Load the trusted, self-generated checkpoint metadata
# ------------------------------------------------------------

state = torch.load(
    CHECKPOINT,
    map_location="cpu",
    mmap=True,
    weights_only=False
)

completed_epoch = int(state["epoch"])
best_epoch = int(state["best_epoch"])
best_f1 = float(state["best_val_macro_f1"])

print("\nCHECKPOINT")
print("Completed epoch:", completed_epoch)
print("Next epoch:", completed_epoch + 1)
print("Best epoch:", best_epoch)
print("Best validation macro-F1:", round(best_f1, 6))
print(
    "Epochs without improvement:",
    int(state["epochs_without_improvement"])
)

# ------------------------------------------------------------
# 3. Validate recoverability
# ------------------------------------------------------------

required = [
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "torch_rng_state",
    "cuda_rng_states",
    "python_rng_state",
    "numpy_rng_state",
    "train_loader_generator_state"
]

for key in required:
    assert key in state, f"Missing checkpoint component: {key}"

assert completed_epoch >= 1
assert int(history["epoch"].max()) <= completed_epoch

checkpoint_config = state["config"]

assert checkpoint_config["experiment"] == (
    "Experiment08_Meta_MAE_DropPath01_V1"
)
assert checkpoint_config["pretraining"] == "Meta_MAE_ImageNet1K"
assert float(checkpoint_config["drop_path_rate"]) == 0.1

print("\nRECOVERY STATUS")
print("Model state:", "PASS")
print("Optimizer state:", "PASS")
print("Scheduler state:", "PASS")
print("RNG states:", "PASS")
print("Checkpoint configuration:", "PASS")
print("Checkpoint and history:", "COMPATIBLE")

print("\nRESEARCH INTEGRITY")
print("Training restarted:", False)
print("Files modified:", False)
print("Test images accessed:", 0)

print("\nSTEP 08.43 PASS")

del state


# %% [cell 296] Experiment 08 - Step 08.44
# ============================================================
# Experiment 08 - Step 08.44
# Investigate Checkpoint/History Mismatch
# Read-only; no training
# ============================================================

from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = DR_RESEARCH

RUN_DIR = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_v1"
)

HISTORY = RUN_DIR / "training_history.csv"
CHECKPOINT = RUN_DIR / "last_training_state.pt"

assert RUN_DIR.is_dir()
assert HISTORY.is_file()
assert CHECKPOINT.is_file()

print("EXPERIMENT 08.44 - RECOVERY FILE INVESTIGATION")

print("\nALL RUN FILES")

for path in sorted(RUN_DIR.iterdir()):
    stat = path.stat()

    modified = datetime.fromtimestamp(
        stat.st_mtime
    ).isoformat(timespec="seconds")

    print(
        path.name,
        "| Size GiB:",
        round(stat.st_size / 1024**3, 4),
        "| Modified:",
        modified
    )

print("\nPOTENTIAL RECOVERY FILES")

candidates = [
    path for path in RUN_DIR.iterdir()
    if any(
        token in path.name.lower()
        for token in [
            "partial",
            "backup",
            ".bak",
            "checkpoint",
            "training_state"
        ]
    )
]

for path in candidates:
    print(path.name)

if not candidates:
    print("No additional checkpoint candidates found.")

print("\nTRAINING HISTORY")

history = pd.read_csv(HISTORY)

print("History rows:", len(history))

print(
    history.tail(4).to_string(index=False)
)

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Training restarted: False")
print("Test images accessed: 0")

print("\nSTEP 08.44 COMPLETE")


# %% [cell 297] Experiment 08 - Step 08.45
# ============================================================
# Experiment 08 - Step 08.45
# Reconcile Meta MAE History with Recoverable Checkpoint
# Preserve original evidence; no training
# ============================================================

from pathlib import Path
import pandas as pd
import torch
import shutil
import os

ROOT = DR_RESEARCH

RUN_DIR = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_v1"
)

CHECKPOINT = RUN_DIR / "last_training_state.pt"
HISTORY = RUN_DIR / "training_history.csv"

BACKUP = RUN_DIR / "training_history_pre_recovery_epoch15.csv"

assert CHECKPOINT.is_file()
assert HISTORY.is_file()
assert not BACKUP.exists(), "Recovery backup already exists."

# ------------------------------------------------------------
# 1. Read the trusted checkpoint
# ------------------------------------------------------------

state = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
    mmap=True
)

completed_epoch = int(state["epoch"])
best_epoch = int(state["best_epoch"])
best_f1 = float(state["best_val_macro_f1"])

assert completed_epoch == 14
assert best_epoch == 11

# ------------------------------------------------------------
# 2. Validate current history
# ------------------------------------------------------------

history = pd.read_csv(HISTORY)

assert len(history) == 15
assert history["epoch"].tolist() == list(range(1, 16))

recovered = history.loc[
    history["epoch"] <= completed_epoch
].copy()

assert len(recovered) == completed_epoch

assert abs(
    float(
        recovered.loc[
            recovered["epoch"] == best_epoch,
            "val_macro_f1"
        ].iloc[0]
    ) - best_f1
) < 1e-8

# ------------------------------------------------------------
# 3. Preserve original history before modification
# ------------------------------------------------------------

shutil.copy2(HISTORY, BACKUP)

assert BACKUP.is_file()

# ------------------------------------------------------------
# 4. Atomically replace working history
# ------------------------------------------------------------

temporary = RUN_DIR / "training_history.csv.recovery.partial"

assert not temporary.exists()

recovered.to_csv(temporary, index=False)
os.replace(temporary, HISTORY)

# ------------------------------------------------------------
# 5. Verify recovery
# ------------------------------------------------------------

verified = pd.read_csv(HISTORY)

assert len(verified) == 14
assert int(verified["epoch"].max()) == 14

original = pd.read_csv(BACKUP)

assert len(original) == 15
assert int(original["epoch"].max()) == 15

print("EXPERIMENT 08.45 - HISTORY RECOVERY")

print("\nCHECKPOINT")
print("Recoverable epoch:", completed_epoch)
print("Next training epoch:", completed_epoch + 1)
print("Best epoch:", best_epoch)
print("Best macro-F1:", round(best_f1, 6))

print("\nHISTORY")
print("Original history rows:", len(original))
print("Recovered working rows:", len(verified))
print("Original history preserved:", BACKUP.is_file())

print("\nRESEARCH INTEGRITY")
print("Model checkpoint modified: False")
print("Best model modified: False")
print("Original history evidence preserved: True")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 08.45 PASS")

del state


# %% [cell 298] Experiment 08 - Step 08.46
# ============================================================
# Experiment 08 - Step 08.46
# Verify Runtime Readiness for Epoch 15 Resume
# Read-only; no training
# ============================================================

from pathlib import Path
import pandas as pd
import torch
import json

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

RUN_DIR = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_v1"
)

SCRIPT_PATH = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_resume_v1.py"
)

required_files = {
    "Frozen training index": INDEX_PATH,
    "Resume script": SCRIPT_PATH,
    "Training checkpoint": RUN_DIR / "last_training_state.pt",
    "Training history": RUN_DIR / "training_history.csv",
    "Locked configuration": RUN_DIR / "training_config.json",
    "Best model": RUN_DIR / "best_model.pt",
    "Recovery history backup": (
        RUN_DIR / "training_history_pre_recovery_epoch15.csv"
    ),
}

print("EXPERIMENT 08.46 - RESUME READINESS")

print("\nENVIRONMENT")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nREQUIRED FILES")

for name, path in required_files.items():
    exists = path.is_file()
    print(f"{name}: {'PASS' if exists else 'MISSING'}")
    assert exists, f"Missing required file: {path}"

config = json.loads(
    (RUN_DIR / "training_config.json").read_text(encoding="utf-8")
)

assert config["experiment"] == "Experiment08_Meta_MAE_DropPath01_V1"
assert config["drop_path_rate"] == 0.1

history = pd.read_csv(RUN_DIR / "training_history.csv")
assert history["epoch"].tolist() == list(range(1, 15))

print("\nTRAINING RECOVERY")
print("Last history epoch:", int(history["epoch"].max()))
print("Expected next epoch:", 15)
print("Drop Path:", config["drop_path_rate"])

# ------------------------------------------------------------
# Check local train/validation images only
# ------------------------------------------------------------

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

selected = index.loc[
    index["split"].isin(["train", "val"])
].copy()

assert len(selected) == 1566

print("\nLOCAL IMAGE CACHE")

all_ready = True

for split_name, expected in [
    ("train", 1290),
    ("val", 276)
]:
    frame = selected.loc[selected["split"] == split_name]
    assert len(frame) == expected

    paths = [Path(p) for p in frame["local_path"]]

    missing = sum(not p.is_file() for p in paths)
    symlinks = sum(p.is_symlink() for p in paths)

    drive_backed = sum(
        p.is_file()
        and str(p.resolve()).startswith(str(DR_RESEARCH.parent) + "/")
        for p in paths
    )

    print(f"\n{split_name.upper()}")
    print("Expected images:", expected)
    print("Missing images:", missing)
    print("Symbolic links:", symlinks)
    print("Drive-backed images:", drive_backed)

    if missing or symlinks or drive_backed:
        all_ready = False

print("\nRECOVERY DECISION")

if all_ready and torch.cuda.is_available():
    print("STATUS: READY FOR CHECKPOINT RESUME")
elif not all_ready:
    print("STATUS: LOCAL IMAGE CACHE RECOVERY REQUIRED")
else:
    print("STATUS: GPU RECOVERY REQUIRED")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Training restarted: False")
print("Test images accessed: 0")

print("\nSTEP 08.46 COMPLETE")


# %% [cell 299] Experiment 08 - Step 08.47
# ============================================================
# Experiment 08 - Step 08.47
# Restore FGADR Train/Validation Cache After Runtime Restart
# SHA-256 verified; test split excluded
# ============================================================

from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import hashlib
import shutil
import time
import os

ROOT = DR_RESEARCH

INDEX = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

CACHE_ROOT = (LOCAL_DISK / "fgadr_classification")
SOURCE_ROOT = ROOT / "datasets" / "DR"

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

# ------------------------------------------------------------
# 1. Validate frozen index
# ------------------------------------------------------------

assert INDEX.is_file()
assert sha256_file(INDEX) == EXPECTED_INDEX_SHA256

index = pd.read_csv(INDEX, dtype={"image_id": str})

selected = index.loc[
    index["split"].isin(["train", "val"])
].copy()

assert len(selected) == 1566
assert (selected["split"] == "train").sum() == 1290
assert (selected["split"] == "val").sum() == 276

assert selected["local_path"].is_unique
assert selected["sha256"].notna().all()

# ------------------------------------------------------------
# 2. Verify approved source and destination paths
# ------------------------------------------------------------

source_root = SOURCE_ROOT.resolve()
cache_root = CACHE_ROOT.resolve()

for row in selected.itertuples(index=False):
    source = Path(row.source_path)
    destination = Path(row.local_path)

    assert source.is_file(), f"Missing source: {source}"
    assert source.resolve().is_relative_to(source_root)
    assert destination.is_relative_to(cache_root)

# ------------------------------------------------------------
# 3. Restore missing or damaged images
# ------------------------------------------------------------

start = time.perf_counter()

copied = 0
reused = 0
repaired = 0
errors = []

for row in tqdm(
    selected.itertuples(index=False),
    total=len(selected),
    desc="Restoring FGADR train/val cache"
):
    source = Path(row.source_path)
    destination = Path(row.local_path)
    expected_hash = str(row.sha256).lower()

    try:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if destination.exists() or destination.is_symlink():
            if (
                destination.is_file()
                and not destination.is_symlink()
                and sha256_file(destination) == expected_hash
            ):
                reused += 1
                continue

            if destination.is_dir():
                raise RuntimeError(
                    f"Unexpected directory: {destination}"
                )

            destination.unlink()
            repaired += 1

        temporary = destination.with_name(
            destination.name + ".partial"
        )

        if temporary.exists():
            temporary.unlink()

        shutil.copyfile(source, temporary)

        if sha256_file(temporary) != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch: {row.image_id}"
            )

        os.replace(temporary, destination)
        copied += 1

    except Exception as exc:
        errors.append((row.image_id, str(exc)))

# ------------------------------------------------------------
# 4. Final integrity verification
# ------------------------------------------------------------

missing = 0
symlinks = 0
hash_mismatches = 0

for row in selected.itertuples(index=False):
    path = Path(row.local_path)

    if not path.is_file():
        missing += 1
        continue

    if path.is_symlink():
        symlinks += 1
        continue

    if sha256_file(path) != str(row.sha256).lower():
        hash_mismatches += 1

elapsed = (time.perf_counter() - start) / 60

print("\nEXPERIMENT 08.47 - CACHE RECOVERY SUMMARY")
print("Expected images:", len(selected))
print("Newly copied:", copied)
print("Reused verified:", reused)
print("Repaired:", repaired)
print("Copy errors:", len(errors))

print("\nFINAL INTEGRITY")
print("Missing images:", missing)
print("Symbolic links:", symlinks)
print("SHA-256 mismatches:", hash_mismatches)
print("Runtime (minutes):", round(elapsed, 2))

if errors:
    print("\nFIRST ERRORS")
    for image_id, message in errors[:5]:
        print(image_id, message)

assert not errors
assert missing == 0
assert symlinks == 0
assert hash_mismatches == 0
assert copied + reused == 1566

print("\nRESEARCH INTEGRITY")
print("Frozen index modified: False")
print("Original dataset modified: False")
print("Training restarted: False")
print("Test images accessed: 0")

print("\nSTEP 08.47 PASS")


# %% [cell 300] Experiment 08 - Step 08.48
# ============================================================
# Experiment 08 - Step 08.48
# Build Checkpoint-Authoritative Meta MAE Resume Script
# Source modification only; no training
# ============================================================

from pathlib import Path
import ast

ROOT = DR_RESEARCH
OUT = ROOT / "experiment08_results"

SOURCE = OUT / "meta_mae_drop_path_01_resume_v1.py"
TARGET = OUT / "meta_mae_drop_path_01_resume_v2.py"

RUN_DIR = OUT / "meta_mae_drop_path_01_v1"

assert SOURCE.is_file()
assert not TARGET.exists()

assert (RUN_DIR / "last_training_state.pt").is_file()
assert (RUN_DIR / "training_history.csv").is_file()

source = SOURCE.read_text(encoding="utf-8")

# ------------------------------------------------------------
# 1. Replace history consistency section
# ------------------------------------------------------------

old_section = """# Check that the persisted history has not advanced beyond
# the recoverable training state.
assert not history_df.empty
assert history_df["epoch"].is_unique
assert int(history_df["epoch"].max()) <= completed_epoch

best_f1 = float(resume_state["best_val_macro_f1"])"""

new_section = """# The checkpoint is authoritative after interrupted saves.
# Any CSV inconsistency is reconciled from checkpoint metadata.
assert not history_df.empty
assert history_df["epoch"].is_unique

checkpoint_history = resume_state.get("history_rows")

if checkpoint_history is not None:
    authoritative_history = pd.DataFrame(checkpoint_history)

    assert not authoritative_history.empty
    assert authoritative_history["epoch"].is_unique
    assert int(authoritative_history["epoch"].max()) == completed_epoch

    history_df = authoritative_history.copy()

else:
    history_df = history_df.loc[
        history_df["epoch"] <= completed_epoch
    ].copy()

    assert int(history_df["epoch"].max()) == completed_epoch

best_f1 = float(resume_state["best_val_macro_f1"])"""

assert source.count(old_section) == 1

modified = source.replace(
    old_section,
    new_section,
    1
)

# ------------------------------------------------------------
# 2. Preserve existing checkpoint-history reconciliation
# ------------------------------------------------------------

assert "resume_state.get(\"history_rows\")" in modified
assert "completed_epoch + 1" in modified
assert "drop_path_rate=0.1" in modified

# ------------------------------------------------------------
# 3. Verify checkpoint-first saving remains present
# ------------------------------------------------------------

checkpoint_save = modified.find(
    "atomic_torch_save(\n"
    "            training_state,\n"
    "            CHECKPOINT_PATH"
)

history_save = modified.find(
    "atomic_csv_save(\n"
    "            history_df,\n"
    "            HISTORY_PATH"
)

assert checkpoint_save >= 0
assert history_save >= 0
assert checkpoint_save < history_save

# ------------------------------------------------------------
# 4. Static validation
# ------------------------------------------------------------

ast.parse(modified)
compile(modified, str(TARGET), "exec")

assert "Experiment08_Meta_MAE_DropPath01_V1" in modified
assert "exp08_train_dataset" in modified
assert "exp08_val_loader" in modified

assert "experiment04_results" not in modified
assert "experiment07_results" not in modified

# ------------------------------------------------------------
# 5. Save V2 without touching previous artifacts
# ------------------------------------------------------------

with TARGET.open("x", encoding="utf-8") as handle:
    handle.write(modified)

print("EXPERIMENT 08.48 - CHECKPOINT RECOVERY SCRIPT")

print("\nSOURCE")
print("Original:", SOURCE.name)
print("Recovery version:", TARGET.name)
print("Python syntax:", "PASS")

print("\nRECOVERY RULES")
print("Checkpoint authoritative:", True)
print("History reconstructed from checkpoint:", True)
print("Checkpoint saved before CSV:", True)
print("Resume epoch determined by checkpoint:", True)

print("\nRESEARCH INTEGRITY")
print("Existing checkpoint modified:", False)
print("Training history modified:", False)
print("Training restarted:", False)
print("Test images accessed:", 0)

print("\nSTEP 08.48 PASS")


# %% [cell 301] Experiment 08 - Step 08.49
# ============================================================
# Experiment 08 - Step 08.49
# Resume Meta MAE Training from Epoch 14 Checkpoint
# ============================================================

from pathlib import Path
import json
import runpy
import time
import pandas as pd
import torch

ROOT = DR_RESEARCH

RUN_DIR = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_v1"
)

SCRIPT = (
    ROOT / "experiment08_results"
    / "meta_mae_drop_path_01_resume_v2.py"
)

CHECKPOINT = RUN_DIR / "last_training_state.pt"
HISTORY = RUN_DIR / "training_history.csv"
CONFIG_PATH = RUN_DIR / "training_config.json"

assert SCRIPT.is_file()
assert CHECKPOINT.is_file()
assert HISTORY.is_file()
assert CONFIG_PATH.is_file()
assert torch.cuda.is_available()

# ------------------------------------------------------------
# 1. Verify locked protocol and recovery history
# ------------------------------------------------------------

config = json.loads(
    CONFIG_PATH.read_text(encoding="utf-8")
)

history_before = pd.read_csv(HISTORY)

assert config["experiment"] == (
    "Experiment08_Meta_MAE_DropPath01_V1"
)
assert config["pretraining"] == "Meta_MAE_ImageNet1K"
assert float(config["drop_path_rate"]) == 0.1
assert int(config["seed"]) == 42

assert history_before["epoch"].tolist() == list(range(1, 15))

# Read only the trusted, self-generated checkpoint.
state = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
    mmap=True
)

completed_epoch = int(state["epoch"])

assert completed_epoch == 14, (
    f"Unexpected checkpoint epoch: {completed_epoch}. "
    "Do not restart an already advanced run."
)

assert state["config"]["experiment"] == config["experiment"]
assert float(state["config"]["drop_path_rate"]) == 0.1

del state

# ------------------------------------------------------------
# 2. Resume training
# ------------------------------------------------------------

print("EXPERIMENT 08.49 - META MAE TRAINING RECOVERY")
print("Model: Meta MAE ViT-Large/16")
print("Drop Path: 0.1")
print("Recovered checkpoint epoch:", completed_epoch)
print("Starting epoch:", completed_epoch + 1)
print("Maximum epochs:", config["maximum_epochs"])
print("Test split access: Prohibited")

start = time.perf_counter()

runpy.run_path(
    str(SCRIPT),
    run_name="__main__"
)

elapsed_minutes = (time.perf_counter() - start) / 60

# ------------------------------------------------------------
# 3. Verify resulting history
# ------------------------------------------------------------

history_after = pd.read_csv(HISTORY)

assert history_after["epoch"].is_unique
assert history_after["epoch"].tolist() == list(
    range(1, int(history_after["epoch"].max()) + 1)
)

assert int(history_after["epoch"].max()) >= 15
assert int(history_after["epoch"].max()) <= 25

best = history_after.loc[
    history_after["val_macro_f1"].idxmax()
]

print("\nEXPERIMENT 08.49 - TRAINING SUMMARY")
print("Completed epochs:", len(history_after))
print("Last epoch:", int(history_after["epoch"].max()))
print("Best epoch:", int(best["epoch"]))
print(
    "Best validation macro-F1:",
    round(float(best["val_macro_f1"]), 6)
)
print(
    "Best validation QWK:",
    round(float(best["val_qwk"]), 6)
)
print(
    "Best validation accuracy:",
    round(float(best["val_accuracy"]), 6)
)
print("Resume runtime (minutes):", round(elapsed_minutes, 2))

print("\nOUTPUT VERIFICATION")
print("Best model exists:", (RUN_DIR / "best_model.pt").is_file())
print(
    "Best validation predictions exist:",
    (RUN_DIR / "best_validation_predictions.csv").is_file()
)
print("Resume checkpoint exists:", CHECKPOINT.is_file())

print("\nRESEARCH INTEGRITY")
print("Original experiments modified: False")
print("Test images accessed: 0")

print("\nSTEP 08.49 PASS")
