# Experiment 04 - RETFound-MAE ViT-L/16 fine-tuning (seed 42)
#
# Source: FGADR_MA_Mask_Audit.ipynb (master notebook, 392 cells), cells 88-104, run step by step in Google Colab (L4 GPU).
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
#
# Checked on 2026-10-28 cell by cell against the master notebook: apart from
# the path, Colab and %pip changes listed above, every code cell is identical
# to the code that was run. Cells marked 'identical to cell N' were verified
# to be exact duplicates.

# %% Paths
from common.paths import ROOT as DR_RESEARCH
from common.paths import CACHE as LOCAL_DISK

# %% [cell 88]

import os
from huggingface_hub import HfApi

MODEL_ID = "YukunZhou/RETFound_mae_natureCFP"

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN is not configured."

api = HfApi(token=token)
model_info = api.model_info(
    repo_id=MODEL_ID,
    files_metadata=True
)

print("STEP 4.1A - RETFOUND ACCESS CHECK")
print("Repository:", model_info.id)
print("Access: PASS")

print("\nMODEL FILES")
for file in model_info.siblings:
    print(file.rfilename, "| Size:", file.size)

print("\nSTEP 4.1A COMPLETED")


# %% [cell 89] Experiment 04 - Step 4.1B

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import torch

import os
from huggingface_hub import HfApi, hf_hub_download

# ============================================================
# Experiment 04 - Step 4.1B
# RETFound Pretrained Weight Download and Verification
# ============================================================

MODEL_ID = "YukunZhou/RETFound_mae_natureCFP"
FILENAME = "RETFound_mae_natureCFP.pth"

PROJECT_ROOT = DR_RESEARCH
WEIGHT_DIR = (
    PROJECT_ROOT / "model_checkpoints" / "RETFound_pretrained"
)

LOCAL_DIR = (LOCAL_DISK / "retfound_pretrained_download")

WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

FINAL_PATH = WEIGHT_DIR / FILENAME
REPORT_PATH = WEIGHT_DIR / "RETFound_Weight_Manifest.json"

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN is unavailable."

api = HfApi(token=token)
info = api.model_info(MODEL_ID, files_metadata=True)

matching = [
    file for file in info.siblings
    if file.rfilename == FILENAME
]

assert len(matching) == 1
expected_size = int(matching[0].size)

assert expected_size == 3952489221

def calculate_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()

print("STEP 4.1B - RETFOUND WEIGHT DOWNLOAD")
print("Repository:", MODEL_ID)
print("Filename:", FILENAME)
print("Expected bytes:", expected_size)
print("Target:", FINAL_PATH)

if FINAL_PATH.is_file() and FINAL_PATH.stat().st_size == expected_size:
    print("\nExisting Drive checkpoint found.")
    print("Skipping duplicate download.")
    source_path = FINAL_PATH
else:
    # Download to the local runtime before transferring to Drive.
    source_path = Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=FILENAME,
            token=token,
            local_dir=str(LOCAL_DIR)
        )
    )

    assert source_path.is_file()
    assert source_path.stat().st_size == expected_size

    print("\nTransferring verified-size checkpoint to Google Drive...")

    temporary_path = WEIGHT_DIR / (FILENAME + ".partial")

    shutil.copyfile(source_path, temporary_path)

    assert temporary_path.stat().st_size == expected_size

    # Verify byte-for-byte integrity before finalizing the copy.
    source_sha256 = calculate_sha256(source_path)
    drive_sha256 = calculate_sha256(temporary_path)

    assert source_sha256 == drive_sha256, (
        "Checkpoint transfer integrity check failed."
    )

    temporary_path.replace(FINAL_PATH)

assert FINAL_PATH.is_file()
assert FINAL_PATH.stat().st_size == expected_size

final_sha256 = calculate_sha256(FINAL_PATH)

report = {
    "experiment": "Experiment04_Step4.1B",
    "repository": MODEL_ID,
    "repository_revision": info.sha,
    "filename": FILENAME,
    "file_size_bytes": expected_size,
    "sha256": final_sha256,
    "local_download_path": str(source_path),
    "persistent_path": str(FINAL_PATH),
    "torch_version": torch.__version__,
    "checkpoint_structure_inspected": False,
    "weights_loaded_into_model": False,
    "created_utc": datetime.now(timezone.utc).isoformat()
}

if REPORT_PATH.exists():
    with open(REPORT_PATH, "r", encoding="utf-8") as handle:
        existing_report = json.load(handle)

    assert existing_report["sha256"] == final_sha256, (
        "Existing checkpoint manifest does not match the file."
    )
else:
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

print("\nRETFOUND DOWNLOAD VERIFICATION")
print("Repository revision:", info.sha)
print("File exists:", FINAL_PATH.is_file())
print("File size (bytes):", FINAL_PATH.stat().st_size)
print("SHA256:", final_sha256)
print("Drive persistence: PASS")
print("Transfer integrity: PASS")
print("Model instantiated: False")

print("\nSTEP 4.1B COMPLETED")


# %% [cell 90] Experiment 04 - Step 4.2A

from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json
import re
import torch

# ============================================================
# Experiment 04 - Step 4.2A
# RETFound Checkpoint Architecture Audit
# ============================================================

ROOT = DR_RESEARCH
WEIGHT_PATH = (
    ROOT
    / "model_checkpoints"
    / "RETFound_pretrained"
    / "RETFound_mae_natureCFP.pth"
)

REPORT_DIR = ROOT / "experiment04_results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "RETFound_Checkpoint_Architecture_Audit.json"

EXPECTED_SIZE = 3952489221

assert WEIGHT_PATH.is_file(), "Checkpoint not found."
assert WEIGHT_PATH.stat().st_size == EXPECTED_SIZE

print("STEP 4.2A - RETFOUND CHECKPOINT AUDIT")
print("Checkpoint:", WEIGHT_PATH.name)
print("File size:", WEIGHT_PATH.stat().st_size)
print("PyTorch:", torch.__version__)

# ============================================================
# 1. Inspect checkpoint structure safely on CPU
# ============================================================

checkpoint = torch.load(
    WEIGHT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

assert isinstance(checkpoint, dict)

print("\nTOP-LEVEL CHECKPOINT KEYS")
print(list(checkpoint.keys())[:30])

if "model" in checkpoint:
    state_dict = checkpoint["model"]
    state_dict_container = "model"
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    state_dict_container = "state_dict"
else:
    state_dict = checkpoint
    state_dict_container = "root"

assert isinstance(state_dict, dict)

tensor_items = {
    key: value
    for key, value in state_dict.items()
    if isinstance(value, torch.Tensor)
}

assert len(tensor_items) > 0

# ============================================================
# 2. Summarize parameter names and shapes
# ============================================================

prefix_counts = Counter(
    key.split(".")[0]
    for key in tensor_items
)

decoder_keys = [
    key for key in tensor_items
    if key.startswith("decoder")
]

encoder_block_ids = sorted({
    int(match.group(1))
    for key in tensor_items
    if (match := re.match(r"^blocks\.(\d+)\.", key))
})

important_keys = [
    "cls_token",
    "pos_embed",
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "norm.weight",
    "norm.bias",
    "head.weight",
    "head.bias",
    "decoder_embed.weight",
    "decoder_pos_embed",
    "decoder_pred.weight"
]

print("\nPARAMETER SUMMARY")
print("State dictionary container:", state_dict_container)
print("Tensor entries:", len(tensor_items))
print("MAE decoder tensor entries:", len(decoder_keys))
print("Encoder block count:", len(encoder_block_ids))
print("Encoder block indices:", encoder_block_ids)

print("\nPARAMETER PREFIX COUNTS")
for prefix, count in prefix_counts.most_common():
    print(f"{prefix}: {count}")

print("\nIMPORTANT PARAMETER SHAPES")
for key in important_keys:
    if key in tensor_items:
        print(key, ":", tuple(tensor_items[key].shape))
    else:
        print(key, ": NOT FOUND")

print("\nFIRST 20 TENSOR KEYS")
for key in list(tensor_items.keys())[:20]:
    print(key, tuple(tensor_items[key].shape))

# ============================================================
# 3. Parameter volume and numeric formats
# ============================================================

total_elements = sum(
    value.numel()
    for value in tensor_items.values()
)

decoder_elements = sum(
    tensor_items[key].numel()
    for key in decoder_keys
)

dtype_counts = Counter(
    str(value.dtype)
    for value in tensor_items.values()
)

print("\nCHECKPOINT STATISTICS")
print("Total tensor elements:", total_elements)
print("Decoder tensor elements:", decoder_elements)
print("Tensor dtypes:", dict(dtype_counts))

# ============================================================
# 4. Save reproducibility report
# ============================================================

report = {
    "experiment": "Experiment04_Step4.2A",
    "repository": "YukunZhou/RETFound_mae_natureCFP",
    "repository_revision": (
        "556830f78214f0e8da35af965292ac5a3180ac47"
    ),
    "checkpoint_filename": WEIGHT_PATH.name,
    "checkpoint_sha256": (
        "e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485"
    ),
    "checkpoint_file_size": EXPECTED_SIZE,
    "state_dict_container": state_dict_container,
    "tensor_entry_count": len(tensor_items),
    "encoder_block_indices": encoder_block_ids,
    "decoder_tensor_count": len(decoder_keys),
    "total_tensor_elements": total_elements,
    "decoder_tensor_elements": decoder_elements,
    "tensor_dtypes": dict(dtype_counts),
    "important_shapes": {
        key: list(tensor_items[key].shape)
        if key in tensor_items else None
        for key in important_keys
    },
    "gpu_model_instantiated": False,
    "classifier_loaded": False,
    "created_utc": datetime.now(timezone.utc).isoformat()
}

if REPORT_PATH.exists():
    raise FileExistsError(
        f"Architecture audit already exists: {REPORT_PATH}"
    )

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nOUTPUT FILE")
print(REPORT_PATH)

print("\nSTEP 4.2A COMPLETED")

# Release checkpoint references before subsequent experiments.
del tensor_items
del state_dict
del checkpoint


# %% [cell 91] Experiment 04 - Step 4.2A Fix

import argparse
from pathlib import Path
import torch

# ============================================================
# Experiment 04 - Step 4.2A Fix
# Safe RETFound Checkpoint Deserialization
# ============================================================

WEIGHT_PATH = (DR_RESEARCH / "model_checkpoints/RETFound_pretrained/RETFound_mae_natureCFP.pth")

assert WEIGHT_PATH.is_file()
assert WEIGHT_PATH.stat().st_size == 3952489221

print("RETFOUND SAFE CHECKPOINT LOADING")
print("PyTorch version:", torch.__version__)
print("Checkpoint:", WEIGHT_PATH.name)

with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(
        WEIGHT_PATH,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

assert isinstance(checkpoint, dict)

print("\nTOP-LEVEL CHECKPOINT KEYS")
print(list(checkpoint.keys()))

if "model" in checkpoint:
    state_dict = checkpoint["model"]
    print("\nState dictionary container: model")
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    print("\nState dictionary container: state_dict")
else:
    state_dict = checkpoint
    print("\nState dictionary container: root")

assert isinstance(state_dict, dict)

tensor_count = sum(
    isinstance(value, torch.Tensor)
    for value in state_dict.values()
)

print("Tensor entries:", tensor_count)

print("\nRETFOUND SAFE LOADING COMPLETED")


# %% [cell 92] Experiment 04 - Step 4.2B

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import re
import json
import math
import torch

# ============================================================
# Experiment 04 - Step 4.2B
# RETFound Checkpoint Architecture Audit
# ============================================================

ROOT = DR_RESEARCH
RESULTS_DIR = ROOT / "experiment04_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = RESULTS_DIR / "RETFound_Checkpoint_Architecture_Audit.json"

assert "checkpoint" in globals()
assert "state_dict" in globals()
assert isinstance(state_dict, dict)

tensors = {
    key: value
    for key, value in state_dict.items()
    if isinstance(value, torch.Tensor)
}

assert len(tensors) == 398

# ============================================================
# 1. Identify parameter categories
# ============================================================

prefix_counts = Counter(
    key.split(".")[0]
    for key in tensors
)

encoder_blocks = sorted({
    int(match.group(1))
    for key in tensors
    if (match := re.match(r"^blocks\.(\d+)\.", key))
})

decoder_keys = [
    key for key in tensors
    if key.startswith("decoder")
]

important_keys = [
    "cls_token",
    "pos_embed",
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "blocks.0.attn.qkv.weight",
    "blocks.0.mlp.fc1.weight",
    "norm.weight",
    "norm.bias",
    "fc_norm.weight",
    "head.weight",
    "head.bias",
    "decoder_embed.weight",
    "decoder_pos_embed",
    "decoder_pred.weight"
]

# ============================================================
# 2. Calculate tensor statistics without copying weights
# ============================================================

total_parameters = sum(
    value.numel()
    for value in tensors.values()
)

decoder_parameters = sum(
    tensors[key].numel()
    for key in decoder_keys
)

dtype_counts = Counter(
    str(value.dtype)
    for value in tensors.values()
)

important_shapes = {
    key: (
        list(tensors[key].shape)
        if key in tensors
        else None
    )
    for key in important_keys
}

print("STEP 4.2B - RETFOUND ARCHITECTURE AUDIT")

print("\nCHECKPOINT STRUCTURE")
print("State dictionary container: model")
print("Tensor entries:", len(tensors))
print("Encoder block count:", len(encoder_blocks))
print("Encoder block indices:", encoder_blocks)
print("Decoder tensor entries:", len(decoder_keys))

print("\nPARAMETER PREFIX COUNTS")
for prefix, count in prefix_counts.most_common():
    print(f"{prefix}: {count}")

print("\nIMPORTANT PARAMETER SHAPES")
for key, shape in important_shapes.items():
    print(f"{key}: {shape if shape is not None else 'NOT FOUND'}")

print("\nMODEL SIZE")
print("All checkpoint tensor elements:", total_parameters)
print("Decoder tensor elements:", decoder_parameters)
print("Tensor dtypes:", dict(dtype_counts))

print("\nFIRST 20 TENSOR KEYS")
for key in list(tensors)[:20]:
    print(key, list(tensors[key].shape))

# ============================================================
# 3. Inspect original training metadata safely
# ============================================================

print("\nCHECKPOINT TRAINING METADATA")
print("Epoch:", checkpoint.get("epoch", "NOT FOUND"))

args = checkpoint.get("args")

if args is not None:
    args_dict = vars(args) if hasattr(args, "__dict__") else {}

    for name in [
        "model",
        "input_size",
        "mask_ratio",
        "norm_pix_loss",
        "blr",
        "batch_size",
        "epochs"
    ]:
        value = args_dict.get(name, "NOT RECORDED")
        print(f"{name}: {value}")
else:
    args_dict = {}

# ============================================================
# 4. Save architecture audit
# ============================================================

report = {
    "experiment": "Experiment04_Step4.2B",
    "repository": "YukunZhou/RETFound_mae_natureCFP",
    "repository_revision": (
        "556830f78214f0e8da35af965292ac5a3180ac47"
    ),
    "checkpoint_sha256": (
        "e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485"
    ),
    "state_dict_container": "model",
    "tensor_entries": len(tensors),
    "encoder_block_indices": encoder_blocks,
    "decoder_tensor_entries": len(decoder_keys),
    "total_tensor_elements": total_parameters,
    "decoder_tensor_elements": decoder_parameters,
    "tensor_dtypes": dict(dtype_counts),
    "important_parameter_shapes": important_shapes,
    "checkpoint_epoch": str(checkpoint.get("epoch")),
    "selected_training_arguments": {
        key: str(args_dict.get(key, "NOT RECORDED"))
        for key in [
            "model",
            "input_size",
            "mask_ratio",
            "norm_pix_loss",
            "blr",
            "batch_size",
            "epochs"
        ]
    },
    "classification_model_instantiated": False,
    "model_weights_transferred": False,
    "created_utc": datetime.now(timezone.utc).isoformat()
}

if REPORT_PATH.exists():
    raise FileExistsError(
        f"Audit output already exists: {REPORT_PATH}"
    )

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nOUTPUT FILE:", REPORT_PATH)
print("STEP 4.2B COMPLETED")


# %% [cell 93] Experiment 04 - Step 4.2C

from pathlib import Path
from datetime import datetime, timezone
import json
import torch
import timm

# ============================================================
# Experiment 04 - Step 4.2C
# RETFound / timm Architecture Compatibility Audit
# ============================================================

ROOT = DR_RESEARCH
REPORT_DIR = ROOT / "experiment04_results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = (
    REPORT_DIR / "RETFound_timm_Compatibility_Audit.json"
)

assert "state_dict" in globals()
assert isinstance(state_dict, dict)
assert "patch_embed.proj.weight" in state_dict
assert "pos_embed" in state_dict

MODEL_NAME = "vit_large_patch16_224"
NUM_CLASSES = 5

print("STEP 4.2C - RETFOUND MODEL COMPATIBILITY")
print("timm version:", timm.__version__)
print("PyTorch version:", torch.__version__)

# Create model structure only, without allocating real parameters.
with torch.device("meta"):
    classification_model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
        global_pool="avg"
    )

target_state = classification_model.state_dict()

encoder_prefixes = (
    "patch_embed.",
    "blocks.",
)

special_encoder_keys = {
    "cls_token",
    "pos_embed"
}

# MAE decoder and reconstruction-only tensors are not
# required for five-class classification.
source_encoder = {
    key: value
    for key, value in state_dict.items()
    if key.startswith(encoder_prefixes)
    or key in special_encoder_keys
}

source_norm = {
    key: value
    for key, value in state_dict.items()
    if key.startswith("norm.")
}

missing_encoder = sorted(
    set(target_state)
    .intersection(
        {
            key for key in target_state
            if key.startswith(encoder_prefixes)
            or key in special_encoder_keys
        }
    )
    - set(source_encoder)
)

shape_mismatches = []

for key, value in source_encoder.items():
    if key not in target_state:
        shape_mismatches.append({
            "key": key,
            "problem": "missing_target_parameter",
            "source_shape": list(value.shape),
            "target_shape": None
        })
    elif tuple(value.shape) != tuple(target_state[key].shape):
        shape_mismatches.append({
            "key": key,
            "problem": "shape_mismatch",
            "source_shape": list(value.shape),
            "target_shape": list(target_state[key].shape)
        })

compatible_encoder = [
    key for key, value in source_encoder.items()
    if key in target_state
    and tuple(value.shape) == tuple(target_state[key].shape)
]

compatible_elements = sum(
    state_dict[key].numel()
    for key in compatible_encoder
)

source_elements = sum(
    tensor.numel()
    for tensor in source_encoder.values()
)

print("\nTARGET MODEL")
print("Architecture:", MODEL_NAME)
print("Classes:", NUM_CLASSES)
print("Pooling:", classification_model.global_pool)
print("Target tensor entries:", len(target_state))
print("Target parameter elements:",
      sum(p.numel() for p in classification_model.parameters()))

print("\nENCODER COMPATIBILITY")
print("Source encoder tensor entries:", len(source_encoder))
print("Compatible encoder tensor entries:", len(compatible_encoder))
print("Missing encoder parameters:", len(missing_encoder))
print("Shape/name incompatibilities:", len(shape_mismatches))
print("Compatible encoder elements:", compatible_elements)
print("Source encoder elements:", source_elements)
print("Compatible element fraction:",
      round(compatible_elements / source_elements, 8))

print("\nNORMALIZATION AND CLASSIFIER")
for key in [
    "norm.weight",
    "norm.bias",
    "fc_norm.weight",
    "fc_norm.bias",
    "head.weight",
    "head.bias"
]:
    source_shape = (
        list(state_dict[key].shape)
        if key in state_dict else None
    )
    target_shape = (
        list(target_state[key].shape)
        if key in target_state else None
    )
    print(
        f"{key}: source={source_shape}, target={target_shape}"
    )

print("\nFIRST INCOMPATIBILITIES")
for item in shape_mismatches[:15]:
    print(item)

print("\nFIRST MISSING ENCODER KEYS")
print(missing_encoder[:15])

report = {
    "experiment": "Experiment04_Step4.2C",
    "model": MODEL_NAME,
    "global_pool": "avg",
    "num_classes": NUM_CLASSES,
    "timm_version": timm.__version__,
    "source_encoder_entries": len(source_encoder),
    "compatible_encoder_entries": len(compatible_encoder),
    "missing_encoder_keys": missing_encoder,
    "shape_mismatches": shape_mismatches,
    "compatible_encoder_elements": compatible_elements,
    "source_encoder_elements": source_elements,
    "encoder_compatible_fraction": (
        compatible_elements / source_elements
    ),
    "gpu_memory_allocated_for_model": False,
    "weights_loaded": False,
    "created_utc": datetime.now(timezone.utc).isoformat()
}

if REPORT_PATH.exists():
    raise FileExistsError(
        f"Report already exists: {REPORT_PATH}"
    )

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nOUTPUT FILE:", REPORT_PATH)
print("STEP 4.2C COMPLETED")


# %% [cell 94] Experiment 04 - Step 4.3A

from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import numpy as np
import torch
import timm

# ============================================================
# Experiment 04 - Step 4.3A
# RETFound Encoder Loading and GPU Forward Smoke Test
# ============================================================

ROOT = DR_RESEARCH
REPORT_DIR = ROOT / "experiment04_results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "RETFound_GPU_Forward_Smoke_Test.json"

assert not REPORT_PATH.exists(), "Smoke-test report already exists."
assert "state_dict" in globals()
assert "val_loader" in globals()
assert torch.cuda.is_available()

MODEL_NAME = "vit_large_patch16_224"
NUM_CLASSES = 5
DEVICE = torch.device("cuda")

EXPECTED_NEW_KEYS = {
    "fc_norm.weight",
    "fc_norm.bias",
    "head.weight",
    "head.bias",
}

# ============================================================
# 1. Extract pretrained encoder tensors
# ============================================================

encoder_state = {
    key: value
    for key, value in state_dict.items()
    if key.startswith(("patch_embed.", "blocks."))
    or key in {"cls_token", "pos_embed"}
}

assert len(encoder_state) == 292

# ============================================================
# 2. Create the classification model on CPU
# ============================================================

torch.manual_seed(42)

retfound_model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES,
    global_pool="avg"
)

# Only the new classification normalization and head
# may remain randomly initialized.
load_result = retfound_model.load_state_dict(
    encoder_state,
    strict=False
)

missing_keys = set(load_result.missing_keys)
unexpected_keys = set(load_result.unexpected_keys)

assert missing_keys == EXPECTED_NEW_KEYS, (
    f"Unexpected missing parameters: {sorted(missing_keys)}"
)

assert not unexpected_keys, (
    f"Unexpected pretrained parameters: {sorted(unexpected_keys)}"
)

# Verify that actual pretrained tensor values were copied.
for key in [
    "patch_embed.proj.weight",
    "blocks.0.attn.qkv.weight",
    "blocks.23.mlp.fc2.weight",
    "pos_embed",
]:
    loaded = retfound_model.state_dict()[key]
    source = encoder_state[key]

    assert torch.equal(loaded, source), (
        f"Pretrained weight verification failed: {key}"
    )

print("STEP 4.3A - RETFOUND GPU FORWARD TEST")
print("Architecture:", MODEL_NAME)
print("Total model parameters:", sum(
    p.numel() for p in retfound_model.parameters()
))
print("Pretrained encoder tensors:", len(encoder_state))
print("Missing parameters:", sorted(missing_keys))
print("Unexpected parameters:", sorted(unexpected_keys))
print("Pretrained weight value checks: PASS")

# ============================================================
# 3. Move model to GPU
# ============================================================

gc.collect()
torch.cuda.empty_cache()

retfound_model = retfound_model.to(DEVICE)
retfound_model.eval()

print("\nGPU INFORMATION")
print("GPU:", torch.cuda.get_device_name(0))
print(
    "Total GPU memory (GiB):",
    round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        3
    )
)

# ============================================================
# 4. Validation-only forward smoke test
# ============================================================

images, labels, image_ids = next(iter(val_loader))

images = images[:1].to(DEVICE, non_blocking=True)
labels = labels[:1].to(DEVICE, non_blocking=True)

assert images.shape == (1, 3, 224, 224)

torch.cuda.reset_peak_memory_stats(DEVICE)
torch.cuda.synchronize()

with torch.inference_mode():
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):
        logits = retfound_model(images)

torch.cuda.synchronize()

assert logits.shape == (1, NUM_CLASSES)
assert torch.isfinite(logits).all()

probabilities = torch.softmax(
    logits.float(),
    dim=1
)

assert torch.allclose(
    probabilities.sum(dim=1),
    torch.ones(1, device=DEVICE),
    atol=1e-5
)

peak_allocated_gib = (
    torch.cuda.max_memory_allocated(DEVICE) / 1024**3
)

current_allocated_gib = (
    torch.cuda.memory_allocated(DEVICE) / 1024**3
)

print("\nFORWARD PASS")
print("Input shape:", tuple(images.shape))
print("Output shape:", tuple(logits.shape))
print("Output dtype:", logits.dtype)
print("Output finite:", bool(torch.isfinite(logits).all()))
print("Probability normalization: PASS")
print("Checkpoint weights loaded: PASS")
print("Forward execution: PASS")

print("\nGPU MEMORY")
print("Current allocated GiB:", round(current_allocated_gib, 3))
print("Peak allocated GiB:", round(peak_allocated_gib, 3))

# ============================================================
# 5. Save smoke-test report
# ============================================================

report = {
    "experiment": "Experiment04_Step4.3A",
    "architecture": MODEL_NAME,
    "pretrained_checkpoint": "RETFound_mae_natureCFP.pth",
    "pretrained_encoder_tensor_count": len(encoder_state),
    "missing_keys": sorted(missing_keys),
    "unexpected_keys": sorted(unexpected_keys),
    "new_layers": sorted(EXPECTED_NEW_KEYS),
    "num_classes": NUM_CLASSES,
    "input_shape": list(images.shape),
    "output_shape": list(logits.shape),
    "forward_autocast_dtype": "bfloat16",
    "gpu_name": torch.cuda.get_device_name(0),
    "peak_allocated_gib": peak_allocated_gib,
    "current_allocated_gib": current_allocated_gib,
    "torch_version": torch.__version__,
    "timm_version": timm.__version__,
    "test_split_evaluated": False,
    "backward_test_performed": False,
    "formal_training_started": False,
    "created_utc": datetime.now(timezone.utc).isoformat()
}

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nOUTPUT FILE:", REPORT_PATH)
print("STEP 4.3A COMPLETED")


# %% [cell 95] Experiment 04 - Step 4.3B

# ============================================================
# Experiment 04 - Step 4.3B
# RETFound Full Fine-Tuning Memory Smoke Test
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import time

import torch
import torch.nn as nn

ROOT = DR_RESEARCH
REPORT_DIR = ROOT / "experiment04_results"
REPORT_PATH = REPORT_DIR / "RETFound_Backward_Memory_Smoke_Test.json"

assert not REPORT_PATH.exists(), "Smoke-test report already exists."
assert "retfound_model" in globals()
assert "train_loader" in globals()
assert torch.cuda.is_available()
assert callable(getattr(retfound_model, "set_grad_checkpointing", None))

DEVICE = torch.device("cuda")
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.05
BATCH_SIZE = 1

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Enable activation recomputation to reduce training memory.
retfound_model.set_grad_checkpointing(enable=True)
retfound_model.train()

criterion = nn.CrossEntropyLoss()

# foreach=False avoids temporary memory spikes in AdamW.
optimizer = torch.optim.AdamW(
    retfound_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    foreach=False
)

images, labels, image_ids = next(iter(train_loader))

images = images[:BATCH_SIZE].to(DEVICE)
labels = labels[:BATCH_SIZE].to(DEVICE)

assert images.shape == (1, 3, 224, 224)
assert labels.shape == (1,)

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(DEVICE)

print("STEP 4.3B - RETFOUND BACKWARD MEMORY TEST")
print("GPU:", torch.cuda.get_device_name(0))
print("Batch size:", BATCH_SIZE)
print("Mixed precision: BF16")
print("Gradient checkpointing: Enabled")
print("Optimizer: AdamW")
print("Optimizer foreach: False")

start_time = time.perf_counter()

try:
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):
        logits = retfound_model(images)
        loss = criterion(logits.float(), labels)

    assert logits.shape == (1, 5)
    assert torch.isfinite(loss)

    loss.backward()

    classifier = retfound_model.head
    assert classifier.weight.grad is not None
    assert torch.isfinite(classifier.weight.grad).all()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        retfound_model.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True
    )

    assert torch.isfinite(gradient_norm)

    parameter_before = classifier.weight.detach().clone()

    optimizer.step()

    parameter_change = float(
        (
            classifier.weight.detach()
            - parameter_before
        ).abs().max().item()
    )

    assert parameter_change > 0

    torch.cuda.synchronize()

    elapsed_seconds = time.perf_counter() - start_time

    peak_allocated_gib = (
        torch.cuda.max_memory_allocated(DEVICE) / 1024**3
    )

    peak_reserved_gib = (
        torch.cuda.max_memory_reserved(DEVICE) / 1024**3
    )

    total_memory_gib = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    )

    print("\nTRAINING STEP RESULTS")
    print("Loss:", round(float(loss.item()), 6))
    print("Gradient norm:", round(float(gradient_norm.item()), 6))
    print("Classifier parameter change:", parameter_change)
    print("Forward pass: PASS")
    print("Backward pass: PASS")
    print("AdamW optimizer step: PASS")

    print("\nGPU MEMORY RESULTS")
    print("Peak allocated GiB:", round(peak_allocated_gib, 3))
    print("Peak reserved GiB:", round(peak_reserved_gib, 3))
    print("Total GPU memory GiB:", round(total_memory_gib, 3))
    print("Smoke-test duration seconds:", round(elapsed_seconds, 2))

    report = {
        "experiment": "Experiment04_Step4.3B",
        "architecture": "vit_large_patch16_224",
        "batch_size": BATCH_SIZE,
        "precision": "bfloat16",
        "gradient_checkpointing": True,
        "optimizer": "AdamW",
        "optimizer_foreach": False,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "total_memory_gib": total_memory_gib,
        "duration_seconds": elapsed_seconds,
        "backward_pass_success": True,
        "optimizer_step_success": True,
        "smoke_test_only": True,
        "formal_training_started": False,
        "test_split_evaluated": False,
        "created_utc": datetime.now(timezone.utc).isoformat()
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nOUTPUT FILE:", REPORT_PATH)
    print("STEP 4.3B COMPLETED")

except torch.cuda.OutOfMemoryError:
    print("\nGPU OUT OF MEMORY")
    print("Full fine-tuning did not fit this configuration.")
    print("Do not interpret this as a model quality result.")
    raise

finally:
    optimizer.zero_grad(set_to_none=True)
    del images, labels
    gc.collect()
    torch.cuda.empty_cache()


# %% [cell 96] Experiment 04 - Step 4.3C

# ============================================================
# Experiment 04 - Step 4.3C
# RETFound Micro-Batch Size 4 Memory Test
# ============================================================

import gc
import time
import json
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn as nn

ROOT = DR_RESEARCH
REPORT_PATH = (
    ROOT / "experiment04_results"
    / "RETFound_Batch4_Memory_Test.json"
)

assert not REPORT_PATH.exists()
assert "retfound_model" in globals()
assert "train_loader" in globals()
assert torch.cuda.is_available()
assert train_loader.batch_size >= 4

DEVICE = torch.device("cuda")
MICRO_BATCH_SIZE = 4

# Release the previous smoke-test optimizer state.
if "optimizer" in globals():
    del optimizer

gc.collect()
torch.cuda.empty_cache()

retfound_model.set_grad_checkpointing(enable=True)
retfound_model.train()
retfound_model.zero_grad(set_to_none=True)

smoke_optimizer = torch.optim.AdamW(
    retfound_model.parameters(),
    lr=1e-5,
    weight_decay=0.05,
    foreach=False
)

smoke_criterion = nn.CrossEntropyLoss()

batch_images, batch_labels, _ = next(iter(train_loader))
batch_images = batch_images[:MICRO_BATCH_SIZE].to(DEVICE)
batch_labels = batch_labels[:MICRO_BATCH_SIZE].to(DEVICE)

assert batch_images.shape == (4, 3, 224, 224)

torch.cuda.reset_peak_memory_stats(DEVICE)
torch.cuda.synchronize()

start = time.perf_counter()

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = retfound_model(batch_images)
    loss = smoke_criterion(outputs.float(), batch_labels)

assert outputs.shape == (4, 5)
assert torch.isfinite(loss)

loss.backward()

grad_norm = torch.nn.utils.clip_grad_norm_(
    retfound_model.parameters(),
    max_norm=1.0,
    error_if_nonfinite=True
)

assert torch.isfinite(grad_norm)

before = retfound_model.head.weight.detach().clone()
smoke_optimizer.step()

parameter_change = float(
    (retfound_model.head.weight.detach() - before)
    .abs().max().item()
)

assert parameter_change > 0

torch.cuda.synchronize()

elapsed = time.perf_counter() - start
peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

print("STEP 4.3C - RETFOUND BATCH 4 TEST")
print("Micro-batch size:", MICRO_BATCH_SIZE)
print("Input shape:", tuple(batch_images.shape))
print("Output shape:", tuple(outputs.shape))
print("Loss:", round(float(loss.item()), 6))
print("Gradient norm:", round(float(grad_norm.item()), 6))
print("Classifier update:", parameter_change)
print("Forward and backward: PASS")
print("Optimizer update: PASS")

print("\nGPU MEMORY RESULTS")
print("Peak allocated GiB:", round(peak_allocated, 3))
print("Peak reserved GiB:", round(peak_reserved, 3))
print("Total memory GiB:", round(total_memory, 3))
print("Step duration seconds:", round(elapsed, 3))

report = {
    "experiment": "Experiment04_Step4.3C",
    "micro_batch_size": MICRO_BATCH_SIZE,
    "precision": "bfloat16",
    "gradient_checkpointing": True,
    "optimizer": "AdamW",
    "peak_allocated_gib": peak_allocated,
    "peak_reserved_gib": peak_reserved,
    "duration_seconds": elapsed,
    "optimizer_step_success": True,
    "smoke_test_only": True,
    "test_split_evaluated": False,
    "created_utc": datetime.now(timezone.utc).isoformat()
}

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

# Remove temporary training states after successful testing.
smoke_optimizer.zero_grad(set_to_none=True)
del smoke_optimizer, batch_images, batch_labels, outputs, loss
gc.collect()
torch.cuda.empty_cache()

print("\nSTEP 4.3C COMPLETED")


# %% [cell 97] Experiment 04 - Step 4.4A

# ============================================================
# Experiment 04 - Step 4.4A
# Training Protocol and Data Integrity Audit
# ============================================================

import hashlib
import json
from pathlib import Path

import torch
import timm
import pandas as pd

ROOT = DR_RESEARCH

SPLIT_PATH = (
    ROOT / "experiment02_results"
    / "FGADR_Split_V2_Frozen.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

EXPECTED_SPLIT_SHA256 = (
    "a3d83242847834ef38c6d1de8c9566556f33792041370807c709a5af4370d49f"
)

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(block)
    return digest.hexdigest()

assert SPLIT_PATH.is_file()
assert INDEX_PATH.is_file()

split_hash = file_sha256(SPLIT_PATH)
index_hash = file_sha256(INDEX_PATH)

assert split_hash == EXPECTED_SPLIT_SHA256
assert index_hash == EXPECTED_INDEX_SHA256

index_frame = pd.read_csv(INDEX_PATH)

assert len(index_frame) == 1842
assert set(index_frame["split"]) == {"train", "val", "test"}

split_counts = (
    index_frame["split"].value_counts().to_dict()
)

assert split_counts == {
    "train": 1290,
    "val": 276,
    "test": 276
}

assert "retfound_model" in globals()
assert "train_loader" in globals()
assert "val_loader" in globals()
assert "test_loader" in globals()

# Verify normalization and model architecture settings.
norm_eps = {
    "blocks.0.norm1": retfound_model.blocks[0].norm1.eps,
    "blocks.0.norm2": retfound_model.blocks[0].norm2.eps,
    "fc_norm": retfound_model.fc_norm.eps,
}

print("STEP 4.4A - TRAINING PROTOCOL AUDIT")

print("\nDATA INTEGRITY")
print("Frozen split SHA256:", split_hash)
print("Training index SHA256:", index_hash)
print("Split counts:", split_counts)
print("Data integrity: PASS")

print("\nMODEL CONFIGURATION")
print("Architecture: vit_large_patch16_224")
print("Global pooling:", retfound_model.global_pool)
print("LayerNorm epsilon values:", norm_eps)
print("Input size: 224")
print("Precision: BF16")
print("Gradient checkpointing: Enabled")
print("Micro-batch size: 4")
print("Gradient accumulation steps: 4")
print("Effective batch size: 16")

print("\nDATA TRANSFORMS")
print("Train transform:")
print(train_loader.dataset.transform)
print("\nValidation transform:")
print(val_loader.dataset.transform)

print("\nDATA LOADER SIZES")
print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))
print("Test loader used for evaluation: False")

print("\nSTEP 4.4A COMPLETED")


# %% [cell 98] Experiment 04 - Step 4.5A

# ============================================================
# Experiment 04 - Step 4.5A
# RETFound Formal Training - Epoch 1
# ============================================================

import os
import gc
import json
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
)
from tqdm.auto import tqdm

# ============================================================
# 1. Locked experiment configuration
# ============================================================

SEED = 42
NUM_CLASSES = 5
MICRO_BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = 16

MAX_EPOCHS = 25
MIN_EPOCHS = 8
PATIENCE = 6

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.05
GRAD_CLIP_NORM = 1.0

DEVICE = torch.device("cuda")
MODEL_NAME = "vit_large_patch16_224"

ROOT = DR_RESEARCH

EXPERIMENT_DIR = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
)

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = EXPERIMENT_DIR / "last_training_state.pt"
HISTORY_PATH = EXPERIMENT_DIR / "training_history.csv"
CONFIG_PATH = EXPERIMENT_DIR / "training_config.json"
PREDICTION_PATH = EXPERIMENT_DIR / "validation_predictions_epoch001.csv"

assert not CHECKPOINT_PATH.exists(), (
    "A training checkpoint already exists. "
    "Do not restart Epoch 1 over an existing experiment."
)
assert not HISTORY_PATH.exists()
assert not CONFIG_PATH.exists()

assert torch.cuda.is_available()
assert "state_dict" in globals()
assert "train_loader" in globals()
assert "val_loader" in globals()

# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False

# ============================================================
# 3. Validate frozen dataset index
# ============================================================

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

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_SHA256

index_df = pd.read_csv(INDEX_PATH)
train_index = index_df[index_df["split"] == "train"]

assert len(train_index) == 1290

class_counts = np.bincount(
    train_index["dr_grade"].to_numpy(dtype=int),
    minlength=NUM_CLASSES
)

assert class_counts.tolist() == [71, 148, 417, 453, 201]

# Match the inverse-square-root weighting used by the baseline.
raw_weights = 1.0 / np.sqrt(class_counts.astype(float))
normalized_weights = raw_weights / raw_weights.mean()

class_weights = torch.tensor(
    normalized_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("STEP 4.5A - RETFOUND FORMAL TRAINING")
print("Class counts:", class_counts.tolist())
print("Class weights:", np.round(normalized_weights, 6).tolist())

# ============================================================
# 4. Rebuild training loader with micro-batch size 4
# ============================================================

train_generator = torch.Generator()
train_generator.manual_seed(SEED)

retfound_train_loader = DataLoader(
    train_loader.dataset,
    batch_size=MICRO_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    generator=train_generator
)

assert len(retfound_train_loader.dataset) == 1290
assert len(retfound_train_loader) == 323
assert len(val_loader.dataset) == 276

print("\nDATA COVERAGE")
print("Training images:", len(retfound_train_loader.dataset))
print("Training micro-batches:", len(retfound_train_loader))
print("Validation images:", len(val_loader.dataset))
print("Effective batch size:", EFFECTIVE_BATCH_SIZE)

# ============================================================
# 5. Remove smoke-test objects
# ============================================================

if "retfound_model" in globals():
    del retfound_model

if "smoke_criterion" in globals():
    del smoke_criterion

if "criterion" in globals():
    del criterion

gc.collect()
torch.cuda.empty_cache()

# ============================================================
# 6. Build fresh RETFound model from original weights
# ============================================================

torch.manual_seed(SEED)

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES,
    global_pool="avg"
)

encoder_state = {
    key: value
    for key, value in state_dict.items()
    if key.startswith(("patch_embed.", "blocks."))
    or key in {"cls_token", "pos_embed"}
}

assert len(encoder_state) == 292

load_result = model.load_state_dict(
    encoder_state,
    strict=False
)

expected_missing = {
    "fc_norm.weight",
    "fc_norm.bias",
    "head.weight",
    "head.bias"
}

assert set(load_result.missing_keys) == expected_missing
assert len(load_result.unexpected_keys) == 0

for key in [
    "patch_embed.proj.weight",
    "blocks.0.attn.qkv.weight",
    "blocks.23.mlp.fc2.weight",
    "pos_embed"
]:
    assert torch.equal(
        model.state_dict()[key],
        encoder_state[key]
    )

# Check the classification setup.
assert model.global_pool == "avg"
assert model.blocks[0].norm1.eps == 1e-6
assert model.fc_norm.eps == 1e-6

model.set_grad_checkpointing(enable=True)
model = model.to(DEVICE)

# ============================================================
# 7. Loss, optimizer, and scheduler
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    reduction="none"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    foreach=False
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_EPOCHS,
    eta_min=1e-7
)

# ============================================================
# 8. Training epoch
# ============================================================

model.train()
optimizer.zero_grad(set_to_none=True)

torch.cuda.reset_peak_memory_stats(DEVICE)

epoch_start = time.perf_counter()

train_loss_numerator = 0.0
train_weight_denominator = 0.0
train_true = []
train_pred = []

micro_batches = len(retfound_train_loader)

progress = tqdm(
    enumerate(retfound_train_loader),
    total=micro_batches,
    desc="RETFound Epoch 1"
)

for batch_idx, (images, labels, image_ids) in progress:

    images = images.to(DEVICE, non_blocking=True)
    labels = labels.to(DEVICE, non_blocking=True)

    # Each accumulation group contains up to four micro-batches.
    group_start = (
        batch_idx // ACCUMULATION_STEPS
    ) * ACCUMULATION_STEPS

    group_end = min(
        group_start + ACCUMULATION_STEPS,
        micro_batches
    )

    # The last accumulation group can contain fewer than 16 images.
    group_sample_count = min(
        EFFECTIVE_BATCH_SIZE,
        len(retfound_train_loader.dataset)
        - group_start * MICRO_BATCH_SIZE
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):
        logits = model(images)

        individual_losses = criterion(
            logits.float(),
            labels
        )

        # Class weights are applied per sample.
        # Normalize the accumulation group by its actual
        # summed class weights to match weighted mean CE.
        sample_weight_sum = class_weights[labels].sum()

        loss_sum = individual_losses.sum()

    # Accumulate unnormalized gradients.
    loss_sum.backward()

    train_loss_numerator += float(loss_sum.detach().item())
    train_weight_denominator += float(
        sample_weight_sum.detach().item()
    )

    predictions = logits.detach().argmax(dim=1)

    train_true.extend(labels.detach().cpu().tolist())
    train_pred.extend(predictions.cpu().tolist())

    is_group_end = (
        (batch_idx + 1) % ACCUMULATION_STEPS == 0
        or (batch_idx + 1) == micro_batches
    )

    if is_group_end:
        # Weighted CE mean is normalized across all samples
        # in this accumulation group.
        # The preceding backward calls accumulated raw gradients.
        group_weight_total = (
            class_weights[
                torch.tensor(
                    train_true[
                        max(
                            0,
                            len(train_true) - group_sample_count
                        ):
                    ],
                    dtype=torch.long,
                    device=DEVICE
                )
            ].sum()
        )

        assert group_weight_total.item() > 0

        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(group_weight_total)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_NORM,
            error_if_nonfinite=True
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    progress.set_postfix(
        loss=f"{float(individual_losses.mean().item()):.3f}"
    )

assert len(train_true) == 1290

train_loss = (
    train_loss_numerator / train_weight_denominator
)

train_accuracy = accuracy_score(train_true, train_pred)
train_macro_f1 = f1_score(
    train_true,
    train_pred,
    average="macro",
    labels=list(range(NUM_CLASSES)),
    zero_division=0
)

# ============================================================
# 9. Validation - no training, no test data
# ============================================================

model.eval()

val_true = []
val_pred = []
val_ids = []
val_probabilities = []

val_loss_numerator = 0.0
val_weight_denominator = 0.0

with torch.inference_mode():

    for images, labels, image_ids in tqdm(
        val_loader,
        desc="Validation Epoch 1"
    ):

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(images)

        individual_losses = criterion(
            logits.float(),
            labels
        )

        probabilities = torch.softmax(
            logits.float(),
            dim=1
        )

        predictions = probabilities.argmax(dim=1)

        val_loss_numerator += float(
            individual_losses.sum().item()
        )

        val_weight_denominator += float(
            class_weights[labels].sum().item()
        )

        val_true.extend(labels.cpu().tolist())
        val_pred.extend(predictions.cpu().tolist())
        val_ids.extend(list(image_ids))
        val_probabilities.extend(
            probabilities.cpu().tolist()
        )

assert len(val_true) == 276
assert len(set(val_ids)) == 276

val_loss = (
    val_loss_numerator / val_weight_denominator
)

val_accuracy = accuracy_score(val_true, val_pred)

val_macro_f1 = f1_score(
    val_true,
    val_pred,
    labels=list(range(NUM_CLASSES)),
    average="macro",
    zero_division=0
)

val_qwk = cohen_kappa_score(
    val_true,
    val_pred,
    labels=list(range(NUM_CLASSES)),
    weights="quadratic"
)

val_confusion = confusion_matrix(
    val_true,
    val_pred,
    labels=list(range(NUM_CLASSES))
)

# Referable DR = DR grade >= 2.
true_referable = np.asarray(val_true) >= 2
pred_referable = np.asarray(val_pred) >= 2

tp = int(np.sum(true_referable & pred_referable))
fn = int(np.sum(true_referable & ~pred_referable))
tn = int(np.sum(~true_referable & ~pred_referable))
fp = int(np.sum(~true_referable & pred_referable))

referable_sensitivity = tp / (tp + fn)
referable_specificity = tn / (tn + fp)

scheduler.step()

torch.cuda.synchronize()

epoch_duration = time.perf_counter() - epoch_start

peak_allocated = (
    torch.cuda.max_memory_allocated(DEVICE) / 1024**3
)

peak_reserved = (
    torch.cuda.max_memory_reserved(DEVICE) / 1024**3
)

# ============================================================
# 10. Save reproducibility files and training state
# ============================================================

config = {
    "experiment": "RETFound_ViT_Large_V1",
    "architecture": MODEL_NAME,
    "pretraining": "RETFound_mae_natureCFP",
    "pretrained_sha256": (
        "e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485"
    ),
    "training_index_sha256": EXPECTED_INDEX_SHA256,
    "seed": SEED,
    "num_classes": NUM_CLASSES,
    "input_size": 224,
    "global_pool": "avg",
    "micro_batch_size": MICRO_BATCH_SIZE,
    "gradient_accumulation_steps": ACCUMULATION_STEPS,
    "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "scheduler": "CosineAnnealingLR",
    "maximum_epochs": MAX_EPOCHS,
    "minimum_epochs": MIN_EPOCHS,
    "patience": PATIENCE,
    "model_selection_metric": "val_macro_f1",
    "class_counts": class_counts.tolist(),
    "class_weights": normalized_weights.tolist(),
    "precision": "bfloat16",
    "gradient_checkpointing": True,
    "training_images": 1290,
    "validation_images": 276,
    "test_images_evaluated": 0
}

history_row = {
    "epoch": 1,
    "train_loss": train_loss,
    "train_accuracy": train_accuracy,
    "train_macro_f1": train_macro_f1,
    "val_loss": val_loss,
    "val_accuracy": val_accuracy,
    "val_macro_f1": val_macro_f1,
    "val_qwk": val_qwk,
    "referable_sensitivity": referable_sensitivity,
    "referable_specificity": referable_specificity,
    "learning_rate": LEARNING_RATE,
    "duration_seconds": epoch_duration,
    "peak_allocated_gib": peak_allocated,
    "peak_reserved_gib": peak_reserved
}

validation_predictions = pd.DataFrame({
    "image_id": val_ids,
    "true_grade": val_true,
    "predicted_grade": val_pred,
    **{
        f"prob_dr{grade}": [
            values[grade] for values in val_probabilities
        ]
        for grade in range(NUM_CLASSES)
    }
})

# Use temporary files so interrupted writes are not
# mistaken for completed checkpoints.
temporary_checkpoint = Path(str(CHECKPOINT_PATH) + ".partial")

training_state = {
    "epoch": 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_val_macro_f1": float(val_macro_f1),
    "best_epoch": 1,
    "epochs_without_improvement": 0,
    "torch_rng_state": torch.get_rng_state(),
    "cuda_rng_states": torch.cuda.get_rng_state_all(),
    "python_rng_state": random.getstate(),
    "train_loader_generator_state": train_generator.get_state(),
    "config": config
}

torch.save(training_state, temporary_checkpoint)
os.replace(temporary_checkpoint, CHECKPOINT_PATH)

with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)

pd.DataFrame([history_row]).to_csv(
    HISTORY_PATH,
    index=False
)

validation_predictions.to_csv(
    PREDICTION_PATH,
    index=False
)

# ============================================================
# 11. Report
# ============================================================

print("\nEPOCH 1 TRAINING RESULTS")
print("Train loss:", round(train_loss, 6))
print("Train accuracy:", round(train_accuracy, 6))
print("Train macro-F1:", round(train_macro_f1, 6))

print("\nEPOCH 1 VALIDATION RESULTS")
print("Validation loss:", round(val_loss, 6))
print("Validation accuracy:", round(val_accuracy, 6))
print("Validation macro-F1:", round(val_macro_f1, 6))
print("Validation QWK:", round(float(val_qwk), 6))
print(
    "Referable DR sensitivity:",
    round(referable_sensitivity, 6)
)
print(
    "Referable DR specificity:",
    round(referable_specificity, 6)
)

print("\nVALIDATION CONFUSION MATRIX")
print(val_confusion)

print("\nTRAINING RUNTIME")
print("Epoch duration (minutes):", round(epoch_duration / 60, 2))
print("Peak allocated GiB:", round(peak_allocated, 3))
print("Peak reserved GiB:", round(peak_reserved, 3))

print("\nCHECKPOINT")
print("Path:", CHECKPOINT_PATH)
print("Exists:", CHECKPOINT_PATH.is_file())
print(
    "Size GiB:",
    round(CHECKPOINT_PATH.stat().st_size / 1024**3, 3)
)
print("Resume epoch:", 2)
print("Test evaluated:", False)

print("\nSTEP 4.5A COMPLETED")


# %% [cell 99] Experiment 04 - Step 4.5B

# ============================================================
# Experiment 04 - Step 4.5B
# RETFound Formal Training Resume: Epochs 2-25
# ============================================================

import os
import gc
import json
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
)
from tqdm.auto import tqdm

# ============================================================
# 1. Paths and locked configuration
# ============================================================

ROOT = DR_RESEARCH

EXPERIMENT_DIR = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
)

CHECKPOINT_PATH = EXPERIMENT_DIR / "last_training_state.pt"
BEST_MODEL_PATH = EXPERIMENT_DIR / "best_model.pt"
HISTORY_PATH = EXPERIMENT_DIR / "training_history.csv"
CONFIG_PATH = EXPERIMENT_DIR / "training_config.json"
INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

assert CHECKPOINT_PATH.is_file()
assert HISTORY_PATH.is_file()
assert CONFIG_PATH.is_file()
assert torch.cuda.is_available()
assert "train_loader" in globals()
assert "val_loader" in globals()

with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
    config = json.load(handle)

SEED = int(config["seed"])
NUM_CLASSES = int(config["num_classes"])
MICRO_BATCH_SIZE = int(config["micro_batch_size"])
ACCUMULATION_STEPS = int(config["gradient_accumulation_steps"])
MAX_EPOCHS = int(config["maximum_epochs"])
MIN_EPOCHS = int(config["minimum_epochs"])
PATIENCE = int(config["patience"])
LEARNING_RATE = float(config["learning_rate"])
WEIGHT_DECAY = float(config["weight_decay"])

assert NUM_CLASSES == 5
assert MICRO_BATCH_SIZE == 4
assert ACCUMULATION_STEPS == 4
assert MAX_EPOCHS == 25
assert MIN_EPOCHS == 8
assert PATIENCE == 6
assert config["architecture"] == "vit_large_patch16_224"

DEVICE = torch.device("cuda")

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

assert file_sha256(INDEX_PATH) == EXPECTED_INDEX_SHA256

# ============================================================
# 2. Release previous model and audit objects
# ============================================================

for name in [
    "model",
    "optimizer",
    "scheduler",
    "criterion",
    "training_state",
    "retfound_model",
    "checkpoint",
    "state_dict",
    "encoder_state",
    "tensors",
    "source_encoder",
    "classification_model"
]:
    if name in globals():
        del globals()[name]

gc.collect()
torch.cuda.empty_cache()

# ============================================================
# 3. Load formal training checkpoint
# ============================================================

print("STEP 4.5B - RETFOUND TRAINING RESUME")
print("Loading formal training checkpoint...")

resume_state = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

completed_epoch = int(resume_state["epoch"])
start_epoch = completed_epoch + 1

assert completed_epoch >= 1

assert (
    resume_state["config"]["training_index_sha256"]
    == EXPECTED_INDEX_SHA256
)

history_df = pd.read_csv(HISTORY_PATH)

# Check that the persisted history has not advanced beyond
# the recoverable training state.
assert not history_df.empty
assert history_df["epoch"].is_unique
assert int(history_df["epoch"].max()) <= completed_epoch

best_f1 = float(resume_state["best_val_macro_f1"])
best_epoch = int(resume_state["best_epoch"])
no_improve = int(resume_state["epochs_without_improvement"])

# Reconcile history if a previous run stopped between
# saving the checkpoint and updating the CSV.
checkpoint_history = resume_state.get("history_rows")

if checkpoint_history is not None:
    history_df = pd.DataFrame(checkpoint_history)

assert int(history_df["epoch"].max()) == completed_epoch, (
    "Checkpoint and history disagree. Stop and inspect."
)

assert len(history_df) == completed_epoch
assert abs(
    float(history_df.loc[
        history_df["epoch"] == best_epoch,
        "val_macro_f1"
    ].iloc[0]) - best_f1
) < 1e-8

print("Completed epoch:", completed_epoch)
print("Resume from epoch:", start_epoch)
print("Previous best epoch:", best_epoch)
print("Previous best validation macro-F1:", round(best_f1, 6))

# ============================================================
# 4. Rebuild data loader
# ============================================================

train_generator = torch.Generator()
train_generator.set_state(
    resume_state["train_loader_generator_state"]
)

resume_train_loader = DataLoader(
    train_loader.dataset,
    batch_size=MICRO_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    generator=train_generator
)

assert len(resume_train_loader.dataset) == 1290
assert len(resume_train_loader) == 323
assert len(val_loader.dataset) == 276

# ============================================================
# 5. Rebuild model and optimizer
# ============================================================

model = timm.create_model(
    config["architecture"],
    pretrained=False,
    num_classes=NUM_CLASSES,
    global_pool="avg"
)

model.load_state_dict(
    resume_state["model_state_dict"],
    strict=True
)

assert model.blocks[0].norm1.eps == 1e-6
assert model.fc_norm.eps == 1e-6

model.set_grad_checkpointing(enable=True)
model = model.to(DEVICE)

class_weights = torch.tensor(
    config["class_weights"],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    reduction="none"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    foreach=False
)

optimizer.load_state_dict(
    resume_state["optimizer_state_dict"]
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_EPOCHS,
    eta_min=1e-7
)

scheduler.load_state_dict(
    resume_state["scheduler_state_dict"]
)

# ============================================================
# 6. Restore random states
# ============================================================

torch.set_rng_state(resume_state["torch_rng_state"].cpu())

torch.cuda.set_rng_state_all(
    [state.cpu() for state in resume_state["cuda_rng_states"]]
)

random.setstate(resume_state["python_rng_state"])

if "numpy_rng_state" in resume_state:
    np.random.set_state(resume_state["numpy_rng_state"])

# Release checkpoint after restoring all required objects.
del resume_state
gc.collect()

# ============================================================
# 7. Atomic save helper
# ============================================================

def atomic_torch_save(payload, path):
    temporary_path = Path(str(path) + ".partial")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)

def atomic_csv_save(frame, path):
    temporary_path = Path(str(path) + ".partial")
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)

# Preserve Epoch 1 as the initial best model.
if not BEST_MODEL_PATH.exists():
    if best_epoch != completed_epoch:
        raise RuntimeError(
            "Best model is missing and the current checkpoint "
            "does not represent the best epoch."
        )

    atomic_torch_save(
        {
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "val_macro_f1": best_f1,
            "config": config
        },
        BEST_MODEL_PATH
    )

print("Model and optimizer resumed: PASS")
print("Best checkpoint available: PASS")
print("Frozen split integrity: PASS")
print("Test split accessed: False")

# ============================================================
# 8. Resume training
# ============================================================

if start_epoch > MAX_EPOCHS:
    print("Maximum training epochs already completed.")

elif completed_epoch >= MIN_EPOCHS and no_improve >= PATIENCE:
    print("Early stopping was already reached.")

else:

    for epoch in range(start_epoch, MAX_EPOCHS + 1):

        epoch_start = time.perf_counter()
        epoch_lr = float(optimizer.param_groups[0]["lr"])

        model.train()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.reset_peak_memory_stats(DEVICE)

        train_true = []
        train_pred = []

        train_loss_numerator = 0.0
        train_weight_denominator = 0.0

        group_weight_total = torch.zeros(
            (),
            device=DEVICE,
            dtype=torch.float32
        )

        train_progress = tqdm(
            resume_train_loader,
            desc=f"RETFound Epoch {epoch}",
            leave=True
        )

        for batch_idx, (images, labels, _) in enumerate(
            train_progress
        ):

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            ):
                logits = model(images)
                individual_losses = criterion(
                    logits.float(),
                    labels
                )

                loss_sum = individual_losses.sum()

            assert torch.isfinite(loss_sum), (
                f"Non-finite loss at epoch {epoch}."
            )

            loss_sum.backward()

            sample_weight_sum = (
                class_weights[labels].sum().detach()
            )

            group_weight_total += sample_weight_sum

            train_loss_numerator += float(
                loss_sum.detach().item()
            )

            train_weight_denominator += float(
                sample_weight_sum.item()
            )

            predictions = logits.detach().argmax(dim=1)

            train_true.extend(labels.cpu().tolist())
            train_pred.extend(
                predictions.cpu().tolist()
            )

            is_group_end = (
                (batch_idx + 1) % ACCUMULATION_STEPS == 0
                or (batch_idx + 1) == len(resume_train_loader)
            )

            if is_group_end:
                assert group_weight_total.item() > 0

                # Normalize the accumulated weighted CE gradient
                # by the actual accumulated sample-weight sum.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_weight_total)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=True
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                group_weight_total.zero_()

            train_progress.set_postfix(
                loss=f"{float(individual_losses.mean().item()):.3f}"
            )

        assert len(train_true) == 1290

        train_loss = (
            train_loss_numerator / train_weight_denominator
        )

        train_accuracy = accuracy_score(
            train_true,
            train_pred
        )

        train_macro_f1 = f1_score(
            train_true,
            train_pred,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0
        )

        # ====================================================
        # 9. Validation
        # ====================================================

        model.eval()

        val_true = []
        val_pred = []
        val_ids = []
        val_probabilities = []

        val_loss_numerator = 0.0
        val_weight_denominator = 0.0

        with torch.inference_mode():

            for images, labels, image_ids in tqdm(
                val_loader,
                desc=f"Validation Epoch {epoch}",
                leave=False
            ):

                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16
                ):
                    logits = model(images)

                individual_losses = criterion(
                    logits.float(),
                    labels
                )

                probabilities = torch.softmax(
                    logits.float(),
                    dim=1
                )

                predictions = probabilities.argmax(dim=1)

                val_loss_numerator += float(
                    individual_losses.sum().item()
                )

                val_weight_denominator += float(
                    class_weights[labels].sum().item()
                )

                val_true.extend(labels.cpu().tolist())
                val_pred.extend(predictions.cpu().tolist())
                val_ids.extend(list(image_ids))
                val_probabilities.extend(
                    probabilities.cpu().tolist()
                )

        assert len(val_true) == 276
        assert len(set(val_ids)) == 276

        val_loss = (
            val_loss_numerator / val_weight_denominator
        )

        val_accuracy = accuracy_score(
            val_true,
            val_pred
        )

        val_macro_f1 = f1_score(
            val_true,
            val_pred,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0
        )

        val_qwk = cohen_kappa_score(
            val_true,
            val_pred,
            labels=list(range(NUM_CLASSES)),
            weights="quadratic"
        )

        true_referable = np.asarray(val_true) >= 2
        pred_referable = np.asarray(val_pred) >= 2

        tp = int(np.sum(
            true_referable & pred_referable
        ))

        fn = int(np.sum(
            true_referable & ~pred_referable
        ))

        tn = int(np.sum(
            ~true_referable & ~pred_referable
        ))

        fp = int(np.sum(
            ~true_referable & pred_referable
        ))

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        is_best = val_macro_f1 > best_f1 + 1e-12

        if is_best:
            best_f1 = float(val_macro_f1)
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        scheduler.step()

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - epoch_start

        peak_allocated = (
            torch.cuda.max_memory_allocated(DEVICE) / 1024**3
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved(DEVICE) / 1024**3
        )

        # ====================================================
        # 10. Persist metrics and checkpoints
        # ====================================================

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "val_qwk": val_qwk,
            "referable_sensitivity": sensitivity,
            "referable_specificity": specificity,
            "learning_rate": epoch_lr,
            "duration_seconds": elapsed,
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved
        }

        history_df = pd.concat(
            [history_df, pd.DataFrame([row])],
            ignore_index=True
        )

        # Save validation predictions for the best epoch.
        if is_best:
            prediction_frame = pd.DataFrame({
                "image_id": val_ids,
                "true_grade": val_true,
                "predicted_grade": val_pred,
                **{
                    f"prob_dr{grade}": [
                        values[grade]
                        for values in val_probabilities
                    ]
                    for grade in range(NUM_CLASSES)
                }
            })

            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1": float(val_macro_f1),
                    "config": config
                },
                BEST_MODEL_PATH
            )

            atomic_csv_save(
                prediction_frame,
                EXPERIMENT_DIR / "best_validation_predictions.csv"
            )

        training_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "epochs_without_improvement": no_improve,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "train_loader_generator_state": (
                train_generator.get_state()
            ),
            "config": config,
            "history_rows": history_df.to_dict(
                orient="records"
            )
        }

        atomic_torch_save(
            training_state,
            CHECKPOINT_PATH
        )

        atomic_csv_save(
            history_df,
            HISTORY_PATH
        )

        del training_state
        gc.collect()

        # ====================================================
        # 11. Epoch summary
        # ====================================================

        print(f"\nEPOCH {epoch} SUMMARY")
        print("Train loss:", round(train_loss, 6))
        print("Train macro-F1:", round(train_macro_f1, 6))
        print("Validation loss:", round(val_loss, 6))
        print("Validation macro-F1:", round(val_macro_f1, 6))
        print("Validation QWK:", round(float(val_qwk), 6))
        print("Best epoch:", best_epoch)
        print("Best macro-F1:", round(best_f1, 6))
        print("Epochs without improvement:", no_improve)
        print("Epoch duration minutes:", round(elapsed / 60, 2))
        print("Peak GPU allocated GiB:", round(peak_allocated, 3))
        print("Best model updated:", is_best)
        print("Resume checkpoint saved: PASS")

        if epoch >= MIN_EPOCHS and no_improve >= PATIENCE:
            print("\nEARLY STOPPING TRIGGERED")
            print("Stopped at epoch:", epoch)
            break

# ============================================================
# 12. Final training summary
# ============================================================

history_df = pd.read_csv(HISTORY_PATH)

print("\nRETFOUND TRAINING SUMMARY")
print("Completed epochs:", int(history_df["epoch"].max()))

best_row = history_df.loc[
    history_df["val_macro_f1"].idxmax()
]

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

print("Best model exists:", BEST_MODEL_PATH.is_file())
print("Resume checkpoint exists:", CHECKPOINT_PATH.is_file())
print("Test set evaluated: False")

print("\nSTEP 4.5B COMPLETED")


# %% [cell 100] Experiment 04 - Step 4.6A

# ============================================================
# Experiment 04 - Step 4.6A
# Paired Validation Prediction Audit
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score
)

ROOT = DR_RESEARCH

EFFICIENTNET_PATH = (
    ROOT / "experiment03_results"
    / "efficientnet_b0_baseline_v1"
    / "validation_predictions.csv"
)

RETFOUND_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_validation_predictions.csv"
)

RETFOUND_HISTORY_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "training_history.csv"
)

OUTPUT_DIR = ROOT / "experiment04_results"
PAIRED_PATH = OUTPUT_DIR / "FGADR_Validation_Paired_Predictions_V1.csv"
AUDIT_PATH = OUTPUT_DIR / "FGADR_Validation_Paired_Audit_V1.json"

for path in [
    EFFICIENTNET_PATH,
    RETFOUND_PATH,
    RETFOUND_HISTORY_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

assert not PAIRED_PATH.exists()
assert not AUDIT_PATH.exists()

efficientnet = pd.read_csv(EFFICIENTNET_PATH)
retfound = pd.read_csv(RETFOUND_PATH)
history = pd.read_csv(RETFOUND_HISTORY_PATH)

print("STEP 4.6A - PAIRED VALIDATION AUDIT")

print("\nINPUT FILES")
print("EfficientNet rows:", len(efficientnet))
print("EfficientNet columns:", efficientnet.columns.tolist())
print("RETFound rows:", len(retfound))
print("RETFound columns:", retfound.columns.tolist())

# ============================================================
# 1. Validate RETFound checkpoint selection
# ============================================================

best_row = history.loc[history["val_macro_f1"].idxmax()]

assert int(best_row["epoch"]) == 16
assert np.isclose(
    best_row["val_macro_f1"],
    0.534207,
    atol=1e-6
)

# ============================================================
# 2. Harmonize prediction column names
# ============================================================

def harmonize_predictions(frame, model_name):
    frame = frame.copy()

    possible_true = [
        "true_grade",
        "true_label",
        "label",
        "dr_grade",
        "y_true"
    ]

    possible_pred = [
        "predicted_grade",
        "pred_grade",
        "predicted_label",
        "prediction",
        "y_pred"
    ]

    true_col = next(
        (col for col in possible_true if col in frame.columns),
        None
    )

    pred_col = next(
        (col for col in possible_pred if col in frame.columns),
        None
    )

    assert "image_id" in frame.columns, (
        f"{model_name}: image_id column not found."
    )

    assert true_col is not None, (
        f"{model_name}: true label column not found."
    )

    assert pred_col is not None, (
        f"{model_name}: prediction column not found."
    )

    frame = frame.rename(
        columns={
            true_col: "true_grade",
            pred_col: f"{model_name}_pred"
        }
    )

    assert len(frame) == 276
    assert frame["image_id"].notna().all()
    assert frame["image_id"].is_unique

    assert frame["true_grade"].notna().all()
    assert frame[f"{model_name}_pred"].notna().all()

    assert set(frame["true_grade"].astype(int)).issubset(
        set(range(5))
    )

    assert set(frame[f"{model_name}_pred"].astype(int)).issubset(
        set(range(5))
    )

    keep_columns = [
        "image_id",
        "true_grade",
        f"{model_name}_pred"
    ]

    return frame[keep_columns]

efficientnet = harmonize_predictions(
    efficientnet,
    "efficientnet"
)

retfound = harmonize_predictions(
    retfound,
    "retfound"
)

# ============================================================
# 3. Pair images and verify ground truth consistency
# ============================================================

paired = efficientnet.merge(
    retfound,
    on="image_id",
    how="outer",
    suffixes=("_efficientnet", "_retfound"),
    indicator=True,
    validate="one_to_one"
)

assert len(paired) == 276
assert paired["_merge"].eq("both").all()

assert (
    paired["true_grade_efficientnet"]
    == paired["true_grade_retfound"]
).all(), "Ground-truth disagreement between models."

paired = paired.drop(columns=["_merge"])

paired = paired.rename(
    columns={"true_grade_efficientnet": "true_grade"}
)

paired = paired.drop(columns=["true_grade_retfound"])

paired = paired.sort_values("image_id").reset_index(drop=True)

assert (
    paired["true_grade"].value_counts().sort_index().tolist()
    == [15, 32, 89, 97, 43]
)

# ============================================================
# 4. Calculate per-model metrics
# ============================================================

results = {}

for model_name in ["efficientnet", "retfound"]:
    y_true = paired["true_grade"].to_numpy(dtype=int)
    y_pred = paired[f"{model_name}_pred"].to_numpy(dtype=int)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(5)),
            zero_division=0
        )
    )

    results[model_name] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=list(range(5)),
                zero_division=0
            )
        ),
        "qwk": float(
            cohen_kappa_score(
                y_true,
                y_pred,
                weights="quadratic"
            )
        ),
        "per_class_f1": f1.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(5))
        ).tolist()
    }

    print(f"\n{model_name.upper()} METRICS")
    print("Accuracy:", round(results[model_name]["accuracy"], 6))
    print("Macro-F1:", round(results[model_name]["macro_f1"], 6))
    print("QWK:", round(results[model_name]["qwk"], 6))
    print("Per-class F1:", np.round(f1, 4).tolist())
    print("Per-class recall:", np.round(recall, 4).tolist())

    print("Confusion matrix:")
    print(np.asarray(results[model_name]["confusion_matrix"]))

assert np.isclose(
    results["efficientnet"]["macro_f1"],
    0.561555,
    atol=1e-5
)

assert np.isclose(
    results["retfound"]["macro_f1"],
    0.534207,
    atol=1e-5
)

# ============================================================
# 5. Paired correctness comparison
# ============================================================

paired["efficientnet_correct"] = (
    paired["efficientnet_pred"] == paired["true_grade"]
)

paired["retfound_correct"] = (
    paired["retfound_pred"] == paired["true_grade"]
)

both_correct = int(
    (paired["efficientnet_correct"] &
     paired["retfound_correct"]).sum()
)

efficientnet_only = int(
    (paired["efficientnet_correct"] &
     ~paired["retfound_correct"]).sum()
)

retfound_only = int(
    (~paired["efficientnet_correct"] &
     paired["retfound_correct"]).sum()
)

both_wrong = int(
    (~paired["efficientnet_correct"] &
     ~paired["retfound_correct"]).sum()
)

assert sum([
    both_correct,
    efficientnet_only,
    retfound_only,
    both_wrong
]) == 276

paired_comparison = {
    "both_correct": both_correct,
    "efficientnet_only_correct": efficientnet_only,
    "retfound_only_correct": retfound_only,
    "both_wrong": both_wrong
}

print("\nPAIRED CORRECTNESS")
for key, value in paired_comparison.items():
    print(f"{key}: {value}")

# ============================================================
# 6. Save reproducible paired audit
# ============================================================

paired.to_csv(PAIRED_PATH, index=False)

audit = {
    "experiment": "Experiment04_Step4.6A",
    "dataset": "FGADR",
    "split": "val",
    "sample_count": len(paired),
    "retfound_selected_epoch": 16,
    "selection_metric": "validation_macro_f1",
    "model_metrics": results,
    "paired_correctness": paired_comparison,
    "test_split_evaluated": False
}

with open(AUDIT_PATH, "w", encoding="utf-8") as handle:
    json.dump(audit, handle, indent=2)

print("\nOUTPUT FILES")
print(PAIRED_PATH)
print(AUDIT_PATH)

print("\nSTEP 4.6A COMPLETED")


# %% [cell 101]

from pathlib import Path

ROOT = DR_RESEARCH

print("GOOGLE DRIVE DIAGNOSTIC")
print("Drive mounted:", DR_RESEARCH.parent.is_dir())
print("Research folder exists:", ROOT.is_dir())

if not ROOT.is_dir():
    pass  # Google Drive mount removed

print("\nEXPERIMENT FOLDERS")

for folder in [
    ROOT / "experiment03_results",
    ROOT / "experiment04_results",
]:
    print(f"\nFolder: {folder}")
    print("Exists:", folder.is_dir())

    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            print(" -", path.name)

print("\nVALIDATION PREDICTION FILES")

if ROOT.is_dir():
    for path in sorted(ROOT.rglob("*validation*prediction*.csv")):
        print(path)

print("\nDIAGNOSTIC COMPLETED")


# %% [cell 102]

from pathlib import Path

ROOT = DR_RESEARCH

folders = {
    "EfficientNet": (
        ROOT / "experiment03_results" / "efficientnet_b0_baseline_v1"
    ),
    "RETFound": (
        ROOT / "experiment04_results" / "retfound_vit_large_v1"
    ),
}

print("STEP 4.6A - TARGETED FILE CHECK")

for name, folder in folders.items():
    print(f"\n{name}")
    print("Folder exists:", folder.is_dir())

    if folder.is_dir():
        for file in sorted(folder.iterdir()):
            print(" -", file.name)

print("\nTARGETED FILE CHECK COMPLETED")


# %% [cell 103] Experiment 04 - Step 4.6B

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix
)

# ============================================================
# Experiment 04 - Step 4.6B
# Paired Validation Performance Analysis
# ============================================================

ROOT = DR_RESEARCH

EFF_PATH = ROOT / "experiment03_results/efficientnet_b0_baseline_v1/validation_predictions.csv"
RET_PATH = ROOT / "experiment04_results/retfound_vit_large_v1/best_validation_predictions.csv"

OUT_DIR = ROOT / "experiment04_results"
PAIRED_PATH = OUT_DIR / "FGADR_Validation_Paired_Predictions_V1.csv"
REPORT_PATH = OUT_DIR / "FGADR_Validation_Paired_Audit_V1.json"

assert EFF_PATH.is_file()
assert RET_PATH.is_file()

eff = pd.read_csv(EFF_PATH)
ret = pd.read_csv(RET_PATH)

print("STEP 4.6B - PAIRED VALIDATION ANALYSIS")
print("EfficientNet columns:", eff.columns.tolist())
print("RETFound columns:", ret.columns.tolist())

def standardize(df, name):
    true_candidates = ["true_grade", "true_label", "label", "dr_grade", "y_true"]
    pred_candidates = ["predicted_grade", "pred_grade", "predicted_label", "prediction", "y_pred"]

    true_col = next((c for c in true_candidates if c in df.columns), None)
    pred_col = next((c for c in pred_candidates if c in df.columns), None)

    assert "image_id" in df.columns
    assert true_col is not None, f"{name}: true label column missing"
    assert pred_col is not None, f"{name}: prediction column missing"

    result = df[["image_id", true_col, pred_col]].copy()
    result.columns = ["image_id", f"{name}_true", f"{name}_pred"]

    assert len(result) == 276
    assert result["image_id"].is_unique
    assert result.notna().all().all()

    return result

eff = standardize(eff, "efficientnet")
ret = standardize(ret, "retfound")

paired = eff.merge(
    ret,
    on="image_id",
    how="outer",
    validate="one_to_one",
    indicator=True
)

assert len(paired) == 276
assert paired["_merge"].eq("both").all()
assert (paired["efficientnet_true"] == paired["retfound_true"]).all()

paired = paired.drop(columns=["_merge", "retfound_true"])
paired = paired.rename(columns={"efficientnet_true": "true_grade"})
paired = paired.sort_values("image_id").reset_index(drop=True)

y_true = paired["true_grade"].to_numpy(dtype=int)

assert np.bincount(y_true, minlength=5).tolist() == [15, 32, 89, 97, 43]

results = {}

for name in ["efficientnet", "retfound"]:
    y_pred = paired[f"{name}_pred"].to_numpy(dtype=int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        zero_division=0
    )

    results[name] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, average="macro",
            labels=[0, 1, 2, 3, 4], zero_division=0
        )),
        "qwk": float(cohen_kappa_score(
            y_true, y_pred, labels=[0, 1, 2, 3, 4],
            weights="quadratic"
        )),
        "per_class_f1": f1.tolist(),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1, 2, 3, 4]
        ).tolist()
    }

    print(f"\n{name.upper()} RESULTS")
    print("Accuracy:", round(results[name]["accuracy"], 6))
    print("Macro-F1:", round(results[name]["macro_f1"], 6))
    print("QWK:", round(results[name]["qwk"], 6))
    print("Per-class F1:", np.round(f1, 4).tolist())
    print("Per-class recall:", np.round(recall, 4).tolist())

    print("Confusion matrix:")
    print(np.array(results[name]["confusion_matrix"]))

assert np.isclose(results["efficientnet"]["macro_f1"], 0.561555, atol=1e-5)
assert np.isclose(results["retfound"]["macro_f1"], 0.534207, atol=1e-5)

paired["efficientnet_correct"] = (
    paired["efficientnet_pred"] == paired["true_grade"]
)
paired["retfound_correct"] = (
    paired["retfound_pred"] == paired["true_grade"]
)

comparison = {
    "both_correct": int((
        paired["efficientnet_correct"] &
        paired["retfound_correct"]
    ).sum()),
    "efficientnet_only_correct": int((
        paired["efficientnet_correct"] &
        ~paired["retfound_correct"]
    ).sum()),
    "retfound_only_correct": int((
        ~paired["efficientnet_correct"] &
        paired["retfound_correct"]
    ).sum()),
    "both_wrong": int((
        ~paired["efficientnet_correct"] &
        ~paired["retfound_correct"]
    ).sum())
}

assert sum(comparison.values()) == 276

print("\nPAIRED CORRECTNESS")
for key, value in comparison.items():
    print(f"{key}: {value}")

assert not PAIRED_PATH.exists(), "Paired predictions already exist"
assert not REPORT_PATH.exists(), "Audit report already exists"

paired.to_csv(PAIRED_PATH, index=False)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "experiment": "Experiment04_Step4.6B",
        "split": "val",
        "sample_count": 276,
        "retfound_best_epoch": 16,
        "model_results": results,
        "paired_correctness": comparison,
        "test_split_evaluated": False
    }, f, indent=2)

print("\nOUTPUT FILES")
print(PAIRED_PATH)
print(REPORT_PATH)
print("\nSTEP 4.6B COMPLETED")


# %% [cell 104] Experiment 04 - Step 4.6C

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, cohen_kappa_score

# ============================================================
# Experiment 04 - Step 4.6C
# Paired Statistical Validation Audit
# ============================================================

ROOT = DR_RESEARCH
RESULT_DIR = ROOT / "experiment04_results"

PAIRED_PATH = RESULT_DIR / "FGADR_Validation_Paired_Predictions_V1.csv"
EFF_PATH = (
    ROOT / "experiment03_results"
    / "efficientnet_b0_baseline_v1"
    / "validation_predictions.csv"
)

REPORT_PATH = RESULT_DIR / "FGADR_Paired_Statistics_V1.json"

assert PAIRED_PATH.is_file()
assert EFF_PATH.is_file()
assert not REPORT_PATH.exists()

paired = pd.read_csv(PAIRED_PATH)
eff_metadata = pd.read_csv(EFF_PATH)[
    ["image_id", "risk_group_id"]
]

df = paired.merge(
    eff_metadata,
    on="image_id",
    how="left",
    validate="one_to_one"
)

assert len(df) == 276
assert df["risk_group_id"].notna().all()
assert df["image_id"].is_unique

y = df["true_grade"].to_numpy(dtype=int)
a = df["efficientnet_pred"].to_numpy(dtype=int)
b = df["retfound_pred"].to_numpy(dtype=int)

correct_a = (a == y)
correct_b = (b == y)

a_only = int(np.sum(correct_a & ~correct_b))
b_only = int(np.sum(~correct_a & correct_b))

# Exact McNemar test via conditional binomial test.
mcnemar_p = float(
    binomtest(
        min(a_only, b_only),
        n=a_only + b_only,
        p=0.5,
        alternative="two-sided"
    ).pvalue
)

# Source-risk-group bootstrap to preserve known
# possible within-group image dependencies.
group_indices = [
    np.flatnonzero(
        df["risk_group_id"].to_numpy() == group
    )
    for group in df["risk_group_id"].unique()
]

rng = np.random.default_rng(42)
n_boot = 2000
delta_f1 = []
delta_qwk = []

for _ in range(n_boot):
    sampled_groups = rng.integers(
        0, len(group_indices), size=len(group_indices)
    )

    idx = np.concatenate([
        group_indices[g] for g in sampled_groups
    ])

    sample_y = y[idx]
    sample_a = a[idx]
    sample_b = b[idx]

    delta_f1.append(
        f1_score(
            sample_y, sample_b,
            labels=[0, 1, 2, 3, 4],
            average="macro", zero_division=0
        )
        - f1_score(
            sample_y, sample_a,
            labels=[0, 1, 2, 3, 4],
            average="macro", zero_division=0
        )
    )

    delta_qwk.append(
        cohen_kappa_score(
            sample_y, sample_b,
            labels=[0, 1, 2, 3, 4],
            weights="quadratic"
        )
        - cohen_kappa_score(
            sample_y, sample_a,
            labels=[0, 1, 2, 3, 4],
            weights="quadratic"
        )
    )

def confidence_interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()

observed_delta_f1 = (
    f1_score(y, b, labels=[0,1,2,3,4],
             average="macro", zero_division=0)
    - f1_score(y, a, labels=[0,1,2,3,4],
               average="macro", zero_division=0)
)

observed_delta_qwk = (
    cohen_kappa_score(
        y, b, labels=[0,1,2,3,4], weights="quadratic"
    )
    - cohen_kappa_score(
        y, a, labels=[0,1,2,3,4], weights="quadratic"
    )
)

report = {
    "experiment": "Experiment04_Step4.6C",
    "dataset": "FGADR",
    "split": "val",
    "n_images": len(df),
    "n_source_risk_groups": len(group_indices),
    "mcnemar_exact_p": mcnemar_p,
    "efficientnet_only_correct": a_only,
    "retfound_only_correct": b_only,
    "bootstrap_replicates": n_boot,
    "bootstrap_unit": "source_risk_group",
    "retfound_minus_efficientnet": {
        "macro_f1": {
            "observed": float(observed_delta_f1),
            "ci95": confidence_interval(delta_f1)
        },
        "qwk": {
            "observed": float(observed_delta_qwk),
            "ci95": confidence_interval(delta_qwk)
        }
    },
    "test_split_evaluated": False
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("STEP 4.6C - STATISTICAL AUDIT")
print("Validation images:", len(df))
print("Source-risk groups:", len(group_indices))
print("McNemar exact p-value:", round(mcnemar_p, 6))
print(
    "Macro-F1 difference:",
    round(observed_delta_f1, 6)
)
print(
    "Macro-F1 95% CI:",
    np.round(confidence_interval(delta_f1), 6).tolist()
)
print(
    "QWK difference:",
    round(observed_delta_qwk, 6)
)
print(
    "QWK 95% CI:",
    np.round(confidence_interval(delta_qwk), 6).tolist()
)
print("Report:", REPORT_PATH)
print("STEP 4.6C COMPLETED")
