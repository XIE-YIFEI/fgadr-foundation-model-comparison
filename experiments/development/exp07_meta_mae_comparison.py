# Experiment 07 - Meta MAE ViT-L/16 fine-tuning and paired attribution vs RETFound
#
# Source: FGADR_MA_Mask_Audit.ipynb (master notebook, 392 cells), cells 194-246, run step by step in Google Colab (L4 GPU).
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
# NOTE: extracted automatically from the master notebook and not yet
# checked by the author against the original runs.

# %% Paths
from common.paths import ROOT as DR_RESEARCH
from common.paths import CACHE as LOCAL_DISK

# %% [cell 194] Experiment 07 - Step 07.1

# ============================================================
# Experiment 07 - Step 07.1
# Inspect Existing RETFound Training Artifacts
# Read-only inspection: no training, no inference
# ============================================================

from pathlib import Path
import json

ROOT = DR_RESEARCH

EXP04 = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
)

EXP03 = (
    ROOT / "experiment03_results"
    / "efficientnet_b0_baseline_v1"
)

print("EXPERIMENT 07.1 - PRETRAINING CONTROL PREPARATION")

for name, directory in [
    ("RETFound", EXP04),
    ("EfficientNet-B0", EXP03)
]:
    print(f"\n=== {name} ===")
    print("Directory exists:", directory.is_dir())

    if not directory.is_dir():
        continue

    # Inspect only top-level artifacts.
    for path in sorted(directory.iterdir()):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 ** 2)

            print(
                f"FILE: {path.name} "
                f"({size_mb:.2f} MB)"
            )
        elif path.is_dir():
            print(f"DIR: {path.name}/")

# Inspect small metadata files without loading model weights.
print("\n=== RETFound metadata ===")

if EXP04.is_dir():
    metadata_files = [
        p for p in EXP04.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".json", ".yaml", ".yml"}
        and p.stat().st_size < 2_000_000
    ]

    for path in sorted(metadata_files):
        print(f"\n--- {path.name} ---")

        if path.suffix.lower() == ".json":
            try:
                data = json.loads(
                    path.read_text(encoding="utf-8")
                )
                print(
                    json.dumps(
                        data,
                        indent=2,
                        ensure_ascii=False
                    )[:6000]
                )
            except Exception as exc:
                print("Could not parse:", str(exc))
        else:
            print(
                path.read_text(
                    encoding="utf-8"
                )[:6000]
            )

print("\nSTEP 07.1 INSPECTION COMPLETE")


# %% [cell 195]

from pathlib import Path
import importlib.metadata as metadata
import json

ROOT = DR_RESEARCH

print("EXPERIMENT 07.2 - PRETRAINING CONTROL INVENTORY")

# Inspect a bounded set of directories without recursive scanning.
directories = [
    ROOT,
    ROOT / "models",
    ROOT / "checkpoints",
    ROOT / "pretrained",
    ROOT / "pretrained_weights",
    ROOT / "experiment04_results",
    ROOT / "experiment04_results/retfound_vit_large_v1",
]

for directory in directories:
    print(f"\nDIRECTORY: {directory}")
    print("Exists:", directory.is_dir())

    if not directory.is_dir():
        continue

    entries = sorted(directory.iterdir(), key=lambda p: p.name)

    for path in entries[:40]:
        kind = "DIR" if path.is_dir() else "FILE"
        size = (
            ""
            if path.is_dir()
            else f" ({path.stat().st_size / 1024**2:.1f} MB)"
        )
        print(f"  [{kind}] {path.name}{size}")

    if len(entries) > 40:
        print(f"  ... {len(entries) - 40} additional entries")

print("\nPACKAGE VERSIONS")
for package in ["torch", "timm", "transformers", "numpy"]:
    try:
        print(package, metadata.version(package))
    except metadata.PackageNotFoundError:
        print(package, "NOT INSTALLED")

print("\nRETFOUND CONFIGURATION")
config_path = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1/training_config.json"
)

config = json.loads(config_path.read_text(encoding="utf-8"))

for key in [
    "architecture",
    "pretraining",
    "pretrained_sha256",
    "input_size",
    "global_pool",
    "seed",
]:
    print(f"{key}: {config.get(key)}")

print("\nSTEP 07.2 INVENTORY COMPLETE")


# %% [cell 196] Experiment 07 - Step 07.3

# ============================================================
# Experiment 07 - Step 07.3
# Inspect Original Checkpoints and Architecture Audits
# Read-only; no training, downloads, or model loading
# ============================================================

from pathlib import Path
import json

ROOT = DR_RESEARCH
CHECKPOINT_DIR = ROOT / "model_checkpoints"
AUDIT_DIR = ROOT / "experiment04_results"

print("EXPERIMENT 07.3 - CHECKPOINT AND ARCHITECTURE AUDIT")

print("\n=== MODEL CHECKPOINTS ===")
print("Directory exists:", CHECKPOINT_DIR.is_dir())

if CHECKPOINT_DIR.is_dir():
    entries = sorted(
        CHECKPOINT_DIR.iterdir(),
        key=lambda p: p.name
    )

    for path in entries[:50]:
        if path.is_file():
            size_mb = path.stat().st_size / (1024 ** 2)
            print(f"FILE: {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"DIR: {path.name}/")

    print("Total direct entries:", len(entries))

print("\n=== RETFOUND AUDIT REPORTS ===")

audit_names = [
    "RETFound_Checkpoint_Architecture_Audit.json",
    "RETFound_timm_Compatibility_Audit.json",
    "RETFound_GPU_Forward_Smoke_Test.json",
]

for filename in audit_names:
    path = AUDIT_DIR / filename

    print(f"\n--- {filename} ---")
    print("Exists:", path.is_file())

    if not path.is_file():
        continue

    assert path.stat().st_size < 2_000_000

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
        print(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            )[:12000]
        )
    except Exception as error:
        print("Read error:", str(error))

print("\nSTEP 07.3 INSPECTION COMPLETE")


# %% [cell 197] Experiment 07 - Step 07.4
# ============================================================
# Experiment 07 - Step 07.4
# Inspect Original Pretrained Checkpoint Inventory
# Read-only; no large checkpoint loading
# ============================================================

from pathlib import Path
import json

ROOT = DR_RESEARCH

PRETRAINED_DIR = (
    ROOT / "model_checkpoints"
    / "RETFound_pretrained"
)

print("EXPERIMENT 07.4 - PRETRAINED WEIGHT INVENTORY")
print("Directory exists:", PRETRAINED_DIR.is_dir())

assert PRETRAINED_DIR.is_dir()

entries = sorted(
    PRETRAINED_DIR.iterdir(),
    key=lambda p: p.name
)

for path in entries[:50]:
    if path.is_file():
        size_mb = path.stat().st_size / (1024 ** 2)
        print(
            f"FILE: {path.name} | "
            f"Size: {size_mb:.2f} MB"
        )
    else:
        print(f"DIR: {path.name}/")

print("Total direct entries:", len(entries))

config_path = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "training_config.json"
)

audit_path = (
    ROOT / "experiment04_results"
    / "RETFound_Checkpoint_Architecture_Audit.json"
)

config = json.loads(
    config_path.read_text(encoding="utf-8")
)
audit = json.loads(
    audit_path.read_text(encoding="utf-8")
)

assert (
    config["pretrained_sha256"]
    == audit["checkpoint_sha256"]
)

assert (
    audit["important_parameter_shapes"]["patch_embed.proj.weight"]
    == [1024, 3, 16, 16]
)

assert len(audit["encoder_block_indices"]) == 24

print("\nARCHITECTURE CHECK")
print("Architecture:", config["architecture"])
print("Pretraining:", config["pretraining"])
print("Checkpoint SHA256:", config["pretrained_sha256"])
print("Encoder blocks:", len(audit["encoder_block_indices"]))
print("Original checkpoint epoch:", audit["checkpoint_epoch"])
print("Repository revision:", audit["repository_revision"])

print("\nSTEP 07.4 PASS")


# %% [cell 198] Experiment 07 - Step 07.5
# ============================================================
# Experiment 07 - Step 07.5
# Verify Existing Manifest and Official MAE Source
# No model download, training, or inference
# ============================================================

from pathlib import Path
import json
import requests

ROOT = DR_RESEARCH

manifest_path = (
    ROOT / "model_checkpoints/RETFound_pretrained"
    / "RETFound_Weight_Manifest.json"
)

official_mae_url = (
    "https://dl.fbaipublicfiles.com/mae/pretrain/"
    "mae_pretrain_vit_large.pth"
)

print("EXPERIMENT 07.5 - PRETRAINING SOURCE CHECK")

print("\n=== RETFOUND MANIFEST ===")
assert manifest_path.is_file()

manifest = json.loads(
    manifest_path.read_text(encoding="utf-8")
)

print(json.dumps(manifest, indent=2)[:8000])

print("\n=== OFFICIAL MAE CHECKPOINT ===")
print("URL:", official_mae_url)

try:
    response = requests.head(
        official_mae_url,
        allow_redirects=True,
        timeout=20
    )

    print("HTTP status:", response.status_code)
    print("Content type:", response.headers.get("Content-Type"))
    print("Content length:", response.headers.get("Content-Length"))
    print("Final URL:", response.url)

    if response.status_code == 200:
        print("Official source is reachable.")
    else:
        print("Source requires further access verification.")

except requests.RequestException as error:
    print("Connection check unavailable:", str(error))

print("\nNo checkpoint downloaded.")
print("No model training or inference performed.")
print("STEP 07.5 INSPECTION COMPLETE")


# %% [cell 199] Experiment 07 - Step 07.6

# ============================================================
# Experiment 07 - Step 07.6
# Download and Freeze Meta MAE ViT-Large Checkpoint
# No model loading, training, inference, or test access
# ============================================================

from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import requests

ROOT = DR_RESEARCH

DEST_DIR = ROOT / "model_checkpoints/Meta_MAE_pretrained"
DEST_DIR.mkdir(parents=True, exist_ok=True)

URL = (
    "https://dl.fbaipublicfiles.com/mae/pretrain/"
    "mae_pretrain_vit_large.pth"
)

FILENAME = "mae_pretrain_vit_large.pth"
CHECKPOINT = DEST_DIR / FILENAME
PARTIAL = DEST_DIR / (FILENAME + ".part")
MANIFEST = DEST_DIR / "Meta_MAE_Weight_Manifest_V1.json"

EXPECTED_SIZE = 1213314565
CHUNK_SIZE = 8 * 1024 * 1024

print("EXPERIMENT 07.6 - META MAE CHECKPOINT FREEZE")
print("Source:", URL)
print("Destination:", CHECKPOINT)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()

# Never overwrite a frozen checkpoint or its manifest.
if CHECKPOINT.exists():
    assert CHECKPOINT.stat().st_size == EXPECTED_SIZE, (
        "Existing checkpoint has unexpected size."
    )
    assert MANIFEST.exists(), (
        "Checkpoint exists without a manifest. "
        "Inspect it before proceeding."
    )

    existing = json.loads(MANIFEST.read_text())
    assert existing["sha256"] == sha256_file(CHECKPOINT)
    assert existing["file_size_bytes"] == EXPECTED_SIZE

    print("Existing checkpoint verified. No download needed.")

else:
    assert not MANIFEST.exists(), (
        "Manifest exists without checkpoint. Manual audit needed."
    )

    initial_size = PARTIAL.stat().st_size if PARTIAL.exists() else 0
    assert initial_size <= EXPECTED_SIZE

    headers = (
        {"Range": f"bytes={initial_size}-"}
        if initial_size > 0 else {}
    )

    started = time.monotonic()

    with requests.get(
        URL,
        headers=headers,
        stream=True,
        timeout=(30, 180),
        allow_redirects=True
    ) as response:

        response.raise_for_status()

        if initial_size > 0:
            assert response.status_code == 206, (
                "Server did not honor resume request."
            )
            content_range = response.headers.get(
                "Content-Range", ""
            )
            assert content_range.startswith(
                f"bytes {initial_size}-"
            ), f"Unexpected Content-Range: {content_range}"

            mode = "ab"
        else:
            assert response.status_code == 200
            mode = "wb"

        with PARTIAL.open(mode) as f:
            downloaded = initial_size
            next_report = initial_size + 100 * 1024 * 1024

            for chunk in response.iter_content(
                chunk_size=CHUNK_SIZE
            ):
                if not chunk:
                    continue

                f.write(chunk)
                downloaded += len(chunk)

                assert downloaded <= EXPECTED_SIZE, (
                    "Download exceeded expected file size."
                )

                if downloaded >= next_report:
                    print(
                        f"Downloaded: "
                        f"{downloaded / 1024**2:.1f} MiB"
                    )
                    next_report += 100 * 1024 * 1024

    assert PARTIAL.stat().st_size == EXPECTED_SIZE, (
        "Incomplete download retained as .part file."
    )

    print("\nComputing SHA256...")
    digest = sha256_file(PARTIAL)

    # Rename the complete temporary file within the same Drive folder.
    os.replace(PARTIAL, CHECKPOINT)

    manifest = {
        "experiment": "07.6",
        "model": "Meta MAE ViT-Large/16",
        "source_url": URL,
        "filename": FILENAME,
        "persistent_path": str(CHECKPOINT),
        "file_size_bytes": EXPECTED_SIZE,
        "sha256": digest,
        "download_completed": True,
        "checkpoint_structure_inspected": False,
        "encoder_compatibility_verified": False,
        "fine_tuning_started": False,
        "test_images_used": 0
    }

    with MANIFEST.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        "Elapsed seconds:",
        round(time.monotonic() - started, 2)
    )

final_manifest = json.loads(
    MANIFEST.read_text(encoding="utf-8")
)

assert CHECKPOINT.is_file()
assert CHECKPOINT.stat().st_size == EXPECTED_SIZE
assert final_manifest["sha256"] == sha256_file(CHECKPOINT)

print("\nCHECKPOINT REPORT")
print("Model:", final_manifest["model"])
print("Size (MiB):", round(EXPECTED_SIZE / 1024**2, 2))
print("SHA256:", final_manifest["sha256"])
print("Saved:", final_manifest["persistent_path"])
print("Manifest:", MANIFEST)
print("Structure inspected: False")
print("Fine-tuning started: False")
print("Test images used: 0")

print("\nSTEP 07.6 PASS")


# %% [cell 200] Experiment 07 - Step 07.7
# ============================================================
# Experiment 07 - Step 07.7
# Meta MAE ViT-L/16 Checkpoint Architecture Audit
# No training, inference, GPU use, or test access
# ============================================================

from pathlib import Path
import argparse
import json
import re
import torch

ROOT = DR_RESEARCH

CHECKPOINT = (
    ROOT / "model_checkpoints/Meta_MAE_pretrained"
    / "mae_pretrain_vit_large.pth"
)

AUDIT_PATH = (
    ROOT / "experiment04_results"
    / "RETFound_Checkpoint_Architecture_Audit.json"
)

MANIFEST_PATH = (
    ROOT / "model_checkpoints/Meta_MAE_pretrained"
    / "Meta_MAE_Weight_Manifest_V1.json"
)

OUT_DIR = ROOT / "experiment07_results"
REPORT_PATH = OUT_DIR / "Meta_MAE_Architecture_Audit_V1.json"

for path in [CHECKPOINT, AUDIT_PATH, MANIFEST_PATH]:
    assert path.is_file(), f"Missing input: {path}"

assert not REPORT_PATH.exists(), "Audit report already exists."

reference = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

assert CHECKPOINT.stat().st_size == manifest["file_size_bytes"]
assert manifest["download_completed"] is True

print("EXPERIMENT 07.7 - META MAE ARCHITECTURE AUDIT")
print("Checkpoint:", CHECKPOINT.name)
print("RETFound reference:", reference["repository"])
print("Device: CPU")

# Allowlist argparse.Namespace for official MAE training metadata.
# Never use weights_only=False for this inspection.
with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        mmap=True,
        weights_only=True
    )

assert isinstance(checkpoint, dict)

print("\nCHECKPOINT CONTAINER KEYS")
print(list(checkpoint.keys()))

assert "model" in checkpoint
state = checkpoint["model"]
assert isinstance(state, dict)

def encoder_key(key):
    return (
        key == "cls_token"
        or key == "pos_embed"
        or key.startswith("patch_embed.")
        or key.startswith("blocks.")
        or key.startswith("norm.")
    )

encoder = {
    key: tensor
    for key, tensor in state.items()
    if encoder_key(key)
}

decoder = {
    key: tensor
    for key, tensor in state.items()
    if key.startswith("decoder_")
    or key.startswith("mask_token")
}

block_ids = sorted({
    int(match.group(1))
    for key in encoder
    if (match := re.match(r"^blocks\.(\d+)\.", key))
})

important_keys = [
    "cls_token",
    "pos_embed",
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "blocks.0.attn.qkv.weight",
    "blocks.0.mlp.fc1.weight",
    "norm.weight",
    "norm.bias"
]

expected_shapes = reference["important_parameter_shapes"]

shape_checks = {}

for key in important_keys:
    expected = expected_shapes[key]
    actual = list(encoder[key].shape) if key in encoder else None
    shape_checks[key] = {
        "expected": expected,
        "actual": actual,
        "match": actual == expected
    }

print("\nIMPORTANT PARAMETER SHAPES")
for key, result in shape_checks.items():
    print(
        key,
        "| actual:", result["actual"],
        "| expected:", result["expected"],
        "| match:", result["match"]
    )

print("\nENCODER SUMMARY")
print("Checkpoint tensor entries:", len(state))
print("Encoder tensor entries:", len(encoder))
print("Decoder tensor entries:", len(decoder))
print("Encoder block count:", len(block_ids))
print("Encoder block indices:", block_ids)
print(
    "Encoder parameter elements:",
    sum(t.numel() for t in encoder.values())
)

checks = {
    "all_important_shapes_match": all(
        result["match"] for result in shape_checks.values()
    ),
    "encoder_entry_count_matches": (
        len(encoder) == reference["source_encoder_entries"]
        if "source_encoder_entries" in reference
        else len(encoder) == 292
    ),
    "encoder_elements_match": (
        sum(t.numel() for t in encoder.values())
        == 303299584
    ),
    "blocks_match": block_ids == list(range(24)),
    "patch_size_matches": (
        list(encoder["patch_embed.proj.weight"].shape)
        == [1024, 3, 16, 16]
    )
}

print("\nARCHITECTURE CHECKS")
for name, passed in checks.items():
    print(name + ":", "PASS" if passed else "FAIL")

passed = all(checks.values())

report = {
    "experiment": "07.7",
    "model": "Meta MAE ViT-Large/16",
    "status": (
        "PRELIMINARY_ARCHITECTURE_PASS"
        if passed else "ARCHITECTURE_REVIEW_REQUIRED"
    ),
    "checkpoint_path": str(CHECKPOINT),
    "checkpoint_sha256_from_manifest": manifest["sha256"],
    "checkpoint_keys": list(checkpoint.keys()),
    "total_tensor_entries": len(state),
    "encoder_tensor_entries": len(encoder),
    "decoder_tensor_entries": len(decoder),
    "encoder_block_count": len(block_ids),
    "encoder_parameter_elements": int(
        sum(t.numel() for t in encoder.values())
    ),
    "important_parameter_shapes": shape_checks,
    "checks": checks,
    "full_key_compatibility_test_completed": False,
    "weights_transferred_to_classifier": False,
    "fine_tuning_started": False,
    "test_images_used": 0
}

# Save the report only if the preliminary audit passes.
if passed:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nReport:", REPORT_PATH)
    print("STEP 07.7 PASS")
else:
    print("\nSTEP 07.7 REVIEW REQUIRED")
    print("No audit report written.")


# %% [cell 201] identical to cell 202, omitted here


# %% [cell 202] Experiment 07 - Step 07.7A
# ============================================================
# Experiment 07 - Step 07.7A
# Identify Additional Parameters and Checkpoint Type
# Read-only metadata inspection
# ============================================================

from pathlib import Path
import torch
import argparse

path = (DR_RESEARCH / "model_checkpoints/Meta_MAE_pretrained/mae_pretrain_vit_large.pth")

print("EXPERIMENT 07.7A - CHECKPOINT KEY AUDIT")

with torch.serialization.safe_globals([argparse.Namespace]):
    ckpt = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=True
    )

state = ckpt["model"]

print("\nCHECKPOINT TOP-LEVEL KEYS")
print(list(ckpt.keys()))

print("\nNON-BLOCK, NON-PATCH PARAMETERS")
for key, tensor in state.items():
    if not key.startswith(("blocks.", "patch_embed.")):
        if isinstance(tensor, torch.Tensor):
            print(f"{key}: {tuple(tensor.shape)}")

print("\nCLASSIFICATION / NORMALIZATION PARAMETERS")
for key in sorted(state):
    if any(
        token in key
        for token in [
            "head", "fc_norm", "norm.",
            "decoder", "mask_token"
        ]
    ):
        print(f"{key}: {tuple(state[key].shape)}")

print("\nCHECKPOINT SUMMARY")
print("Total tensor entries:", len(state))
print(
    "Total tensor elements:",
    sum(t.numel() for t in state.values())
)
print(
    "Decoder present:",
    any(k.startswith("decoder_") for k in state)
)
print(
    "Classification head present:",
    any(k.startswith("head.") for k in state)
)
print(
    "Global-pooling norm present:",
    any(k.startswith("fc_norm.") for k in state)
)

print("\nSTEP 07.7A INSPECTION COMPLETE")


# %% [cell 203] Experiment 07 - Step 07.7B

# ============================================================
# Experiment 07 - Step 07.7B
# Exact Encoder Key and Shape Comparison
# Read-only, CPU, memory-mapped checkpoints
# ============================================================

from pathlib import Path
import argparse
import gc
import torch

ROOT = DR_RESEARCH

RET_PATH = (
    ROOT / "model_checkpoints/RETFound_pretrained"
    / "RETFound_mae_natureCFP.pth"
)
META_PATH = (
    ROOT / "model_checkpoints/Meta_MAE_pretrained"
    / "mae_pretrain_vit_large.pth"
)

assert RET_PATH.is_file()
assert META_PATH.is_file()

def encoder_key(key):
    return (
        key in {"cls_token", "pos_embed"}
        or key.startswith("patch_embed.")
        or key.startswith("blocks.")
        or key.startswith("norm.")
    )

def read_encoder_shapes(path):
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=True
        )

    assert isinstance(checkpoint, dict)
    assert "model" in checkpoint

    state = checkpoint["model"]

    shapes = {
        key: tuple(value.shape)
        for key, value in state.items()
        if encoder_key(key)
        and isinstance(value, torch.Tensor)
    }

    del checkpoint, state
    gc.collect()

    return shapes

print("EXPERIMENT 07.7B - EXACT ENCODER COMPARISON")

print("\nReading RETFound encoder metadata...")
ret_shapes = read_encoder_shapes(RET_PATH)

print("Reading Meta MAE encoder metadata...")
meta_shapes = read_encoder_shapes(META_PATH)

ret_keys = set(ret_shapes)
meta_keys = set(meta_shapes)

only_ret = sorted(ret_keys - meta_keys)
only_meta = sorted(meta_keys - ret_keys)

shared = sorted(ret_keys & meta_keys)

shape_mismatch = [
    (key, ret_shapes[key], meta_shapes[key])
    for key in shared
    if ret_shapes[key] != meta_shapes[key]
]

print("\nENCODER COUNTS")
print("RETFound encoder keys:", len(ret_keys))
print("Meta MAE encoder keys:", len(meta_keys))
print("Shared encoder keys:", len(shared))

print("\nONLY IN RETFOUND")
for key in only_ret:
    print(key, ret_shapes[key])
if not only_ret:
    print("None")

print("\nONLY IN META MAE")
for key in only_meta:
    print(key, meta_shapes[key])
if not only_meta:
    print("None")

print("\nSHAPE MISMATCHES")
for key, ret_shape, meta_shape in shape_mismatch:
    print(key, "RETFound:", ret_shape, "Meta:", meta_shape)
if not shape_mismatch:
    print("None")

print("\nSHARED PARAMETER ELEMENTS")
print(
    sum(
        1 if len(ret_shapes[k]) == 0
        else __import__("math").prod(ret_shapes[k])
        for k in shared
        if ret_shapes[k] == meta_shapes[k]
    )
)

print("\nARCHITECTURE COMPARISON")
print("Identical key sets:", ret_keys == meta_keys)
print("All shared shapes match:", len(shape_mismatch) == 0)
print("Extra RETFound keys:", len(only_ret))
print("Extra Meta MAE keys:", len(only_meta))

print("\nSTEP 07.7B INSPECTION COMPLETE")


# %% [cell 204] Experiment 07 - Step 07.8

# ============================================================
# Experiment 07 - Step 07.8
# Meta MAE ViT-L/16 Classifier Loading Smoke Test
# No fine-tuning or dataset inference
# ============================================================

from pathlib import Path
import argparse
import gc
import json

import torch
import timm

ROOT = DR_RESEARCH

CHECKPOINT = (
    ROOT / "model_checkpoints/Meta_MAE_pretrained"
    / "mae_pretrain_vit_large.pth"
)

REPORT = (
    ROOT / "experiment07_results"
    / "Meta_MAE_Classifier_Load_Audit_V1.json"
)

assert CHECKPOINT.is_file()
assert not REPORT.exists(), "Audit already exists."
assert torch.cuda.is_available()

print("EXPERIMENT 07.8 - CLASSIFIER LOADING AUDIT")
print("Torch:", torch.__version__)
print("Timm:", timm.__version__)
print("GPU:", torch.cuda.get_device_name(0))

# Reproduce the RETFound classifier architecture.
model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg"
)

target_state = model.state_dict()

with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        mmap=True,
        weights_only=True
    )

source_state = checkpoint["model"]

# Select every source tensor whose key and shape exactly
# match the actual classification model.
compatible = {
    key: tensor
    for key, tensor in source_state.items()
    if key in target_state
    and tuple(tensor.shape) == tuple(target_state[key].shape)
}

unmatched_source = sorted(
    set(source_state) - set(compatible)
)
uninitialized_target = sorted(
    set(target_state) - set(compatible)
)

expected_missing = {
    "fc_norm.weight",
    "fc_norm.bias",
    "head.weight",
    "head.bias"
}

expected_unused = {
    "norm.weight",
    "norm.bias"
}

print("\nWEIGHT COMPATIBILITY")
print("Source tensors:", len(source_state))
print("Compatible tensors:", len(compatible))
print("Unused source keys:", unmatched_source)
print("New classifier keys:", uninitialized_target)

assert len(compatible) == 292
assert unmatched_source == sorted(expected_unused)
assert uninitialized_target == sorted(expected_missing)

compatible_elements = sum(
    tensor.numel() for tensor in compatible.values()
)

assert compatible_elements == 303299584

load_result = model.load_state_dict(
    compatible,
    strict=False
)

assert sorted(load_result.missing_keys) == sorted(expected_missing)
assert not load_result.unexpected_keys

print("\nLOAD RESULT")
print("Loaded encoder tensors:", len(compatible))
print("Loaded encoder elements:", compatible_elements)
print("Missing keys:", load_result.missing_keys)
print("Unexpected keys:", load_result.unexpected_keys)

# Run a synthetic forward pass only.
del compatible, checkpoint, source_state, target_state
gc.collect()

model = model.to("cuda").eval()

with torch.inference_mode():
    synthetic_input = torch.zeros(
        1, 3, 224, 224,
        device="cuda"
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):
        output = model(synthetic_input)

    assert tuple(output.shape) == (1, 5)
    assert torch.isfinite(output).all()

print("\nSYNTHETIC FORWARD TEST")
print("Input shape:", (1, 3, 224, 224))
print("Output shape:", tuple(output.shape))
print("Finite output:", True)

report = {
    "experiment": "07.8",
    "status": "CLASSIFIER_LOADING_PASS",
    "model": "Meta MAE ViT-Large/16",
    "architecture": "vit_large_patch16_224",
    "global_pool": "avg",
    "num_classes": 5,
    "compatible_encoder_tensors": 292,
    "compatible_encoder_elements": compatible_elements,
    "unused_source_keys": unmatched_source,
    "new_classifier_keys": uninitialized_target,
    "synthetic_forward_shape": list(output.shape),
    "real_images_evaluated": 0,
    "fine_tuning_started": False,
    "test_images_used": 0
}

REPORT.parent.mkdir(parents=True, exist_ok=True)

with REPORT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

del model, synthetic_input, output
gc.collect()
torch.cuda.empty_cache()

print("\nReport:", REPORT)
print("STEP 07.8 PASS")


# %% [cell 205] Experiment 07 - Step 07.9

# ============================================================
# Experiment 07 - Step 07.9
# Fine-Tuning Protocol Reproducibility Audit
# Read-only; no model loading, training, or inference
# ============================================================

from pathlib import Path
import json
import pandas as pd

ROOT = DR_RESEARCH

RET_DIR = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
)

EFF_DIR = (
    ROOT / "experiment03_results"
    / "efficientnet_b0_baseline_v1"
)

NOTEBOOK_DIR = ROOT / "notebooks"

print("EXPERIMENT 07.9 - PROTOCOL AUDIT")

# 1. Existing RETFound training configuration
ret_config_path = RET_DIR / "training_config.json"

assert ret_config_path.is_file()

ret_config = json.loads(
    ret_config_path.read_text(encoding="utf-8")
)

print("\n=== RETFOUND TRAINING CONFIG ===")
print(json.dumps(ret_config, indent=2))

# 2. RETFound training history structure
history_path = RET_DIR / "training_history.csv"

print("\n=== RETFOUND TRAINING HISTORY ===")
print("Exists:", history_path.is_file())

if history_path.is_file():
    history = pd.read_csv(history_path)

    print("Columns:", history.columns.tolist())
    print("Rows:", len(history))
    print("\nFirst 3 epochs:")
    print(history.head(3).to_string(index=False))

    print("\nLast 3 epochs:")
    print(history.tail(3).to_string(index=False))

# 3. Previous EfficientNet training protocol,
# inspected only to identify reusable preprocessing details.
eff_config_path = EFF_DIR / "training_config.json"

print("\n=== EFFICIENTNET TRAINING CONFIG ===")
print("Exists:", eff_config_path.is_file())

if eff_config_path.is_file():
    eff_config = json.loads(
        eff_config_path.read_text(encoding="utf-8")
    )
    print(json.dumps(eff_config, indent=2))

# 4. Locate existing notebook/script names.
# Do not recursively scan datasets or checkpoints.
print("\n=== EXISTING NOTEBOOK/SCRIPT INVENTORY ===")
print("Directory exists:", NOTEBOOK_DIR.is_dir())

if NOTEBOOK_DIR.is_dir():
    entries = sorted(
        NOTEBOOK_DIR.iterdir(),
        key=lambda p: p.name
    )

    for path in entries[:60]:
        kind = "DIR" if path.is_dir() else "FILE"
        print(f"{kind}: {path.name}")

    print("Total direct entries:", len(entries))

print("\n=== PROTOCOL GAPS TO VERIFY ===")

fields = [
    "train_transform",
    "val_transform",
    "train_transforms",
    "val_transforms",
    "normalization",
    "image_mean",
    "image_std",
    "drop_path_rate",
    "label_smoothing",
    "scheduler",
    "warmup_epochs",
    "gradient_clip_norm",
    "seed",
]

for field in fields:
    value = ret_config.get(field, "NOT RECORDED")
    print(f"{field}: {value}")

print("\nTest images evaluated:", ret_config.get(
    "test_images_evaluated", "NOT RECORDED"
))

print("\nSTEP 07.9 INSPECTION COMPLETE")


# %% [cell 206] Experiment 07 - Step 07.10
# ============================================================
# Experiment 07 - Step 07.10
# Recover RETFound Training Protocol Evidence
# Read-only inspection
# ============================================================

from pathlib import Path
from IPython import get_ipython

ROOT = DR_RESEARCH
AUDIT_DIR = ROOT / "experiment04_results"

print("EXPERIMENT 07.10 - PROTOCOL RECOVERY")

# Inspect existing Experiment 04 metadata filenames.
print("\n=== EXPERIMENT 04 METADATA FILES ===")

for path in sorted(AUDIT_DIR.iterdir()):
    if path.is_file() and path.suffix.lower() in {
        ".json", ".yaml", ".yml", ".py", ".ipynb"
    }:
        print(path.name)

# Search only current Colab execution history.
print("\n=== CURRENT SESSION CODE HISTORY ===")

ip = get_ipython()

keywords = [
    "retfound_vit_large_v1",
    "RETFound_ViT_Large_V1",
    "gradient_checkpointing",
    "CosineAnnealingLR",
]

matches = []

if ip is not None:
    history = ip.user_ns.get("In", [])

    for cell_number, source in enumerate(history):
        if not isinstance(source, str):
            continue

        if any(word in source for word in keywords):
            matches.append((cell_number, source))

print("Matching cells:", len(matches))

for cell_number, source in matches[-12:]:
    print(f"\n--- CELL {cell_number} ---")
    print(source[:3500])

print("\n=== REQUIRED PROTOCOL DETAILS ===")
print("Training transform: NOT YET VERIFIED")
print("Validation transform: NOT YET VERIFIED")
print("Initialization sequence: NOT YET VERIFIED")
print("Scheduler step timing: NOT YET VERIFIED")

print("\nSTEP 07.10 INSPECTION COMPLETE")


# %% [cell 207] Step 07.11A - Locate Original Colab Notebooks
# ============================================================
# Step 07.11A - Locate Original Colab Notebooks
# Read-only, bounded folder inspection
# ============================================================

from pathlib import Path

folders = [
    (DR_RESEARCH.parent / "Colab Notebooks"),
    (DR_RESEARCH / "notebooks"),
    DR_RESEARCH,
    LOCAL_DISK,
]

keywords = [
    "retfound", "experiment04", "experiment_04",
    "fgadr", "vit_large", "training", "untitled"
]

print("EXPERIMENT 07.11A - NOTEBOOK DISCOVERY")

for folder in folders:
    print(f"\nDIRECTORY: {folder}")
    print("Exists:", folder.is_dir())

    if not folder.is_dir():
        continue

    entries = sorted(folder.iterdir(), key=lambda p: p.name)

    matches = [
        p for p in entries
        if p.suffix.lower() in {".ipynb", ".py"}
        or any(k in p.name.lower() for k in keywords)
    ]

    print("Direct entries:", len(entries))
    print("Candidate files/folders:", len(matches))

    for p in matches[:80]:
        kind = "DIR" if p.is_dir() else "FILE"
        print(f"{kind}: {p.name}")

print("\nSTEP 07.11A COMPLETE")


# %% [cell 208] Experiment 07 - Step 07.11B
# ============================================================
# Experiment 07 - Step 07.11B
# Locate RETFound Training Code in Existing Notebooks
# Read-only inspection
# ============================================================

from pathlib import Path
import json

NOTEBOOK_DIR = (DR_RESEARCH.parent / "Colab Notebooks")

notebook_names = [
    "Untitled0.ipynb",
    "FGADR_MA_Mask_Audit.ipynb",
]

keywords = [
    "retfound",
    "vit_large_patch16_224",
    "gradient_checkpointing",
    "cosineannealinglr",
    "optimizer.step",
    "scheduler.step",
    "randomresizedcrop",
    "randomhorizontalflip",
    "normalize",
    "dataloader",
    "train_transform",
    "val_transform",
]

print("EXPERIMENT 07.11B - NOTEBOOK CODE DISCOVERY")

for name in notebook_names:
    path = NOTEBOOK_DIR / name

    print(f"\n=== {name} ===")
    print("Exists:", path.is_file())

    if not path.is_file():
        continue

    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    print("Total cells:", len(cells))

    matches = []

    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        found = [
            word for word in keywords
            if word in source.lower()
        ]

        if found:
            matches.append({
                "cell": index,
                "keywords": found,
                "source": source,
            })

    print("Matching code cells:", len(matches))

    for item in matches[:20]:
        print(f"\nCELL {item['cell']}")
        print("Keywords:", ", ".join(item["keywords"]))
        print("Code preview:")
        print(item["source"][:1200])

    if len(matches) > 20:
        print(f"... {len(matches) - 20} additional matching cells")

print("\nSTEP 07.11B INSPECTION COMPLETE")


# %% [cell 209] Experiment 07 - Step 07.12
# ============================================================
# Experiment 07 - Step 07.12
# Recover Original RETFound Training Source
# Read-only notebook inspection
# ============================================================

from pathlib import Path
import json
import hashlib

notebook_path = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert notebook_path.is_file()

notebook = json.loads(
    notebook_path.read_text(encoding="utf-8")
)

cells = notebook["cells"]

targets = [84, 97, 98, 99]

print("EXPERIMENT 07.12 - RETFOUND SOURCE RECOVERY")

for index in targets:
    assert index < len(cells)

    cell = cells[index]
    assert cell["cell_type"] == "code"

    source = "".join(cell.get("source", []))
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()

    print(f"\n{'=' * 65}")
    print(f"CELL {index}")
    print("Characters:", len(source))
    print("Source SHA256:", digest)
    print(f"{'=' * 65}\n")

    print(source)

print("\nSTEP 07.12 SOURCE RECOVERY COMPLETE")


# %% [cell 210] Experiment 07 - Step 07.13
# ============================================================
# Experiment 07 - Step 07.13
# Frozen Train/Validation Data Path Preflight
# Read-only; no training or inference
# ============================================================

from pathlib import Path
import pandas as pd

ROOT = DR_RESEARCH
INDEX = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

assert INDEX.is_file()

df = pd.read_csv(INDEX, dtype={"image_id": str})
assert df["image_id"].is_unique

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()

assert len(train) == 1290
assert len(val) == 276
assert set(train["risk_group_id"]).isdisjoint(
    set(val["risk_group_id"])
)

print("EXPERIMENT 07.13 - DATA PATH PREFLIGHT")

print("\nINDEX")
print("Training images:", len(train))
print("Validation images:", len(val))
print("Training class counts:",
      train["dr_grade"].value_counts().sort_index().to_dict())

for split_name, frame in [("train", train), ("val", val)]:
    print(f"\n{split_name.upper()} SAMPLE PATHS")

    for row in frame.head(3).itertuples(index=False):
        local = Path(str(row.local_path))
        source = Path(str(row.source_path))

        print("Image ID:", row.image_id)
        print("Local exists:", local.is_file())
        print("Source exists:", source.is_file())
        print("Local:", local)
        print("Source:", source)
        print("---")

print("\nSTEP 07.13 INSPECTION COMPLETE")


# %% [cell 211] Experiment 07 - Step 07.14
# ============================================================
# Experiment 07 - Step 07.14
# Restore Frozen Classification Image Paths
# Train + Validation only; no image copying
# ============================================================

from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

active = index[index["split"].isin(["train", "val"])].copy()

assert len(active) == 1566
assert active["image_id"].is_unique
assert (active["split"] == "train").sum() == 1290
assert (active["split"] == "val").sum() == 276

created = 0
reused = 0

for row in active.itertuples(index=False):
    source = Path(str(row.source_path))
    target = Path(str(row.local_path))

    assert source.is_file(), f"Missing source: {source}"
    assert str(target).startswith(str(LOCAL_DISK / "fgadr_classification"))

    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_symlink():
        assert target.resolve() == source.resolve(), (
            f"Incorrect symlink: {target}"
        )
        reused += 1

    elif target.exists():
        assert target.samefile(source), (
            f"Unexpected existing target: {target}"
        )
        reused += 1

    else:
        target.symlink_to(source)
        created += 1

print("EXPERIMENT 07.14 - DATA PATH RESTORATION")
print("Training images:", 1290)
print("Validation images:", 276)
print("Created symlinks:", created)
print("Reused existing links:", reused)
print("Total active image paths:", created + reused)

assert created + reused == 1566

# Confirm representative train/validation images are readable.
for split in ["train", "val"]:
    row = active[active["split"] == split].iloc[0]
    path = Path(str(row["local_path"]))

    assert path.is_file()

    with Image.open(path) as image:
        image.verify()

    print(
        f"{split.upper()} sample verified:",
        row["image_id"]
    )

print("Test images accessed: 0")
print("Original Drive images modified: 0")
print("\nSTEP 07.14 PASS")


# %% [cell 212] Experiment 07 - Step 07.15
# ============================================================
# Experiment 07 - Step 07.15
# Meta MAE Training DataLoader Preflight
# Original RETFound preprocessing protocol
# ============================================================

from pathlib import Path
import random
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

SEED = 42
IMAGE_SIZE = 224
MICRO_BATCH_SIZE = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

data = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

train_frame = data[data["split"] == "train"].copy()
val_frame = data[data["split"] == "val"].copy()

assert len(train_frame) == 1290
assert len(val_frame) == 276

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
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

eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

class FGADRClassificationDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(str(row["local_path"]))

        with Image.open(path) as source:
            image = source.convert("RGB")

        return (
            self.transform(image),
            torch.tensor(int(row["dr_grade"]), dtype=torch.long),
            str(row["image_id"])
        )

train_dataset = FGADRClassificationDataset(
    train_frame, train_transform
)

val_dataset = FGADRClassificationDataset(
    val_frame, eval_transform
)

train_generator = torch.Generator()
train_generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=MICRO_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    generator=train_generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("EXPERIMENT 07.15 - DATALOADER PREFLIGHT")

train_images, train_labels, train_ids = next(iter(train_loader))
val_images, val_labels, val_ids = next(iter(val_loader))

assert train_images.shape == (4, 3, 224, 224)
assert val_images.shape == (16, 3, 224, 224)
assert torch.isfinite(train_images).all()
assert torch.isfinite(val_images).all()

sample_a = val_dataset[0][0]
sample_b = val_dataset[0][0]

assert torch.equal(sample_a, sample_b)

print("\nDATASET")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Training micro-batches:", len(train_loader))
print("Validation batches:", len(val_loader))

print("\nBATCH CHECK")
print("Train shape:", tuple(train_images.shape))
print("Validation shape:", tuple(val_images.shape))
print("Train labels:", train_labels.tolist())
print("Validation labels:", val_labels.tolist())
print("Train finite:", bool(torch.isfinite(train_images).all()))
print("Validation finite:", bool(torch.isfinite(val_images).all()))
print("Validation deterministic:", bool(torch.equal(sample_a, sample_b)))

print("\nPREPROCESSING")
print("Training:", train_transform)
print("Validation:", eval_transform)

print("\nTest images accessed: 0")
print("Model training started: False")
print("\nSTEP 07.15 PASS")


# %% [cell 213] Experiment 07 - Step 07.16
# ============================================================
# Experiment 07 - Step 07.16
# Lock Meta MAE Fine-Tuning Protocol
# No training, no model loading
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = DR_RESEARCH

EXPERIMENT_DIR = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1"
)

CONFIG_PATH = EXPERIMENT_DIR / "training_config.json"

RET_CONFIG_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "training_config.json"
)

META_MANIFEST_PATH = (
    ROOT / "model_checkpoints"
    / "Meta_MAE_pretrained"
    / "Meta_MAE_Weight_Manifest_V1.json"
)

assert RET_CONFIG_PATH.is_file()
assert META_MANIFEST_PATH.is_file()
assert not CONFIG_PATH.exists(), "Configuration already exists."

ret_config = json.loads(
    RET_CONFIG_PATH.read_text(encoding="utf-8")
)

meta_manifest = json.loads(
    META_MANIFEST_PATH.read_text(encoding="utf-8")
)

EXPECTED_META_SHA256 = (
    "d2ceff4892889d0c0f5418fa8bcb02e6a25a37fc3bdc0619c3cacd7d3d05c39f"
)

print("EXPERIMENT 07.16 - PROTOCOL LOCK")

print("\nMETA MAE MANIFEST")
print(json.dumps(meta_manifest, indent=2)[:3000])

config = dict(ret_config)

config.update({
    "experiment": "Experiment07_Meta_MAE_ViT_Large_V1",
    "pretraining": "Meta_MAE_ImageNet1K",
    "pretrained_sha256": EXPECTED_META_SHA256,
    "pretrained_checkpoint": (
        "model_checkpoints/Meta_MAE_pretrained/"
        "mae_pretrain_vit_large.pth"
    ),
    "reference_experiment": "RETFound_ViT_Large_V1",
    "reference_protocol": "Experiment04_Steps_4.5A_4.5B",
    "gradient_clip_norm": 1.0,
    "optimizer_foreach": False,
    "scheduler_t_max": 25,
    "scheduler_eta_min": 1e-7,
    "train_transform": [
        "Resize(224,224,BICUBIC)",
        "RandomHorizontalFlip(p=0.5)",
        "RandomRotation(degrees=10,BILINEAR,fill=0)",
        "ToTensor",
        "Normalize(ImageNet)"
    ],
    "val_transform": [
        "Resize(224,224,BICUBIC)",
        "ToTensor",
        "Normalize(ImageNet)"
    ],
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "weighted_ce_reduction": "none",
    "gradient_accumulation_normalization": (
        "sum_of_class_weights_per_accumulation_group"
    ),
    "training_seed_reset_before_model_init": True,
    "test_images_evaluated": 0,
    "status": "CONFIG_LOCKED_NOT_TRAINED",
    "created_utc": datetime.now(timezone.utc).isoformat()
})

locked_fields = [
    "architecture",
    "training_index_sha256",
    "seed",
    "num_classes",
    "input_size",
    "global_pool",
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
    "model_selection_metric",
    "class_counts",
    "class_weights",
    "precision",
    "gradient_checkpointing",
    "training_images",
    "validation_images"
]

for field in locked_fields:
    assert config[field] == ret_config[field], (
        f"Protocol mismatch: {field}"
    )

assert config["pretrained_sha256"] == EXPECTED_META_SHA256
assert config["test_images_evaluated"] == 0

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

with CONFIG_PATH.open("x", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("\nPROTOCOL COMPARISON")
print("Matched RETFound fields:", len(locked_fields))
print("Architecture:", config["architecture"])
print("Pretraining:", config["pretraining"])
print("Learning rate:", config["learning_rate"])
print("Effective batch size:", config["effective_batch_size"])
print("Gradient clip norm:", config["gradient_clip_norm"])
print("Train images:", config["training_images"])
print("Validation images:", config["validation_images"])
print("Test images:", config["test_images_evaluated"])

print("\nConfiguration:", CONFIG_PATH)
print("Training started:", False)
print("STEP 07.16 PASS")


# %% [cell 214] Experiment 07 - Step 07.17
# ============================================================
# Experiment 07 - Step 07.17
# Meta MAE Formal Fine-Tuning - Epoch 1
# Matched RETFound training protocol
# ============================================================

import os
import gc
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score
)
from tqdm.auto import tqdm

ROOT = DR_RESEARCH

EXP_DIR = ROOT / "experiment07_results/meta_mae_vit_large_v1"
CONFIG_PATH = EXP_DIR / "training_config.json"
CHECKPOINT_PATH = EXP_DIR / "last_training_state.pt"
HISTORY_PATH = EXP_DIR / "training_history.csv"
PREDICTION_PATH = EXP_DIR / "validation_predictions_epoch001.csv"

WEIGHT_PATH = (
    ROOT / "model_checkpoints/Meta_MAE_pretrained"
    / "mae_pretrain_vit_large.pth"
)

assert torch.cuda.is_available()
assert CONFIG_PATH.is_file()
assert WEIGHT_PATH.is_file()
assert not CHECKPOINT_PATH.exists(), "Training checkpoint already exists."
assert not HISTORY_PATH.exists(), "Training history already exists."
assert not PREDICTION_PATH.exists(), "Epoch 1 predictions already exist."

assert "train_dataset" in globals()
assert "val_dataset" in globals()

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

assert config["experiment"] == "Experiment07_Meta_MAE_ViT_Large_V1"
assert config["status"] == "CONFIG_LOCKED_NOT_TRAINED"
assert config["pretraining"] == "Meta_MAE_ImageNet1K"
assert config["num_classes"] == 5
assert config["micro_batch_size"] == 4
assert config["gradient_accumulation_steps"] == 4
assert config["test_images_evaluated"] == 0
assert len(train_dataset) == 1290
assert len(val_dataset) == 276

SEED = int(config["seed"])
DEVICE = torch.device("cuda")
ACCUMULATION_STEPS = 4
GRAD_CLIP_NORM = float(config["gradient_clip_norm"])

print("EXPERIMENT 07.17 - META MAE FORMAL TRAINING")
print("GPU:", torch.cuda.get_device_name(0))
print("Architecture:", config["architecture"])
print("Epoch: 1 /", config["maximum_epochs"])

# ------------------------------------------------------------
# 1. Restore the original training initialization sequence
# ------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False

train_generator = torch.Generator()
train_generator.manual_seed(SEED)

formal_train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    generator=train_generator
)

formal_val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

assert len(formal_train_loader) == 323
assert len(formal_val_loader) == 18

# ------------------------------------------------------------
# 2. Create a fresh ViT-Large classifier
# ------------------------------------------------------------

# Release disposable model objects from previous smoke tests.
for name in ["model", "optimizer", "scheduler", "criterion"]:
    if name in globals():
        del globals()[name]

gc.collect()
torch.cuda.empty_cache()

# Match the original RETFound model initialization seed.
torch.manual_seed(SEED)

model = timm.create_model(
    config["architecture"],
    pretrained=False,
    num_classes=5,
    global_pool="avg"
)

# ------------------------------------------------------------
# 3. Load official Meta MAE encoder weights
# ------------------------------------------------------------

with torch.serialization.safe_globals([argparse.Namespace]):
    pretrained_checkpoint = torch.load(
        WEIGHT_PATH,
        map_location="cpu",
        mmap=True,
        weights_only=True
    )

source_state = pretrained_checkpoint["model"]

encoder_state = {
    key: value
    for key, value in source_state.items()
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
assert not load_result.unexpected_keys

assert model.blocks[0].norm1.eps == 1e-6
assert model.fc_norm.eps == 1e-6

print("\nPRETRAINED ENCODER")
print("Loaded tensors:", len(encoder_state))
print("New classifier parameters:", sorted(expected_missing))
print("Unexpected parameters:", load_result.unexpected_keys)

del encoder_state, source_state, pretrained_checkpoint
gc.collect()

model.set_grad_checkpointing(enable=True)
model = model.to(DEVICE)

# ------------------------------------------------------------
# 4. Loss, optimizer, scheduler
# ------------------------------------------------------------

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
    lr=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"]),
    foreach=False
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=int(config["maximum_epochs"]),
    eta_min=float(config["scheduler_eta_min"])
)

# ------------------------------------------------------------
# 5. Training epoch
# ------------------------------------------------------------

model.train()
optimizer.zero_grad(set_to_none=True)

torch.cuda.reset_peak_memory_stats(DEVICE)
start_time = time.perf_counter()

train_true = []
train_pred = []

train_loss_numerator = 0.0
train_weight_denominator = 0.0

group_weight_total = torch.zeros(
    (), device=DEVICE, dtype=torch.float32
)

optimizer_steps = 0

for batch_idx, (images, labels, _) in enumerate(
    tqdm(formal_train_loader, desc="Meta MAE Epoch 1")
):
    images = images.to(DEVICE, non_blocking=True)
    labels = labels.to(DEVICE, non_blocking=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):
        logits = model(images)
        individual_losses = criterion(logits.float(), labels)
        loss_sum = individual_losses.sum()

    assert torch.isfinite(loss_sum), "Non-finite training loss."

    loss_sum.backward()

    sample_weight_sum = class_weights[labels].sum().detach()

    group_weight_total += sample_weight_sum

    train_loss_numerator += float(loss_sum.detach().item())
    train_weight_denominator += float(sample_weight_sum.item())

    predictions = logits.detach().argmax(dim=1)

    train_true.extend(labels.cpu().tolist())
    train_pred.extend(predictions.cpu().tolist())

    is_group_end = (
        (batch_idx + 1) % ACCUMULATION_STEPS == 0
        or (batch_idx + 1) == len(formal_train_loader)
    )

    if is_group_end:
        assert group_weight_total.item() > 0

        # Match the RETFound weighted gradient accumulation.
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(group_weight_total)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_NORM,
            error_if_nonfinite=True
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        group_weight_total.zero_()
        optimizer_steps += 1

assert len(train_true) == 1290
assert optimizer_steps == 81

train_loss = train_loss_numerator / train_weight_denominator
train_accuracy = accuracy_score(train_true, train_pred)

train_macro_f1 = f1_score(
    train_true,
    train_pred,
    labels=list(range(5)),
    average="macro",
    zero_division=0
)

# ------------------------------------------------------------
# 6. Validation - frozen validation images only
# ------------------------------------------------------------

model.eval()

val_true = []
val_pred = []
val_ids = []
val_probabilities = []

val_loss_numerator = 0.0
val_weight_denominator = 0.0

with torch.inference_mode():
    for images, labels, image_ids in tqdm(
        formal_val_loader,
        desc="Meta MAE Validation Epoch 1"
    ):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(images)

        individual_losses = criterion(logits.float(), labels)

        probabilities = torch.softmax(logits.float(), dim=1)
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
        val_probabilities.extend(probabilities.cpu().tolist())

assert len(val_true) == 276
assert len(set(val_ids)) == 276

val_loss = val_loss_numerator / val_weight_denominator
val_accuracy = accuracy_score(val_true, val_pred)

val_macro_f1 = f1_score(
    val_true,
    val_pred,
    labels=list(range(5)),
    average="macro",
    zero_division=0
)

val_qwk = cohen_kappa_score(
    val_true,
    val_pred,
    labels=list(range(5)),
    weights="quadratic"
)

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

elapsed = time.perf_counter() - start_time

peak_allocated = (
    torch.cuda.max_memory_allocated(DEVICE) / 1024**3
)
peak_reserved = (
    torch.cuda.max_memory_reserved(DEVICE) / 1024**3
)

# ------------------------------------------------------------
# 7. Save recoverable training state
# ------------------------------------------------------------

history_row = {
    "epoch": 1,
    "train_loss": train_loss,
    "train_accuracy": train_accuracy,
    "train_macro_f1": train_macro_f1,
    "val_loss": val_loss,
    "val_accuracy": val_accuracy,
    "val_macro_f1": val_macro_f1,
    "val_qwk": float(val_qwk),
    "referable_sensitivity": referable_sensitivity,
    "referable_specificity": referable_specificity,
    "learning_rate": float(config["learning_rate"]),
    "duration_seconds": elapsed,
    "peak_allocated_gib": peak_allocated,
    "peak_reserved_gib": peak_reserved
}

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
    "numpy_rng_state": np.random.get_state(),
    "train_loader_generator_state": train_generator.get_state(),
    "config": config,
    "history_rows": [history_row]
}

predictions_frame = pd.DataFrame({
    "image_id": val_ids,
    "true_grade": val_true,
    "predicted_grade": val_pred,
    **{
        f"prob_dr{grade}": [
            row[grade] for row in val_probabilities
        ]
        for grade in range(5)
    }
})

# The temporary checkpoint prevents partial writes from being
# mistaken for a completed training state.
temporary_checkpoint = Path(
    str(CHECKPOINT_PATH) + ".partial"
)

torch.save(training_state, temporary_checkpoint)
os.replace(temporary_checkpoint, CHECKPOINT_PATH)

pd.DataFrame([history_row]).to_csv(
    HISTORY_PATH,
    index=False
)

predictions_frame.to_csv(
    PREDICTION_PATH,
    index=False
)

# ------------------------------------------------------------
# 8. Report
# ------------------------------------------------------------

print("\nEPOCH 1 TRAINING RESULTS")
print("Train loss:", round(train_loss, 6))
print("Train accuracy:", round(train_accuracy, 6))
print("Train macro-F1:", round(train_macro_f1, 6))
print("Optimizer steps:", optimizer_steps)

print("\nEPOCH 1 VALIDATION RESULTS")
print("Validation loss:", round(val_loss, 6))
print("Validation accuracy:", round(val_accuracy, 6))
print("Validation macro-F1:", round(val_macro_f1, 6))
print("Validation QWK:", round(float(val_qwk), 6))
print("Referable sensitivity:", round(referable_sensitivity, 6))
print("Referable specificity:", round(referable_specificity, 6))

print("\nRESOURCE USAGE")
print("Epoch duration minutes:", round(elapsed / 60, 2))
print("Peak GPU allocated GiB:", round(peak_allocated, 3))
print("Peak GPU reserved GiB:", round(peak_reserved, 3))

print("\nPERSISTENCE")
print("Checkpoint saved:", CHECKPOINT_PATH.is_file())
print("History saved:", HISTORY_PATH.is_file())
print("Validation predictions saved:", PREDICTION_PATH.is_file())
print("Checkpoint size GiB:",
      round(CHECKPOINT_PATH.stat().st_size / 1024**3, 3))

print("\nTest images evaluated: 0")
print("Next epoch: 2")
print("STEP 07.17 EPOCH 1 COMPLETED")


# %% [cell 215] Experiment 07 - Step 07.18
# ============================================================
# Experiment 07 - Step 07.18
# Training I/O and Checkpoint Preflight
# Read-only; no training or inference
# ============================================================

from pathlib import Path
import shutil
import pandas as pd

ROOT = DR_RESEARCH

EXP_DIR = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

CHECKPOINT = EXP_DIR / "last_training_state.pt"
HISTORY = EXP_DIR / "training_history.csv"
PREDICTIONS = EXP_DIR / "validation_predictions_epoch001.csv"

print("EXPERIMENT 07.18 - I/O PREFLIGHT")

# Verify persisted epoch-one artifacts without loading weights.
for path in [CHECKPOINT, HISTORY, PREDICTIONS]:
    assert path.is_file(), f"Missing artifact: {path.name}"
    print(
        path.name,
        "| Size MiB:",
        round(path.stat().st_size / 1024**2, 2)
    )

history = pd.read_csv(HISTORY)

assert len(history) == 1
assert int(history.iloc[0]["epoch"]) == 1

print("\nTRAINING STATE")
print("Completed epoch:", int(history.iloc[-1]["epoch"]))
print(
    "Validation macro-F1:",
    round(float(history.iloc[-1]["val_macro_f1"]), 6)
)

# Inspect train and validation files only.
index = pd.read_csv(INDEX_PATH)
active = index[index["split"].isin(["train", "val"])]

assert len(active) == 1566

sample_paths = [
    Path(str(path))
    for path in active["local_path"].head(3)
]

print("\nIMAGE STORAGE")
for path in sample_paths:
    print("Path:", path)
    print("Is symlink:", path.is_symlink())
    print("Target:", path.resolve())

# Estimate the amount of source image data.
total_bytes = sum(
    Path(str(path)).stat().st_size
    for path in active["source_path"]
)

disk = shutil.disk_usage(str(LOCAL_DISK))

print("\nLOCAL STORAGE")
print("Train/val source size GiB:",
      round(total_bytes / 1024**3, 3))
print("Free /content GiB:",
      round(disk.free / 1024**3, 3))
print(
    "Enough free space for image cache:",
    disk.free > total_bytes * 1.5
)

print("\nTraining resumed:", False)
print("Test images accessed:", 0)
print("STEP 07.18 INSPECTION COMPLETE")


# %% [cell 216] Experiment 07 - Step 07.19

# ============================================================
# Experiment 07 - Step 07.19
# Cache Frozen Train/Validation Images on Local Disk
# No model training, no test images, no source modifications
# ============================================================

from pathlib import Path
import shutil
import os
import pandas as pd
from tqdm.auto import tqdm

ROOT = DR_RESEARCH
INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

active = index[index["split"].isin(["train", "val"])].copy()

assert len(active) == 1566
assert active["image_id"].is_unique

print("EXPERIMENT 07.19 - LOCAL IMAGE CACHE")
print("Images to inspect:", len(active))

copied = 0
already_cached = 0
total_bytes = 0

for row in tqdm(
    active.itertuples(index=False),
    total=len(active),
    desc="Caching train/validation images"
):
    source = Path(str(row.source_path))
    target = Path(str(row.local_path))

    assert source.is_file(), f"Missing source: {source}"
    assert target.parent.is_relative_to(
        (LOCAL_DISK / "fgadr_classification")
    )

    assert target.name == source.name
    target.parent.mkdir(parents=True, exist_ok=True)

    source_size = source.stat().st_size

    # Existing regular files are never overwritten.
    if target.is_file() and not target.is_symlink():
        assert target.stat().st_size == source_size, (
            f"Existing file size mismatch: {target}"
        )
        already_cached += 1
        total_bytes += source_size
        continue

    # Only replace a symlink pointing to the expected source.
    assert target.is_symlink(), (
        f"Unexpected target state: {target}"
    )
    assert target.resolve() == source.resolve(), (
        f"Symlink target mismatch: {target}"
    )

    temporary = target.with_name(target.name + ".cache_partial")

    # Never replace an unexpected partial cache.
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(
            f"Unexpected partial cache file: {temporary}"
        )

    try:
        shutil.copyfile(source, temporary)

        assert temporary.stat().st_size == source_size, (
            f"Incomplete copy: {target}"
        )

        # Atomically replace the symlink with the real local file.
        os.replace(temporary, target)

    except Exception:
        if temporary.is_file():
            temporary.unlink()
        raise

    assert target.is_file()
    assert not target.is_symlink()

    copied += 1
    total_bytes += source_size

print("\nLOCAL CACHE SUMMARY")
print("Newly cached:", copied)
print("Already cached:", already_cached)
print("Total images:", copied + already_cached)
print("Total image GiB:", round(total_bytes / 1024**3, 3))

assert copied + already_cached == 1566

# Confirm every active path is now a regular local file.
regular_files = sum(
    Path(str(p)).is_file()
    and not Path(str(p)).is_symlink()
    for p in active["local_path"]
)

assert regular_files == 1566

print("Regular local files:", regular_files)
print("Drive sources modified:", False)
print("Test images accessed:", 0)
print("Model training started:", False)
print("\nSTEP 07.19 PASS")


# %% [cell 217] identical to cell 220, omitted here


# %% [cell 218] identical to cell 223, omitted here


# %% [cell 219]
from pathlib import Path

PROJECT_ROOT = DR_RESEARCH

assert PROJECT_ROOT.is_dir(), "Project directory not found."

print("Google Drive connection: PASS")
print("Project directory:", PROJECT_ROOT)


# %% [cell 220] Experiment 07 - Step 07.20
# ============================================================
# Experiment 07 - Step 07.20
# Epoch 1 Resume Checkpoint Integrity Audit
# Read-only; no training or inference
# ============================================================

from pathlib import Path
import json
import gc

import pandas as pd
import torch

ROOT = DR_RESEARCH
EXP_DIR = ROOT / "experiment07_results/meta_mae_vit_large_v1"

CHECKPOINT_PATH = EXP_DIR / "last_training_state.pt"
CONFIG_PATH = EXP_DIR / "training_config.json"
HISTORY_PATH = EXP_DIR / "training_history.csv"

assert CHECKPOINT_PATH.is_file()
assert CONFIG_PATH.is_file()
assert HISTORY_PATH.is_file()

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
history = pd.read_csv(HISTORY_PATH)

print("EXPERIMENT 07.20 - CHECKPOINT RESUME AUDIT")

# Load only a checkpoint created by this trusted project.
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    mmap=True,
    weights_only=False
)

required_keys = {
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "best_val_macro_f1",
    "best_epoch",
    "epochs_without_improvement",
    "torch_rng_state",
    "cuda_rng_states",
    "python_rng_state",
    "numpy_rng_state",
    "train_loader_generator_state",
    "config",
    "history_rows",
}

missing = sorted(required_keys - set(checkpoint))
assert not missing, f"Missing checkpoint fields: {missing}"

assert checkpoint["epoch"] == 1
assert checkpoint["best_epoch"] == 1
assert checkpoint["epochs_without_improvement"] == 0
assert len(history) == 1
assert int(history.iloc[0]["epoch"]) == 1

saved_f1 = float(checkpoint["best_val_macro_f1"])
history_f1 = float(history.iloc[0]["val_macro_f1"])

assert abs(saved_f1 - history_f1) < 1e-8
assert checkpoint["config"] == config
assert len(checkpoint["history_rows"]) == 1

model_state = checkpoint["model_state_dict"]

assert tuple(model_state["head.weight"].shape) == (5, 1024)
assert tuple(model_state["head.bias"].shape) == (5,)
assert tuple(model_state["pos_embed"].shape) == (1, 197, 1024)

optimizer_state = checkpoint["optimizer_state_dict"]
scheduler_state = checkpoint["scheduler_state_dict"]

assert len(optimizer_state["state"]) > 0
assert len(optimizer_state["param_groups"]) == 1
assert scheduler_state["last_epoch"] == 1

assert isinstance(
    checkpoint["train_loader_generator_state"], torch.Tensor
)

assert len(checkpoint["cuda_rng_states"]) > 0

print("\nRECOVERY STATE")
print("Completed epoch:", checkpoint["epoch"])
print("Resume from epoch:", checkpoint["epoch"] + 1)
print("Best epoch:", checkpoint["best_epoch"])
print("Best validation macro-F1:", round(saved_f1, 6))

print("\nMODEL AND OPTIMIZER")
print("Model state tensors:", len(model_state))
print("Optimizer parameter states:", len(optimizer_state["state"]))
print("Scheduler last epoch:", scheduler_state["last_epoch"])
print(
    "Saved learning rate:",
    optimizer_state["param_groups"][0]["lr"]
)

print("\nRANDOM STATE")
print("Torch RNG:", "PRESENT")
print("CUDA RNG:", "PRESENT")
print("Python RNG:", "PRESENT")
print("NumPy RNG:", "PRESENT")
print("DataLoader generator RNG:", "PRESENT")

print("\nVALIDATION")
print("Checkpoint/history match:", True)
print("Training resumed:", False)
print("Test images accessed:", 0)

del checkpoint, model_state, optimizer_state, scheduler_state
gc.collect()

print("\nSTEP 07.20 PASS")


# %% [cell 221]
from pathlib import Path
import torch

ROOT = DR_RESEARCH

CHECKPOINT = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1"
    / "last_training_state.pt"
)

LOCAL_DATA = (LOCAL_DISK / "fgadr_classification")

print("EXPERIMENT 07.20A - RUNTIME PREFLIGHT")

print("GPU available:", torch.cuda.is_available())
print("Checkpoint exists:", CHECKPOINT.is_file())
print("Local data directory exists:", LOCAL_DATA.is_dir())

sample = LOCAL_DATA / "2/0000_1.png"

print("Sample image exists:", sample.is_file())
print("Sample is symlink:", sample.is_symlink())

print("Train dataset in memory:", "train_dataset" in globals())
print("Validation dataset in memory:", "val_dataset" in globals())

print("\nSTEP 07.20A COMPLETE")


# %% [cell 222] Experiment 07 - Step 07.20B
# ============================================================
# Experiment 07 - Step 07.20B
# Restore Local Train/Validation Image Cache
# After Colab Runtime Restart
# ============================================================

from pathlib import Path
import shutil
import pandas as pd
from tqdm.auto import tqdm

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

LOCAL_ROOT = (LOCAL_DISK / "fgadr_classification")

assert INDEX_PATH.is_file()

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

active = index[index["split"].isin(["train", "val"])].copy()

assert len(active) == 1566
assert active["image_id"].is_unique
assert (active["split"] == "train").sum() == 1290
assert (active["split"] == "val").sum() == 276

print("EXPERIMENT 07.20B - RESTORE LOCAL DATA")

copied = 0
reused = 0

for row in tqdm(
    active.itertuples(index=False),
    total=len(active),
    desc="Restoring train/validation images"
):
    source = Path(str(row.source_path))
    target = Path(str(row.local_path))

    assert source.is_file(), f"Missing source: {source}"

    # Ensure all writes remain inside the local cache directory.
    assert target.is_relative_to(LOCAL_ROOT)
    assert target.name == source.name

    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_symlink():
        raise RuntimeError(
            f"Unexpected symlink: {target}"
        )

    if target.exists():
        assert target.is_file()
        assert target.stat().st_size == source.stat().st_size
        reused += 1
        continue

    temporary = target.with_name(target.name + ".partial")

    assert not temporary.exists(), (
        f"Unexpected partial file: {temporary}"
    )

    shutil.copyfile(source, temporary)

    assert temporary.stat().st_size == source.stat().st_size

    temporary.replace(target)
    copied += 1

regular_files = sum(
    Path(str(p)).is_file()
    and not Path(str(p)).is_symlink()
    for p in active["local_path"]
)

assert regular_files == 1566

print("\nRESTORATION SUMMARY")
print("Newly copied:", copied)
print("Existing local files:", reused)
print("Total local images:", regular_files)
print("Train images:", 1290)
print("Validation images:", 276)
print("Test images accessed:", 0)
print("Epoch 1 retrained:", False)

print("\nSTEP 07.20B PASS")


# %% [cell 223]

pass  # Google Drive mount removed

print("Google Drive reconnected successfully.")


# %% [cell 224] Experiment 07 - Step 07.20D
# ============================================================
# Experiment 07 - Step 07.20D
# Runtime Recovery Status Check
# Read-only; no training or data copying
# ============================================================

from pathlib import Path
import torch

ROOT = DR_RESEARCH

CHECKPOINT = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1"
    / "last_training_state.pt"
)

LOCAL_ROOT = (LOCAL_DISK / "fgadr_classification")

print("EXPERIMENT 07.20D - RUNTIME STATUS")

print("Google Drive connected:", ROOT.is_dir())
print("GPU available:", torch.cuda.is_available())
print("Epoch checkpoint exists:", CHECKPOINT.is_file())

print("\nLOCAL IMAGE CACHE")
print("Cache directory exists:", LOCAL_ROOT.is_dir())

sample = LOCAL_ROOT / "2/0000_1.png"
print("Sample image exists:", sample.is_file())
print("Sample is symlink:", sample.is_symlink())

print("\nPYTHON ENVIRONMENT")
print("Train dataset exists:", "train_dataset" in globals())
print("Validation dataset exists:", "val_dataset" in globals())

print("\nSTEP 07.20D COMPLETE")


# %% [cell 225] Experiment 07 - Step 07.20E
# ============================================================
# Experiment 07 - Step 07.20E
# Parallel Local Cache Restoration
# Train and validation only
# ============================================================

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import pandas as pd
import shutil
import os

ROOT = DR_RESEARCH
INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

LOCAL_ROOT = (LOCAL_DISK / "fgadr_classification")
MAX_WORKERS = 8

df = pd.read_csv(INDEX_PATH, dtype={"image_id": str})
active = df[df["split"].isin(["train", "val"])].copy()

assert len(active) == 1566
assert active["image_id"].is_unique

def restore_image(row):
    source = Path(str(row.source_path))
    target = Path(str(row.local_path))

    assert target.is_relative_to(LOCAL_ROOT)
    assert source.is_file()
    assert target.name == source.name

    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() or target.is_symlink():
        assert target.is_file()
        assert not target.is_symlink()
        assert target.stat().st_size == source.stat().st_size
        return "reused"

    temporary = target.with_name(target.name + ".partial")

    # Do not overwrite a partial file unexpectedly.
    if temporary.exists():
        if temporary.is_symlink():
            raise RuntimeError(f"Unexpected symlink: {temporary}")
        temporary.unlink()

    shutil.copyfile(source, temporary)

    assert temporary.stat().st_size == source.stat().st_size

    os.replace(temporary, target)

    return "copied"

print("EXPERIMENT 07.20E - PARALLEL CACHE RESTORATION")

rows = list(active.itertuples(index=False))

results = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(restore_image, row) for row in rows]

    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Restoring local images"
    ):
        results.append(future.result())

copied = results.count("copied")
reused = results.count("reused")

assert copied + reused == 1566

for path in active["local_path"]:
    local = Path(str(path))
    assert local.is_file()
    assert not local.is_symlink()

print("\nRESTORATION SUMMARY")
print("Newly copied:", copied)
print("Reused:", reused)
print("Total local images:", copied + reused)
print("Train images:", 1290)
print("Validation images:", 276)
print("Test images accessed:", 0)
print("Epoch 1 retrained:", False)

print("\nSTEP 07.20E PASS")


# %% [cell 226] Experiment 07 - Step 07.20F
# ============================================================
# Experiment 07 - Step 07.20F
# Rebuild Classification Datasets After Runtime Restart
# Read-only; no training or checkpoint modifications
# ============================================================

from pathlib import Path
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

data = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

train_frame = data[data["split"] == "train"].copy()
val_frame = data[data["split"] == "val"].copy()

assert len(train_frame) == 1290
assert len(val_frame) == 276
assert train_frame["image_id"].is_unique
assert val_frame["image_id"].is_unique

assert set(train_frame["risk_group_id"]).isdisjoint(
    set(val_frame["risk_group_id"])
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
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

eval_transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

class FGADRClassificationDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(str(row["local_path"]))

        with Image.open(path) as source:
            image = source.convert("RGB")

        return (
            self.transform(image),
            torch.tensor(
                int(row["dr_grade"]),
                dtype=torch.long
            ),
            str(row["image_id"])
        )

train_dataset = FGADRClassificationDataset(
    train_frame,
    train_transform
)

val_dataset = FGADRClassificationDataset(
    val_frame,
    eval_transform
)

print("EXPERIMENT 07.20F - DATASET RECOVERY")

# Validate the cached paths without opening all images.
for dataset in [train_dataset, val_dataset]:
    for index in [0, len(dataset) - 1]:
        path = Path(str(dataset.frame.iloc[index]["local_path"]))
        assert path.is_file()
        assert not path.is_symlink()

# These are inspection loaders only.
# Epoch 2 will reconstruct its own seeded training loader.
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

train_images, train_labels, _ = next(iter(train_loader))
val_images, val_labels, _ = next(iter(val_loader))

assert train_images.shape == (4, 3, 224, 224)
assert val_images.shape == (16, 3, 224, 224)
assert torch.isfinite(train_images).all()
assert torch.isfinite(val_images).all()

print("\nDATASET STATUS")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Training micro-batches:", len(train_loader))
print("Validation batches:", len(val_loader))

print("\nBATCH CHECK")
print("Train batch shape:", tuple(train_images.shape))
print("Validation batch shape:", tuple(val_images.shape))
print("Train finite:", bool(torch.isfinite(train_images).all()))
print("Validation finite:", bool(torch.isfinite(val_images).all()))

print("\nCHECKPOINT")
print("Epoch 1 retrained:", False)
print("Checkpoint modified:", False)
print("Test images accessed:", 0)

print("\nSTEP 07.20F PASS")


# %% [cell 227] Experiment 07 - Step 07.21
# ============================================================
# Experiment 07 - Step 07.21
# Resume Meta MAE Fine-Tuning - Epoch 2 Only
# Matched RETFound protocol; no test-set access
# ============================================================

import os
import gc
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score
)
from tqdm.auto import tqdm

ROOT = DR_RESEARCH
EXP_DIR = ROOT / "experiment07_results/meta_mae_vit_large_v1"

CONFIG_PATH = EXP_DIR / "training_config.json"
CHECKPOINT_PATH = EXP_DIR / "last_training_state.pt"
BEST_PATH = EXP_DIR / "best_model.pt"
HISTORY_PATH = EXP_DIR / "training_history.csv"
BEST_PRED_PATH = EXP_DIR / "best_validation_predictions.csv"

assert all(p.is_file() for p in [
    CONFIG_PATH, CHECKPOINT_PATH, HISTORY_PATH
])
assert "train_dataset" in globals()
assert "val_dataset" in globals()
assert len(train_dataset) == 1290
assert len(val_dataset) == 276
assert torch.cuda.is_available()

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
history = pd.read_csv(HISTORY_PATH)

assert len(history) == 1
assert int(history.iloc[-1]["epoch"]) == 1
assert config["num_classes"] == 5
assert config["test_images_evaluated"] == 0

DEVICE = torch.device("cuda")
EPOCH = 2
SEED = int(config["seed"])
ACCUM_STEPS = int(config["gradient_accumulation_steps"])
GRAD_CLIP = float(config["gradient_clip_norm"])

print("EXPERIMENT 07.21 - META MAE EPOCH 2 RESUME")

# ------------------------------------------------------------
# 1. Load the trusted Epoch 1 recovery state
# ------------------------------------------------------------

resume = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    mmap=True,
    weights_only=False
)

assert int(resume["epoch"]) == 1
assert resume["config"] == config
assert int(resume["best_epoch"]) == 1

best_f1 = float(resume["best_val_macro_f1"])
best_epoch = int(resume["best_epoch"])
no_improve = int(resume["epochs_without_improvement"])

assert abs(
    best_f1 - float(history.iloc[0]["val_macro_f1"])
) < 1e-8

# ------------------------------------------------------------
# 2. Reconstruct data loaders
# ------------------------------------------------------------

train_generator = torch.Generator()
train_generator.set_state(
    resume["train_loader_generator_state"]
)

resume_train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    generator=train_generator
)

resume_val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

assert len(resume_train_loader) == 323
assert len(resume_val_loader) == 18

# ------------------------------------------------------------
# 3. Reconstruct model and optimizer
# ------------------------------------------------------------

for name in ["model", "optimizer", "scheduler", "criterion"]:
    if name in globals():
        del globals()[name]

gc.collect()
torch.cuda.empty_cache()

model = timm.create_model(
    config["architecture"],
    pretrained=False,
    num_classes=5,
    global_pool="avg"
)

model.load_state_dict(
    resume["model_state_dict"],
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
    lr=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"]),
    foreach=False
)

optimizer.load_state_dict(
    resume["optimizer_state_dict"]
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=int(config["maximum_epochs"]),
    eta_min=float(config["scheduler_eta_min"])
)

scheduler.load_state_dict(
    resume["scheduler_state_dict"]
)

assert scheduler.last_epoch == 1

# ------------------------------------------------------------
# 4. Preserve Epoch 1 best model before training
# ------------------------------------------------------------

def atomic_torch_save(payload, destination):
    temporary = Path(str(destination) + ".partial")
    assert not temporary.exists(), (
        f"Unexpected partial checkpoint: {temporary}"
    )
    torch.save(payload, temporary)
    os.replace(temporary, destination)

def atomic_csv_save(frame, destination):
    temporary = Path(str(destination) + ".partial")
    assert not temporary.exists(), (
        f"Unexpected partial CSV: {temporary}"
    )
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)

if not BEST_PATH.exists():
    assert best_epoch == 1

    atomic_torch_save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "val_macro_f1": best_f1,
            "config": config
        },
        BEST_PATH
    )

if not BEST_PRED_PATH.exists():
    epoch1_pred = EXP_DIR / "validation_predictions_epoch001.csv"
    assert epoch1_pred.is_file()

    predictions_epoch1 = pd.read_csv(epoch1_pred)
    assert len(predictions_epoch1) == 276

    atomic_csv_save(
        predictions_epoch1,
        BEST_PRED_PATH
    )

# ------------------------------------------------------------
# 5. Restore random states after model reconstruction
# ------------------------------------------------------------

torch.set_rng_state(resume["torch_rng_state"].cpu())

torch.cuda.set_rng_state_all([
    state.cpu()
    for state in resume["cuda_rng_states"]
])

random.setstate(resume["python_rng_state"])

if "numpy_rng_state" in resume:
    np.random.set_state(resume["numpy_rng_state"])

# Check learning rate before starting.
epoch_lr = float(optimizer.param_groups[0]["lr"])

assert abs(
    epoch_lr - 9.960967771506666e-06
) < 1e-12

print("Resume checkpoint: PASS")
print("Completed epoch: 1")
print("Starting epoch: 2")
print("Learning rate:", epoch_lr)
print("Previous best macro-F1:", round(best_f1, 6))

# Release CPU checkpoint references.
del resume
gc.collect()

# ------------------------------------------------------------
# 6. Train Epoch 2
# ------------------------------------------------------------

model.train()
optimizer.zero_grad(set_to_none=True)
torch.cuda.reset_peak_memory_stats(DEVICE)

epoch_start = time.perf_counter()

train_true = []
train_pred = []

train_loss_num = 0.0
train_weight_den = 0.0

group_weight_total = torch.zeros(
    (), device=DEVICE, dtype=torch.float32
)

optimizer_steps = 0

for batch_idx, (images, labels, _) in enumerate(
    tqdm(
        resume_train_loader,
        desc="Meta MAE Epoch 2",
        total=323
    )
):
    images = images.to(DEVICE, non_blocking=True)
    labels = labels.to(DEVICE, non_blocking=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):
        logits = model(images)
        individual_losses = criterion(
            logits.float(), labels
        )
        loss_sum = individual_losses.sum()

    assert torch.isfinite(loss_sum)

    loss_sum.backward()

    sample_weight = class_weights[labels].sum().detach()

    group_weight_total += sample_weight

    train_loss_num += float(loss_sum.detach().item())
    train_weight_den += float(sample_weight.item())

    predictions = logits.detach().argmax(dim=1)
    train_true.extend(labels.cpu().tolist())
    train_pred.extend(predictions.cpu().tolist())

    is_group_end = (
        (batch_idx + 1) % ACCUM_STEPS == 0
        or (batch_idx + 1) == len(resume_train_loader)
    )

    if is_group_end:
        assert group_weight_total.item() > 0

        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(group_weight_total)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP,
            error_if_nonfinite=True
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        group_weight_total.zero_()
        optimizer_steps += 1

assert len(train_true) == 1290
assert optimizer_steps == 81

train_loss = train_loss_num / train_weight_den
train_accuracy = accuracy_score(train_true, train_pred)

train_macro_f1 = f1_score(
    train_true,
    train_pred,
    labels=list(range(5)),
    average="macro",
    zero_division=0
)

# ------------------------------------------------------------
# 7. Validate Epoch 2
# ------------------------------------------------------------

model.eval()

val_true = []
val_pred = []
val_ids = []
val_probabilities = []

val_loss_num = 0.0
val_weight_den = 0.0

with torch.inference_mode():
    for images, labels, image_ids in tqdm(
        resume_val_loader,
        desc="Meta MAE Validation Epoch 2"
    ):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(images)

        losses = criterion(logits.float(), labels)
        probabilities = torch.softmax(
            logits.float(), dim=1
        )
        predictions = probabilities.argmax(dim=1)

        val_loss_num += float(losses.sum().item())
        val_weight_den += float(
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

val_loss = val_loss_num / val_weight_den
val_accuracy = accuracy_score(val_true, val_pred)

val_macro_f1 = f1_score(
    val_true,
    val_pred,
    labels=list(range(5)),
    average="macro",
    zero_division=0
)

val_qwk = cohen_kappa_score(
    val_true,
    val_pred,
    labels=list(range(5)),
    weights="quadratic"
)

true_ref = np.asarray(val_true) >= 2
pred_ref = np.asarray(val_pred) >= 2

tp = int(np.sum(true_ref & pred_ref))
fn = int(np.sum(true_ref & ~pred_ref))
tn = int(np.sum(~true_ref & ~pred_ref))
fp = int(np.sum(~true_ref & pred_ref))

ref_sensitivity = tp / (tp + fn)
ref_specificity = tn / (tn + fp)

is_best = val_macro_f1 > best_f1 + 1e-12

if is_best:
    best_f1 = float(val_macro_f1)
    best_epoch = EPOCH
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

# ------------------------------------------------------------
# 8. Save best model, history, and recovery state
# ------------------------------------------------------------

row = {
    "epoch": EPOCH,
    "train_loss": train_loss,
    "train_accuracy": train_accuracy,
    "train_macro_f1": train_macro_f1,
    "val_loss": val_loss,
    "val_accuracy": val_accuracy,
    "val_macro_f1": val_macro_f1,
    "val_qwk": float(val_qwk),
    "referable_sensitivity": ref_sensitivity,
    "referable_specificity": ref_specificity,
    "learning_rate": epoch_lr,
    "duration_seconds": elapsed,
    "peak_allocated_gib": peak_allocated,
    "peak_reserved_gib": peak_reserved
}

updated_history = pd.concat(
    [history, pd.DataFrame([row])],
    ignore_index=True
)

assert updated_history["epoch"].tolist() == [1, 2]

if is_best:
    pred_frame = pd.DataFrame({
        "image_id": val_ids,
        "true_grade": val_true,
        "predicted_grade": val_pred,
        **{
            f"prob_dr{grade}": [
                p[grade] for p in val_probabilities
            ]
            for grade in range(5)
        }
    })

    atomic_torch_save(
        {
            "epoch": EPOCH,
            "model_state_dict": model.state_dict(),
            "val_macro_f1": float(val_macro_f1),
            "config": config
        },
        BEST_PATH
    )

    atomic_csv_save(pred_frame, BEST_PRED_PATH)

training_state = {
    "epoch": EPOCH,
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
    "history_rows": updated_history.to_dict(
        orient="records"
    )
}

atomic_torch_save(training_state, CHECKPOINT_PATH)
atomic_csv_save(updated_history, HISTORY_PATH)

# ------------------------------------------------------------
# 9. Final report
# ------------------------------------------------------------

print("\nEPOCH 2 TRAINING RESULTS")
print("Train loss:", round(train_loss, 6))
print("Train accuracy:", round(train_accuracy, 6))
print("Train macro-F1:", round(train_macro_f1, 6))
print("Optimizer steps:", optimizer_steps)

print("\nEPOCH 2 VALIDATION RESULTS")
print("Validation loss:", round(val_loss, 6))
print("Validation accuracy:", round(val_accuracy, 6))
print("Validation macro-F1:", round(val_macro_f1, 6))
print("Validation QWK:", round(float(val_qwk), 6))
print("Referable sensitivity:", round(ref_sensitivity, 6))
print("Referable specificity:", round(ref_specificity, 6))

print("\nTRAINING PERFORMANCE")
print("Epoch duration minutes:", round(elapsed / 60, 2))
print("Previous epoch minutes:", 19.16)
print("Peak allocated GiB:", round(peak_allocated, 3))
print("Peak reserved GiB:", round(peak_reserved, 3))

print("\nCHECKPOINT STATUS")
print("Best epoch:", best_epoch)
print("Best validation macro-F1:", round(best_f1, 6))
print("Best model updated:", is_best)
print("Resume checkpoint saved:", CHECKPOINT_PATH.is_file())
print("Training history epochs:", updated_history["epoch"].tolist())
print("Next epoch:", 3)

print("\nTest images evaluated: 0")
print("STEP 07.21 EPOCH 2 COMPLETED")


# %% [cell 228] Experiment 07 - Step 07.22
# ============================================================
# Experiment 07 - Step 07.22
# Meta MAE Formal Training Resume: Epochs 3-25
# Automatic checkpointing and early stopping
# ============================================================

import os
import gc
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm.auto import tqdm

ROOT = DR_RESEARCH
EXP_DIR = ROOT / "experiment07_results/meta_mae_vit_large_v1"

CONFIG_PATH = EXP_DIR / "training_config.json"
CHECKPOINT_PATH = EXP_DIR / "last_training_state.pt"
BEST_PATH = EXP_DIR / "best_model.pt"
HISTORY_PATH = EXP_DIR / "training_history.csv"
BEST_PRED_PATH = EXP_DIR / "best_validation_predictions.csv"

assert all(p.is_file() for p in [
    CONFIG_PATH, CHECKPOINT_PATH, BEST_PATH,
    HISTORY_PATH, BEST_PRED_PATH
])

assert "train_dataset" in globals()
assert "val_dataset" in globals()
assert len(train_dataset) == 1290
assert len(val_dataset) == 276
assert torch.cuda.is_available()

# Check local cache without reading Drive images.
for dataset in [train_dataset, val_dataset]:
    for idx in [0, len(dataset) - 1]:
        p = Path(str(dataset.frame.iloc[idx]["local_path"]))
        assert p.is_file() and not p.is_symlink()

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
history = pd.read_csv(HISTORY_PATH)

assert config["experiment"] == "Experiment07_Meta_MAE_ViT_Large_V1"
assert config["test_images_evaluated"] == 0
assert config["architecture"] == "vit_large_patch16_224"
assert config["gradient_accumulation_steps"] == 4
assert history["epoch"].is_unique

DEVICE = torch.device("cuda")
MAX_EPOCHS = int(config["maximum_epochs"])
MIN_EPOCHS = int(config["minimum_epochs"])
PATIENCE = int(config["patience"])
ACCUM_STEPS = int(config["gradient_accumulation_steps"])
MICRO_BATCH_SIZE = int(config["micro_batch_size"])
GRAD_CLIP = float(config["gradient_clip_norm"])
NUM_CLASSES = int(config["num_classes"])

print("EXPERIMENT 07.22 - AUTOMATIC TRAINING RESUME")

# ------------------------------------------------------------
# 1. Recover checkpoint
# ------------------------------------------------------------

# This checkpoint was created by the trusted project.
resume = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    mmap=True,
    weights_only=False
)

completed_epoch = int(resume["epoch"])
start_epoch = completed_epoch + 1

assert int(history["epoch"].max()) == completed_epoch
assert len(history) == completed_epoch
assert resume["config"] == config

best_f1 = float(resume["best_val_macro_f1"])
best_epoch = int(resume["best_epoch"])
no_improve = int(resume["epochs_without_improvement"])

assert abs(
    float(history.loc[
        history["epoch"] == best_epoch,
        "val_macro_f1"
    ].iloc[0]) - best_f1
) < 1e-8

assert start_epoch >= 3, "Epoch 2 must be completed first."

print("Completed epoch:", completed_epoch)
print("Resume from epoch:", start_epoch)
print("Previous best epoch:", best_epoch)
print("Previous best macro-F1:", round(best_f1, 6))

# ------------------------------------------------------------
# 2. Restore training DataLoader generator state
# ------------------------------------------------------------

train_generator = torch.Generator()
train_generator.set_state(
    resume["train_loader_generator_state"]
)

resume_train_loader = DataLoader(
    train_dataset,
    batch_size=MICRO_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    generator=train_generator
)

resume_val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

assert len(resume_train_loader) == 323
assert len(resume_val_loader) == 18

# ------------------------------------------------------------
# 3. Restore model, optimizer and scheduler
# ------------------------------------------------------------

for name in ["model", "optimizer", "scheduler", "criterion"]:
    if name in globals():
        del globals()[name]

gc.collect()
torch.cuda.empty_cache()

model = timm.create_model(
    config["architecture"],
    pretrained=False,
    num_classes=NUM_CLASSES,
    global_pool="avg"
)

model.load_state_dict(
    resume["model_state_dict"],
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
    lr=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"]),
    foreach=False
)

optimizer.load_state_dict(
    resume["optimizer_state_dict"]
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_EPOCHS,
    eta_min=float(config["scheduler_eta_min"])
)

scheduler.load_state_dict(
    resume["scheduler_state_dict"]
)

assert scheduler.last_epoch == completed_epoch

# ------------------------------------------------------------
# 4. Restore random states
# ------------------------------------------------------------

torch.set_rng_state(resume["torch_rng_state"].cpu())

torch.cuda.set_rng_state_all([
    state.cpu() for state in resume["cuda_rng_states"]
])

random.setstate(resume["python_rng_state"])
np.random.set_state(resume["numpy_rng_state"])

torch.backends.cudnn.benchmark = False

del resume
gc.collect()

# ------------------------------------------------------------
# 5. Atomic saving helpers
# ------------------------------------------------------------

def atomic_torch_save(payload, destination):
    temporary = Path(str(destination) + ".partial")
    if temporary.exists():
        raise RuntimeError(
            f"Unresolved partial checkpoint: {temporary}"
        )
    torch.save(payload, temporary)
    os.replace(temporary, destination)

def atomic_csv_save(frame, destination):
    temporary = Path(str(destination) + ".partial")
    if temporary.exists():
        raise RuntimeError(
            f"Unresolved partial CSV: {temporary}"
        )
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)

# ------------------------------------------------------------
# 6. Training loop
# ------------------------------------------------------------

if start_epoch > MAX_EPOCHS:
    print("Maximum epochs already completed.")

elif completed_epoch >= MIN_EPOCHS and no_improve >= PATIENCE:
    print("Early stopping already reached.")

else:
    for epoch in range(start_epoch, MAX_EPOCHS + 1):

        epoch_start = time.perf_counter()
        epoch_lr = float(optimizer.param_groups[0]["lr"])

        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(DEVICE)

        train_true = []
        train_pred = []
        train_loss_num = 0.0
        train_weight_den = 0.0

        group_weight_total = torch.zeros(
            (), device=DEVICE, dtype=torch.float32
        )

        optimizer_steps = 0

        for batch_idx, (images, labels, _) in enumerate(
            tqdm(
                resume_train_loader,
                desc=f"Meta MAE Epoch {epoch}"
            )
        ):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            ):
                logits = model(images)
                losses = criterion(logits.float(), labels)
                loss_sum = losses.sum()

            assert torch.isfinite(loss_sum)

            loss_sum.backward()

            batch_weight = class_weights[labels].sum().detach()
            group_weight_total += batch_weight

            train_loss_num += float(loss_sum.detach().item())
            train_weight_den += float(batch_weight.item())

            predictions = logits.detach().argmax(dim=1)
            train_true.extend(labels.cpu().tolist())
            train_pred.extend(predictions.cpu().tolist())

            is_group_end = (
                (batch_idx + 1) % ACCUM_STEPS == 0
                or (batch_idx + 1) == len(resume_train_loader)
            )

            if is_group_end:
                assert group_weight_total.item() > 0

                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_weight_total)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP,
                    error_if_nonfinite=True
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                group_weight_total.zero_()
                optimizer_steps += 1

        assert len(train_true) == 1290
        assert optimizer_steps == 81

        train_loss = train_loss_num / train_weight_den
        train_accuracy = accuracy_score(train_true, train_pred)
        train_macro_f1 = f1_score(
            train_true,
            train_pred,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_true = []
        val_pred = []
        val_ids = []
        val_probabilities = []

        val_loss_num = 0.0
        val_weight_den = 0.0

        with torch.inference_mode():
            for images, labels, image_ids in tqdm(
                resume_val_loader,
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

                losses = criterion(logits.float(), labels)
                probabilities = torch.softmax(
                    logits.float(), dim=1
                )
                predictions = probabilities.argmax(dim=1)

                val_loss_num += float(losses.sum().item())
                val_weight_den += float(
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

        val_loss = val_loss_num / val_weight_den
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

        true_ref = np.asarray(val_true) >= 2
        pred_ref = np.asarray(val_pred) >= 2

        tp = int(np.sum(true_ref & pred_ref))
        fn = int(np.sum(true_ref & ~pred_ref))
        tn = int(np.sum(~true_ref & ~pred_ref))
        fp = int(np.sum(~true_ref & pred_ref))

        ref_sensitivity = tp / (tp + fn)
        ref_specificity = tn / (tn + fp)

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

        # ----------------------------------------------------
        # Save best model and predictions
        # ----------------------------------------------------

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "val_qwk": float(val_qwk),
            "referable_sensitivity": ref_sensitivity,
            "referable_specificity": ref_specificity,
            "learning_rate": epoch_lr,
            "duration_seconds": elapsed,
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved
        }

        history = pd.concat(
            [history, pd.DataFrame([row])],
            ignore_index=True
        )

        if is_best:
            pred_frame = pd.DataFrame({
                "image_id": val_ids,
                "true_grade": val_true,
                "predicted_grade": val_pred,
                **{
                    f"prob_dr{grade}": [
                        p[grade] for p in val_probabilities
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
                BEST_PATH
            )

            atomic_csv_save(pred_frame, BEST_PRED_PATH)

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
            "history_rows": history.to_dict(
                orient="records"
            )
        }

        atomic_torch_save(training_state, CHECKPOINT_PATH)
        atomic_csv_save(history, HISTORY_PATH)

        del training_state
        gc.collect()

        print(f"\nEPOCH {epoch} SUMMARY")
        print("Train loss:", round(train_loss, 6))
        print("Train macro-F1:", round(train_macro_f1, 6))
        print("Validation loss:", round(val_loss, 6))
        print("Validation macro-F1:", round(val_macro_f1, 6))
        print("Validation QWK:", round(float(val_qwk), 6))
        print("Referable sensitivity:", round(ref_sensitivity, 6))
        print("Referable specificity:", round(ref_specificity, 6))
        print("Best epoch:", best_epoch)
        print("Best macro-F1:", round(best_f1, 6))
        print("Epochs without improvement:", no_improve)
        print("Epoch duration minutes:", round(elapsed / 60, 2))
        print("Checkpoint saved: PASS")

        if epoch >= MIN_EPOCHS and no_improve >= PATIENCE:
            print("\nEARLY STOPPING TRIGGERED")
            print("Stopped at epoch:", epoch)
            break

# ------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------

history = pd.read_csv(HISTORY_PATH)
best_row = history.loc[history["val_macro_f1"].idxmax()]

print("\nMETA MAE TRAINING SUMMARY")
print("Completed epochs:", int(history["epoch"].max()))
print("Best epoch:", int(best_row["epoch"]))
print(
    "Best validation macro-F1:",
    round(float(best_row["val_macro_f1"]), 6)
)
print(
    "Best validation accuracy:",
    round(float(best_row["val_accuracy"]), 6)
)
print(
    "Best validation QWK:",
    round(float(best_row["val_qwk"]), 6)
)
print("Best model exists:", BEST_PATH.is_file())
print("Resume checkpoint exists:", CHECKPOINT_PATH.is_file())
print("Test images evaluated: 0")
print("\nSTEP 07.22 COMPLETED")


# %% [cell 229] Experiment 07 - Step 07.23
# ============================================================
# Experiment 07 - Step 07.23
# Paired ViT-L Validation Prediction Audit
# CSV-only; no model inference or test-set access
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
)

ROOT = DR_RESEARCH

RET_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_validation_predictions.csv"
)

META_PATH = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1"
    / "best_validation_predictions.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

for path in [RET_PATH, META_PATH, INDEX_PATH]:
    assert path.is_file(), f"Missing file: {path}"

ret = pd.read_csv(RET_PATH, dtype={"image_id": str})
meta = pd.read_csv(META_PATH, dtype={"image_id": str})
index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

print("EXPERIMENT 07.23 - PAIRED VALIDATION AUDIT")

required = {
    "image_id", "true_grade", "predicted_grade",
    "prob_dr0", "prob_dr1", "prob_dr2",
    "prob_dr3", "prob_dr4"
}

for name, frame in [("RETFound", ret), ("Meta MAE", meta)]:
    assert required.issubset(frame.columns)
    assert len(frame) == 276
    assert frame["image_id"].is_unique
    assert frame["true_grade"].between(0, 4).all()
    assert frame["predicted_grade"].between(0, 4).all()
    print(f"{name} predictions: {len(frame)}")

val_index = index.loc[
    index["split"] == "val",
    ["image_id", "dr_grade", "risk_group_id"]
].copy()

assert len(val_index) == 276
assert val_index["image_id"].is_unique

paired = (
    val_index
    .merge(
        ret.rename(columns={
            "true_grade": "ret_true",
            "predicted_grade": "ret_pred"
        })[["image_id", "ret_true", "ret_pred"]],
        on="image_id",
        how="inner",
        validate="one_to_one"
    )
    .merge(
        meta.rename(columns={
            "true_grade": "meta_true",
            "predicted_grade": "meta_pred"
        })[["image_id", "meta_true", "meta_pred"]],
        on="image_id",
        how="inner",
        validate="one_to_one"
    )
)

assert len(paired) == 276
assert set(paired["image_id"]) == set(val_index["image_id"])

assert (
    paired["dr_grade"].astype(int)
    == paired["ret_true"].astype(int)
).all()

assert (
    paired["dr_grade"].astype(int)
    == paired["meta_true"].astype(int)
).all()

y = paired["dr_grade"].to_numpy(dtype=int)

def evaluate(predictions):
    pred = predictions.to_numpy(dtype=int)

    true_ref = y >= 2
    pred_ref = pred >= 2

    sensitivity = np.mean(pred_ref[true_ref])
    specificity = np.mean(~pred_ref[~true_ref])

    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(
            y, pred, average="macro",
            labels=[0, 1, 2, 3, 4],
            zero_division=0
        ),
        "qwk": cohen_kappa_score(
            y, pred,
            labels=[0, 1, 2, 3, 4],
            weights="quadratic"
        ),
        "referable_sensitivity": sensitivity,
        "referable_specificity": specificity
    }

ret_metrics = evaluate(paired["ret_pred"])
meta_metrics = evaluate(paired["meta_pred"])

print("\nPAIRED VALIDATION METRICS")
print(
    f"{'Metric':<26} {'RETFound':>12} "
    f"{'Meta MAE':>12} {'Difference':>12}"
)

for key in ret_metrics:
    a = ret_metrics[key]
    b = meta_metrics[key]
    print(
        f"{key:<26} {a:>12.6f} "
        f"{b:>12.6f} {b-a:>+12.6f}"
    )

ret_correct = paired["ret_pred"].to_numpy() == y
meta_correct = paired["meta_pred"].to_numpy() == y

print("\nPAIRED CORRECTNESS")
print("Both correct:", int(np.sum(ret_correct & meta_correct)))
print("RETFound only correct:", int(np.sum(ret_correct & ~meta_correct)))
print("Meta MAE only correct:", int(np.sum(~ret_correct & meta_correct)))
print("Both wrong:", int(np.sum(~ret_correct & ~meta_correct)))

print("\nVALIDATION COHORT")
print("Images:", len(paired))
print("Source-risk groups:", paired["risk_group_id"].nunique())
print("Test images used:", 0)
print("Model inference performed:", False)

print("\nSTEP 07.23 PASS")


# %% [cell 230] Experiment 07 - Step 07.24
# ============================================================
# Experiment 07 - Step 07.24
# Paired Source-Risk-Group Bootstrap
# 2,000 resamples; validation CSVs only
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score
)

ROOT = DR_RESEARCH

RET_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_validation_predictions.csv"
)

META_PATH = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1"
    / "best_validation_predictions.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

OUT_DIR = ROOT / "experiment07_results"
OUT_CSV = OUT_DIR / "ViTL_Paired_Bootstrap_V1.csv"
OUT_JSON = OUT_DIR / "ViTL_Paired_Bootstrap_Report_V1.json"

assert all(p.is_file() for p in [
    RET_PATH, META_PATH, INDEX_PATH
])
assert not OUT_CSV.exists()
assert not OUT_JSON.exists()

ret = pd.read_csv(RET_PATH, dtype={"image_id": str})
meta = pd.read_csv(META_PATH, dtype={"image_id": str})

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

val = index.loc[
    index["split"] == "val",
    ["image_id", "risk_group_id", "dr_grade"]
].copy()

assert len(val) == 276
assert val["risk_group_id"].notna().all()

for frame in [ret, meta]:
    assert len(frame) == 276
    assert frame["image_id"].is_unique

paired = (
    val.merge(
        ret[["image_id", "true_grade", "predicted_grade"]]
        .rename(columns={
            "true_grade": "ret_true",
            "predicted_grade": "ret_pred"
        }),
        on="image_id",
        validate="one_to_one"
    ).merge(
        meta[["image_id", "true_grade", "predicted_grade"]]
        .rename(columns={
            "true_grade": "meta_true",
            "predicted_grade": "meta_pred"
        }),
        on="image_id",
        validate="one_to_one"
    )
)

assert len(paired) == 276
assert (paired["dr_grade"] == paired["ret_true"]).all()
assert (paired["dr_grade"] == paired["meta_true"]).all()

y = paired["dr_grade"].to_numpy(dtype=int)
ret_pred = paired["ret_pred"].to_numpy(dtype=int)
meta_pred = paired["meta_pred"].to_numpy(dtype=int)

def metrics(y_true, y_pred):
    ref_true = y_true >= 2
    ref_pred = y_pred >= 2

    return np.array([
        accuracy_score(y_true, y_pred),
        f1_score(
            y_true, y_pred,
            labels=[0, 1, 2, 3, 4],
            average="macro",
            zero_division=0
        ),
        cohen_kappa_score(
            y_true, y_pred,
            labels=[0, 1, 2, 3, 4],
            weights="quadratic"
        ),
        np.mean(ref_pred[ref_true])
        if ref_true.any() else np.nan,
        np.mean(~ref_pred[~ref_true])
        if (~ref_true).any() else np.nan
    ], dtype=float)

metric_names = [
    "accuracy",
    "macro_f1",
    "qwk",
    "referable_sensitivity",
    "referable_specificity"
]

point_diff = (
    metrics(y, meta_pred) - metrics(y, ret_pred)
)

groups = paired["risk_group_id"].astype(str).to_numpy()
unique_groups = np.unique(groups)

assert len(unique_groups) == 239

group_indices = [
    np.flatnonzero(groups == group)
    for group in unique_groups
]

N_BOOT = 2000
SEED = 20261020

rng = np.random.default_rng(SEED)
samples = np.empty((N_BOOT, len(metric_names)))

print("EXPERIMENT 07.24 - PAIRED CLUSTER BOOTSTRAP")
print("Validation images:", len(y))
print("Source-risk groups:", len(unique_groups))
print("Bootstrap resamples:", N_BOOT)

for b in range(N_BOOT):
    selected_groups = rng.integers(
        0, len(unique_groups),
        size=len(unique_groups)
    )

    sample_idx = np.concatenate([
        group_indices[g] for g in selected_groups
    ])

    sample_y = y[sample_idx]

    samples[b] = (
        metrics(sample_y, meta_pred[sample_idx])
        - metrics(sample_y, ret_pred[sample_idx])
    )

assert np.isfinite(samples).all()

lower = np.percentile(samples, 2.5, axis=0)
upper = np.percentile(samples, 97.5, axis=0)

summary = pd.DataFrame({
    "metric": metric_names,
    "meta_minus_retfound": point_diff,
    "ci95_lower": lower,
    "ci95_upper": upper,
    "ci_excludes_zero": (
        (lower > 0) | (upper < 0)
    )
})

report = {
    "experiment": "07.24",
    "method": "paired_source_risk_group_bootstrap",
    "resamples": N_BOOT,
    "seed": SEED,
    "validation_images": len(y),
    "source_risk_groups": len(unique_groups),
    "comparison": "Meta MAE minus RETFound",
    "ci_method": "percentile_95",
    "note": (
        "Validation only. Source-risk groups are "
        "not independently verified patient IDs. "
        "No multiple-comparison adjustment."
    ),
    "results": summary.to_dict(orient="records"),
    "test_images_used": 0
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

with OUT_CSV.open("x", encoding="utf-8") as f:
    summary.to_csv(f, index=False)

with OUT_JSON.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nBOOTSTRAP RESULTS")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nOUTPUT FILES")
print("CSV:", OUT_CSV)
print("JSON:", OUT_JSON)
print("Test images used:", 0)
print("Model inference performed:", False)
print("\nSTEP 07.24 PASS")


# %% [cell 231] Experiment 07 - Step 07.25
# ============================================================
# Experiment 07 - Step 07.25
# Recover Existing MA Attribution Analysis Protocol
# Read-only inventory; no model inference
# ============================================================

from pathlib import Path
import json

ROOT = DR_RESEARCH
EXP05 = ROOT / "experiment05_results"

print("EXPERIMENT 07.25 - MA ATTRIBUTION PROTOCOL INVENTORY")
print("Experiment 05 exists:", EXP05.is_dir())

assert EXP05.is_dir()

# Inspect only Experiment 05 and its immediate subfolders.
folders = [EXP05]

folders.extend(
    p for p in sorted(EXP05.iterdir())
    if p.is_dir()
)

print("\nEXPERIMENT 05 ARTIFACTS")

for folder in folders:
    print(f"\nDIRECTORY: {folder.relative_to(ROOT)}")

    entries = sorted(folder.iterdir(), key=lambda p: p.name)

    for path in entries[:70]:
        if path.is_dir():
            print("DIR :", path.name)
        else:
            print(
                "FILE:",
                path.name,
                "| MiB:",
                round(path.stat().st_size / 1024**2, 3)
            )

    if len(entries) > 70:
        print("Additional entries:", len(entries) - 70)

print("\nSMALL JSON PROTOCOL FILES")

for folder in folders:
    for path in sorted(folder.glob("*.json")):
        if path.stat().st_size > 100_000:
            continue

        print(f"\n--- {path.relative_to(ROOT)} ---")

        try:
            content = json.loads(
                path.read_text(encoding="utf-8")
            )

            if isinstance(content, dict):
                print("Keys:", sorted(content.keys()))

                for key, value in content.items():
                    if any(term in key.lower() for term in [
                        "method", "layer", "target",
                        "mask", "subset", "spatial",
                        "seed", "bootstrap", "threshold",
                        "eligible", "kernel", "control"
                    ]):
                        print(f"{key}: {str(value)[:400]}")

            else:
                print("JSON type:", type(content).__name__)

        except Exception as error:
            print("Read error:", str(error))

print("\nModel inference performed: False")
print("Test images accessed: 0")
print("STEP 07.25 INSPECTION COMPLETE")


# %% [cell 232] Experiment 07 - Step 07.26
# ============================================================
# Experiment 07 - Step 07.26
# Recover RETFound CAM Storage and Evaluation Protocol
# Read-only; no inference or training
# ============================================================

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = DR_RESEARCH
EXP05 = ROOT / "experiment05_results"
GRADCAM = EXP05 / "gradcam"

files = {
    "cohort": EXP05 / "FGADR_Validation_MA_Cohort_V1.csv",
    "ret_attribution": GRADCAM / "RETFound_MA_Attribution_V1.csv",
    "ret_maps": GRADCAM / "RETFound_MA_CAM_Maps_V1.npz",
    "ret_report": GRADCAM / "RETFound_MA_Attribution_Report_V1.json",
    "mask_protocol": EXP05 / "MA_Mask_Resampling_Protocol_V1.json",
    "spatial_report": EXP05 / "MA_Spatial_Null_Retinal_Constrained_Report_V1.json",
    "spatial_results": EXP05 / "MA_Spatial_Null_Retinal_Constrained_V1.csv",
}

assert all(p.is_file() for p in files.values())

print("EXPERIMENT 07.26 - MA ATTRIBUTION PROTOCOL AUDIT")

for name in ["cohort", "ret_attribution", "spatial_results"]:
    frame = pd.read_csv(files[name], dtype={"image_id": str})

    print(f"\n=== {name.upper()} ===")
    print("Rows:", len(frame))
    print("Columns:", frame.columns.tolist())
    print("First two records:")
    print(frame.head(2).to_string(index=False, max_cols=20))

for name in ["ret_report", "mask_protocol", "spatial_report"]:
    report = json.loads(
        files[name].read_text(encoding="utf-8")
    )

    print(f"\n=== {name.upper()} ===")
    print(json.dumps(report, indent=2, ensure_ascii=False)[:9000])

print("\n=== RETFOUND CAM MAP CONTAINER ===")

with np.load(files["ret_maps"], allow_pickle=False) as maps:
    print("Archive entries:", len(maps.files))
    print("First 15 keys:", maps.files[:15])

    # Metadata inspection only. Do not recompute CAM values.
    print("Archive size MiB:",
          round(files["ret_maps"].stat().st_size / 1024**2, 3))

print("\nNew CAM inference performed: False")
print("Test images accessed: 0")
print("STEP 07.26 INSPECTION COMPLETE")


# %% [cell 233] Experiment 07 - Step 07.27
# ============================================================
# Experiment 07 - Step 07.27
# CAM Format and Risk-Group Consistency Audit
# Read-only; no model inference or training
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
EXP05 = ROOT / "experiment05_results"

COHORT_PATH = EXP05 / "FGADR_Validation_MA_Cohort_V1.csv"
SPATIAL_PATH = EXP05 / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
CAM_PATH = EXP05 / "gradcam/RETFound_MA_CAM_Maps_V1.npz"
INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})
spatial = pd.read_csv(
    SPATIAL_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)
index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

print("EXPERIMENT 07.27 - CAM AND GROUP AUDIT")

# ------------------------------------------------------------
# 1. Frozen validation membership
# ------------------------------------------------------------

val_index = index.loc[index["split"] == "val"].copy()

primary = cohort.loc[
    cohort["primary_ma_candidate"].astype(str).str.lower() == "true"
].copy()

eligible = spatial.loc[
    spatial["eligible"].astype(str).str.lower() == "true"
].copy()

assert len(val_index) == 276
assert len(primary) == 156
assert len(eligible) == 137

assert primary["image_id"].is_unique
assert eligible["image_id"].is_unique

assert set(primary["image_id"]).issubset(
    set(val_index["image_id"])
)

assert set(eligible["image_id"]).issubset(
    set(primary["image_id"])
)

print("\nCOHORT STATUS")
print("Validation images:", len(val_index))
print("Primary MA images:", len(primary))
print("Spatial-control eligible images:", len(eligible))

# ------------------------------------------------------------
# 2. Risk-group audit
# ------------------------------------------------------------

print("\nRISK-GROUP STATUS")
print(
    "Spatial CSV missing group IDs:",
    int(spatial["risk_group_id"].isna().sum())
)

group_lookup = val_index[
    ["image_id", "risk_group_id"]
].rename(columns={
    "risk_group_id": "frozen_risk_group_id"
})

recovered = spatial.merge(
    group_lookup,
    on="image_id",
    how="left",
    validate="one_to_one"
)

print(
    "Frozen index missing group IDs:",
    int(recovered["frozen_risk_group_id"].isna().sum())
)

print(
    "Eligible frozen risk groups:",
    recovered.loc[
        recovered["eligible"].astype(str).str.lower() == "true",
        "frozen_risk_group_id"
    ].nunique()
)

both_present = recovered[
    recovered["risk_group_id"].notna()
    & recovered["frozen_risk_group_id"].notna()
]

print(
    "Existing group ID mismatches:",
    int((
        both_present["risk_group_id"]
        != both_present["frozen_risk_group_id"]
    ).sum())
)

# ------------------------------------------------------------
# 3. Existing RETFound CAM storage
# ------------------------------------------------------------

print("\nRETFOUND CAM FORMAT")

with np.load(CAM_PATH, allow_pickle=False) as archive:
    ids = archive["image_ids"]
    maps = archive["retfound_cam"]

    ids = [str(x) for x in ids.tolist()]

    print("Image ID array shape:", len(ids))
    print("CAM array shape:", maps.shape)
    print("CAM dtype:", maps.dtype)
    print("CAM finite:", bool(np.isfinite(maps).all()))
    print("CAM minimum:", float(maps.min()))
    print("CAM maximum:", float(maps.max()))

    print(
        "CAM IDs match primary cohort:",
        set(ids) == set(primary["image_id"])
    )

    print(
        "CAM IDs unique:",
        len(ids) == len(set(ids))
    )

    assert len(ids) == 156
    assert len(ids) == len(set(ids))
    assert set(ids) == set(primary["image_id"])
    assert np.isfinite(maps).all()

print("\nOriginal files modified: False")
print("Model inference performed: False")
print("Test images accessed: 0")
print("STEP 07.27 PASS")


# %% [cell 234] Experiment 07 - Step 07.28
# ============================================================
# Experiment 07 - Step 07.28
# Recover Original RETFound Grad-CAM Implementation
# Read-only notebook inspection; no model inference
# ============================================================

from pathlib import Path
import json
import re

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert NOTEBOOK.is_file(), "Original research notebook not found"

with NOTEBOOK.open("r", encoding="utf-8") as file:
    notebook = json.load(file)

cells = notebook.get("cells", [])

patterns = [
    r"register_forward_hook",
    r"register_full_backward_hook",
    r"retain_grad",
    r"GradCAM",
    r"grad_cam",
    r"gradcam",
    r"reshape_transform",
    r"retfound_cam",
    r"blocks\[23\]",
    r"blocks\[-1\]"
]

matches = []

for index, cell in enumerate(cells):
    if cell.get("cell_type") != "code":
        continue

    source = "".join(cell.get("source", []))

    matched = [
        pattern for pattern in patterns
        if re.search(pattern, source, flags=re.IGNORECASE)
    ]

    if matched:
        matches.append((index, source, matched))

print("EXPERIMENT 07.28 - ORIGINAL GRADCAM CODE AUDIT")
print("Notebook cells:", len(cells))
print("Matching code cells:", len(matches))

for index, source, matched in matches:
    print("\n" + "=" * 65)
    print("NOTEBOOK CELL:", index)
    print("MATCHED PATTERNS:", matched)
    print("SOURCE LENGTH:", len(source))

    lines = source.splitlines()

    if len(source) <= 15000:
        print(source)
    else:
        matching_lines = set()

        for line_no, line in enumerate(lines):
            if any(
                re.search(p, line, flags=re.IGNORECASE)
                for p in patterns
            ):
                matching_lines.update(
                    range(
                        max(0, line_no - 12),
                        min(len(lines), line_no + 22)
                    )
                )

        for line_no in sorted(matching_lines):
            print(f"{line_no + 1:04d}: {lines[line_no]}")

print("\nModel inference performed: False")
print("Existing CAM files modified: False")
print("Test images accessed: 0")
print("STEP 07.28 INSPECTION COMPLETE")


# %% [cell 235] Experiment 07 - Step 07.29
# ============================================================
# Experiment 07 - Step 07.29
# Recover Final RETFound Batch Grad-CAM Core
# Read-only notebook extraction
# ============================================================

from pathlib import Path
import json

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

with NOTEBOOK.open("r", encoding="utf-8") as f:
    notebook = json.load(f)

cell = notebook["cells"][120]

assert cell["cell_type"] == "code"

source = "".join(cell["source"])
lines = source.splitlines()

assert "Experiment 05" in source
assert "Step 5.4H" in source
assert "retfound_cam" in source

print("EXPERIMENT 07.29 - FINAL CAM IMPLEMENTATION")

# Print only the core batch-inference and Grad-CAM logic.
start_line = 275
end_line = 490

print("Source cell:", 120)
print("Total source lines:", len(lines))

for i in range(start_line - 1, min(end_line, len(lines))):
    print(f"{i + 1:04d}: {lines[i]}")

print("\nNotebook modified: False")
print("Model inference performed: False")
print("Test images accessed: 0")
print("STEP 07.29 INSPECTION COMPLETE")


# %% [cell 236] Experiment 07 - Step 07.30
# ============================================================
# Experiment 07 - Step 07.30
# Meta MAE Grad-CAM Pilot
# Exact validation batch context; read-only
# ============================================================

from pathlib import Path
import gc

import cv2
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from torchvision import transforms

ROOT = DR_RESEARCH

MODEL_PATH = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1/best_model.pt"
)
PRED_PATH = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1/best_validation_predictions.csv"
)
INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)
COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)
RET_ATTR_PATH = (
    ROOT / "experiment05_results/gradcam"
    / "RETFound_MA_Attribution_V1.csv"
)

assert torch.cuda.is_available()
assert all(p.is_file() for p in [
    MODEL_PATH, PRED_PATH, INDEX_PATH,
    COHORT_PATH, RET_ATTR_PATH
])

pred = pd.read_csv(PRED_PATH, dtype={"image_id": str})
index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})
cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})
ret_attr = pd.read_csv(RET_ATTR_PATH, dtype={"image_id": str})

assert len(pred) == 276
assert pred["image_id"].is_unique

# Match original validation batch ordering.
frame = pred.iloc[:16].merge(
    index[["image_id", "split", "dr_grade", "local_path"]],
    on="image_id",
    how="left",
    sort=False,
    validate="one_to_one"
)

assert len(frame) == 16
assert frame["split"].eq("val").all()
assert (
    frame["true_grade"].astype(int)
    == frame["dr_grade"].astype(int)
).all()

primary_ids = set(
    cohort.loc[cohort["primary_ma_candidate"], "image_id"]
)
positions = [
    i for i, image_id in enumerate(frame["image_id"])
    if image_id in primary_ids
]

assert positions, "No primary MA image in the first batch."

position = positions[0]
row = frame.iloc[position]
image_id = str(row["image_id"])
target_class = int(row["true_grade"])

transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

images = []

for item in frame.itertuples(index=False):
    path = Path(str(item.local_path))
    assert path.is_file(), f"Local image missing: {path}"

    with Image.open(path) as img:
        images.append(transform(img.convert("RGB")))

batch = torch.stack(images).to("cuda")

# Release previous training objects if present.
for name in ["model", "optimizer", "scheduler"]:
    if name in globals():
        del globals()[name]

gc.collect()
torch.cuda.empty_cache()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    mmap=True,
    weights_only=True
)

assert int(checkpoint["epoch"]) == 24

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg"
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

del checkpoint

model.eval()
model.set_grad_checkpointing(enable=False)
model.requires_grad_(False)
model = model.cuda()

captured = {}

def activation_hook(module, inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation

handle = model.blocks[-1].register_forward_hook(
    activation_hook
)

try:
    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(batch)

    probabilities = torch.softmax(
        logits.float(), dim=1
    ).detach().cpu().numpy()

    expected = frame[
        [f"prob_dr{i}" for i in range(5)]
    ].to_numpy(dtype=np.float32)

    max_difference = float(
        np.max(np.abs(probabilities - expected))
    )

    predictions_match = bool(np.array_equal(
        probabilities.argmax(axis=1),
        frame["predicted_grade"].to_numpy(dtype=int)
    ))

    assert predictions_match, "Prediction mismatch"
    assert max_difference <= 1e-4, (
        f"Probability mismatch: {max_difference}"
    )

    activation = captured["activation"]
    assert activation.shape == (16, 197, 1024)

    gradient = torch.autograd.grad(
        outputs=logits[position, target_class],
        inputs=activation,
        retain_graph=False,
        create_graph=False
    )[0]

    spatial_activation = (
        activation[position, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    spatial_gradient = (
        gradient[position, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    channel_weights = spatial_gradient.mean(
        dim=(1, 2), keepdim=True
    )

    raw_cam = torch.relu(
        (channel_weights * spatial_activation).sum(dim=0)
    ).cpu().numpy()

    raw_max = float(raw_cam.max())
    assert np.isfinite(raw_cam).all()
    assert raw_max > 1e-12

    cam = cv2.resize(
        raw_cam, (224, 224),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    cam /= float(cam.max())

    lesion_row = cohort.set_index("image_id").loc[image_id]

    with Image.open(lesion_row["resolved_mask_path"]) as img:
        mask = np.asarray(img)

    if mask.ndim == 3:
        assert np.array_equal(mask[..., 0], mask[..., 1])
        assert np.array_equal(mask[..., 0], mask[..., 2])
        mask = mask[..., 0]

    weighted_mask = cv2.resize(
        (
            mask > int(lesion_row["reference_threshold"])
        ).astype(np.float32),
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    assert weighted_mask.sum() > 0

    ret_row = ret_attr.set_index("image_id").loc[image_id]
    assert np.isclose(
        weighted_mask.sum(),
        float(ret_row["mask_weight_sum"]),
        atol=1e-4
    )

    mean_full = float(cam.mean())
    mean_ma = float(
        (cam * weighted_mask).sum() / weighted_mask.sum()
    )
    enrichment = mean_ma / mean_full

    assert np.isfinite(enrichment)
    assert np.isfinite(cam).all()
    assert cam.shape == (224, 224)

    print("EXPERIMENT 07.30 - META MAE CAM PILOT")
    print("Image ID:", image_id)
    print("True grade:", target_class)
    print("Meta predicted grade:", int(probabilities[position].argmax()))
    print("Validation batch size:", len(frame))
    print("Predictions match:", predictions_match)
    print("Maximum probability difference:", max_difference)

    print("\nCAM QUALITY")
    print("Target layer: blocks.23")
    print("CAM shape:", cam.shape)
    print("CAM minimum:", float(cam.min()))
    print("CAM maximum:", float(cam.max()))
    print("CAM standard deviation:", float(cam.std()))
    print("Raw CAM maximum:", raw_max)
    print("MA mask weight sum:", float(weighted_mask.sum()))
    print("Meta MAE enrichment:", round(enrichment, 6))
    print("RETFound enrichment:", round(float(ret_row["ma_enrichment"]), 6))

    print("\nTest images accessed: 0")
    print("Research files modified: False")
    print("STEP 07.30 PASS")

finally:
    handle.remove()
    captured.clear()
    del model
    del batch
    gc.collect()
    torch.cuda.empty_cache()


# %% [cell 237] Experiment 07 - Step 07.31
# ============================================================
# Experiment 07 - Step 07.31
# Meta MAE Full MA Grad-CAM Attribution
# 156 primary images, original validation batch context
# Resumable per-batch caching; no test-set access
# ============================================================

from pathlib import Path
import os
import gc
import json

import cv2
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

ROOT = DR_RESEARCH

MODEL_PATH = ROOT / (
    "experiment07_results/meta_mae_vit_large_v1/best_model.pt"
)
PRED_PATH = ROOT / (
    "experiment07_results/meta_mae_vit_large_v1/"
    "best_validation_predictions.csv"
)
INDEX_PATH = ROOT / (
    "experiment03_results/FGADR_Training_Index_V1.csv"
)
COHORT_PATH = ROOT / (
    "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
)
RET_PATH = ROOT / (
    "experiment05_results/gradcam/"
    "RETFound_MA_Attribution_V1.csv"
)

OUT = ROOT / "experiment07_results/gradcam"
CACHE = OUT / "meta_mae_batch_cache_v1"

CSV_PATH = OUT / "Meta_MAE_MA_Attribution_V1.csv"
MAP_PATH = OUT / "Meta_MAE_MA_CAM_Maps_V1.npz"
REPORT_PATH = OUT / "Meta_MAE_MA_Attribution_Report_V1.json"

assert torch.cuda.is_available()

for path in [
    MODEL_PATH, PRED_PATH, INDEX_PATH,
    COHORT_PATH, RET_PATH
]:
    assert path.is_file(), f"Missing: {path}"

# Protect completed research artifacts.
assert not any(p.exists() for p in [
    CSV_PATH, MAP_PATH, REPORT_PATH
]), "Final outputs already exist; inspect them before rerunning."

OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Restore frozen cohort and validation ordering
# ------------------------------------------------------------

pred = pd.read_csv(PRED_PATH, dtype={"image_id": str})
index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)
cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})
ret = pd.read_csv(RET_PATH, dtype={"image_id": str})

assert len(pred) == len(cohort) == 276
assert len(ret) == 156
assert pred["image_id"].is_unique
assert index["image_id"].is_unique

validation = pred.merge(
    index[
        ["image_id", "split", "dr_grade", "local_path"]
    ],
    on="image_id",
    how="left",
    sort=False,
    validate="one_to_one"
)

assert len(validation) == 276
assert validation["split"].eq("val").all()
assert (
    validation["true_grade"].astype(int)
    == validation["dr_grade"].astype(int)
).all()

primary = cohort.loc[
    cohort["primary_ma_candidate"].astype(str).str.lower() == "true"
].copy()

assert len(primary) == 156
assert primary["image_id"].is_unique

primary_lookup = primary.set_index("image_id")
primary_ids = set(primary["image_id"])

assert primary_ids.issubset(set(validation["image_id"]))
assert primary_ids == set(ret["image_id"])

ret_lookup = ret.set_index("image_id")

BATCH_SIZE = 16
N_BATCHES = 18
TOLERANCE = 1e-4

batch_frames = [
    validation.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].copy()
    for i in range(N_BATCHES)
]

assert sum(len(f) for f in batch_frames) == 276

transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_image_batch(frame):
    images = []

    for row in frame.itertuples(index=False):
        path = Path(str(row.local_path))

        # Only use the restored local validation cache.
        assert path.is_file(), f"Local image missing: {path}"
        assert not path.is_symlink()

        with Image.open(path) as img:
            images.append(transform(img.convert("RGB")))

    return torch.stack(images).cuda()

def load_weighted_mask(row):
    with Image.open(row["resolved_mask_path"]) as img:
        mask = np.asarray(img)

    if mask.ndim == 3:
        assert np.array_equal(mask[..., 0], mask[..., 1])
        assert np.array_equal(mask[..., 0], mask[..., 2])
        mask = mask[..., 0]

    binary = (
        mask > int(row["reference_threshold"])
    ).astype(np.float32)

    weighted = cv2.resize(
        binary,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    assert float(weighted.sum()) > 0
    return weighted

def batch_cache_path(batch_index):
    return CACHE / f"batch_{batch_index:03d}.npz"

# ------------------------------------------------------------
# 2. Audit previously completed batches
# ------------------------------------------------------------

pending = []

for batch_index, frame in enumerate(batch_frames):
    expected_ids = [
        image_id
        for image_id in frame["image_id"].tolist()
        if image_id in primary_ids
    ]

    if not expected_ids:
        continue

    path = batch_cache_path(batch_index)

    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            cached_ids = archive["image_ids"].astype(str).tolist()
            cached_maps = archive["meta_mae_cam"]
            cached_records = json.loads(
                str(archive["records_json"].item())
            )

            assert cached_ids == expected_ids
            assert cached_maps.shape == (
                len(expected_ids), 224, 224
            )
            assert np.isfinite(cached_maps).all()
            assert len(cached_records) == len(expected_ids)
            assert [
                r["image_id"] for r in cached_records
            ] == expected_ids

    else:
        pending.append(batch_index)

print("EXPERIMENT 07.31 - META MAE FULL MA ATTRIBUTION")
print("Validation images:", len(validation))
print("Primary MA images:", len(primary))
print("Validation batches:", N_BATCHES)
print("Pending batches:", len(pending))
print("Cached batches:", 18 - len(pending))
print("Target layer: blocks.23")
print("Attribution target: ground_truth_grade_logit")

# ------------------------------------------------------------
# 3. Load best model only if processing is needed
# ------------------------------------------------------------

if pending:
    for name in ["model", "optimizer", "scheduler"]:
        if name in globals():
            del globals()[name]

    gc.collect()
    torch.cuda.empty_cache()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        mmap=True,
        weights_only=True
    )

    assert int(checkpoint["epoch"]) == 24

    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=5,
        global_pool="avg"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    del checkpoint

    model.eval()
    model.set_grad_checkpointing(enable=False)
    model.requires_grad_(False)
    model = model.cuda()

    captured = {}

    def activation_hook(module, inputs, output):
        activation = output.detach().requires_grad_(True)
        captured["activation"] = activation
        return activation

    handle = model.blocks[-1].register_forward_hook(
        activation_hook
    )

    try:
        for batch_index in tqdm(
            pending,
            desc="Meta MAE validation batches"
        ):
            frame = batch_frames[batch_index]

            target_positions = [
                position
                for position, image_id in enumerate(
                    frame["image_id"].tolist()
                )
                if image_id in primary_ids
            ]

            images = load_image_batch(frame)
            captured.clear()

            with torch.enable_grad():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16
                ):
                    logits = model(images)

            probabilities = torch.softmax(
                logits.float(), dim=1
            ).detach().cpu().numpy()

            expected = frame[
                [f"prob_dr{i}" for i in range(5)]
            ].to_numpy(dtype=np.float32)

            max_difference = float(
                np.max(np.abs(probabilities - expected))
            )

            assert np.array_equal(
                probabilities.argmax(axis=1),
                frame["predicted_grade"].to_numpy(dtype=int)
            ), f"Prediction mismatch: batch {batch_index}"

            assert max_difference <= TOLERANCE, (
                f"Probability mismatch: batch {batch_index}, "
                f"difference={max_difference}"
            )

            activation = captured["activation"]

            assert activation.shape == (
                len(frame), 197, 1024
            )

            records = []
            cams = []
            ids = []

            for target_number, position in enumerate(
                target_positions
            ):
                row = frame.iloc[position]
                image_id = str(row["image_id"])
                target_class = int(row["true_grade"])

                keep_graph = (
                    target_number < len(target_positions) - 1
                )

                gradient = torch.autograd.grad(
                    outputs=logits[position, target_class],
                    inputs=activation,
                    retain_graph=keep_graph,
                    create_graph=False
                )[0]

                a = (
                    activation[position, 1:, :]
                    .detach().float()
                    .transpose(0, 1)
                    .reshape(1024, 14, 14)
                )

                g = (
                    gradient[position, 1:, :]
                    .detach().float()
                    .transpose(0, 1)
                    .reshape(1024, 14, 14)
                )

                weights = g.mean(
                    dim=(1, 2),
                    keepdim=True
                )

                raw_cam = torch.relu(
                    (weights * a).sum(dim=0)
                ).cpu().numpy()

                assert np.isfinite(raw_cam).all()

                raw_max = float(raw_cam.max())

                assert raw_max > 1e-12, (
                    f"Degenerate CAM: {image_id}"
                )

                cam = cv2.resize(
                    raw_cam,
                    (224, 224),
                    interpolation=cv2.INTER_LINEAR
                ).astype(np.float32)

                cam /= float(cam.max())

                assert np.isfinite(cam).all()
                assert cam.shape == (224, 224)

                mask = load_weighted_mask(
                    primary_lookup.loc[image_id]
                )

                assert np.isclose(
                    float(mask.sum()),
                    float(
                        ret_lookup.loc[
                            image_id, "mask_weight_sum"
                        ]
                    ),
                    rtol=1e-5,
                    atol=1e-4
                )

                full_mean = float(cam.mean())
                ma_mean = float(
                    (cam * mask).sum() / mask.sum()
                )

                assert full_mean > 1e-12

                enrichment = ma_mean / full_mean

                assert np.isfinite(enrichment)

                records.append({
                    "image_id": image_id,
                    "true_grade": target_class,
                    "model": "meta_mae",
                    "target_layer": "blocks.23",
                    "predicted_grade": int(
                        probabilities[position].argmax()
                    ),
                    "correct": bool(
                        probabilities[position].argmax()
                        == target_class
                    ),
                    "mask_weight_sum": float(mask.sum()),
                    "mean_cam_full": full_mean,
                    "mean_cam_ma": ma_mean,
                    "ma_enrichment": enrichment,
                    "raw_cam_max": raw_max,
                    "cam_std": float(cam.std()),
                    "batch_index": batch_index,
                    "batch_position": position,
                    "probability_max_difference": float(
                        np.max(
                            np.abs(
                                probabilities[position]
                                - expected[position]
                            )
                        )
                    )
                })

                ids.append(image_id)
                cams.append(cam)

            cache_file = batch_cache_path(batch_index)
            partial = Path(str(cache_file) + ".partial")

            assert not cache_file.exists()
            assert not partial.exists()

            with partial.open("wb") as f:
                np.savez_compressed(
                    f,
                    image_ids=np.asarray(ids, dtype="U64"),
                    meta_mae_cam=np.stack(cams).astype(np.float32),
                    records_json=np.asarray(
                        json.dumps(records)
                    )
                )

            os.replace(partial, cache_file)

            del images, logits, probabilities, activation
            del gradient, a, g, raw_cam, cam
            captured.clear()

    finally:
        handle.remove()
        captured.clear()
        del model
        gc.collect()
        torch.cuda.empty_cache()

# ------------------------------------------------------------
# 4. Consolidate all completed batch caches
# ------------------------------------------------------------

all_records = []
all_ids = []
all_maps = []

for batch_index, frame in enumerate(batch_frames):
    expected_ids = [
        image_id
        for image_id in frame["image_id"].tolist()
        if image_id in primary_ids
    ]

    if not expected_ids:
        continue

    path = batch_cache_path(batch_index)
    assert path.is_file()

    with np.load(path, allow_pickle=False) as archive:
        ids = archive["image_ids"].astype(str).tolist()
        maps = archive["meta_mae_cam"].copy()
        records = json.loads(
            str(archive["records_json"].item())
        )

    assert ids == expected_ids
    assert len(records) == len(ids)
    assert maps.shape == (len(ids), 224, 224)

    all_ids.extend(ids)
    all_maps.extend(maps)
    all_records.extend(records)

assert len(all_ids) == 156
assert len(set(all_ids)) == 156
assert set(all_ids) == primary_ids

result = pd.DataFrame(all_records)
result = result.sort_values("image_id").reset_index(drop=True)

map_lookup = dict(zip(all_ids, all_maps))
sorted_ids = result["image_id"].tolist()

final_maps = np.stack([
    map_lookup[image_id]
    for image_id in sorted_ids
]).astype(np.float32)

assert final_maps.shape == (156, 224, 224)
assert np.isfinite(final_maps).all()
assert result["ma_enrichment"].notna().all()
assert result["probability_max_difference"].max() <= TOLERANCE

# ------------------------------------------------------------
# 5. Save final results atomically
# ------------------------------------------------------------

csv_temp = Path(str(CSV_PATH) + ".partial")
map_temp = Path(str(MAP_PATH) + ".partial")

assert not csv_temp.exists()
assert not map_temp.exists()

result.to_csv(csv_temp, index=False)

with map_temp.open("wb") as f:
    np.savez_compressed(
        f,
        image_ids=np.asarray(sorted_ids, dtype="U64"),
        meta_mae_cam=final_maps
    )

os.replace(map_temp, MAP_PATH)
os.replace(csv_temp, CSV_PATH)

report = {
    "experiment": "Experiment07_Step07.31",
    "model": "Meta_MAE_ViT_Large",
    "checkpoint": str(MODEL_PATH),
    "checkpoint_epoch": 24,
    "target_layer": "blocks.23",
    "attribution_target": "ground_truth_grade_logit",
    "precision": "bfloat16",
    "validation_batch_size": 16,
    "primary_images": 156,
    "validation_batches": 18,
    "probability_tolerance": TOLERANCE,
    "max_probability_difference": float(
        result["probability_max_difference"].max()
    ),
    "cam_shape": list(final_maps.shape),
    "test_split_evaluated": False
}

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

# ------------------------------------------------------------
# 6. Summary
# ------------------------------------------------------------

print("\nEXPERIMENT 07.31 - FINAL RESULTS")
print("Processed MA images:", len(result))
print("CAM shape:", final_maps.shape)
print("All CAMs finite:", bool(np.isfinite(final_maps).all()))
print(
    "Maximum probability difference:",
    float(result["probability_max_difference"].max())
)

print("\nMETA MAE MA ENRICHMENT")
print("Mean:", round(float(result["ma_enrichment"].mean()), 6))
print("Median:", round(float(result["ma_enrichment"].median()), 6))
print(
    "CAM standard deviation minimum:",
    round(float(result["cam_std"].min()), 6)
)

print("\nOUTPUT STATUS")
print("Attribution CSV saved:", CSV_PATH.is_file())
print("CAM NPZ saved:", MAP_PATH.is_file())
print("Report saved:", REPORT_PATH.is_file())
print("RETFound CAM recomputed: False")
print("Test images evaluated: 0")

print("\nSTEP 07.31 PASS")


# %% [cell 238] Experiment 07 - Step 07.32
# ============================================================
# Experiment 07 - Step 07.32
# Paired ViT-L MA Attribution Audit
# Read-only; no model inference or mask recomputation
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

RET_DIR = ROOT / "experiment05_results/gradcam"
META_DIR = ROOT / "experiment07_results/gradcam"

RET_CSV = RET_DIR / "RETFound_MA_Attribution_V1.csv"
META_CSV = META_DIR / "Meta_MAE_MA_Attribution_V1.csv"

RET_NPZ = RET_DIR / "RETFound_MA_CAM_Maps_V1.npz"
META_NPZ = META_DIR / "Meta_MAE_MA_CAM_Maps_V1.npz"

INDEX_PATH = (
    ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
)

for path in [RET_CSV, META_CSV, RET_NPZ, META_NPZ, INDEX_PATH]:
    assert path.is_file(), f"Missing: {path}"

ret = pd.read_csv(RET_CSV, dtype={"image_id": str})
meta = pd.read_csv(META_CSV, dtype={"image_id": str})
index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(ret) == len(meta) == 156
assert ret["image_id"].is_unique
assert meta["image_id"].is_unique
assert set(ret["image_id"]) == set(meta["image_id"])

fields = [
    "image_id",
    "true_grade",
    "predicted_grade",
    "mask_weight_sum",
    "mean_cam_full",
    "mean_cam_ma",
    "ma_enrichment",
    "cam_std"
]

paired = ret[fields].merge(
    meta[fields],
    on="image_id",
    how="inner",
    validate="one_to_one",
    suffixes=("_ret", "_meta")
)

paired = paired.merge(
    index[["image_id", "split", "dr_grade", "risk_group_id"]],
    on="image_id",
    how="left",
    validate="one_to_one"
)

assert len(paired) == 156
assert paired["split"].eq("val").all()
assert paired["risk_group_id"].notna().all()

assert (
    paired["true_grade_ret"].astype(int)
    == paired["true_grade_meta"].astype(int)
).all()

assert (
    paired["true_grade_ret"].astype(int)
    == paired["dr_grade"].astype(int)
).all()

assert np.allclose(
    paired["mask_weight_sum_ret"],
    paired["mask_weight_sum_meta"],
    atol=1e-4,
    rtol=1e-5
)

with np.load(RET_NPZ, allow_pickle=False) as archive:
    ret_ids = archive["image_ids"].astype(str).tolist()
    ret_maps = archive["retfound_cam"].copy()

with np.load(META_NPZ, allow_pickle=False) as archive:
    meta_ids = archive["image_ids"].astype(str).tolist()
    meta_maps = archive["meta_mae_cam"].copy()

assert ret_ids == meta_ids
assert set(ret_ids) == set(paired["image_id"])
assert ret_maps.shape == meta_maps.shape == (156, 224, 224)
assert np.isfinite(ret_maps).all()
assert np.isfinite(meta_maps).all()

for name, maps in [
    ("ret", ret_maps),
    ("meta", meta_maps)
]:
    mean_lookup = dict(
        zip(ret_ids, maps.mean(axis=(1, 2), dtype=np.float64))
    )

    expected_means = paired["image_id"].map(
        mean_lookup
    ).to_numpy(dtype=float)

    recorded_means = paired[
        f"mean_cam_full_{name}"
    ].to_numpy(dtype=float)

    assert np.allclose(
        expected_means,
        recorded_means,
        atol=1e-6,
        rtol=1e-4
    )

    assert np.isfinite(
        paired[f"ma_enrichment_{name}"]
    ).all()

paired["delta_enrichment"] = (
    paired["ma_enrichment_meta"]
    - paired["ma_enrichment_ret"]
)

paired["ret_correct"] = (
    paired["predicted_grade_ret"] == paired["dr_grade"]
)

paired["meta_correct"] = (
    paired["predicted_grade_meta"] == paired["dr_grade"]
)

paired = paired.sort_values("image_id").reset_index(drop=True)

print("EXPERIMENT 07.32 - PAIRED MA ATTRIBUTION AUDIT")

print("\nPAIRING STATUS")
print("Paired MA images:", len(paired))
print("Source-risk groups:", paired["risk_group_id"].nunique())
print("Ground-truth consistency: PASS")
print("MA mask area consistency: PASS")
print("CAM image alignment: PASS")
print("CAM mean consistency: PASS")

print("\nRAW MA ENRICHMENT")
for name, column in [
    ("RETFound", "ma_enrichment_ret"),
    ("Meta MAE", "ma_enrichment_meta")
]:
    values = paired[column]
    print(f"\n{name}")
    print("Mean:", round(float(values.mean()), 6))
    print("Median:", round(float(values.median()), 6))
    print("Enrichment > 1:", int((values > 1).sum()))

delta = paired["delta_enrichment"]

print("\nPAIRED RAW DIFFERENCE (META - RETFOUND)")
print("Mean difference:", round(float(delta.mean()), 6))
print("Median difference:", round(float(delta.median()), 6))
print("Meta higher images:", int((delta > 0).sum()))
print("RETFound higher images:", int((delta < 0).sum()))
print("Equal images:", int((delta == 0).sum()))

print("\nCLASSIFICATION ON PRIMARY MA COHORT")
print("RETFound correct:", int(paired["ret_correct"].sum()))
print("Meta MAE correct:", int(paired["meta_correct"].sum()))

print("\nResearch files modified: False")
print("Model inference performed: False")
print("Test images accessed: 0")
print("STEP 07.32 PASS")


# %% [cell 239] Experiment 07 - Step 07.33
# ============================================================
# Experiment 07 - Step 07.33
# Retinal-Constrained Spatial Null: RETFound vs Meta MAE
# Reuse existing CAM maps; validation only
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm.auto import tqdm

ROOT = DR_RESEARCH
EXP05 = ROOT / "experiment05_results"
OUT = ROOT / "experiment07_results"

COHORT = EXP05 / "FGADR_Validation_MA_Cohort_V1.csv"
INDEX = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
REFERENCE = EXP05 / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
RET_MAPS = EXP05 / "gradcam/RETFound_MA_CAM_Maps_V1.npz"
META_MAPS = OUT / "gradcam/Meta_MAE_MA_CAM_Maps_V1.npz"

RESULT = OUT / "ViTL_MA_Spatial_Null_Paired_V1.csv"

assert all(p.is_file() for p in [
    COHORT, INDEX, REFERENCE, RET_MAPS, META_MAPS
])
assert not RESULT.exists(), "Output already exists; do not overwrite."

cohort = pd.read_csv(COHORT, dtype={"image_id": str})
index = pd.read_csv(
    INDEX, dtype={"image_id": str, "risk_group_id": str}
)
reference = pd.read_csv(
    REFERENCE, dtype={"image_id": str}
).set_index("image_id")

primary = (
    cohort.loc[
        cohort["primary_ma_candidate"].astype(str).str.lower() == "true"
    ]
    .merge(
        index[[
            "image_id", "split", "risk_group_id",
            "local_path", "dr_grade"
        ]],
        on="image_id",
        how="inner",
        validate="one_to_one"
    )
    .set_index("image_id")
)

assert len(primary) == 156
assert primary["split"].eq("val").all()
assert primary["risk_group_id"].notna().all()

with np.load(RET_MAPS, allow_pickle=False) as data:
    ret_ids = data["image_ids"].astype(str).tolist()
    ret_cams = data["retfound_cam"].copy()

with np.load(META_MAPS, allow_pickle=False) as data:
    meta_ids = data["image_ids"].astype(str).tolist()
    meta_cams = data["meta_mae_cam"].copy()

assert ret_ids == meta_ids
assert set(ret_ids) == set(primary.index)
assert ret_cams.shape == meta_cams.shape == (156, 224, 224)

SIZE = 224
MIN_COVERAGE = 0.95
MIN_NULL = 100

def load_mask(row):
    with Image.open(row["resolved_mask_path"]) as image:
        mask = np.asarray(image)

    if mask.ndim == 3:
        assert np.array_equal(mask[..., 0], mask[..., 1])
        assert np.array_equal(mask[..., 0], mask[..., 2])
        mask = mask[..., 0]

    assert mask.shape == (1280, 1280)

    binary = (
        mask > int(row["reference_threshold"])
    ).astype(np.float32)

    weighted = cv2.resize(
        binary, (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    ).astype(np.float64)

    assert weighted.sum() > 0
    return weighted

def retinal_foreground(row):
    path = Path(str(row["local_path"]))
    assert path.is_file() and not path.is_symlink()

    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (SIZE, SIZE), Image.Resampling.BICUBIC
            )
        )

    candidate = (rgb.max(axis=2) > 20).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)
    )

    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        candidate, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    assert contours

    foreground = np.zeros((SIZE, SIZE), dtype=np.uint8)

    cv2.drawContours(
        foreground,
        [max(contours, key=cv2.contourArea)],
        -1, 1, thickness=-1
    )

    foreground = cv2.erode(foreground, kernel)
    return foreground.astype(np.float64)

def fft_shift_values(values, mask):
    return np.fft.ifft2(
        np.fft.fft2(values.astype(np.float64))
        * np.conj(np.fft.fft2(mask))
    ).real

def shift_geometry(mask, foreground):
    ys, xs = np.nonzero(mask > 0)

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())

    dx = np.arange(-xmin, SIZE - xmax, dtype=int)
    dy = np.arange(-ymin, SIZE - ymax, dtype=int)

    shifts_y, shifts_x = np.meshgrid(
        dy % SIZE, dx % SIZE, indexing="ij"
    )

    coverage_map = (
        fft_shift_values(foreground, mask) / mask.sum()
    )

    permitted = (
        coverage_map[shifts_y, shifts_x] >= MIN_COVERAGE
    )

    permitted &= ~(
        (shifts_y == 0) & (shifts_x == 0)
    )

    observed_coverage = float(
        (foreground * mask).sum() / mask.sum()
    )

    coords_y = shifts_y[permitted]
    coords_x = shifts_x[permitted]

    eligible = (
        observed_coverage >= MIN_COVERAGE
        and len(coords_y) >= MIN_NULL
    )

    return {
        "eligible": eligible,
        "coverage": observed_coverage,
        "null_shifts": len(coords_y),
        "coords_y": coords_y,
        "coords_x": coords_x
    }

def spatial_metrics(cam, mask, geometry):
    scores = fft_shift_values(cam, mask)
    scores /= float(mask.sum()) * float(cam.mean())

    observed = float(scores[0, 0])

    null = scores[
        geometry["coords_y"],
        geometry["coords_x"]
    ]

    assert len(null) >= MIN_NULL
    assert np.isfinite(null).all()

    return {
        "observed_enrichment": observed,
        "null_mean_enrichment": float(null.mean()),
        "null_adjusted_enrichment": float(
            observed - null.mean()
        ),
        "spatial_percentile": float(
            100 * np.mean(null < observed)
        ),
        "upper_tail_fraction": float(
            (1 + np.count_nonzero(null >= observed))
            / (len(null) + 1)
        )
    }

print("EXPERIMENT 07.33 - RETINAL-CONSTRAINED SPATIAL NULL")

records = []

# Reuse only the previously eligible analysis cohort.
eligible_ids = set(
    reference.index[reference["eligible"].astype(bool)]
)

assert len(eligible_ids) == 137

for i, image_id in enumerate(
    tqdm(ret_ids, desc="Paired MA spatial analysis")
):
    if image_id not in eligible_ids:
        continue

    row = primary.loc[image_id]
    old = reference.loc[image_id]

    mask = load_mask(row)
    foreground = retinal_foreground(row)
    geometry = shift_geometry(mask, foreground)

    assert geometry["eligible"], (
        f"Eligibility mismatch: {image_id}"
    )

    assert geometry["null_shifts"] == int(
        old["null_shifts"]
    ), f"Null shift mismatch: {image_id}"

    assert np.isclose(
        geometry["coverage"],
        float(old["observed_coverage"]),
        atol=1e-6
    )

    ret_result = spatial_metrics(
        ret_cams[i], mask, geometry
    )

    meta_result = spatial_metrics(
        meta_cams[i], mask, geometry
    )

    # Verify exact compatibility with the historical
    # RETFound spatial-null analysis.
    for metric in ret_result:
        old_value = float(old[f"retfound_{metric}"])

        assert np.isclose(
            ret_result[metric],
            old_value,
            atol=1e-5,
            rtol=1e-4
        ), f"RETFound protocol mismatch: {image_id}, {metric}"

    record = {
        "image_id": image_id,
        "true_grade": int(row["true_grade"]),
        "risk_group_id": str(row["risk_group_id"]),
        "eligible": True,
        "null_shifts": geometry["null_shifts"]
    }

    for prefix, values in [
        ("retfound", ret_result),
        ("meta_mae", meta_result)
    ]:
        for key, value in values.items():
            record[f"{prefix}_{key}"] = value

    record["delta_null_adjusted"] = (
        meta_result["null_adjusted_enrichment"]
        - ret_result["null_adjusted_enrichment"]
    )

    record["delta_spatial_percentile"] = (
        meta_result["spatial_percentile"]
        - ret_result["spatial_percentile"]
    )

    records.append(record)

results = pd.DataFrame(records).sort_values(
    "image_id"
).reset_index(drop=True)

assert len(results) == 137
assert results["image_id"].is_unique
assert results["risk_group_id"].nunique() == 120
assert np.isfinite(
    results["delta_null_adjusted"]
).all()

# Save only after completing all consistency checks.
with RESULT.open("x", encoding="utf-8") as f:
    results.to_csv(f, index=False)

print("\nPROTOCOL VERIFICATION")
print("Eligible images:", len(results))
print("Source-risk groups:", results["risk_group_id"].nunique())
print("RETFound historical metrics reproduced: PASS")
print("Shared null geometry: PASS")

print("\nSPATIAL-NULL RESULTS")

for model_name in ["retfound", "meta_mae"]:
    print(f"\n{model_name.upper()}")

    print(
        "Mean null-adjusted enrichment:",
        round(float(
            results[
                f"{model_name}_null_adjusted_enrichment"
            ].mean()
        ), 6)
    )

    print(
        "Median spatial percentile:",
        round(float(
            results[
                f"{model_name}_spatial_percentile"
            ].median()
        ), 4)
    )

    print(
        "Images above 95th null percentile:",
        int((
            results[
                f"{model_name}_upper_tail_fraction"
            ] <= 0.05
        ).sum())
    )

print("\nPAIRED DIFFERENCE (META - RETFOUND)")

print(
    "Mean null-adjusted difference:",
    round(float(results["delta_null_adjusted"].mean()), 6)
)

print(
    "Mean spatial percentile difference:",
    round(float(results["delta_spatial_percentile"].mean()), 6)
)

print(
    "Meta higher null-adjusted images:",
    int((results["delta_null_adjusted"] > 0).sum())
)

print(
    "RETFound higher null-adjusted images:",
    int((results["delta_null_adjusted"] < 0).sum())
)

print("\nOUTPUT STATUS")
print("Results saved:", RESULT.is_file())
print("Model inference performed: False")
print("Test images accessed: 0")
print("STEP 07.33 PASS")


# %% [cell 240] Experiment 07 - Step 07.34
# ============================================================
# Experiment 07 - Step 07.34
# Paired Source-Risk-Group Bootstrap for MA Attribution
# Same 137-image cohort; CSV-only analysis
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
OUT = ROOT / "experiment07_results"

SPATIAL_PATH = OUT / "ViTL_MA_Spatial_Null_Paired_V1.csv"

RET_PATH = (
    ROOT / "experiment05_results/gradcam"
    / "RETFound_MA_Attribution_V1.csv"
)

META_PATH = (
    OUT / "gradcam/Meta_MAE_MA_Attribution_V1.csv"
)

CSV_OUT = OUT / "ViTL_MA_Bootstrap_V1.csv"
JSON_OUT = OUT / "ViTL_MA_Bootstrap_Report_V1.json"

assert all(p.is_file() for p in [
    SPATIAL_PATH, RET_PATH, META_PATH
])

assert not CSV_OUT.exists()
assert not JSON_OUT.exists()

spatial = pd.read_csv(
    SPATIAL_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

ret = pd.read_csv(RET_PATH, dtype={"image_id": str})
meta = pd.read_csv(META_PATH, dtype={"image_id": str})

data = (
    spatial
    .merge(
        ret[["image_id", "ma_enrichment"]].rename(
            columns={"ma_enrichment": "ret_raw"}
        ),
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        meta[["image_id", "ma_enrichment"]].rename(
            columns={"ma_enrichment": "meta_raw"}
        ),
        on="image_id",
        validate="one_to_one"
    )
)

assert len(data) == 137
assert data["risk_group_id"].notna().all()
assert data["risk_group_id"].nunique() == 120
assert data["image_id"].is_unique

data["delta_raw"] = data["meta_raw"] - data["ret_raw"]

assert np.allclose(
    data["delta_null_adjusted"],
    data["meta_mae_null_adjusted_enrichment"]
    - data["retfound_null_adjusted_enrichment"],
    atol=1e-8
)

assert np.allclose(
    data["delta_spatial_percentile"],
    data["meta_mae_spatial_percentile"]
    - data["retfound_spatial_percentile"],
    atol=1e-8
)

metrics = [
    "delta_raw",
    "delta_null_adjusted",
    "delta_spatial_percentile"
]

grouped = data.groupby("risk_group_id", sort=True)

group_sizes = grouped.size().to_numpy(dtype=float)
group_sums = grouped[metrics].sum().to_numpy(dtype=float)

N_GROUPS = len(group_sizes)
N_BOOT = 2000
SEED = 20261020

assert N_GROUPS == 120

rng = np.random.default_rng(SEED)
draws = np.empty((N_BOOT, len(metrics)), dtype=float)

for i in range(N_BOOT):
    selected = rng.integers(
        0, N_GROUPS, size=N_GROUPS
    )

    draws[i] = (
        group_sums[selected].sum(axis=0)
        / group_sizes[selected].sum()
    )

point = data[metrics].mean().to_numpy()
lower = np.quantile(draws, 0.025, axis=0)
upper = np.quantile(draws, 0.975, axis=0)

summary = pd.DataFrame({
    "metric": metrics,
    "mean_meta_minus_retfound": point,
    "ci95_lower": lower,
    "ci95_upper": upper,
    "ci_excludes_zero": (lower > 0) | (upper < 0)
})

report = {
    "experiment": "Experiment07_Step07.34",
    "comparison": "Meta MAE minus RETFound",
    "method": "paired_source_risk_group_cluster_bootstrap",
    "images": 137,
    "source_risk_groups": 120,
    "bootstrap_resamples": N_BOOT,
    "seed": SEED,
    "confidence_interval": "percentile_95",
    "multiple_comparison_adjustment": False,
    "clinical_lesion_validation": False,
    "test_split_evaluated": False,
    "results": summary.to_dict(orient="records")
}

with CSV_OUT.open("x", encoding="utf-8") as f:
    summary.to_csv(f, index=False)

with JSON_OUT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 07.34 - PAIRED MA ATTRIBUTION BOOTSTRAP")
print("Eligible images:", len(data))
print("Source-risk groups:", N_GROUPS)
print("Bootstrap resamples:", N_BOOT)

print("\nPAIRED BOOTSTRAP RESULTS")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nOUTPUT STATUS")
print("CSV saved:", CSV_OUT.is_file())
print("JSON saved:", JSON_OUT.is_file())
print("Model inference performed: False")
print("Test images accessed: 0")
print("STEP 07.34 PASS")


# %% [cell 241] Experiment 07 - Step 07.35
# ============================================================
# Experiment 07 - Step 07.35
# Model-Level MA Spatial Enrichment Bootstrap
# Validation-only; existing spatial results
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

INPUT = (
    ROOT / "experiment07_results"
    / "ViTL_MA_Spatial_Null_Paired_V1.csv"
)

assert INPUT.is_file()

data = pd.read_csv(
    INPUT,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(data) == 137
assert data["risk_group_id"].notna().all()
assert data["risk_group_id"].nunique() == 120

columns = {
    "RETFound": "retfound_null_adjusted_enrichment",
    "Meta MAE": "meta_mae_null_adjusted_enrichment"
}

grouped = data.groupby("risk_group_id", sort=True)
sizes = grouped.size().to_numpy(dtype=float)

sums = grouped[list(columns.values())].sum().to_numpy(
    dtype=float
)

N_BOOT = 2000
SEED = 20261020

rng = np.random.default_rng(SEED)
draws = np.empty((N_BOOT, 2), dtype=float)

for i in range(N_BOOT):
    selected = rng.integers(0, len(sizes), size=len(sizes))

    draws[i] = (
        sums[selected].sum(axis=0)
        / sizes[selected].sum()
    )

print("EXPERIMENT 07.35 - MODEL-LEVEL SPATIAL AUDIT")
print("Eligible images:", len(data))
print("Source-risk groups:", len(sizes))
print("Bootstrap resamples:", N_BOOT)

print("\nMODEL-LEVEL NULL-ADJUSTED ENRICHMENT")

for j, (name, column) in enumerate(columns.items()):
    mean = float(data[column].mean())
    low, high = np.quantile(draws[:, j], [0.025, 0.975])

    print(f"\n{name}")
    print("Mean:", round(mean, 6))
    print("95% CI:", (round(float(low), 6),
                      round(float(high), 6)))
    print("CI excludes zero:", bool(low > 0 or high < 0))
    print("CI entirely above zero:", bool(low > 0))

print("\nPAIRED CONSISTENCY")
print(
    "Meta minus RETFound:",
    round(float(
        (data[columns["Meta MAE"]]
         - data[columns["RETFound"]]).mean()
    ), 6)
)

print("\nResearch files modified: False")
print("Model inference performed: False")
print("Test images accessed: 0")
print("STEP 07.35 PASS")


# %% [cell 242] Experiment 07 - Step 07.36
# ============================================================
# Experiment 07 - Step 07.36
# Classification-Conditioned MA Attribution Audit
# CSV-only; exploratory subgroup analysis
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

SPATIAL = ROOT / (
    "experiment07_results/"
    "ViTL_MA_Spatial_Null_Paired_V1.csv"
)

RET = ROOT / (
    "experiment05_results/gradcam/"
    "RETFound_MA_Attribution_V1.csv"
)

META = ROOT / (
    "experiment07_results/gradcam/"
    "Meta_MAE_MA_Attribution_V1.csv"
)

for path in [SPATIAL, RET, META]:
    assert path.is_file()

spatial = pd.read_csv(
    SPATIAL,
    dtype={"image_id": str, "risk_group_id": str}
)

ret = pd.read_csv(RET, dtype={"image_id": str})
meta = pd.read_csv(META, dtype={"image_id": str})

data = (
    spatial
    .merge(
        ret[["image_id", "true_grade", "predicted_grade"]]
        .rename(columns={
            "true_grade": "ret_true",
            "predicted_grade": "ret_pred"
        }),
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        meta[["image_id", "true_grade", "predicted_grade"]]
        .rename(columns={
            "true_grade": "meta_true",
            "predicted_grade": "meta_pred"
        }),
        on="image_id",
        validate="one_to_one"
    )
)

assert len(data) == 137
assert data["image_id"].is_unique
assert data["risk_group_id"].nunique() == 120
assert (data["ret_true"] == data["meta_true"]).all()
assert (data["ret_true"] == data["true_grade"]).all()

data["ret_correct"] = data["ret_pred"] == data["true_grade"]
data["meta_correct"] = data["meta_pred"] == data["true_grade"]

subsets = {
    "All eligible images": data,
    "Both models correct": data[
        data["ret_correct"] & data["meta_correct"]
    ],
    "Both models incorrect": data[
        ~data["ret_correct"] & ~data["meta_correct"]
    ],
    "RETFound only correct": data[
        data["ret_correct"] & ~data["meta_correct"]
    ],
    "Meta MAE only correct": data[
        ~data["ret_correct"] & data["meta_correct"]
    ]
}

print("EXPERIMENT 07.36 - CLASSIFICATION-CONDITIONED AUDIT")

print("\nSUBGROUP ATTRIBUTION")

for name, subset in subsets.items():
    print(f"\n{name}")
    print("Images:", len(subset))
    print("Source-risk groups:", subset["risk_group_id"].nunique())

    if len(subset) == 0:
        continue

    print(
        "RETFound adjusted mean:",
        round(float(
            subset["retfound_null_adjusted_enrichment"].mean()
        ), 6)
    )

    print(
        "Meta MAE adjusted mean:",
        round(float(
            subset["meta_mae_null_adjusted_enrichment"].mean()
        ), 6)
    )

    print(
        "Meta minus RETFound:",
        round(float(
            subset["delta_null_adjusted"].mean()
        ), 6)
    )

print("\nBY TRUE DR GRADE")

summary = data.groupby("true_grade").agg(
    images=("image_id", "size"),
    risk_groups=("risk_group_id", "nunique"),
    ret_adjusted=(
        "retfound_null_adjusted_enrichment", "mean"
    ),
    meta_adjusted=(
        "meta_mae_null_adjusted_enrichment", "mean"
    ),
    meta_minus_ret=("delta_null_adjusted", "mean")
)

print(summary.to_string(float_format=lambda x: f"{x:.6f}"))

print("\nResearch files modified: False")
print("Model inference performed: False")
print("Test images accessed: 0")
print("STEP 07.36 PASS")


# %% [cell 243] Experiment 07 - Step 07.37
# ============================================================
# Experiment 07 - Step 07.37
# Both-Correct MA Attribution: Exploratory Cluster Bootstrap
# CSV-only; no inference, no file modifications
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

SPATIAL = ROOT / (
    "experiment07_results/ViTL_MA_Spatial_Null_Paired_V1.csv"
)
RET = ROOT / (
    "experiment05_results/gradcam/RETFound_MA_Attribution_V1.csv"
)
META = ROOT / (
    "experiment07_results/gradcam/Meta_MAE_MA_Attribution_V1.csv"
)

spatial = pd.read_csv(
    SPATIAL, dtype={"image_id": str, "risk_group_id": str}
)
ret = pd.read_csv(RET, dtype={"image_id": str})
meta = pd.read_csv(META, dtype={"image_id": str})

data = (
    spatial
    .merge(
        ret[["image_id", "correct"]].rename(
            columns={"correct": "ret_correct"}
        ),
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        meta[["image_id", "correct"]].rename(
            columns={"correct": "meta_correct"}
        ),
        on="image_id",
        validate="one_to_one"
    )
)

assert len(data) == 137
assert data["risk_group_id"].nunique() == 120
assert data["image_id"].is_unique

for column in ["ret_correct", "meta_correct"]:
    assert data[column].notna().all()
    if data[column].dtype != bool:
        data[column] = data[column].astype(str).str.lower().map(
            {"true": True, "false": False}
        )
    assert data[column].notna().all()

both = data.loc[
    data["ret_correct"] & data["meta_correct"]
].copy()

assert len(both) == 78
assert both["risk_group_id"].nunique() == 72

metric_columns = [
    "delta_null_adjusted",
    "delta_spatial_percentile"
]

groups = both.groupby("risk_group_id", sort=True)

sizes = groups.size().to_numpy(dtype=float)
sums = groups[metric_columns].sum().to_numpy(dtype=float)

N_BOOT = 2000
SEED = 20261020

rng = np.random.default_rng(SEED)
draws = np.empty((N_BOOT, len(metric_columns)))

for i in range(N_BOOT):
    selected = rng.integers(
        0, len(sizes), size=len(sizes)
    )
    draws[i] = (
        sums[selected].sum(axis=0)
        / sizes[selected].sum()
    )

print("EXPERIMENT 07.37 - BOTH-CORRECT BOOTSTRAP")
print("Images:", len(both))
print("Source-risk groups:", len(sizes))
print("Bootstrap resamples:", N_BOOT)

print("\nEXPLORATORY PAIRED RESULTS")

for j, column in enumerate(metric_columns):
    point = float(both[column].mean())
    low, high = np.quantile(draws[:, j], [0.025, 0.975])

    print(f"\n{column}")
    print("Meta minus RETFound:", round(point, 6))
    print(
        "95% CI:",
        (round(float(low), 6), round(float(high), 6))
    )
    print("CI includes zero:", bool(low <= 0 <= high))

print("\nINTERPRETATION LIMITS")
print("Post-hoc correctness-conditioned subset: True")
print("Confirmatory analysis: False")
print("Multiple-comparison adjustment: False")
print("Research files modified: False")
print("Model inference performed: False")
print("Test images accessed: 0")
print("\nSTEP 07.37 PASS")


# %% [cell 244] Experiment 07 - Step 07.38
# ============================================================
# Experiment 07 - Step 07.38
# Common 7x7 CAM Resolution Sensitivity
# Reuse Step 07.33 functions and frozen spatial geometry
# ============================================================

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = DR_RESEARCH

REFERENCE_PATH = (
    ROOT / "experiment07_results"
    / "ViTL_MA_Spatial_Null_Paired_V1.csv"
)

required_functions = [
    "load_mask",
    "retinal_foreground",
    "shift_geometry",
    "spatial_metrics"
]

for name in required_functions:
    assert name in globals() and callable(globals()[name]), (
        f"Missing function: {name}. Restore Step 07.33 functions."
    )

assert "primary" in globals()
assert "ret_ids" in globals()
assert "ret_cams" in globals()
assert "meta_cams" in globals()

reference = pd.read_csv(
    REFERENCE_PATH,
    dtype={"image_id": str, "risk_group_id": str}
).set_index("image_id")

assert len(reference) == 137

def matched_resolution_cam(cam):
    coarse = cv2.resize(
        cam.astype(np.float32),
        (7, 7),
        interpolation=cv2.INTER_AREA
    )

    restored = cv2.resize(
        coarse,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float64)

    assert np.isfinite(restored).all()
    assert restored.mean() > 1e-12

    return restored

records = []

print("EXPERIMENT 07.38 - COMMON RESOLUTION SENSITIVITY")

for i, image_id in enumerate(ret_ids):
    if image_id not in reference.index:
        continue

    row = primary.loc[image_id]

    mask = load_mask(row)
    foreground = retinal_foreground(row)
    geometry = shift_geometry(mask, foreground)

    assert geometry["eligible"]
    assert geometry["null_shifts"] == int(
        reference.loc[image_id, "null_shifts"]
    )

    ret_result = spatial_metrics(
        matched_resolution_cam(ret_cams[i]),
        mask,
        geometry
    )

    meta_result = spatial_metrics(
        matched_resolution_cam(meta_cams[i]),
        mask,
        geometry
    )

    records.append({
        "image_id": image_id,
        "risk_group_id": str(row["risk_group_id"]),
        "ret_adjusted": ret_result["null_adjusted_enrichment"],
        "meta_adjusted": meta_result["null_adjusted_enrichment"],
        "delta_adjusted": (
            meta_result["null_adjusted_enrichment"]
            - ret_result["null_adjusted_enrichment"]
        ),
        "delta_percentile": (
            meta_result["spatial_percentile"]
            - ret_result["spatial_percentile"]
        )
    })

data = pd.DataFrame(records)

assert len(data) == 137
assert data["risk_group_id"].nunique() == 120

metrics = ["delta_adjusted", "delta_percentile"]

grouped = data.groupby("risk_group_id", sort=True)
sizes = grouped.size().to_numpy(dtype=float)
sums = grouped[metrics].sum().to_numpy(dtype=float)

rng = np.random.default_rng(20261020)
draws = np.empty((2000, 2))

for b in range(2000):
    selected = rng.integers(0, len(sizes), size=len(sizes))
    draws[b] = (
        sums[selected].sum(axis=0)
        / sizes[selected].sum()
    )

print("\nCOHORT")
print("Eligible images:", len(data))
print("Source-risk groups:", data["risk_group_id"].nunique())
print("Common CAM grid: 7x7")

print("\nMODEL-LEVEL RESULTS")
print("RETFound mean adjusted:", round(data["ret_adjusted"].mean(), 6))
print("Meta MAE mean adjusted:", round(data["meta_adjusted"].mean(), 6))

print("\nPAIRED RESOLUTION SENSITIVITY")

for j, metric in enumerate(metrics):
    mean = float(data[metric].mean())
    low, high = np.quantile(draws[:, j], [0.025, 0.975])

    print(f"\n{metric}")
    print("Mean Meta minus RETFound:", round(mean, 6))
    print("95% CI:", (round(float(low), 6), round(float(high), 6)))
    print("CI includes zero:", bool(low <= 0 <= high))

print("\nRESEARCH INTEGRITY")
print("Original CAMs recomputed: False")
print("Research files modified: False")
print("Test images accessed: 0")
print("STEP 07.38 PASS")


# %% [cell 245] Experiment 07 - Step 07.39
# ============================================================
# Experiment 07 - Step 07.39
# Persist Common-Resolution Sensitivity Results
# No recomputation or model inference
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
OUT = ROOT / "experiment07_results"

CSV_PATH = OUT / "ViTL_MA_Common_Resolution_V1.csv"
REPORT_PATH = OUT / "ViTL_MA_Common_Resolution_Report_V1.json"

assert "data" in globals()
assert "draws" in globals()

assert len(data) == 137
assert data["risk_group_id"].nunique() == 120
assert draws.shape == (2000, 2)

assert not CSV_PATH.exists()
assert not REPORT_PATH.exists()

metrics = ["delta_adjusted", "delta_percentile"]

summary = {}

for j, metric in enumerate(metrics):
    low, high = np.quantile(
        draws[:, j], [0.025, 0.975]
    )

    summary[metric] = {
        "mean": float(data[metric].mean()),
        "ci95_lower": float(low),
        "ci95_upper": float(high),
        "ci_contains_zero": bool(low <= 0 <= high)
    }

assert np.isclose(
    summary["delta_adjusted"]["mean"],
    -0.021061,
    atol=1e-5
)

assert np.isclose(
    summary["delta_percentile"]["mean"],
    -5.03757,
    atol=1e-5
)

report = {
    "experiment": "Experiment07_Step07.38",
    "analysis": "common_7x7_cam_resolution_sensitivity",
    "comparison": "Meta MAE minus RETFound",
    "primary_cam_grid": [14, 14],
    "common_grid": [7, 7],
    "method": "saved_CAM_postprocessing",
    "downsampling": "cv2.INTER_AREA",
    "upsampling": "cv2.INTER_LINEAR",
    "eligible_images": 137,
    "source_risk_groups": 120,
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 20261020,
    "results": summary,
    "test_split_evaluated": False
}

with CSV_PATH.open("x", encoding="utf-8") as f:
    data.to_csv(f, index=False)

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 07.39 - RESULT PERSISTENCE")
print("CSV saved:", CSV_PATH.is_file())
print("Report saved:", REPORT_PATH.is_file())
print("Eligible images:", len(data))
print("Source-risk groups:", data["risk_group_id"].nunique())
print("Bootstrap draws reused:", len(draws))
print("Model inference performed: False")
print("Test images accessed: 0")
print("STEP 07.39 PASS")


# %% [cell 246] Experiment 07 - Step 07.40
# ============================================================
# Experiment 07 - Step 07.40
# Final Results Integrity Audit
# Read-only; no training or inference
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
OUT = ROOT / "experiment07_results"

files = {
    "classification_bootstrap":
        OUT / "ViTL_Paired_Bootstrap_V1.csv",

    "classification_report":
        OUT / "ViTL_Paired_Bootstrap_Report_V1.json",

    "ma_bootstrap":
        OUT / "ViTL_MA_Bootstrap_V1.csv",

    "ma_bootstrap_report":
        OUT / "ViTL_MA_Bootstrap_Report_V1.json",

    "spatial_null":
        OUT / "ViTL_MA_Spatial_Null_Paired_V1.csv",

    "resolution_sensitivity":
        OUT / "ViTL_MA_Common_Resolution_V1.csv",

    "resolution_report":
        OUT / "ViTL_MA_Common_Resolution_Report_V1.json",
}

print("EXPERIMENT 07.40 - FINAL RESULTS INTEGRITY")

for name, path in files.items():
    assert path.is_file(), f"Missing: {name}"
    print(f"{name}: PASS")

classification = pd.read_csv(
    files["classification_bootstrap"]
)
ma = pd.read_csv(files["ma_bootstrap"])
spatial = pd.read_csv(files["spatial_null"])
resolution = pd.read_csv(files["resolution_sensitivity"])

classification_report = json.loads(
    files["classification_report"].read_text(encoding="utf-8")
)
ma_report = json.loads(
    files["ma_bootstrap_report"].read_text(encoding="utf-8")
)
resolution_report = json.loads(
    files["resolution_report"].read_text(encoding="utf-8")
)

assert len(classification) == 5
assert len(ma) == 3
assert len(spatial) == 137
assert len(resolution) == 137

assert classification_report["validation_images"] == 276
assert classification_report["source_risk_groups"] == 239

assert ma_report["images"] == 137
assert ma_report["source_risk_groups"] == 120

assert resolution_report["eligible_images"] == 137
assert resolution_report["source_risk_groups"] == 120

def check_metric(frame, metric, value_column, name, expected):
    row = frame.loc[frame["metric"] == metric]
    assert len(row) == 1, f"Missing or duplicated: {name}"
    value = float(row.iloc[0][value_column])
    assert np.isclose(value, expected, atol=1e-5), (
        f"Unexpected value for {name}: {value}"
    )
    print(f"{name}: {value:+.6f} PASS")

print("\nKEY METRICS")

check_metric(
    classification, "macro_f1",
    "meta_minus_retfound",
    "Classification Macro F1", 0.097232
)

check_metric(
    ma, "delta_raw",
    "mean_meta_minus_retfound",
    "Raw MA Enrichment", 0.489849
)

check_metric(
    ma, "delta_null_adjusted",
    "mean_meta_minus_retfound",
    "Adjusted MA Enrichment", -0.056152
)

check_metric(
    ma, "delta_spatial_percentile",
    "mean_meta_minus_retfound",
    "Spatial Percentile", -3.485040
)

assert np.isclose(
    resolution["delta_adjusted"].mean(),
    -0.021061,
    atol=1e-5
)

assert np.isclose(
    spatial["delta_null_adjusted"].mean(),
    -0.056152,
    atol=1e-5
)

print("\nDATASET STATUS")
print("Classification validation images:", 276)
print("MA eligible images:", len(spatial))
print("MA source-risk groups:", spatial["risk_group_id"].nunique())
print("Resolution analysis images:", len(resolution))

print("\nRESEARCH INTEGRITY")
print("Existing files modified: False")
print("Models retrained: False")
print("CAM recomputed: False")
print("Test images accessed: 0")

print("\nSTEP 07.40 PASS")
