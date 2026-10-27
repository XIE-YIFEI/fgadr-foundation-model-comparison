# Experiment 10 - parameter-randomisation pilot and first production batches
#
# Source: FGADR_MA_Mask_Audit.ipynb (master notebook, 392 cells), cells 332-389, run step by step in Google Colab (L4 GPU).
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

# %% [cell 332] Experiment 10 - Step 10.00
# ============================================================
# Experiment 10 - Step 10.00
# Mount Google Drive and Verify Research Environment
# ============================================================

from pathlib import Path
import torch

# Mount Google Drive
pass  # Google Drive mount removed

# Project root
ROOT = DR_RESEARCH

# Required experiment directories
required_dirs = {
    "Experiment 03": ROOT / "experiment03_results",
    "Experiment 05": ROOT / "experiment05_results",
    "Experiment 07": ROOT / "experiment07_results",
    "Experiment 08": ROOT / "experiment08_results",
    "Experiment 09": ROOT / "experiment09_results",
}

print("\nEXPERIMENT 10.00 - ENVIRONMENT CHECK")

print("\nGOOGLE DRIVE")
print("Project root exists:", ROOT.is_dir())

assert ROOT.is_dir(), "Google Drive project root not found."

print("\nEXPERIMENT DIRECTORIES")

for name, path in required_dirs.items():
    exists = path.is_dir()
    print(f"{name}: {'PASS' if exists else 'MISSING'}")
    assert exists, f"Missing directory: {path}"

print("\nGPU ENVIRONMENT")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory (GiB):",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        )
    )

print("\nRESEARCH INTEGRITY")
print("Research files modified: False")
print("Model training started: False")
print("Test images accessed: 0")

print("\nSTEP 10.00 PASS")


# %% [cell 333] Experiment 10 - Step 10.01
# ============================================================
# Experiment 10 - Step 10.01
# Grad-CAM Implementation and Target Audit
# Read-only; no model execution
# ============================================================

from pathlib import Path
import json

ROOT = DR_RESEARCH

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

REPORTS = {
    "RETFound": (
        ROOT / "experiment05_results/gradcam"
        / "RETFound_MA_Attribution_Report_V1.json"
    ),
    "Meta MAE": (
        ROOT / "experiment07_results/gradcam"
        / "Meta_MAE_MA_Attribution_Report_V1.json"
    )
}

assert NOTEBOOK.is_file(), "Research notebook not found."

with NOTEBOOK.open("r", encoding="utf-8") as f:
    notebook = json.load(f)

terms = [
    "register_forward_hook",
    "register_full_backward_hook",
    "register_backward_hook",
    "retain_grad",
    "gradcam",
    "grad_cam",
    "target_layer",
    "target_class",
    "predicted_class",
    "activations",
    "gradients"
]

print("EXPERIMENT 10.01 - GRADCAM IMPLEMENTATION AUDIT")

print("\nNOTEBOOK")
print("Notebook:", NOTEBOOK.name)
print("Total cells:", len(notebook["cells"]))

matches = []

for cell_number, cell in enumerate(notebook["cells"], start=1):
    if cell.get("cell_type") != "code":
        continue

    source = cell.get("source", [])
    code = "".join(source) if isinstance(source, list) else str(source)

    lines = code.splitlines()

    found = [
        (i + 1, line.strip())
        for i, line in enumerate(lines)
        if any(term in line.lower() for term in terms)
    ]

    if found:
        matches.append((cell_number, found))

print("\nMATCHING CELLS:", len(matches))

for cell_number, found in matches:
    print(f"\nCell {cell_number}")
    print("Matching lines:", len(found))

    for line_number, text in found[:8]:
        print(f"  L{line_number}: {text[:160]}")

print("\nATTRIBUTION REPORTS")

for name, path in REPORTS.items():
    assert path.is_file(), f"Missing report: {path}"

    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    print(f"\n{name}")
    print("Top-level keys:", list(report.keys()))

    for key, value in report.items():
        if any(
            token in key.lower()
            for token in [
                "target",
                "layer",
                "hook",
                "class",
                "gradcam",
                "method",
                "checkpoint",
                "normaliz"
            ]
        ):
            if isinstance(value, (str, int, float, bool)):
                print(f"{key}: {value}")
            elif isinstance(value, dict):
                print(f"{key}: {list(value.keys())}")

print("\nRESEARCH INTEGRITY")
print("Notebook code executed: False")
print("Model parameters modified: False")
print("Files modified: False")
print("Images accessed: 0")
print("Model inference performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.01 COMPLETE")


# %% [cell 334] Experiment 10 - Step 10.02
# ============================================================
# Experiment 10 - Step 10.02
# Audit Original Grad-CAM Computation
# Read-only notebook inspection
# ============================================================

from pathlib import Path
import json

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert NOTEBOOK.is_file()

with NOTEBOOK.open("r", encoding="utf-8") as handle:
    notebook = json.load(handle)

TARGET_CELLS = {
    121: "RETFound Formal Grad-CAM",
    238: "Meta MAE Formal Grad-CAM"
}

KEYWORDS = [
    "register_forward_hook",
    "retain_grad",
    "autograd.grad",
    "gradients",
    "gradient",
    "activation",
    "target_class",
    "target_layer",
    "blocks[-1]",
    "reshape",
    "interpolate",
    "relu",
    "cam =",
    "cam /",
    "cam.max",
    "cam.min",
    "normaliz"
]

print("EXPERIMENT 10.02 - GRADCAM COMPUTATION AUDIT")

for cell_number, label in TARGET_CELLS.items():

    cell = notebook["cells"][cell_number - 1]

    assert cell["cell_type"] == "code"

    source = cell["source"]
    code = "".join(source) if isinstance(source, list) else str(source)

    lines = code.splitlines()

    print("\n" + "=" * 72)
    print(label)
    print("Cell number:", cell_number)
    print("Total code lines:", len(lines))

    matched_lines = [
        i for i, line in enumerate(lines)
        if any(
            keyword.lower() in line.lower()
            for keyword in KEYWORDS
        )
    ]

    intervals = []

    for i in matched_lines:
        intervals.append((
            max(0, i - 5),
            min(len(lines), i + 9)
        ))

    merged = []

    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (
                merged[-1][0],
                max(merged[-1][1], end)
            )
        else:
            merged.append((start, end))

    print("\nRELEVANT CODE")

    for start, end in merged:

        print(f"\n--- Lines {start + 1}-{end} ---")

        for i in range(start, end):
            print(f"{i + 1:04d}: {lines[i]}")

print("\nRESEARCH INTEGRITY")
print("Notebook code executed: False")
print("Models loaded: False")
print("Parameters modified: False")
print("Images accessed: 0")
print("Test images accessed: 0")

print("\nSTEP 10.02 COMPLETE")


# %% [cell 335] Experiment 10 - Step 10.03
# ============================================================
# Experiment 10 - Step 10.03
# Original Model and CAM Artifact Integrity
# Read-only; no model loading or inference
# ============================================================

from pathlib import Path
import json
import numpy as np

ROOT = DR_RESEARCH

ARTIFACTS = {
    "RETFound checkpoint": (
        ROOT / "experiment04_results"
        / "retfound_vit_large_v1"
        / "best_model.pt"
    ),
    "Meta MAE checkpoint": (
        ROOT / "experiment07_results"
        / "meta_mae_vit_large_v1"
        / "best_model.pt"
    ),
    "RETFound CAM": (
        ROOT / "experiment05_results"
        / "gradcam"
        / "RETFound_MA_CAM_Maps_V1.npz"
    ),
    "Meta MAE CAM": (
        ROOT / "experiment07_results"
        / "gradcam"
        / "Meta_MAE_MA_CAM_Maps_V1.npz"
    ),
    "Validation index": (
        ROOT / "experiment03_results"
        / "FGADR_Training_Index_V1.csv"
    )
}

print("EXPERIMENT 10.03 - ARTIFACT INTEGRITY")

print("\nREQUIRED ARTIFACTS")

for name, path in ARTIFACTS.items():
    exists = path.is_file()
    print(f"{name}: {'PASS' if exists else 'MISSING'}")

    assert exists, f"Missing artifact: {path}"

    print(
        "  Size (MiB):",
        round(path.stat().st_size / (1024 ** 2), 2)
    )

# ------------------------------------------------------------
# Inspect saved CAM arrays without model execution
# ------------------------------------------------------------

cam_specs = [
    ("RETFound", ARTIFACTS["RETFound CAM"], "retfound_cam"),
    ("Meta MAE", ARTIFACTS["Meta MAE CAM"], "meta_mae_cam")
]

image_id_sets = []

print("\nGRADCAM ARRAY INTEGRITY")

for model_name, path, cam_key in cam_specs:

    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"image_ids", cam_key}

        image_ids = [
            str(value) for value in data["image_ids"]
        ]

        cams = data[cam_key]

        assert cams.shape == (156, 224, 224)
        assert len(image_ids) == 156
        assert len(set(image_ids)) == 156

        assert np.isfinite(cams).all()

        maxima = cams.max(axis=(1, 2))
        minima = cams.min(axis=(1, 2))
        standard_deviations = cams.std(axis=(1, 2))

        assert minima.min() >= -1e-6
        assert maxima.max() <= 1.000001

        print(f"\n{model_name}")
        print("CAM shape:", cams.shape)
        print("Unique image IDs:", len(set(image_ids)))
        print("CAM dtype:", cams.dtype)
        print("Non-finite values:", int((~np.isfinite(cams)).sum()))
        print("Degenerate CAMs:", int((maxima <= 1e-12).sum()))
        print(
            "Median CAM standard deviation:",
            round(float(np.median(standard_deviations)), 6)
        )

        image_id_sets.append(set(image_ids))

assert image_id_sets[0] == image_id_sets[1]
assert len(image_id_sets[0]) == 156

print("\nCROSS-MODEL MATCHING")
print("Matching image populations: PASS")
print("Matched image count: 156")

print("\nRESEARCH INTEGRITY")
print("Checkpoint weights modified: False")
print("Models loaded: False")
print("Model inference performed: False")
print("Images accessed: 0")
print("Test images accessed: 0")

print("\nSTEP 10.03 PASS")


# %% [cell 336] Experiment 10 - Step 10.04
# ============================================================
# Experiment 10 - Step 10.04
# Checkpoint Structure and Parameter Audit
# Read-only; CPU loading only
# ============================================================

from pathlib import Path
import gc
import hashlib
import torch

ROOT = DR_RESEARCH

CHECKPOINTS = {
    "RETFound": (
        ROOT / "experiment04_results"
        / "retfound_vit_large_v1"
        / "best_model.pt"
    ),
    "Meta MAE": (
        ROOT / "experiment07_results"
        / "meta_mae_vit_large_v1"
        / "best_model.pt"
    )
}

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()

print("EXPERIMENT 10.04 - CHECKPOINT STRUCTURE AUDIT")

print("\nPYTORCH VERSION")
print(torch.__version__)

for model_name, path in CHECKPOINTS.items():

    assert path.is_file()

    print("\n" + "=" * 65)
    print("MODEL:", model_name)
    print("Checkpoint:", path.name)

    print("\nCHECKPOINT HASH")
    print("SHA-256:", sha256_file(path))

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    assert isinstance(checkpoint, dict)

    print("\nTOP-LEVEL KEYS")
    print(list(checkpoint.keys())[:30])

    # Detect the state dictionary without modifying it.
    candidate_keys = [
        "model_state_dict",
        "state_dict",
        "model",
        "model_state"
    ]

    state = None
    state_source = None

    for key in candidate_keys:
        value = checkpoint.get(key)

        if (
            isinstance(value, dict)
            and len(value) > 0
            and all(
                isinstance(v, torch.Tensor)
                for v in value.values()
            )
        ):
            state = value
            state_source = key
            break

    if state is None:
        if all(
            isinstance(v, torch.Tensor)
            for v in checkpoint.values()
        ):
            state = checkpoint
            state_source = "checkpoint_root"

    assert state is not None, (
        "Could not identify tensor state dictionary."
    )

    print("\nSTATE DICTIONARY")
    print("State source:", state_source)
    print("Parameter/buffer entries:", len(state))

    print("\nFIRST 15 TENSOR KEYS")

    for key in list(state.keys())[:15]:
        print(key, tuple(state[key].shape))

    print("\nLAST 15 TENSOR KEYS")

    for key in list(state.keys())[-15:]:
        print(key, tuple(state[key].shape))

    print("\nMODEL COMPONENTS")

    components = {
        "patch_embed": [],
        "blocks.0": [],
        "blocks.11": [],
        "blocks.17": [],
        "blocks.23": [],
        "norm": [],
        "fc_norm": [],
        "head": []
    }

    for key in state:
        for component in components:
            if key.startswith(component + "."):
                components[component].append(key)

    for component, keys in components.items():
        print(
            f"{component}: {len(keys)} tensor entries"
        )

    print("\nCLASSIFICATION HEAD")

    for key in state:
        if key.startswith("head."):
            print(key, tuple(state[key].shape))

    del state
    del checkpoint
    gc.collect()

print("\nRESEARCH INTEGRITY")
print("Checkpoint files modified: False")
print("Models initialized: False")
print("Model inference performed: False")
print("Parameters randomized: False")
print("Test images accessed: 0")

print("\nSTEP 10.04 COMPLETE")


# %% [cell 337] Experiment 10 - Step 10.04A
# ============================================================
# Experiment 10 - Step 10.04A
# Complete Meta MAE Checkpoint Structure Audit
# Read-only; CPU loading
# ============================================================

from pathlib import Path
import torch
import gc

ROOT = DR_RESEARCH

CHECKPOINT_PATH = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1"
    / "best_model.pt"
)

assert CHECKPOINT_PATH.is_file()

print("EXPERIMENT 10.04A - META MAE CHECKPOINT AUDIT")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

print("\nTOP-LEVEL KEYS")
print(list(checkpoint.keys()))

assert "model_state_dict" in checkpoint

state = checkpoint["model_state_dict"]

print("\nSTATE DICTIONARY")
print("Tensor entries:", len(state))

print("\nMODEL COMPONENTS")

components = [
    "patch_embed",
    "blocks.0",
    "blocks.11",
    "blocks.17",
    "blocks.23",
    "norm",
    "fc_norm",
    "head"
]

for component in components:
    count = sum(
        key.startswith(component + ".")
        for key in state.keys()
    )
    print(f"{component}: {count} tensor entries")

print("\nMODEL STRUCTURE")

for key in [
    "cls_token",
    "pos_embed",
    "patch_embed.proj.weight",
    "blocks.0.attn.qkv.weight",
    "blocks.23.attn.qkv.weight",
    "fc_norm.weight",
    "head.weight",
    "head.bias"
]:
    assert key in state, f"Missing parameter: {key}"
    print(key, tuple(state[key].shape))

print("\nCHECKPOINT METADATA")

for key in ["epoch", "val_macro_f1", "config"]:
    if key in checkpoint:
        value = checkpoint[key]
        if isinstance(value, (int, float, str)):
            print(f"{key}: {value}")
        elif isinstance(value, dict):
            print(f"{key} keys: {list(value.keys())}")

assert len(state) == 296
assert tuple(state["head.weight"].shape) == (5, 1024)
assert tuple(state["pos_embed"].shape) == (1, 197, 1024)

print("\nSTRUCTURAL CHECK: PASS")

del state
del checkpoint
gc.collect()

print("\nRESEARCH INTEGRITY")
print("Checkpoint modified: False")
print("Models initialized: False")
print("Model inference performed: False")
print("Parameters randomized: False")
print("Test images accessed: 0")

print("\nSTEP 10.04A COMPLETE")


# %% [cell 338] Experiment 10 - Step 10.05
# ============================================================
# Experiment 10 - Step 10.05
# Cross-Model Parameter Schema and Configuration Audit
# Read-only; CPU; no model inference
# ============================================================

from pathlib import Path
import torch
import gc

ROOT = DR_RESEARCH

PATHS = {
    "RETFound": (
        ROOT / "experiment04_results"
        / "retfound_vit_large_v1"
        / "best_model.pt"
    ),
    "Meta MAE": (
        ROOT / "experiment07_results"
        / "meta_mae_vit_large_v1"
        / "best_model.pt"
    )
}

schemas = {}
configs = {}

print("EXPERIMENT 10.05 - CROSS-MODEL SCHEMA AUDIT")

for name, path in PATHS.items():

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    state = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})

    assert isinstance(state, dict)
    assert isinstance(config, dict)

    schemas[name] = {
        key: (tuple(value.shape), str(value.dtype))
        for key, value in state.items()
    }

    configs[name] = {
        key: config.get(key)
        for key in [
            "architecture",
            "num_classes",
            "input_size",
            "global_pool",
            "precision",
            "training_index_sha256",
            "test_images_evaluated"
        ]
    }

    print(f"\n{name}")
    print("Tensor entries:", len(schemas[name]))

    for key, value in configs[name].items():
        print(f"{key}: {value}")

    del checkpoint
    del state
    gc.collect()

ret_schema = schemas["RETFound"]
meta_schema = schemas["Meta MAE"]

ret_keys = set(ret_schema)
meta_keys = set(meta_schema)

print("\nPARAMETER SCHEMA COMPARISON")
print("Identical tensor keys:", ret_keys == meta_keys)

missing_in_meta = sorted(ret_keys - meta_keys)
missing_in_ret = sorted(meta_keys - ret_keys)

print("Only in RETFound:", missing_in_ret)
print("Only in Meta MAE:", missing_in_meta)

shared = sorted(ret_keys & meta_keys)

shape_differences = [
    key for key in shared
    if ret_schema[key] != meta_schema[key]
]

print("Shape or dtype differences:", len(shape_differences))

for key in shape_differences[:20]:
    print(
        key,
        "RETFound:", ret_schema[key],
        "Meta MAE:", meta_schema[key]
    )

assert ret_keys == meta_keys
assert not shape_differences

print("\nCONFIGURATION COMPARISON")

for key in [
    "architecture",
    "num_classes",
    "input_size",
    "global_pool",
    "training_index_sha256"
]:
    a = configs["RETFound"][key]
    b = configs["Meta MAE"][key]

    assert a == b, (
        f"Configuration mismatch: {key}: {a} vs {b}"
    )

    print(f"{key}: MATCH")

assert configs["RETFound"]["global_pool"] == "avg"
assert configs["Meta MAE"]["global_pool"] == "avg"

print("\nMODEL SCHEMA: PASS")
print("CONFIGURATION MATCH: PASS")

print("\nRESEARCH INTEGRITY")
print("Checkpoint files modified: False")
print("Models initialized: False")
print("Model inference performed: False")
print("Parameters randomized: False")
print("Images accessed: 0")
print("Test images accessed: 0")

print("\nSTEP 10.05 COMPLETE")


# %% [cell 339] Experiment 10 - Step 10.06
# ============================================================
# Experiment 10 - Step 10.06
# Freeze Parameter Randomization Protocol
# No model inference or parameter modification
# ============================================================

from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

ROOT = DR_RESEARCH

OUT = ROOT / "experiment10_results"
PROTOCOL_PATH = OUT / "Experiment10_Randomization_Protocol_V1.json"

assert not PROTOCOL_PATH.exists(), (
    "Protocol already exists. Refusing to overwrite."
)

OUT.mkdir(parents=True, exist_ok=True)

stages = {
    "R0": [],
    "R1": ["head"],
    "R2": ["head", "fc_norm", "blocks.23"],
    "R3": (
        ["head", "fc_norm"]
        + [f"blocks.{i}" for i in range(18, 24)]
    ),
    "R4": (
        ["head", "fc_norm"]
        + [f"blocks.{i}" for i in range(12, 24)]
    ),
    "R5": (
        ["head", "fc_norm"]
        + [f"blocks.{i}" for i in range(24)]
    ),
    "R6": (
        ["head", "fc_norm"]
        + [f"blocks.{i}" for i in range(24)]
        + ["patch_embed", "cls_token", "pos_embed"]
    )
}

# Confirm cumulative component membership.
previous = set()

for stage, components in stages.items():
    current = set(components)

    assert previous.issubset(current)
    assert len(components) == len(current)

    previous = current

# These are protocol seeds, not execution results.
randomization_seeds = [1042, 2042, 3042]

protocol = {
    "experiment": "Experiment10_Attribution_Sanity_Checks",
    "protocol_version": "V1",
    "status": "frozen_before_randomization",
    "created_utc": datetime.now(timezone.utc).isoformat(),

    "models": {
        "RETFound": {
            "checkpoint": (
                "experiment04_results/"
                "retfound_vit_large_v1/best_model.pt"
            ),
            "checkpoint_sha256": (
                "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
            )
        },
        "Meta_MAE": {
            "checkpoint": (
                "experiment07_results/"
                "meta_mae_vit_large_v1/best_model.pt"
            ),
            "checkpoint_sha256": None
        }
    },

    "architecture": "vit_large_patch16_224",
    "num_classes": 5,
    "global_pool": "avg",

    "attribution": {
        "method": "GradCAM",
        "target_layer": "blocks.23",
        "target_class": "ground_truth_grade",
        "input_size": 224,
        "precision": "bfloat16",
        "cam_resolution": [224, 224],
        "original_cam_files": {
            "RETFound": (
                "experiment05_results/gradcam/"
                "RETFound_MA_CAM_Maps_V1.npz"
            ),
            "Meta_MAE": (
                "experiment07_results/gradcam/"
                "Meta_MAE_MA_CAM_Maps_V1.npz"
            )
        }
    },

    "randomization": {
        "design": "cascading_parameter_randomization",
        "stages": stages,
        "randomization_seeds": randomization_seeds,
        "repetitions_per_stage": len(randomization_seeds),
        "independent_repetitions": True,
        "cumulative_modules_within_each_repetition": True,
        "initialization": (
            "Reinitialize selected model components using "
            "their native PyTorch/timm initialization methods; "
            "verify implementation before pilot."
        )
    },

    "evaluation": {
        "pilot_before_full_run": True,
        "cohort": "frozen_validation_ma_candidates",
        "validation_image_count": 156,
        "primary_similarity_metrics": [
            "spearman_correlation",
            "ssim"
        ],
        "diagnostics": [
            "degenerate_cam_rate",
            "near_constant_cam_rate",
            "logit_change",
            "probability_change"
        ],
        "cluster_unit": "source_risk_group",
        "test_set_access": False
    },

    "limitations": [
        "Sanity checks measure parameter sensitivity, not clinical accuracy",
        "Image-level pixels are not independent observations",
        "Source-risk groups are not verified patient identities",
        "Randomized-model degenerate CAMs require explicit reporting",
        "R6 includes token and positional parameters that require "
        "special initialization handling",
        "Randomized-model outputs do not define new pretrained baselines"
    ]
}

with PROTOCOL_PATH.open("x", encoding="utf-8") as handle:
    json.dump(
        protocol,
        handle,
        indent=2,
        ensure_ascii=False
    )

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    verified = json.load(handle)

assert verified["randomization"]["stages"] == stages
assert verified["evaluation"]["test_set_access"] is False

digest = hashlib.sha256(
    PROTOCOL_PATH.read_bytes()
).hexdigest()

print("EXPERIMENT 10.06 - PROTOCOL FREEZE")

print("\nRANDOMIZATION STAGES")

for stage, components in stages.items():
    print(stage, "Components:", len(components))

print("\nREPRODUCIBILITY")
print("Randomization seeds:", randomization_seeds)
print("Repetitions per stage:", len(randomization_seeds))

print("\nPROTOCOL")
print("Saved:", PROTOCOL_PATH.is_file())
print("SHA-256:", digest)

print("\nRESEARCH INTEGRITY")
print("Original checkpoint files modified: False")
print("Original CAM files modified: False")
print("Model inference performed: False")
print("Parameters randomized: False")
print("Test images accessed: 0")

print("\nSTEP 10.06 PASS")


# %% [cell 340] Experiment 10 - Step 10.07
# ============================================================
# Experiment 10 - Step 10.07
# Checkpoint SHA-256 and Protocol Integrity Audit
# Read-only source files; create independent verification record
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = DR_RESEARCH

OUT = ROOT / "experiment10_results"

PROTOCOL_PATH = (
    OUT / "Experiment10_Randomization_Protocol_V1.json"
)

VERIFICATION_PATH = (
    OUT / "Experiment10_Checkpoint_Verification_V1.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
)

EXPECTED_RETFOUND_SHA256 = (
    "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
)

CHECKPOINTS = {
    "RETFound": (
        ROOT / "experiment04_results"
        / "retfound_vit_large_v1"
        / "best_model.pt"
    ),
    "Meta_MAE": (
        ROOT / "experiment07_results"
        / "meta_mae_vit_large_v1"
        / "best_model.pt"
    )
}

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()

print("EXPERIMENT 10.07 - CHECKPOINT VERIFICATION")

assert PROTOCOL_PATH.is_file()
assert not VERIFICATION_PATH.exists(), (
    "Verification file already exists. Refusing to overwrite."
)

protocol_sha256 = sha256_file(PROTOCOL_PATH)

assert protocol_sha256 == EXPECTED_PROTOCOL_SHA256, (
    "Frozen protocol hash mismatch."
)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert protocol["status"] == "frozen_before_randomization"
assert protocol["randomization"]["randomization_seeds"] == [
    1042, 2042, 3042
]

assert protocol["models"]["RETFound"]["checkpoint_sha256"] == (
    EXPECTED_RETFOUND_SHA256
)

assert protocol["models"]["Meta_MAE"]["checkpoint_sha256"] is None

print("\nPROTOCOL")
print("SHA-256:", protocol_sha256)
print("Integrity: PASS")

checkpoint_hashes = {}

print("\nCHECKPOINTS")

for model_name, path in CHECKPOINTS.items():
    assert path.is_file(), f"Missing checkpoint: {path}"

    digest = sha256_file(path)
    checkpoint_hashes[model_name] = digest

    print(f"\n{model_name}")
    print("Size (MiB):", round(path.stat().st_size / (1024 ** 2), 2))
    print("SHA-256:", digest)

assert checkpoint_hashes["RETFound"] == (
    EXPECTED_RETFOUND_SHA256
), "RETFound checkpoint integrity mismatch."

print("\nRETFOUND INTEGRITY: PASS")
print("META MAE HASH RECORDED: PASS")

verification = {
    "experiment": "Experiment10_Attribution_Sanity_Checks",
    "record": "Checkpoint_Verification_V1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "protocol_sha256": protocol_sha256,
    "checkpoint_sha256": checkpoint_hashes,
    "retfound_reference_hash_verified": True,
    "meta_mae_hash_calculated": True,
    "original_protocol_modified": False,
    "checkpoints_modified": False,
    "test_split_accessed": False
}

with VERIFICATION_PATH.open("x", encoding="utf-8") as handle:
    json.dump(
        verification,
        handle,
        indent=2,
        ensure_ascii=False
    )

with VERIFICATION_PATH.open("r", encoding="utf-8") as handle:
    saved = json.load(handle)

assert saved["checkpoint_sha256"] == checkpoint_hashes

print("\nVERIFICATION RECORD")
print("Saved:", VERIFICATION_PATH.is_file())
print("SHA-256:", sha256_file(VERIFICATION_PATH))

print("\nRESEARCH INTEGRITY")
print("Frozen protocol modified: False")
print("Checkpoint files modified: False")
print("Models initialized: False")
print("Parameters randomized: False")
print("Images accessed: 0")
print("Test images accessed: 0")

print("\nSTEP 10.07 PASS")


# %% [cell 341] Experiment 10 - Step 10.08
# ============================================================
# Experiment 10 - Step 10.08
# ViT Randomization Initialization Capability Audit
# No original model loading or modification
# ============================================================

import importlib.util
import inspect
import torch
import torch.nn as nn

print("EXPERIMENT 10.08 - INITIALIZATION CAPABILITY AUDIT")

print("\nENVIRONMENT")
print("PyTorch:", torch.__version__)

timm_spec = importlib.util.find_spec("timm")

assert timm_spec is not None, (
    "timm is not installed in this runtime."
)

import timm

print("timm:", timm.__version__)

# ------------------------------------------------------------
# 1. Inspect model architecture registration
# ------------------------------------------------------------

print("\nMODEL ARCHITECTURE")

available_models = timm.list_models(
    "vit_large_patch16_224"
)

print("Available matches:", available_models)

assert "vit_large_patch16_224" in available_models

# ------------------------------------------------------------
# 2. Inspect initialization methods
# ------------------------------------------------------------

modules = {
    "Linear": nn.Linear(8, 4),
    "LayerNorm": nn.LayerNorm(8),
    "Conv2d": nn.Conv2d(3, 8, kernel_size=3)
}

print("\nMODULE INITIALIZATION METHODS")

for name, module in modules.items():

    has_reset = callable(
        getattr(module, "reset_parameters", None)
    )

    print(f"\n{name}")
    print("Has reset_parameters:", has_reset)

    assert has_reset

    source = inspect.getsource(
        type(module).reset_parameters
    )

    print("Initialization source:")
    print(source[:1800])

# ------------------------------------------------------------
# 3. Confirm LayerNorm default initialization
# ------------------------------------------------------------

print("\nLAYERNORM DEFAULT INITIALIZATION")

layer_norm = nn.LayerNorm(8)

with torch.no_grad():
    layer_norm.weight.fill_(2.0)
    layer_norm.bias.fill_(3.0)

layer_norm.reset_parameters()

weight_is_one = torch.allclose(
    layer_norm.weight,
    torch.ones_like(layer_norm.weight)
)

bias_is_zero = torch.allclose(
    layer_norm.bias,
    torch.zeros_like(layer_norm.bias)
)

print("Weight reset to ones:", weight_is_one)
print("Bias reset to zeros:", bias_is_zero)

assert weight_is_one
assert bias_is_zero

print("LayerNorm reset is deterministic: PASS")

# ------------------------------------------------------------
# 4. Confirm direct parameter objects need special handling
# ------------------------------------------------------------

print("\nDIRECT PARAMETER INITIALIZATION")

direct_parameters = {
    "cls_token": nn.Parameter(torch.zeros(1, 1, 8)),
    "pos_embed": nn.Parameter(torch.zeros(1, 5, 8))
}

for name, parameter in direct_parameters.items():

    has_reset = callable(
        getattr(parameter, "reset_parameters", None)
    )

    print(
        name,
        "shape:", tuple(parameter.shape),
        "has reset_parameters:", has_reset
    )

    assert not has_reset

print("\nAUDIT CONCLUSIONS")
print("Linear initialization requires explicit policy.")
print("Conv2d initialization requires explicit policy.")
print("LayerNorm reset is not random parameter sampling.")
print("Direct parameters require explicit initialization.")
print("Frozen protocol remains unchanged.")

print("\nRESEARCH INTEGRITY")
print("Original checkpoints loaded: False")
print("Original model parameters modified: False")
print("Original CAM files modified: False")
print("Images accessed: 0")
print("Test images accessed: 0")

print("\nSTEP 10.08 PASS")


# %% [cell 342] Experiment 10 - Step 10.09
# ============================================================
# Experiment 10 - Step 10.09
# timm ViT Architecture and Initialization Audit
# Source inspection only; no model instantiation
# ============================================================

import inspect
import timm
import torch

from timm.models.vision_transformer import (
    VisionTransformer,
    Block,
    Attention,
    Mlp
)
from timm.layers import PatchEmbed

print("EXPERIMENT 10.09 - TIMM VIT SOURCE AUDIT")

print("\nENVIRONMENT")
print("PyTorch:", torch.__version__)
print("timm:", timm.__version__)

classes = {
    "VisionTransformer": VisionTransformer,
    "Block": Block,
    "Attention": Attention,
    "Mlp": Mlp,
    "PatchEmbed": PatchEmbed
}

print("\nCLASS INITIALIZATION SIGNATURES")

for name, cls in classes.items():
    print(f"\n{name}")
    print(inspect.signature(cls.__init__))

print("\nRESET METHODS")

for name, cls in classes.items():
    reset = getattr(cls, "reset_parameters", None)

    locally_defined = (
        "reset_parameters" in cls.__dict__
    )

    print(
        f"{name}: "
        f"callable={callable(reset)}, "
        f"locally_defined={locally_defined}"
    )

print("\nVISION TRANSFORMER INITIALIZATION SOURCE")

for method_name in [
    "init_weights",
    "_pos_embed",
    "forward_features",
    "forward_head"
]:
    method = getattr(
        VisionTransformer, method_name, None
    )

    assert callable(method), (
        f"Missing VisionTransformer method: {method_name}"
    )

    source = inspect.getsource(method)

    print(f"\n--- {method_name} ---")
    print(source[:6500])

print("\nBLOCK FORWARD SOURCE")
print(inspect.getsource(Block.forward)[:4500])

print("\nPATCH EMBEDDING FORWARD SOURCE")
print(inspect.getsource(PatchEmbed.forward)[:4500])

print("\nCHECKS")
print(
    "VisionTransformer forward_head available:",
    callable(VisionTransformer.forward_head)
)
print(
    "VisionTransformer init_weights available:",
    callable(VisionTransformer.init_weights)
)

print("\nRESEARCH INTEGRITY")
print("ViT-Large instantiated: False")
print("Original checkpoints loaded: False")
print("Original model weights modified: False")
print("Parameters randomized: False")
print("Images accessed: 0")
print("Test images accessed: 0")

print("\nSTEP 10.09 COMPLETE")


# %% [cell 343] Experiment 10 - Step 10.10
# ============================================================
# Experiment 10 - Step 10.10
# Native timm ViT Initialization Policy Audit
# Source inspection only; no model instantiation
# ============================================================

import inspect
import timm
import torch
import timm.models.vision_transformer as vit

print("EXPERIMENT 10.10 - NATIVE INITIALIZATION AUDIT")

print("\nENVIRONMENT")
print("PyTorch:", torch.__version__)
print("timm:", timm.__version__)

TARGET_FUNCTIONS = [
    "get_init_weights_vit",
    "init_weights_vit_timm",
    "init_weights_vit_jax",
    "init_weights_vit_moco",
]

for function_name in TARGET_FUNCTIONS:

    function = getattr(vit, function_name, None)

    print("\n" + "=" * 68)
    print("FUNCTION:", function_name)

    if function is None:
        print("Function unavailable in this timm version.")
        continue

    assert callable(function)

    try:
        print("Signature:", inspect.signature(function))
    except (TypeError, ValueError):
        print("Signature unavailable.")

    source = inspect.getsource(function)

    print("\nSOURCE")
    print(source[:7500])

print("\nVISION TRANSFORMER CONSTRUCTOR INITIALIZATION")

constructor = inspect.getsource(vit.VisionTransformer.__init__)
lines = constructor.splitlines()

keywords = [
    "init_weights",
    "weight_init",
    "patch_embed",
    "fc_norm",
    "head =",
    "cls_token",
    "pos_embed",
    "global_pool"
]

indices = [
    i
    for i, line in enumerate(lines)
    if any(word in line for word in keywords)
]

intervals = []

for i in indices:
    intervals.append((
        max(0, i - 3),
        min(len(lines), i + 6)
    ))

merged = []

for start, end in sorted(intervals):
    if merged and start <= merged[-1][1]:
        merged[-1] = (
            merged[-1][0],
            max(merged[-1][1], end)
        )
    else:
        merged.append((start, end))

for start, end in merged:
    print(f"\n--- Constructor lines {start + 1}-{end} ---")

    for i in range(start, end):
        print(f"{i + 1:04d}: {lines[i]}")

print("\nRESEARCH INTEGRITY")
print("Original checkpoints loaded: False")
print("ViT-Large instantiated: False")
print("Parameters randomized: False")
print("Images accessed: 0")
print("Test images accessed: 0")

print("\nSTEP 10.10 COMPLETE")


# %% [cell 344] Experiment 10 - Step 10.11
# ============================================================
# Experiment 10 - Step 10.11
# Parameter Reinitialization Unit Test
# Synthetic network only; no research model loading
# ============================================================

import copy
import random
import numpy as np
import torch
import torch.nn as nn

from timm.layers import trunc_normal_

print("EXPERIMENT 10.11 - REINITIALIZATION UNIT TEST")

# ------------------------------------------------------------
# 1. Define an explicit initialization policy
# ------------------------------------------------------------

def reinitialize_module(module):

    with torch.no_grad():

        for submodule in module.modules():

            if isinstance(submodule, nn.Linear):

                trunc_normal_(
                    submodule.weight,
                    std=0.02
                )

                if submodule.bias is not None:
                    nn.init.zeros_(submodule.bias)

            elif isinstance(submodule, nn.Conv2d):

                submodule.reset_parameters()

            elif isinstance(submodule, nn.LayerNorm):

                submodule.reset_parameters()


def reinitialize_direct_parameter(parameter, name):

    with torch.no_grad():

        if name == "cls_token":

            nn.init.normal_(
                parameter,
                mean=0.0,
                std=1e-6
            )

        elif name == "pos_embed":

            trunc_normal_(
                parameter,
                std=0.02
            )

        else:
            raise ValueError(
                f"Unsupported direct parameter: {name}"
            )


# ------------------------------------------------------------
# 2. Construct a small synthetic network
# ------------------------------------------------------------

class SyntheticNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            3, 8, kernel_size=4, stride=4
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(8),
                nn.Linear(8, 8),
                nn.GELU(),
                nn.Linear(8, 8)
            )
            for _ in range(3)
        ])

        self.fc_norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 5)

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, 8)
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, 5, 8)
        )


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_stage(model, components):

    for component in components:

        if component in ["cls_token", "pos_embed"]:

            parameter = getattr(model, component)

            reinitialize_direct_parameter(
                parameter,
                component
            )

        elif component == "patch_embed":

            reinitialize_module(
                model.patch_embed
            )

        elif component == "fc_norm":

            reinitialize_module(
                model.fc_norm
            )

        elif component == "head":

            reinitialize_module(
                model.head
            )

        elif component.startswith("blocks."):

            index = int(component.split(".")[1])

            reinitialize_module(
                model.blocks[index]
            )

        else:
            raise ValueError(
                f"Unsupported component: {component}"
            )


def model_parameters(model):

    return {
        name: value.detach().clone()
        for name, value in model.named_parameters()
    }


def parameters_equal(first, second, prefix):

    names = [
        name for name in first
        if name == prefix or name.startswith(prefix + ".")
    ]

    assert names, f"No parameters found for {prefix}"

    return all(
        torch.equal(first[name], second[name])
        for name in names
    )


# ------------------------------------------------------------
# 3. Reproducibility across identical seeds
# ------------------------------------------------------------

set_seed(9001)
reference = SyntheticNetwork()

baseline = copy.deepcopy(reference)

def independent_run(seed):

    set_seed(seed)

    model = copy.deepcopy(baseline)

    apply_stage(
        model,
        ["head", "blocks.2", "cls_token"]
    )

    return model_parameters(model)


same_a = independent_run(1042)
same_b = independent_run(1042)
different = independent_run(2042)

assert all(
    torch.equal(same_a[key], same_b[key])
    for key in same_a
)

assert not parameters_equal(
    same_a, different, "head"
)

assert not parameters_equal(
    same_a, different, "blocks.2"
)

print("\nRANDOM SEED CHECK")
print("Same seed reproducibility: PASS")
print("Different seed variability: PASS")


# ------------------------------------------------------------
# 4. Verify cumulative stages
# ------------------------------------------------------------

set_seed(1042)
cumulative = copy.deepcopy(baseline)

apply_stage(
    cumulative,
    ["head"]
)

r1 = model_parameters(cumulative)

apply_stage(
    cumulative,
    ["fc_norm", "blocks.2"]
)

r2 = model_parameters(cumulative)

apply_stage(
    cumulative,
    ["blocks.1"]
)

r3 = model_parameters(cumulative)

assert parameters_equal(
    r1, r2, "head"
)

assert parameters_equal(
    r2, r3, "head"
)

assert parameters_equal(
    r2, r3, "blocks.2"
)

assert parameters_equal(
    r2, r3, "fc_norm"
)

assert parameters_equal(
    r1,
    model_parameters(baseline),
    "blocks.1"
)

assert not parameters_equal(
    r3,
    model_parameters(baseline),
    "blocks.1"
)

print("\nCUMULATIVE STAGE CHECK")
print("R1 parameters retained in R2: PASS")
print("R2 parameters retained in R3: PASS")
print("Previously unchanged blocks preserved: PASS")

# ------------------------------------------------------------
# 5. Verify direct parameter initialization
# ------------------------------------------------------------

assert not parameters_equal(
    r1, r2, "blocks.2"
)

assert parameters_equal(
    r1, r2, "cls_token"
)

assert parameters_equal(
    r1, r2, "pos_embed"
)

print("\nDIRECT PARAMETER CHECK")
print("CLS token unchanged across later stages: PASS")
print("Position embedding unchanged: PASS")

print("\nRESEARCH INTEGRITY")
print("Synthetic network only: True")
print("Original checkpoints loaded: False")
print("Original model weights modified: False")
print("Images accessed: 0")
print("Test images accessed: 0")

print("\nSTEP 10.11 PASS")


# %% [cell 345] Experiment 10 - Step 10.12
# ============================================================
# Experiment 10 - Step 10.12
# Real ViT-Large Module Coverage Audit
# No checkpoint loading, randomization, or inference
# ============================================================

import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import timm

ROOT = DR_RESEARCH

PROTOCOL_PATH = (
    ROOT / "experiment10_results"
    / "Experiment10_Randomization_Protocol_V1.json"
)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

stages = protocol["randomization"]["stages"]

print("EXPERIMENT 10.12 - REAL VIT MODULE COVERAGE AUDIT")

# Build on the meta device to avoid allocating full model weights.
model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

print("\nMODEL STRUCTURE")
print("Architecture:", type(model).__name__)
print("Transformer blocks:", len(model.blocks))
print("Global pool:", model.global_pool)

assert len(model.blocks) == 24
assert model.global_pool == "avg"

# ------------------------------------------------------------
# 1. Check actual normalization module classes
# ------------------------------------------------------------

print("\nNORMALIZATION MODULE TYPES")

normalization_modules = []

for name, module in model.named_modules():
    if "norm" in name and hasattr(module, "weight"):
        if isinstance(module.weight, nn.Parameter):
            normalization_modules.append(
                (name, type(module).__name__)
            )

print("Normalization modules:", len(normalization_modules))

for name, module_type in normalization_modules[:12]:
    print(name, module_type)

assert isinstance(model.fc_norm, nn.LayerNorm)

for block in model.blocks:
    assert isinstance(block.norm1, nn.LayerNorm)
    assert isinstance(block.norm2, nn.LayerNorm)

print("LayerNorm compatibility: PASS")

# ------------------------------------------------------------
# 2. Enumerate all model parameters
# ------------------------------------------------------------

parameter_names = [
    name for name, _ in model.named_parameters()
]

assert len(parameter_names) == len(set(parameter_names))

print("\nPARAMETER INVENTORY")
print("Named parameters:", len(parameter_names))

# ------------------------------------------------------------
# 3. Match each stage to its exact parameter set
# ------------------------------------------------------------

def belongs_to_component(parameter_name, component):

    return (
        parameter_name == component
        or parameter_name.startswith(component + ".")
    )


stage_parameters = {}

previous_parameters = set()

print("\nRANDOMIZATION STAGE COVERAGE")

for stage_name, components in stages.items():

    selected = set()

    component_counts = {}

    for component in components:

        matched = {
            name for name in parameter_names
            if belongs_to_component(name, component)
        }

        assert matched, (
            f"No parameters matched component: {component}"
        )

        assert not selected.intersection(matched), (
            f"Overlapping component: {component}"
        )

        selected.update(matched)
        component_counts[component] = len(matched)

    assert previous_parameters.issubset(selected)

    stage_parameters[stage_name] = selected

    newly_added = selected - previous_parameters

    print(f"\n{stage_name}")
    print("Components:", len(components))
    print("Selected parameter tensors:", len(selected))
    print("Newly selected tensors:", len(newly_added))

    previous_parameters = selected

# ------------------------------------------------------------
# 4. Verify final-stage parameter coverage
# ------------------------------------------------------------

r6_parameters = stage_parameters["R6"]

uncovered = sorted(
    set(parameter_names) - r6_parameters
)

print("\nFINAL COVERAGE")
print("Total parameter tensors:", len(parameter_names))
print("R6 parameter tensors:", len(r6_parameters))
print("Uncovered parameter tensors:", len(uncovered))

for name in uncovered:
    print("Uncovered:", name)

assert len(parameter_names) == 296
assert not uncovered, (
    "R6 does not cover all named model parameters."
)

print("\nR6 COMPLETE PARAMETER COVERAGE: PASS")

# ------------------------------------------------------------
# 5. Inspect layer parameter types
# ------------------------------------------------------------

module_types = Counter()

for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
        module_types[type(module).__name__] += 1

print("\nPARAMETERIZED MODULE TYPES")

for module_type, count in sorted(module_types.items()):
    print(module_type, count)

print("\nRESEARCH INTEGRITY")
print("Model constructed on meta device: True")
print("Research checkpoints loaded: False")
print("Research model weights modified: False")
print("Parameters randomized: False")
print("Images accessed: 0")
print("Model inference performed: False")
print("Test images accessed: 0")

del model

print("\nSTEP 10.12 PASS")


# %% [cell 346] Experiment 10 - Step 10.13
# ============================================================
# Experiment 10 - Step 10.13
# Real ViT Component Reinitialization Audit
# CPU-only; no original checkpoint loading
# ============================================================

import copy
import gc
import torch
import torch.nn as nn
import timm

from timm.layers import trunc_normal_

print("EXPERIMENT 10.13 - REAL VIT REINITIALIZATION AUDIT")

torch.manual_seed(9001)

# Construct architecture without allocating full weights.
model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

assert len(model.blocks) == 24

# Materialize only the required components on CPU.
head = model.head.to_empty(device="cpu")
fc_norm = model.fc_norm.to_empty(device="cpu")
block23 = model.blocks[23].to_empty(device="cpu")

def initialize_component(module):

    with torch.no_grad():

        for submodule in module.modules():

            if isinstance(submodule, nn.Linear):
                trunc_normal_(submodule.weight, std=0.02)

                if submodule.bias is not None:
                    nn.init.zeros_(submodule.bias)

            elif isinstance(submodule, nn.Conv2d):
                submodule.reset_parameters()

            elif isinstance(submodule, nn.LayerNorm):
                submodule.reset_parameters()


def tensor_snapshot(module):

    return {
        name: value.detach().clone()
        for name, value in module.named_parameters()
    }


def tensors_identical(first, second):

    return (
        first.keys() == second.keys()
        and all(
            torch.equal(first[key], second[key])
            for key in first
        )
    )


# ------------------------------------------------------------
# 1. Verify identical seeds
# ------------------------------------------------------------

torch.manual_seed(1042)
initialize_component(head)
head_a = tensor_snapshot(head)

torch.manual_seed(1042)
initialize_component(head)
head_b = tensor_snapshot(head)

assert tensors_identical(head_a, head_b)

torch.manual_seed(2042)
initialize_component(head)
head_c = tensor_snapshot(head)

assert not tensors_identical(head_a, head_c)

print("\nHEAD INITIALIZATION")
print("Same seed reproducibility: PASS")
print("Different seed variability: PASS")

# ------------------------------------------------------------
# 2. Verify final block initialization
# ------------------------------------------------------------

torch.manual_seed(1042)
initialize_component(block23)
block_a = tensor_snapshot(block23)

torch.manual_seed(1042)
initialize_component(block23)
block_b = tensor_snapshot(block23)

assert tensors_identical(block_a, block_b)

torch.manual_seed(2042)
initialize_component(block23)
block_c = tensor_snapshot(block23)

assert not tensors_identical(block_a, block_c)

print("\nBLOCK 23 INITIALIZATION")
print("Same seed reproducibility: PASS")
print("Different seed variability: PASS")
print("Parameter tensors:", len(block_a))

# ------------------------------------------------------------
# 3. Verify final normalization
# ------------------------------------------------------------

with torch.no_grad():
    fc_norm.weight.fill_(7.0)
    fc_norm.bias.fill_(9.0)

initialize_component(fc_norm)

assert torch.equal(
    fc_norm.weight,
    torch.ones_like(fc_norm.weight)
)

assert torch.equal(
    fc_norm.bias,
    torch.zeros_like(fc_norm.bias)
)

print("\nFINAL NORMALIZATION")
print("Weight initialized to ones: PASS")
print("Bias initialized to zeros: PASS")

# ------------------------------------------------------------
# 4. Verify direct token initialization
# ------------------------------------------------------------

cls_a = torch.empty((1, 1, 1024), device="cpu")
cls_b = torch.empty_like(cls_a)

pos_a = torch.empty((1, 197, 1024), device="cpu")
pos_b = torch.empty_like(pos_a)

torch.manual_seed(1042)
nn.init.normal_(cls_a, std=1e-6)
trunc_normal_(pos_a, std=0.02)

torch.manual_seed(1042)
nn.init.normal_(cls_b, std=1e-6)
trunc_normal_(pos_b, std=0.02)

assert torch.equal(cls_a, cls_b)
assert torch.equal(pos_a, pos_b)

print("\nDIRECT TOKEN PARAMETERS")
print("CLS token reproducibility: PASS")
print("Position embedding reproducibility: PASS")
print("CLS token standard deviation:", float(cls_a.std()))
print("Position embedding standard deviation:", float(pos_a.std()))

# ------------------------------------------------------------
# 5. Cleanup
# ------------------------------------------------------------

del model
del head
del fc_norm
del block23
del head_a, head_b, head_c
del block_a, block_b, block_c
gc.collect()

print("\nRESEARCH INTEGRITY")
print("Original checkpoints loaded: False")
print("Original model weights modified: False")
print("Medical images accessed: 0")
print("Model inference performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.13 PASS")


# %% [cell 347] Experiment 10 - Step 10.14
# ============================================================
# Experiment 10 - Step 10.14
# Full RETFound Model R1 Reinitialization Audit
# Independent GPU model; no inference
# ============================================================

from pathlib import Path
import gc
import torch
import timm
from timm.layers import trunc_normal_

ROOT = DR_RESEARCH

CHECKPOINT_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_model.pt"
)

assert CHECKPOINT_PATH.is_file()
assert torch.cuda.is_available()

DEVICE = torch.device("cuda")

print("EXPERIMENT 10.14 - FULL MODEL R1 AUDIT")

print("\nENVIRONMENT")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("timm:", timm.__version__)

# ------------------------------------------------------------
# 1. Load the frozen source checkpoint on CPU
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

assert "model_state_dict" in checkpoint

source_state = checkpoint["model_state_dict"]

assert len(source_state) == 296

# ------------------------------------------------------------
# 2. Build an independent model on the meta device
# ------------------------------------------------------------

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

# Allocate uninitialized storage on GPU.
# The next load_state_dict operation initializes all tensors.
model = model.to_empty(device=DEVICE)

load_result = model.load_state_dict(
    source_state,
    strict=True,
    assign=False
)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()

print("\nMODEL LOADING")
print("State dictionary entries:", len(source_state))
print("Missing keys:", len(load_result.missing_keys))
print("Unexpected keys:", len(load_result.unexpected_keys))
print("Strict checkpoint loading: PASS")

# ------------------------------------------------------------
# 3. Verify exact baseline parameter equality
# ------------------------------------------------------------

def verify_source_equality(model, source, excluded=()):

    excluded = set(excluded)
    mismatched = []

    with torch.no_grad():

        for name, tensor in model.state_dict().items():

            if name in excluded:
                continue

            source_tensor = source[name]

            equal = torch.equal(
                tensor.detach().cpu(),
                source_tensor
            )

            if not equal:
                mismatched.append(name)

    return mismatched


baseline_mismatches = verify_source_equality(
    model,
    source_state
)

assert not baseline_mismatches, baseline_mismatches

print("\nBASELINE INTEGRITY")
print("Exact matching tensors:", len(source_state))
print("Baseline parameter comparison: PASS")

# ------------------------------------------------------------
# 4. Reinitialize only the classification head
# ------------------------------------------------------------

original_head_weight = (
    model.head.weight.detach().clone()
)

original_head_bias = (
    model.head.bias.detach().clone()
)

torch.manual_seed(1042)
torch.cuda.manual_seed_all(1042)

with torch.no_grad():

    trunc_normal_(
        model.head.weight,
        std=0.02
    )

    torch.nn.init.zeros_(
        model.head.bias
    )

assert not torch.equal(
    model.head.weight,
    original_head_weight
)

# Both original and randomized biases may be zero.
# A replacement does not guarantee a numerical change.

print("\nR1 HEAD REINITIALIZATION")
print(
    "Head weight changed:",
    not torch.equal(
        model.head.weight,
        original_head_weight
    )
)
print(
    "Head bias changed:",
    not torch.equal(
        model.head.bias,
        original_head_bias
    )
)
print(
    "Head bias is zero:",
    bool(torch.count_nonzero(model.head.bias) == 0)
)

# ------------------------------------------------------------
# 5. Confirm every unselected tensor is unchanged
# ------------------------------------------------------------

excluded_names = (
    "head.weight",
    "head.bias"
)

remaining_mismatches = verify_source_equality(
    model,
    source_state,
    excluded=excluded_names
)

assert not remaining_mismatches, remaining_mismatches

print("\nR1 PARAMETER ISOLATION")
print("Selected parameter tensors:", len(excluded_names))
print(
    "Unselected parameter tensors:",
    len(source_state) - len(excluded_names)
)
print("Unselected tensor mismatches:", len(remaining_mismatches))
print("R1 isolation: PASS")

# ------------------------------------------------------------
# 6. Confirm model state is numerically valid
# ------------------------------------------------------------

nonfinite = []

with torch.no_grad():

    for name, tensor in model.named_parameters():

        if not torch.isfinite(tensor).all().item():
            nonfinite.append(name)

assert not nonfinite

print("\nNUMERICAL VALIDITY")
print("Non-finite parameter tensors:", len(nonfinite))
print("Numerical validity: PASS")

# ------------------------------------------------------------
# 7. Release the temporary model
# ------------------------------------------------------------

del original_head_weight
del original_head_bias
del model
del source_state
del checkpoint

gc.collect()
torch.cuda.empty_cache()

print("\nCLEANUP")
print(
    "GPU allocated memory (GiB):",
    round(
        torch.cuda.memory_allocated()
        / (1024 ** 3),
        3
    )
)

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Temporary model randomized: R1 only")
print("Research output files created: False")
print("Medical images accessed: 0")
print("Model inference performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.14 PASS")


# %% [cell 348] Experiment 10 - Step 10.15
# ============================================================
# Experiment 10 - Step 10.15
# Full RETFound R1-to-R2 Cumulative Reinitialization Audit
# Independent model; no inference or image access
# ============================================================

from pathlib import Path
import gc
import torch
import torch.nn as nn
import timm
from timm.layers import trunc_normal_

ROOT = DR_RESEARCH

CHECKPOINT_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_model.pt"
)

assert CHECKPOINT_PATH.is_file()
assert torch.cuda.is_available()

DEVICE = torch.device("cuda")
SEED = 1042

print("EXPERIMENT 10.15 - R1 TO R2 CUMULATIVE AUDIT")

# ------------------------------------------------------------
# 1. Load checkpoint and independent full model
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]

assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

result = model.load_state_dict(
    source_state,
    strict=True
)

assert not result.missing_keys
assert not result.unexpected_keys

model.eval()

def verify_unchanged(model, source, excluded):

    mismatched = []

    for name, tensor in model.state_dict().items():

        if name in excluded:
            continue

        if not torch.equal(
            tensor.detach().cpu(),
            source[name]
        ):
            mismatched.append(name)

    return mismatched


assert not verify_unchanged(
    model, source_state, excluded=set()
)

print("\nBASELINE")
print("Strict loading: PASS")
print("Exact checkpoint equality: PASS")

# ------------------------------------------------------------
# 2. Apply R1: Classification Head
# ------------------------------------------------------------

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

with torch.no_grad():
    trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)

head_after_r1 = {
    name: tensor.detach().clone()
    for name, tensor in model.head.named_parameters()
}

assert not verify_unchanged(
    model,
    source_state,
    excluded={"head.weight", "head.bias"}
)

print("\nR1")
print("Head reinitialized: PASS")
print("Other parameters unchanged: PASS")

# ------------------------------------------------------------
# 3. Apply incremental R2 modules
# ------------------------------------------------------------

def initialize_module(module):

    with torch.no_grad():

        for submodule in module.modules():

            if isinstance(submodule, nn.Linear):

                trunc_normal_(
                    submodule.weight,
                    std=0.02
                )

                if submodule.bias is not None:
                    nn.init.zeros_(submodule.bias)

            elif isinstance(submodule, nn.Conv2d):

                submodule.reset_parameters()

            elif isinstance(submodule, nn.LayerNorm):

                submodule.reset_parameters()


block_before = {
    name: tensor.detach().clone()
    for name, tensor in model.blocks[23].named_parameters()
}

fc_norm_before = {
    name: tensor.detach().clone()
    for name, tensor in model.fc_norm.named_parameters()
}

initialize_module(model.fc_norm)
initialize_module(model.blocks[23])

# ------------------------------------------------------------
# 4. Verify Head remains unchanged
# ------------------------------------------------------------

for name, tensor in model.head.named_parameters():

    assert torch.equal(
        tensor.detach(),
        head_after_r1[name]
    )

print("\nR2 CUMULATIVE CONSISTENCY")
print("R1 Head retained exactly: PASS")

# ------------------------------------------------------------
# 5. Verify Block 23 and fc_norm
# ------------------------------------------------------------

changed_block_parameters = []

for name, tensor in model.blocks[23].named_parameters():

    if not torch.equal(
        tensor.detach(),
        block_before[name]
    ):
        changed_block_parameters.append(name)

assert changed_block_parameters

assert torch.equal(
    model.fc_norm.weight,
    torch.ones_like(model.fc_norm.weight)
)

assert torch.equal(
    model.fc_norm.bias,
    torch.zeros_like(model.fc_norm.bias)
)

print("\nR2 MODULES")
print(
    "Block 23 changed parameter tensors:",
    len(changed_block_parameters)
)
print("Block 23 reinitialization: PASS")
print("fc_norm reset to defaults: PASS")

# ------------------------------------------------------------
# 6. Verify parameter isolation
# ------------------------------------------------------------

r2_selected = {
    "head.weight",
    "head.bias",
    "fc_norm.weight",
    "fc_norm.bias"
}

r2_selected.update(
    f"blocks.23.{name}"
    for name in block_before
)

assert len(r2_selected) == 16

mismatched = verify_unchanged(
    model,
    source_state,
    excluded=r2_selected
)

assert not mismatched, mismatched

print("\nR2 PARAMETER ISOLATION")
print("Selected parameter tensors:", len(r2_selected))
print("Unselected parameter tensors:", 296 - len(r2_selected))
print("Unselected mismatches:", len(mismatched))
print("R2 isolation: PASS")

# ------------------------------------------------------------
# 7. Verify numerical validity
# ------------------------------------------------------------

nonfinite = [
    name
    for name, tensor in model.named_parameters()
    if not torch.isfinite(tensor).all().item()
]

assert not nonfinite

print("\nNUMERICAL VALIDITY")
print("Non-finite parameters:", len(nonfinite))
print("Numerical validity: PASS")

# ------------------------------------------------------------
# 8. Cleanup
# ------------------------------------------------------------

del model
del checkpoint
del source_state
del head_after_r1
del block_before
del fc_norm_before

gc.collect()
torch.cuda.empty_cache()

print("\nCLEANUP")
print(
    "GPU allocated memory (GiB):",
    round(torch.cuda.memory_allocated() / (1024 ** 3), 3)
)

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Temporary model randomized: R1 then R2")
print("Original CAM files modified: False")
print("Medical images accessed: 0")
print("Model inference performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.15 PASS")


# %% [cell 349] Experiment 10 - Step 10.16
# ============================================================
# Experiment 10 - Step 10.16
# Validation Pilot Cohort and Input Availability Audit
# Read-only; no image decoding or model inference
# ============================================================

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

RET_CAM_PATH = (
    ROOT / "experiment05_results/gradcam"
    / "RETFound_MA_CAM_Maps_V1.npz"
)

META_CAM_PATH = (
    ROOT / "experiment07_results/gradcam"
    / "Meta_MAE_MA_CAM_Maps_V1.npz"
)

PROTOCOL_PATH = (
    ROOT / "experiment10_results"
    / "Experiment10_Randomization_Protocol_V1.json"
)

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_SHA256

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert protocol["evaluation"]["validation_image_count"] == 156
assert protocol["evaluation"]["test_set_access"] is False

index = pd.read_csv(
    INDEX_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str
    }
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val"),
    [
        "image_id",
        "dr_grade",
        "risk_group_id",
        "local_path",
        "source_path"
    ]
].copy()

assert len(validation) == 276

candidates = cohort[["image_id"]].merge(
    validation,
    on="image_id",
    validate="one_to_one"
)

assert len(candidates) == 156
assert candidates["risk_group_id"].notna().all()

with np.load(RET_CAM_PATH, allow_pickle=False) as data:
    ret_ids = set(map(str, data["image_ids"]))

with np.load(META_CAM_PATH, allow_pickle=False) as data:
    meta_ids = set(map(str, data["image_ids"]))

candidate_ids = set(candidates["image_id"])

assert candidate_ids == ret_ids
assert candidate_ids == meta_ids

# Select two deterministic pilot images per DR grade.
pilot = (
    candidates.loc[
        candidates["dr_grade"].isin([1, 2, 3, 4])
    ]
    .sort_values(
        ["dr_grade", "risk_group_id", "image_id"]
    )
    .groupby("dr_grade", group_keys=False)
    .head(2)
    .reset_index(drop=True)
)

assert len(pilot) == 8
assert pilot["risk_group_id"].nunique() == 8

print("EXPERIMENT 10.16 - PILOT INPUT AUDIT")

print("\nDATA INTEGRITY")
print("Frozen validation images:", len(validation))
print("MA candidates:", len(candidates))
print("RETFound CAM IDs matched: PASS")
print("Meta MAE CAM IDs matched: PASS")

print("\nPILOT COHORT")
print(
    pilot[
        ["image_id", "dr_grade", "risk_group_id"]
    ].to_string(index=False)
)

print("\nINPUT AVAILABILITY")

available = 0
missing = []

for row in pilot.itertuples(index=False):

    local = Path(row.local_path)
    source = Path(row.source_path)

    local_ok = local.is_file()
    source_ok = source.is_file()

    if local_ok or source_ok:
        available += 1
    else:
        missing.append(row.image_id)

    print(
        f"{row.image_id}: "
        f"local={local_ok}, source={source_ok}"
    )

print("\nSUMMARY")
print("Available pilot images:", available)
print("Missing pilot images:", len(missing))

assert not missing, (
    f"Missing image files: {missing}"
)

print("Input availability: PASS")

print("\nRESEARCH INTEGRITY")
print("Image pixels decoded: False")
print("Model inference performed: False")
print("Parameters randomized: False")
print("Research files modified: False")
print("Test images accessed: 0")

print("\nSTEP 10.16 PASS")


# %% [cell 350] Experiment 10 - Step 10.16A
# ============================================================
# Experiment 10 - Step 10.16A
# Diagnose Validation Cohort Join Failure
# Read-only; no image decoding or inference
# ============================================================

from pathlib import Path
import pandas as pd

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

print("EXPERIMENT 10.16A - COHORT JOIN DIAGNOSTIC")

print("\nINDEX")
print("Total rows:", len(index))
print("Unique image IDs:", index["image_id"].nunique())
print("Validation rows:", int(index["split"].eq("val").sum()))
print("Duplicate image IDs:", int(index["image_id"].duplicated().sum()))

print("\nCOHORT")
print("Total rows:", len(cohort))
print("Unique image IDs:", cohort["image_id"].nunique())
print("Duplicate image IDs:", int(cohort["image_id"].duplicated().sum()))
print("Columns:", list(cohort.columns))

print("\nCOHORT SPLIT DISTRIBUTION")

if "split" in cohort.columns:
    print(cohort["split"].value_counts(dropna=False).to_string())
else:
    print("No split column in cohort.")

print("\nMATCHING DIAGNOSTIC")

validation = index.loc[
    index["split"].eq("val")
].copy()

validation_ids = set(validation["image_id"])
cohort_ids = set(cohort["image_id"])

matched_ids = validation_ids & cohort_ids

print("Validation unique IDs:", len(validation_ids))
print("Cohort unique IDs:", len(cohort_ids))
print("Matched unique IDs:", len(matched_ids))
print("Cohort IDs outside validation:", len(cohort_ids - validation_ids))
print("Validation IDs outside cohort:", len(validation_ids - cohort_ids))

joined = cohort[["image_id"]].merge(
    validation[
        ["image_id", "dr_grade", "risk_group_id"]
    ],
    on="image_id",
    how="inner",
    validate="many_to_one"
)

print("\nJOIN RESULT")
print("Joined rows:", len(joined))
print("Joined unique IDs:", joined["image_id"].nunique())
print("Duplicate joined rows:", int(joined["image_id"].duplicated().sum()))

print("\nMOST FREQUENT COHORT IDS")
print(
    cohort["image_id"]
    .value_counts()
    .head(10)
    .to_string()
)

print("\nDUPLICATE RECORD EXAMPLES")

duplicate_rows = cohort.loc[
    cohort["image_id"].duplicated(keep=False)
]

if duplicate_rows.empty:
    print("No duplicate cohort IDs.")
else:
    print(
        duplicate_rows.head(10).to_string(index=False)
    )

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images decoded: 0")
print("Model inference performed: False")
print("Parameters randomized: False")
print("Test images accessed: 0")

print("\nSTEP 10.16A DIAGNOSTIC COMPLETE")


# %% [cell 351] Experiment 10 - Step 10.16B
# ============================================================
# Experiment 10 - Step 10.16B
# Corrected Validation Pilot Cohort Audit
# Read-only; no image decoding or model inference
# ============================================================

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

RET_CAM_PATH = (
    ROOT / "experiment05_results/gradcam"
    / "RETFound_MA_CAM_Maps_V1.npz"
)

META_CAM_PATH = (
    ROOT / "experiment07_results/gradcam"
    / "Meta_MAE_MA_CAM_Maps_V1.npz"
)

PROTOCOL_PATH = (
    ROOT / "experiment10_results"
    / "Experiment10_Randomization_Protocol_V1.json"
)

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_SHA256

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert protocol["evaluation"]["validation_image_count"] == 156
assert protocol["evaluation"]["test_set_access"] is False

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val"),
    [
        "image_id",
        "dr_grade",
        "risk_group_id",
        "local_path",
        "source_path"
    ]
].copy()

assert len(validation) == 276
assert len(cohort) == 276

# Validate the candidate indicator before using it.
candidate_flag = (
    cohort["primary_ma_candidate"]
    .astype("string")
    .str.strip()
    .str.lower()
)

assert candidate_flag.isin(
    ["true", "false", "1", "0"]
).all(), "Unexpected primary_ma_candidate values."

candidate_mask = candidate_flag.isin(["true", "1"])

primary_cohort = cohort.loc[
    candidate_mask,
    ["image_id", "true_grade"]
].copy()

assert len(primary_cohort) == 156, (
    f"Unexpected primary candidate count: {len(primary_cohort)}"
)

candidates = primary_cohort.merge(
    validation,
    on="image_id",
    validate="one_to_one"
)

assert len(candidates) == 156

assert (
    candidates["true_grade"]
    == candidates["dr_grade"]
).all()

assert candidates["risk_group_id"].notna().all()

with np.load(RET_CAM_PATH, allow_pickle=False) as data:
    ret_ids = set(map(str, data["image_ids"]))

with np.load(META_CAM_PATH, allow_pickle=False) as data:
    meta_ids = set(map(str, data["image_ids"]))

candidate_ids = set(candidates["image_id"])

assert len(ret_ids) == 156
assert len(meta_ids) == 156
assert candidate_ids == ret_ids
assert candidate_ids == meta_ids

# Deterministic selection: two images from each DR grade.
pilot = (
    candidates.loc[
        candidates["dr_grade"].isin([1, 2, 3, 4])
    ]
    .sort_values(
        ["dr_grade", "risk_group_id", "image_id"]
    )
    .groupby("dr_grade", group_keys=False)
    .head(2)
    .reset_index(drop=True)
)

assert len(pilot) == 8

# Pilot observations should come from distinct source-risk groups.
assert pilot["risk_group_id"].nunique() == 8

print("EXPERIMENT 10.16B - CORRECTED PILOT AUDIT")

print("\nDATA INTEGRITY")
print("Validation images:", len(validation))
print("Cohort records:", len(cohort))
print("Primary MA candidates:", len(candidates))
print("RETFound CAM ID match: PASS")
print("Meta MAE CAM ID match: PASS")

print("\nPILOT COHORT")
print(
    pilot[
        ["image_id", "dr_grade", "risk_group_id"]
    ].to_string(index=False)
)

print("\nIMAGE FILE AVAILABILITY")

available = 0
missing = []

for row in pilot.itertuples(index=False):

    local = Path(row.local_path)
    source = Path(row.source_path)

    local_ok = local.is_file()
    source_ok = source.is_file()

    if local_ok or source_ok:
        available += 1
    else:
        missing.append(row.image_id)

    print(
        f"{row.image_id}: "
        f"local={local_ok}, source={source_ok}"
    )

print("\nSUMMARY")
print("Available pilot images:", available)
print("Missing pilot images:", len(missing))

assert not missing, (
    f"Missing image files: {missing}"
)

print("Input availability: PASS")

print("\nRESEARCH INTEGRITY")
print("Original files modified: False")
print("Image pixels decoded: False")
print("Models loaded: False")
print("Model inference performed: False")
print("Parameters randomized: False")
print("Test images accessed: 0")

print("\nSTEP 10.16B PASS")


# %% [cell 352] Experiment 10 - Step 10.16C
# ============================================================
# Experiment 10 - Step 10.16C
# Group-Distinct Validation Pilot Selection
# Read-only; no image decoding or inference
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

RET_CAM_PATH = (
    ROOT / "experiment05_results/gradcam"
    / "RETFound_MA_CAM_Maps_V1.npz"
)

META_CAM_PATH = (
    ROOT / "experiment07_results/gradcam"
    / "Meta_MAE_MA_CAM_Maps_V1.npz"
)

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

validation = index.loc[
    index["split"].eq("val"),
    [
        "image_id",
        "dr_grade",
        "risk_group_id",
        "local_path",
        "source_path"
    ]
].copy()

flag = (
    cohort["primary_ma_candidate"]
    .astype("string")
    .str.strip()
    .str.lower()
)

assert flag.isin(["true", "false", "1", "0"]).all()

primary_ids = cohort.loc[
    flag.isin(["true", "1"]),
    "image_id"
]

assert len(primary_ids) == 156
assert primary_ids.is_unique

candidates = validation.loc[
    validation["image_id"].isin(primary_ids)
].copy()

assert len(candidates) == 156
assert candidates["risk_group_id"].notna().all()

with np.load(RET_CAM_PATH, allow_pickle=False) as data:
    ret_ids = set(map(str, data["image_ids"]))

with np.load(META_CAM_PATH, allow_pickle=False) as data:
    meta_ids = set(map(str, data["image_ids"]))

assert set(candidates["image_id"]) == ret_ids == meta_ids

# Select two images per grade from distinct groups.
selected_rows = []
used_groups = set()

for grade in [1, 2, 3, 4]:

    grade_frame = (
        candidates.loc[candidates["dr_grade"].eq(grade)]
        .sort_values(["risk_group_id", "image_id"])
    )

    selected_this_grade = 0

    for _, row in grade_frame.iterrows():

        group_id = str(row["risk_group_id"])

        if group_id in used_groups:
            continue

        selected_rows.append(row)
        used_groups.add(group_id)
        selected_this_grade += 1

        if selected_this_grade == 2:
            break

    assert selected_this_grade == 2, (
        f"Insufficient distinct risk groups for DR Grade {grade}"
    )

pilot = pd.DataFrame(selected_rows).reset_index(drop=True)

assert len(pilot) == 8
assert pilot["image_id"].is_unique
assert pilot["risk_group_id"].nunique() == 8

assert pilot["dr_grade"].value_counts().to_dict() == {
    1: 2,
    2: 2,
    3: 2,
    4: 2
}

print("EXPERIMENT 10.16C - GROUP-DISTINCT PILOT AUDIT")

print("\nPILOT COHORT")
print(
    pilot[
        ["image_id", "dr_grade", "risk_group_id"]
    ].to_string(index=False)
)

print("\nGROUP INTEGRITY")
print("Selected images:", len(pilot))
print("Distinct source-risk groups:", pilot["risk_group_id"].nunique())
print("Grade distribution:", pilot["dr_grade"].value_counts().sort_index().to_dict())
print("Group-distinct selection: PASS")

print("\nIMAGE AVAILABILITY")

missing = []

for row in pilot.itertuples(index=False):

    local_ok = Path(row.local_path).is_file()
    source_ok = Path(row.source_path).is_file()

    if not (local_ok or source_ok):
        missing.append(row.image_id)

    print(
        f"{row.image_id}: "
        f"local={local_ok}, source={source_ok}"
    )

print("\nSUMMARY")
print("Available images:", len(pilot) - len(missing))
print("Missing images:", len(missing))

assert not missing, f"Missing pilot images: {missing}"

print("\nRESEARCH INTEGRITY")
print("Original files modified: False")
print("Image pixels decoded: False")
print("Models loaded: False")
print("Model inference performed: False")
print("Parameters randomized: False")
print("Test images accessed: 0")

print("\nSTEP 10.16C PASS")


# %% [cell 353] Experiment 10 - Step 10.17

# ============================================================
# Experiment 10 - Step 10.17
# RETFound Baseline Grad-CAM Reproduction Pilot
# Read-only research artifacts; validation images only
# ============================================================

from pathlib import Path
import gc
import numpy as np
import pandas as pd
import cv2
import torch
import timm

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = DR_RESEARCH

CHECKPOINT_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1/best_model.pt"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

CAM_PATH = (
    ROOT / "experiment05_results/gradcam"
    / "RETFound_MA_CAM_Maps_V1.npz"
)

PILOT_IDS = [
    "1301_1",
    "0421_1",
    "1190_1",
    "0732_1",
    "0001_2",
    "0011_3",
    "0032_3",
    "0043_3"
]

EXPECTED_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

assert torch.cuda.is_available()

device = torch.device("cuda")

print("EXPERIMENT 10.17 - RETFOUND BASELINE CAM REPRODUCTION")

# ------------------------------------------------------------
# 1. Verify the frozen validation pilot
# ------------------------------------------------------------

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(PILOT_IDS) == 8
assert len(set(PILOT_IDS)) == 8

for image_id, grade in zip(PILOT_IDS, EXPECTED_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    candidate_value = str(
        cohort_lookup.loc[image_id, "primary_ma_candidate"]
    ).strip().lower()

    assert candidate_value in {"true", "1"}

assert (
    validation.loc[PILOT_IDS, "risk_group_id"].nunique() == 8
)

# ------------------------------------------------------------
# 2. Load original saved CAM references
# ------------------------------------------------------------

with np.load(CAM_PATH, allow_pickle=False) as archive:
    saved_ids = [
        str(value) for value in archive["image_ids"]
    ]

    id_to_position = {
        image_id: position
        for position, image_id in enumerate(saved_ids)
    }

    assert len(id_to_position) == 156
    assert set(PILOT_IDS).issubset(id_to_position)

    reference_cams = {
        image_id: archive["retfound_cam"][
            id_to_position[image_id]
        ].astype(np.float32).copy()
        for image_id in PILOT_IDS
    }

print("\nREFERENCE")
print("Saved CAMs matched:", len(reference_cams))

# ------------------------------------------------------------
# 3. Reconstruct original image preprocessing
# ------------------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------------------
# 4. Load independent RETFound model
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

state = checkpoint["model_state_dict"]

assert len(state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=device)

load_result = model.load_state_dict(
    state,
    strict=True
)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

del checkpoint, state
gc.collect()

print("\nMODEL")
print("Strict checkpoint loading: PASS")
print("Model evaluation mode: PASS")
print("Parameters frozen: PASS")

# ------------------------------------------------------------
# 5. Register the original Grad-CAM activation hook
# ------------------------------------------------------------

captured = {}

def activation_hook(module, inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation

hook_handle = model.blocks[-1].register_forward_hook(
    activation_hook
)

# ------------------------------------------------------------
# 6. Reproduce Grad-CAM for each pilot image
# ------------------------------------------------------------

records = []

try:
    for image_id, target_grade in zip(
        PILOT_IDS, EXPECTED_GRADES
    ):
        row = validation.loc[image_id]

        local_path = Path(row["local_path"])
        source_path = Path(row["source_path"])

        image_path = (
            local_path if local_path.is_file()
            else source_path
        )

        assert image_path.is_file()

        with Image.open(image_path) as image:
            image_tensor = preprocess(
                image.convert("RGB")
            ).unsqueeze(0).to(device)

        captured.clear()

        with torch.enable_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            ):
                logits = model(image_tensor)

            assert logits.shape == (1, 5)
            assert torch.isfinite(logits).all().item()

            activation = captured["activation"]

            assert activation.shape == (1, 197, 1024)

            gradient = torch.autograd.grad(
                outputs=logits[0, target_grade],
                inputs=activation,
                retain_graph=False,
                create_graph=False
            )[0]

        spatial_activation = (
            activation[0, 1:, :]
            .detach()
            .float()
            .transpose(0, 1)
            .reshape(1024, 14, 14)
        )

        spatial_gradient = (
            gradient[0, 1:, :]
            .detach()
            .float()
            .transpose(0, 1)
            .reshape(1024, 14, 14)
        )

        channel_weights = spatial_gradient.mean(
            dim=(1, 2),
            keepdim=True
        )

        raw_cam = torch.relu(
            (channel_weights * spatial_activation).sum(dim=0)
        ).cpu().numpy()

        assert np.isfinite(raw_cam).all()

        raw_max = float(raw_cam.max())

        if raw_max <= 1e-12:
            print(f"{image_id}: DEGENERATE CAM")
            records.append({
                "image_id": image_id,
                "true_grade": target_grade,
                "degenerate": True,
                "cam_mae": np.nan,
                "cam_max_abs_error": np.nan,
                "cam_pearson": np.nan,
                "cam_cosine": np.nan
            })
        else:
            reproduced = cv2.resize(
                raw_cam,
                (224, 224),
                interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)

            reproduced /= float(reproduced.max())

            reference = reference_cams[image_id]

            assert np.isfinite(reproduced).all()
            assert reproduced.shape == reference.shape

            difference = reproduced - reference

            mae = float(np.mean(np.abs(difference)))
            max_error = float(np.max(np.abs(difference)))

            a = reproduced.astype(np.float64).ravel()
            b = reference.astype(np.float64).ravel()

            centered_a = a - a.mean()
            centered_b = b - b.mean()

            pearson_denominator = (
                np.linalg.norm(centered_a)
                * np.linalg.norm(centered_b)
            )

            cosine_denominator = (
                np.linalg.norm(a) * np.linalg.norm(b)
            )

            pearson = (
                float(
                    np.dot(centered_a, centered_b)
                    / pearson_denominator
                )
                if pearson_denominator > 1e-12
                else np.nan
            )

            cosine = (
                float(np.dot(a, b) / cosine_denominator)
                if cosine_denominator > 1e-12
                else np.nan
            )

            predicted_grade = int(
                logits[0].float().argmax().item()
            )

            records.append({
                "image_id": image_id,
                "true_grade": target_grade,
                "predicted_grade": predicted_grade,
                "degenerate": False,
                "cam_mae": mae,
                "cam_max_abs_error": max_error,
                "cam_pearson": pearson,
                "cam_cosine": cosine
            })

            print(
                f"{image_id}: "
                f"MAE={mae:.8f}, "
                f"Pearson={pearson:.6f}, "
                f"Cosine={cosine:.6f}, "
                f"MaxError={max_error:.6f}"
            )

        del image_tensor, logits, activation, gradient
        del spatial_activation, spatial_gradient
        del channel_weights, raw_cam
        captured.clear()
        torch.cuda.empty_cache()

finally:
    hook_handle.remove()
    captured.clear()
    del model
    gc.collect()
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# 7. Summarize reproducibility without hiding failures
# ------------------------------------------------------------

results = pd.DataFrame(records)

assert len(results) == 8

print("\nREPRODUCTION SUMMARY")
print("Images processed:", len(results))
print("Degenerate CAMs:", int(results["degenerate"].sum()))

valid = results.loc[~results["degenerate"]]

if not valid.empty:
    print(
        "Mean CAM MAE:",
        round(float(valid["cam_mae"].mean()), 8)
    )
    print(
        "Maximum CAM absolute error:",
        round(float(valid["cam_max_abs_error"].max()), 8)
    )
    print(
        "Mean Pearson:",
        round(float(valid["cam_pearson"].mean()), 6)
    )
    print(
        "Minimum Pearson:",
        round(float(valid["cam_pearson"].min()), 6)
    )
    print(
        "Mean cosine:",
        round(float(valid["cam_cosine"].mean()), 6)
    )

print("\nRESEARCH INTEGRITY")
print("Original checkpoints modified: False")
print("Original CAM files modified: False")
print("New output files created: False")
print("Validation images decoded:", len(results))
print("Parameters randomized: False")
print("Training performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.17 EXECUTION COMPLETE")
print("Reproduction acceptance: PENDING REVIEW")


# %% [cell 354] Experiment 10 - Step 10.17
# ============================================================
# Experiment 10 - Step 10.17
# RETFound Baseline Grad-CAM Reproduction Pilot
# Validation images only; no parameter randomization
# ============================================================

from pathlib import Path
import gc
import numpy as np
import pandas as pd
import cv2
import torch
import timm

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ============================================================
# 1. Paths and frozen pilot cohort
# ============================================================

ROOT = DR_RESEARCH

CHECKPOINT_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_model.pt"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

CAM_PATH = (
    ROOT / "experiment05_results"
    / "gradcam"
    / "RETFound_MA_CAM_Maps_V1.npz"
)

PILOT_IDS = [
    "1301_1",
    "0421_1",
    "1190_1",
    "0732_1",
    "0001_2",
    "0011_3",
    "0032_3",
    "0043_3"
]

EXPECTED_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

assert torch.cuda.is_available(), "CUDA is required."

DEVICE = torch.device("cuda")

for path in [
    CHECKPOINT_PATH,
    INDEX_PATH,
    COHORT_PATH,
    CAM_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

print("EXPERIMENT 10.17 - RETFOUND BASELINE CAM REPRODUCTION")

print("\nENVIRONMENT")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("timm:", timm.__version__)

# ============================================================
# 2. Verify frozen validation images
# ============================================================

index = pd.read_csv(
    INDEX_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str
    }
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276
assert len(PILOT_IDS) == 8
assert len(set(PILOT_IDS)) == 8

for image_id, expected_grade in zip(
    PILOT_IDS,
    EXPECTED_GRADES
):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index

    row = validation.loc[image_id]

    assert int(row["dr_grade"]) == expected_grade

    candidate_value = str(
        cohort_lookup.loc[
            image_id,
            "primary_ma_candidate"
        ]
    ).strip().lower()

    assert candidate_value in {"true", "1"}

assert (
    validation.loc[
        PILOT_IDS,
        "risk_group_id"
    ].nunique() == 8
)

print("\nVALIDATION COHORT")
print("Frozen validation images:", len(validation))
print("Pilot images:", len(PILOT_IDS))
print("Distinct source-risk groups:", 8)
print("Pilot cohort verification: PASS")

# ============================================================
# 3. Load saved reference CAMs
# ============================================================

with np.load(CAM_PATH, allow_pickle=False) as archive:

    assert "image_ids" in archive.files
    assert "retfound_cam" in archive.files

    saved_ids = [
        str(value)
        for value in archive["image_ids"]
    ]

    assert len(saved_ids) == 156
    assert len(set(saved_ids)) == 156

    id_to_position = {
        image_id: position
        for position, image_id in enumerate(saved_ids)
    }

    assert set(PILOT_IDS).issubset(
        set(id_to_position)
    )

    reference_cams = {}

    for image_id in PILOT_IDS:

        position = id_to_position[image_id]

        reference = archive[
            "retfound_cam"
        ][position].astype(np.float32).copy()

        assert reference.shape == (224, 224)
        assert np.isfinite(reference).all()

        reference_cams[image_id] = reference

print("\nREFERENCE CAMS")
print("Reference CAMs loaded:", len(reference_cams))
print("Reference integrity: PASS")

# ============================================================
# 4. Reconstruct original preprocessing
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("\nPREPROCESSING")
print("Image size: 224 x 224")
print("Resize interpolation: BICUBIC")
print("Normalization: ImageNet")
print("Preprocessing configured: PASS")

# ============================================================
# 5. Load original RETFound model
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

assert "model_state_dict" in checkpoint

source_state = checkpoint["model_state_dict"]

assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

load_result = model.load_state_dict(
    source_state,
    strict=True
)

assert len(load_result.missing_keys) == 0
assert len(load_result.unexpected_keys) == 0

model.eval()
model.requires_grad_(False)

assert not model.training

print("\nMODEL")
print("Checkpoint loaded: PASS")
print("Parameter tensors:", len(source_state))
print("Strict state loading: PASS")
print("Evaluation mode: PASS")
print("Model parameters frozen: PASS")

del source_state
del checkpoint

gc.collect()

# ============================================================
# 6. Register original Grad-CAM hook
# ============================================================

captured = {}

def activation_hook(module, inputs, output):

    activation = (
        output.detach()
        .requires_grad_(True)
    )

    captured["activation"] = activation

    return activation


hook_handle = (
    model.blocks[-1]
    .register_forward_hook(activation_hook)
)

print("\nGRADCAM CONFIGURATION")
print("Target layer: blocks.23")
print("Target class: ground-truth DR grade")
print("Patch tokens: 196")
print("Feature channels: 1024")
print("Precision: BF16")
print("Grad-CAM hook registered: PASS")

# ============================================================
# 7. Define CAM similarity calculations
# ============================================================

def calculate_similarity(reproduced, reference):

    assert reproduced.shape == reference.shape
    assert np.isfinite(reproduced).all()
    assert np.isfinite(reference).all()

    difference = (
        reproduced.astype(np.float64)
        - reference.astype(np.float64)
    )

    mae = float(
        np.mean(np.abs(difference))
    )

    max_error = float(
        np.max(np.abs(difference))
    )

    a = reproduced.astype(
        np.float64
    ).ravel()

    b = reference.astype(
        np.float64
    ).ravel()

    centered_a = a - a.mean()
    centered_b = b - b.mean()

    pearson_denominator = (
        np.linalg.norm(centered_a)
        * np.linalg.norm(centered_b)
    )

    cosine_denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if pearson_denominator > 1e-12:
        pearson = float(
            np.dot(centered_a, centered_b)
            / pearson_denominator
        )
    else:
        pearson = np.nan

    if cosine_denominator > 1e-12:
        cosine = float(
            np.dot(a, b)
            / cosine_denominator
        )
    else:
        cosine = np.nan

    return {
        "cam_mae": mae,
        "cam_max_abs_error": max_error,
        "cam_pearson": pearson,
        "cam_cosine": cosine
    }


# ============================================================
# 8. Reproduce Grad-CAM for 8 validation images
# ============================================================

records = []

try:

    for image_id, target_grade in zip(
        PILOT_IDS,
        EXPECTED_GRADES
    ):

        row = validation.loc[image_id]

        local_path = Path(row["local_path"])
        source_path = Path(row["source_path"])

        image_path = (
            local_path
            if local_path.is_file()
            else source_path
        )

        assert image_path.is_file(), (
            f"Missing image: {image_id}"
        )

        with Image.open(image_path) as image:

            image_tensor = preprocess(
                image.convert("RGB")
            ).unsqueeze(0).to(DEVICE)

        captured.clear()

        with torch.enable_grad():

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            ):

                logits = model(image_tensor)

            assert logits.shape == (1, 5)

            assert torch.isfinite(
                logits
            ).all().item()

            assert "activation" in captured

            activation = captured["activation"]

            assert activation.shape == (
                1, 197, 1024
            )

            gradient = torch.autograd.grad(
                outputs=logits[0, target_grade],
                inputs=activation,
                retain_graph=False,
                create_graph=False
            )[0]

        spatial_activation = (
            activation[0, 1:, :]
            .detach()
            .float()
            .transpose(0, 1)
            .reshape(1024, 14, 14)
        )

        spatial_gradient = (
            gradient[0, 1:, :]
            .detach()
            .float()
            .transpose(0, 1)
            .reshape(1024, 14, 14)
        )

        channel_weights = (
            spatial_gradient.mean(
                dim=(1, 2),
                keepdim=True
            )
        )

        raw_cam = torch.relu(
            (
                channel_weights
                * spatial_activation
            ).sum(dim=0)
        ).detach().cpu().numpy()

        assert np.isfinite(raw_cam).all()

        raw_max = float(raw_cam.max())

        predicted_grade = int(
            logits[0].detach()
            .float()
            .argmax()
            .item()
        )

        if raw_max <= 1e-12:

            record = {
                "image_id": image_id,
                "true_grade": target_grade,
                "predicted_grade": predicted_grade,
                "degenerate": True,
                "cam_mae": np.nan,
                "cam_max_abs_error": np.nan,
                "cam_pearson": np.nan,
                "cam_cosine": np.nan
            }

            print(
                f"{image_id}: DEGENERATE CAM"
            )

        else:

            reproduced = cv2.resize(
                raw_cam,
                (224, 224),
                interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)

            reproduced_max = float(
                reproduced.max()
            )

            assert reproduced_max > 1e-12

            reproduced /= reproduced_max

            reference = reference_cams[image_id]

            similarity = calculate_similarity(
                reproduced,
                reference
            )

            record = {
                "image_id": image_id,
                "true_grade": target_grade,
                "predicted_grade": predicted_grade,
                "degenerate": False,
                **similarity
            }

            print(
                f"{image_id}: "
                f"MAE={similarity['cam_mae']:.8f}, "
                f"Pearson={similarity['cam_pearson']:.6f}, "
                f"Cosine={similarity['cam_cosine']:.6f}, "
                f"MaxError={similarity['cam_max_abs_error']:.6f}"
            )

        records.append(record)

        del image_tensor
        del logits
        del activation
        del gradient
        del spatial_activation
        del spatial_gradient
        del channel_weights
        del raw_cam

        captured.clear()

        torch.cuda.empty_cache()

finally:

    hook_handle.remove()
    captured.clear()

    del model

    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 9. Reproduction summary
# ============================================================

results = pd.DataFrame(records)

assert len(results) == 8

print("\nREPRODUCTION SUMMARY")

print("Images processed:", len(results))

print(
    "Degenerate CAMs:",
    int(results["degenerate"].sum())
)

valid = results.loc[
    ~results["degenerate"]
].copy()

print("Valid CAMs:", len(valid))

if not valid.empty:

    print(
        "Mean CAM MAE:",
        round(
            float(valid["cam_mae"].mean()),
            8
        )
    )

    print(
        "Maximum CAM absolute error:",
        round(
            float(
                valid["cam_max_abs_error"].max()
            ),
            8
        )
    )

    print(
        "Mean Pearson:",
        round(
            float(
                valid["cam_pearson"].mean()
            ),
            6
        )
    )

    print(
        "Minimum Pearson:",
        round(
            float(
                valid["cam_pearson"].min()
            ),
            6
        )
    )

    print(
        "Mean cosine:",
        round(
            float(
                valid["cam_cosine"].mean()
            ),
            6
        )
    )

    print(
        "Minimum cosine:",
        round(
            float(
                valid["cam_cosine"].min()
            ),
            6
        )
    )

print("\nPER-IMAGE RESULTS")

print(
    results[
        [
            "image_id",
            "true_grade",
            "predicted_grade",
            "degenerate",
            "cam_mae",
            "cam_pearson",
            "cam_cosine",
            "cam_max_abs_error"
        ]
    ].to_string(index=False)
)

# ============================================================
# 10. Research integrity
# ============================================================

print("\nRESEARCH INTEGRITY")

print("Original checkpoint modified: False")
print("Original CAM files modified: False")
print("Research files created: False")
print("Validation images decoded:", len(results))
print("Parameters randomized: False")
print("Training performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.17 EXECUTION COMPLETE")
print("Reproduction acceptance: PENDING REVIEW")


# %% [cell 355] Experiment 10 - Step 10.18
# ============================================================
# Experiment 10 - Step 10.18
# Meta MAE Baseline Grad-CAM Reproduction Pilot
# Validation only; read-only research artifacts
# ============================================================

from pathlib import Path
import gc
import hashlib
import numpy as np
import pandas as pd
import cv2
import torch
import timm

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = DR_RESEARCH

CHECKPOINT_PATH = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1/best_model.pt"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

CAM_PATH = (
    ROOT / "experiment07_results/gradcam"
    / "Meta_MAE_MA_CAM_Maps_V1.npz"
)

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

EXPECTED_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

assert torch.cuda.is_available()

for path in [CHECKPOINT_PATH, INDEX_PATH, COHORT_PATH, CAM_PATH]:
    assert path.is_file(), f"Missing file: {path}"

device = torch.device("cuda")

print("EXPERIMENT 10.18 - META MAE BASELINE CAM REPRODUCTION")

print("\nENVIRONMENT")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("timm:", timm.__version__)

# ------------------------------------------------------------
# 1. Validate original cohort and grades
# ------------------------------------------------------------

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276

for image_id, grade in zip(PILOT_IDS, EXPECTED_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    flag = str(
        cohort_lookup.loc[image_id, "primary_ma_candidate"]
    ).strip().lower()

    assert flag in {"true", "1"}

assert validation.loc[
    PILOT_IDS, "risk_group_id"
].nunique() == 8

print("\nPILOT COHORT")
print("Validation images:", len(validation))
print("Pilot images:", len(PILOT_IDS))
print("Distinct source-risk groups:", 8)
print("Cohort verification: PASS")

# ------------------------------------------------------------
# 2. Load original Meta MAE CAM reference
# ------------------------------------------------------------

with np.load(CAM_PATH, allow_pickle=False) as archive:
    image_ids = [
        str(value) for value in archive["image_ids"]
    ]

    assert len(image_ids) == 156
    assert len(set(image_ids)) == 156

    position_lookup = {
        value: i for i, value in enumerate(image_ids)
    }

    assert set(PILOT_IDS).issubset(position_lookup)

    references = {
        image_id: archive["meta_mae_cam"][
            position_lookup[image_id]
        ].astype(np.float32).copy()
        for image_id in PILOT_IDS
    }

for image_id, cam in references.items():
    assert cam.shape == (224, 224)
    assert np.isfinite(cam).all()

print("\nREFERENCE CAMS")
print("Matched reference CAMs:", len(references))
print("Reference integrity: PASS")

# ------------------------------------------------------------
# 3. Match original preprocessing
# ------------------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------------------
# 4. Load frozen Meta MAE checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

assert len(checkpoint["model_state_dict"]) == 296
assert checkpoint.get("epoch") == 24

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=device)

load_result = model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

del checkpoint
gc.collect()

print("\nMODEL")
print("Checkpoint epoch: 24")
print("Strict checkpoint loading: PASS")
print("Evaluation mode: PASS")
print("Parameters frozen: PASS")

# ------------------------------------------------------------
# 5. Exact original activation hook
# ------------------------------------------------------------

captured = {}

def activation_hook(module, inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation

hook = model.blocks[-1].register_forward_hook(
    activation_hook
)

# ------------------------------------------------------------
# 6. Spatial CAM similarity
# ------------------------------------------------------------

def compare_cams(current, reference):
    a = current.astype(np.float64).ravel()
    b = reference.astype(np.float64).ravel()

    mae = float(np.mean(np.abs(a - b)))
    max_error = float(np.max(np.abs(a - b)))

    a_centered = a - a.mean()
    b_centered = b - b.mean()

    pearson_den = (
        np.linalg.norm(a_centered)
        * np.linalg.norm(b_centered)
    )

    cosine_den = np.linalg.norm(a) * np.linalg.norm(b)

    pearson = (
        float(np.dot(a_centered, b_centered) / pearson_den)
        if pearson_den > 1e-12 else np.nan
    )

    cosine = (
        float(np.dot(a, b) / cosine_den)
        if cosine_den > 1e-12 else np.nan
    )

    return mae, max_error, pearson, cosine

# ------------------------------------------------------------
# 7. Reproduce Grad-CAM
# ------------------------------------------------------------

records = []

try:
    for image_id, grade in zip(PILOT_IDS, EXPECTED_GRADES):
        row = validation.loc[image_id]

        local_path = Path(row["local_path"])
        source_path = Path(row["source_path"])

        image_path = (
            local_path if local_path.is_file()
            else source_path
        )

        assert image_path.is_file()

        with Image.open(image_path) as image:
            input_tensor = preprocess(
                image.convert("RGB")
            ).unsqueeze(0).to(device)

        captured.clear()

        with torch.enable_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            ):
                logits = model(input_tensor)

            assert logits.shape == (1, 5)
            assert torch.isfinite(logits).all().item()

            activation = captured["activation"]
            assert activation.shape == (1, 197, 1024)

            gradient = torch.autograd.grad(
                outputs=logits[0, grade],
                inputs=activation,
                retain_graph=False,
                create_graph=False
            )[0]

        a = (
            activation[0, 1:, :]
            .detach().float()
            .transpose(0, 1)
            .reshape(1024, 14, 14)
        )

        g = (
            gradient[0, 1:, :]
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
        ).detach().cpu().numpy()

        assert np.isfinite(raw_cam).all()

        raw_max = float(raw_cam.max())

        predicted = int(
            logits[0].detach().float().argmax().item()
        )

        if raw_max <= 1e-12:
            records.append({
                "image_id": image_id,
                "true_grade": grade,
                "predicted_grade": predicted,
                "degenerate": True,
                "cam_mae": np.nan,
                "cam_max_abs_error": np.nan,
                "cam_pearson": np.nan,
                "cam_cosine": np.nan
            })

            print(f"{image_id}: DEGENERATE CAM")

        else:
            cam = cv2.resize(
                raw_cam,
                (224, 224),
                interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)

            assert float(cam.max()) > 1e-12
            cam /= float(cam.max())

            mae, max_error, pearson, cosine = compare_cams(
                cam,
                references[image_id]
            )

            records.append({
                "image_id": image_id,
                "true_grade": grade,
                "predicted_grade": predicted,
                "degenerate": False,
                "cam_mae": mae,
                "cam_max_abs_error": max_error,
                "cam_pearson": pearson,
                "cam_cosine": cosine
            })

            print(
                f"{image_id}: "
                f"MAE={mae:.8f}, "
                f"Pearson={pearson:.6f}, "
                f"Cosine={cosine:.6f}, "
                f"MaxError={max_error:.6f}"
            )

        del input_tensor, logits, activation, gradient
        del a, g, weights, raw_cam

        captured.clear()
        torch.cuda.empty_cache()

finally:
    hook.remove()
    captured.clear()

    del model

    gc.collect()
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# 8. Report all observations
# ------------------------------------------------------------

results = pd.DataFrame(records)

assert len(results) == 8

valid = results.loc[~results["degenerate"]]

print("\nREPRODUCTION SUMMARY")
print("Images processed:", len(results))
print("Valid CAMs:", len(valid))
print("Degenerate CAMs:", int(results["degenerate"].sum()))

if not valid.empty:
    print("Mean CAM MAE:", round(float(valid["cam_mae"].mean()), 8))
    print(
        "Maximum CAM absolute error:",
        round(float(valid["cam_max_abs_error"].max()), 8)
    )
    print(
        "Mean Pearson:",
        round(float(valid["cam_pearson"].mean()), 6)
    )
    print(
        "Minimum Pearson:",
        round(float(valid["cam_pearson"].min()), 6)
    )
    print(
        "Mean cosine:",
        round(float(valid["cam_cosine"].mean()), 6)
    )

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Original CAM files modified: False")
print("New research files created: False")
print("Validation images decoded:", len(results))
print("Parameters randomized: False")
print("Test images accessed: 0")

print("\nSTEP 10.18 EXECUTION COMPLETE")
print("Reproduction acceptance: PENDING REVIEW")


# %% [cell 356] Experiment 10 - Step 10.19
# ============================================================
# Experiment 10 - Step 10.19
# RETFound R1 Head Reinitialization Pilot
# Seed 1042; eight frozen validation images; read-only files
# ============================================================

from pathlib import Path
import gc
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import timm

from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.layers import trunc_normal_

ROOT = DR_RESEARCH

CHECKPOINT_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1/best_model.pt"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

CAM_PATH = (
    ROOT / "experiment05_results/gradcam"
    / "RETFound_MA_CAM_Maps_V1.npz"
)

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available()

print("EXPERIMENT 10.19 - RETFOUND R1 PILOT")
print("Model: RETFound")
print("Randomization stage: R1")
print("Randomization seed:", SEED)

# ------------------------------------------------------------
# 1. Verify validation cohort and reference CAMs
# ------------------------------------------------------------

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    flag = str(
        cohort_lookup.loc[image_id, "primary_ma_candidate"]
    ).strip().lower()

    assert flag in {"true", "1"}

assert validation.loc[
    PILOT_IDS, "risk_group_id"
].nunique() == 8

with np.load(CAM_PATH, allow_pickle=False) as archive:
    positions = {
        str(image_id): i
        for i, image_id in enumerate(archive["image_ids"])
    }

    assert len(positions) == 156

    reference_cams = {
        image_id: archive["retfound_cam"][
            positions[image_id]
        ].astype(np.float32).copy()
        for image_id in PILOT_IDS
    }

print("\nCOHORT INTEGRITY")
print("Pilot validation images:", len(PILOT_IDS))
print("Distinct source-risk groups:", 8)
print("Original CAM references: PASS")

# ------------------------------------------------------------
# 2. Image preprocessing
# ------------------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------------------
# 3. Load independent trained model
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]

assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

load_result = model.load_state_dict(
    source_state,
    strict=True
)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

original_head_weight = model.head.weight.detach().clone()
original_head_bias = model.head.bias.detach().clone()

del checkpoint, source_state
gc.collect()

# ------------------------------------------------------------
# 4. Original Grad-CAM function
# ------------------------------------------------------------

captured = {}

def activation_hook(module, inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation

hook = model.blocks[-1].register_forward_hook(
    activation_hook
)

def compute_cam(input_tensor, target_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(input_tensor)

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]

        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, target_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    a = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    g = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = g.mean(dim=(1, 2), keepdim=True)

    raw_cam = torch.relu(
        (weights * a).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw_cam).all()

    raw_max = float(raw_cam.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw_cam,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12
        cam /= float(cam.max())

        assert np.isfinite(cam).all()

    captured.clear()

    return output_logits, cam, raw_max

# ------------------------------------------------------------
# 5. Compare two valid CAMs
# ------------------------------------------------------------

def compare_cams(first, second):
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    a_centered = a - a.mean()
    b_centered = b - b.mean()

    denominator = (
        np.linalg.norm(a_centered)
        * np.linalg.norm(b_centered)
    )

    pearson = (
        float(np.dot(a_centered, b_centered) / denominator)
        if denominator > 1e-12 else np.nan
    )

    spearman = float(
        spearmanr(a, b).statistic
    )

    ssim = float(
        structural_similarity(
            first,
            second,
            data_range=1.0
        )
    )

    mae = float(np.mean(np.abs(a - b)))

    return {
        "pearson": pearson,
        "spearman": spearman,
        "ssim": ssim,
        "mae": mae
    }

# ------------------------------------------------------------
# 6. Process original and R1 models
# ------------------------------------------------------------

records = []

try:
    for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
        row = validation.loc[image_id]

        local = Path(row["local_path"])
        source = Path(row["source_path"])

        image_path = local if local.is_file() else source
        assert image_path.is_file()

        with Image.open(image_path) as image:
            input_tensor = preprocess(
                image.convert("RGB")
            ).unsqueeze(0).to(DEVICE)

        # Restore exact original classification head.
        with torch.no_grad():
            model.head.weight.copy_(original_head_weight)
            model.head.bias.copy_(original_head_bias)

        base_logits, base_cam, base_raw_max = compute_cam(
            input_tensor, grade
        )

        assert base_cam is not None, (
            f"Baseline CAM degenerate: {image_id}"
        )

        reference_metrics = compare_cams(
            base_cam,
            reference_cams[image_id]
        )

        # Reset seed for every image so R1 uses the same
        # randomized head for all eight pilot images.
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        with torch.no_grad():
            trunc_normal_(model.head.weight, std=0.02)
            nn.init.zeros_(model.head.bias)

        randomized_logits, randomized_cam, random_raw_max = (
            compute_cam(input_tensor, grade)
        )

        logit_l2 = float(
            np.linalg.norm(
                randomized_logits - base_logits
            )
        )

        probability_base = torch.softmax(
            torch.from_numpy(base_logits), dim=0
        ).numpy()

        probability_random = torch.softmax(
            torch.from_numpy(randomized_logits), dim=0
        ).numpy()

        probability_l1 = float(
            np.abs(
                probability_random - probability_base
            ).sum()
        )

        degenerate = randomized_cam is None

        if degenerate:
            similarity = {
                "pearson": np.nan,
                "spearman": np.nan,
                "ssim": np.nan,
                "mae": np.nan
            }
        else:
            similarity = compare_cams(
                base_cam, randomized_cam
            )

        record = {
            "image_id": image_id,
            "true_grade": grade,
            "baseline_pred": int(base_logits.argmax()),
            "r1_pred": int(randomized_logits.argmax()),
            "baseline_ref_pearson": reference_metrics["pearson"],
            "logit_l2": logit_l2,
            "probability_l1": probability_l1,
            "r1_degenerate": degenerate,
            "baseline_raw_max": base_raw_max,
            "r1_raw_max": random_raw_max,
            "cam_pearson": similarity["pearson"],
            "cam_spearman": similarity["spearman"],
            "cam_ssim": similarity["ssim"],
            "cam_mae": similarity["mae"]
        }

        records.append(record)

        print(
            f"{image_id}: "
            f"R1Degenerate={degenerate}, "
            f"LogitL2={logit_l2:.5f}, "
            f"Pearson={similarity['pearson']:.6f}, "
            f"Spearman={similarity['spearman']:.6f}, "
            f"SSIM={similarity['ssim']:.6f}"
        )

        del input_tensor
        torch.cuda.empty_cache()

finally:
    hook.remove()
    captured.clear()
    del model, original_head_weight, original_head_bias
    gc.collect()
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# 7. Pilot diagnostics
# ------------------------------------------------------------

results = pd.DataFrame(records)

assert len(results) == 8
assert results["image_id"].is_unique

print("\nPILOT SUMMARY")
print("Images:", len(results))
print("Randomization stage: R1")
print("Seed:", SEED)
print(
    "Degenerate randomized CAMs:",
    int(results["r1_degenerate"].sum())
)
print(
    "Minimum baseline-reference Pearson:",
    round(float(results["baseline_ref_pearson"].min()), 6)
)
print(
    "Mean logit L2 change:",
    round(float(results["logit_l2"].mean()), 6)
)
print(
    "Mean probability L1 change:",
    round(float(results["probability_l1"].mean()), 6)
)

valid = results.loc[~results["r1_degenerate"]]

if len(valid):
    for column in [
        "cam_pearson",
        "cam_spearman",
        "cam_ssim",
        "cam_mae"
    ]:
        print(
            f"Mean {column}:",
            round(float(valid[column].mean()), 6)
        )

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Original CAM files modified: False")
print("Output files created: False")
print("Randomization: Head only")
print("Validation images decoded:", len(results))
print("Test images accessed: 0")

print("\nSTEP 10.19 EXECUTION COMPLETE")
print("Pilot interpretation: PENDING REVIEW")


# %% [cell 357] Experiment 10 - Step 10.20
# ============================================================
# Experiment 10 - Step 10.20
# Meta MAE R1 Classification Head Randomization Pilot
# Eight validation images; seed 1042; no file writes
# ============================================================

from pathlib import Path
import gc
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import timm

from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.layers import trunc_normal_

ROOT = DR_RESEARCH

CHECKPOINT_PATH = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1/best_model.pt"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

CAM_PATH = (
    ROOT / "experiment07_results/gradcam"
    / "Meta_MAE_MA_CAM_Maps_V1.npz"
)

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available()

for path in [
    CHECKPOINT_PATH,
    INDEX_PATH,
    COHORT_PATH,
    CAM_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

print("EXPERIMENT 10.20 - META MAE R1 PILOT")
print("Model: Meta MAE")
print("Randomization stage: R1")
print("Randomization seed:", SEED)

# ------------------------------------------------------------
# 1. Verify frozen validation cohort
# ------------------------------------------------------------

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276
assert len(PILOT_IDS) == 8
assert len(set(PILOT_IDS)) == 8

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    flag = str(
        cohort_lookup.loc[image_id, "primary_ma_candidate"]
    ).strip().lower()

    assert flag in {"true", "1"}

assert (
    validation.loc[PILOT_IDS, "risk_group_id"].nunique()
    == 8
)

with np.load(CAM_PATH, allow_pickle=False) as archive:
    assert "meta_mae_cam" in archive.files
    assert "image_ids" in archive.files

    positions = {
        str(image_id): i
        for i, image_id in enumerate(archive["image_ids"])
    }

    assert len(positions) == 156

    references = {
        image_id: archive["meta_mae_cam"][
            positions[image_id]
        ].astype(np.float32).copy()
        for image_id in PILOT_IDS
    }

print("\nCOHORT INTEGRITY")
print("Pilot validation images:", len(PILOT_IDS))
print("Distinct source-risk groups:", 8)
print("Original CAM references: PASS")

# ------------------------------------------------------------
# 2. Original image preprocessing
# ------------------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------------------
# 3. Load independent Meta MAE model
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

assert checkpoint["epoch"] == 24

source_state = checkpoint["model_state_dict"]

assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

result = model.load_state_dict(
    source_state,
    strict=True
)

assert not result.missing_keys
assert not result.unexpected_keys

model.eval()
model.requires_grad_(False)

original_head_weight = model.head.weight.detach().clone()
original_head_bias = model.head.bias.detach().clone()

del checkpoint, source_state
gc.collect()

print("\nMODEL")
print("Checkpoint epoch: 24")
print("Strict loading: PASS")
print("Original Head snapshot: PASS")

# ------------------------------------------------------------
# 4. Original Grad-CAM computation
# ------------------------------------------------------------

captured = {}

def activation_hook(module, inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation

hook = model.blocks[-1].register_forward_hook(
    activation_hook
)

def compute_cam(input_tensor, target_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(input_tensor)

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, target_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    a = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    g = (
        gradient[0, 1:, :]
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
    ).detach().cpu().numpy()

    assert np.isfinite(raw_cam).all()

    raw_max = float(raw_cam.max())

    output_logits = (
        logits[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw_cam,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12

        cam /= float(cam.max())

        assert np.isfinite(cam).all()

    captured.clear()

    return output_logits, cam, raw_max

# ------------------------------------------------------------
# 5. CAM similarity metrics
# ------------------------------------------------------------

def compare_cams(first, second):
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    centered_a = a - a.mean()
    centered_b = b - b.mean()

    denominator = (
        np.linalg.norm(centered_a)
        * np.linalg.norm(centered_b)
    )

    pearson = (
        float(
            np.dot(centered_a, centered_b)
            / denominator
        )
        if denominator > 1e-12
        else np.nan
    )

    spearman_result = spearmanr(a, b).statistic
    spearman = float(spearman_result)

    ssim = float(
        structural_similarity(
            first,
            second,
            data_range=1.0
        )
    )

    mae = float(np.mean(np.abs(a - b)))

    return {
        "pearson": pearson,
        "spearman": spearman,
        "ssim": ssim,
        "mae": mae
    }

# ------------------------------------------------------------
# 6. Apply one fixed R1 head across all images
# ------------------------------------------------------------

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

with torch.no_grad():
    trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)

random_head_weight = model.head.weight.detach().clone()
random_head_bias = model.head.bias.detach().clone()

assert not torch.equal(
    random_head_weight,
    original_head_weight
)

print("\nR1 PARAMETER CHECK")
print("Head weight reinitialized: PASS")
print("Head seed:", SEED)
print("Shared randomized Head across all images: PASS")

records = []

try:
    for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
        row = validation.loc[image_id]

        local_path = Path(row["local_path"])
        source_path = Path(row["source_path"])

        image_path = (
            local_path
            if local_path.is_file()
            else source_path
        )

        assert image_path.is_file()

        with Image.open(image_path) as image:
            input_tensor = preprocess(
                image.convert("RGB")
            ).unsqueeze(0).to(DEVICE)

        # Restore original Head to compute baseline.
        with torch.no_grad():
            model.head.weight.copy_(original_head_weight)
            model.head.bias.copy_(original_head_bias)

        base_logits, base_cam, base_raw_max = compute_cam(
            input_tensor, grade
        )

        assert base_cam is not None

        reference_metrics = compare_cams(
            base_cam,
            references[image_id]
        )

        # Apply the same precomputed randomized Head.
        with torch.no_grad():
            model.head.weight.copy_(random_head_weight)
            model.head.bias.copy_(random_head_bias)

        random_logits, random_cam, random_raw_max = compute_cam(
            input_tensor, grade
        )

        logit_l2 = float(
            np.linalg.norm(
                random_logits - base_logits
            )
        )

        baseline_probabilities = torch.softmax(
            torch.from_numpy(base_logits), dim=0
        ).numpy()

        random_probabilities = torch.softmax(
            torch.from_numpy(random_logits), dim=0
        ).numpy()

        probability_l1 = float(
            np.abs(
                random_probabilities
                - baseline_probabilities
            ).sum()
        )

        degenerate = random_cam is None

        if degenerate:
            similarity = {
                "pearson": np.nan,
                "spearman": np.nan,
                "ssim": np.nan,
                "mae": np.nan
            }
        else:
            similarity = compare_cams(
                base_cam, random_cam
            )

        record = {
            "image_id": image_id,
            "true_grade": grade,
            "baseline_pred": int(base_logits.argmax()),
            "r1_pred": int(random_logits.argmax()),
            "baseline_ref_pearson": reference_metrics["pearson"],
            "logit_l2": logit_l2,
            "probability_l1": probability_l1,
            "r1_degenerate": degenerate,
            "baseline_raw_max": base_raw_max,
            "r1_raw_max": random_raw_max,
            "cam_pearson": similarity["pearson"],
            "cam_spearman": similarity["spearman"],
            "cam_ssim": similarity["ssim"],
            "cam_mae": similarity["mae"]
        }

        records.append(record)

        print(
            f"{image_id}: "
            f"R1Degenerate={degenerate}, "
            f"LogitL2={logit_l2:.5f}, "
            f"Pearson={similarity['pearson']:.6f}, "
            f"Spearman={similarity['spearman']:.6f}, "
            f"SSIM={similarity['ssim']:.6f}"
        )

        del input_tensor
        torch.cuda.empty_cache()

finally:
    hook.remove()
    captured.clear()

    del model
    del original_head_weight, original_head_bias
    del random_head_weight, random_head_bias

    gc.collect()
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# 7. Report all pilot observations
# ------------------------------------------------------------

results = pd.DataFrame(records)

assert len(results) == 8
assert results["image_id"].is_unique

valid = results.loc[
    ~results["r1_degenerate"]
]

print("\nPILOT SUMMARY")
print("Images:", len(results))
print("Randomization stage: R1")
print("Seed:", SEED)
print(
    "Degenerate randomized CAMs:",
    int(results["r1_degenerate"].sum())
)
print(
    "Minimum baseline-reference Pearson:",
    round(
        float(results["baseline_ref_pearson"].min()),
        6
    )
)
print(
    "Mean logit L2 change:",
    round(float(results["logit_l2"].mean()), 6)
)
print(
    "Mean probability L1 change:",
    round(float(results["probability_l1"].mean()), 6)
)

if not valid.empty:
    for column in [
        "cam_pearson",
        "cam_spearman",
        "cam_ssim",
        "cam_mae"
    ]:
        print(
            f"Mean {column}:",
            round(float(valid[column].mean()), 6)
        )

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Original CAM files modified: False")
print("New output files created: False")
print("Randomization: Head only")
print("Validation images decoded:", len(results))
print("Test images accessed: 0")

print("\nSTEP 10.20 EXECUTION COMPLETE")
print("Pilot interpretation: PENDING REVIEW")


# %% [cell 358] Experiment 10 - Step 10.21
# ============================================================
# Experiment 10 - Step 10.21
# RETFound R1-to-R2 Cumulative Grad-CAM Pilot
# Validation only; no checkpoint or artifact writes
# ============================================================

from pathlib import Path
import gc
import hashlib
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import timm

from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.layers import trunc_normal_

ROOT = DR_RESEARCH

CHECKPOINT_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1/best_model.pt"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

PROTOCOL_PATH = (
    ROOT / "experiment10_results"
    / "Experiment10_Randomization_Protocol_V1.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
)

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available()

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

assert sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA256

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert protocol["randomization"]["stages"]["R1"] == ["head"]
assert set(protocol["randomization"]["stages"]["R2"]) == {
    "head", "fc_norm", "blocks.23"
}
assert SEED in protocol["randomization"]["randomization_seeds"]

print("EXPERIMENT 10.21 - RETFOUND R1 TO R2 PILOT")
print("Seed:", SEED)
print("Protocol integrity: PASS")

# ------------------------------------------------------------
# 1. Verify the exact pilot cohort
# ------------------------------------------------------------

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    flag = str(
        cohort_lookup.loc[image_id, "primary_ma_candidate"]
    ).strip().lower()

    assert flag in {"true", "1"}

assert validation.loc[
    PILOT_IDS, "risk_group_id"
].nunique() == 8

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------------------
# 2. Load trained RETFound model
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

result = model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

assert not result.missing_keys
assert not result.unexpected_keys

model.eval()
model.requires_grad_(False)

del checkpoint
gc.collect()

# ------------------------------------------------------------
# 3. Grad-CAM function
# ------------------------------------------------------------

captured = {}

def activation_hook(module, inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation

hook = model.blocks[-1].register_forward_hook(
    activation_hook
)

def calculate_cam(tensor, grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(tensor)

        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]

        gradient = torch.autograd.grad(
            logits[0, grade],
            activation,
            retain_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2),
        keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12
        cam /= float(cam.max())

    captured.clear()

    return output_logits, cam, raw_max

def similarity(first, second):
    if first is None or second is None:
        return {
            "spearman": np.nan,
            "ssim": np.nan,
            "pearson": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    center_a = a - a.mean()
    center_b = b - b.mean()

    denominator = (
        np.linalg.norm(center_a)
        * np.linalg.norm(center_b)
    )

    pearson = (
        float(np.dot(center_a, center_b) / denominator)
        if denominator > 1e-12 else np.nan
    )

    spearman = float(spearmanr(a, b).statistic)

    ssim = float(
        structural_similarity(
            first,
            second,
            data_range=1.0
        )
    )

    return {
        "spearman": spearman,
        "ssim": ssim,
        "pearson": pearson
    }

# ------------------------------------------------------------
# 4. Produce R0 and R1 on the same frozen inputs
# ------------------------------------------------------------

inputs = []

for image_id in PILOT_IDS:
    row = validation.loc[image_id]
    local = Path(row["local_path"])
    source = Path(row["source_path"])
    image_path = local if local.is_file() else source

    assert image_path.is_file()

    with Image.open(image_path) as image:
        inputs.append(
            preprocess(image.convert("RGB")).unsqueeze(0)
        )

outputs = {"R0": [], "R1": [], "R2": []}

def evaluate_stage(stage):
    for tensor, grade in zip(inputs, PILOT_GRADES):
        logits, cam, raw_max = calculate_cam(
            tensor.to(DEVICE),
            grade
        )
        outputs[stage].append((logits, cam, raw_max))

try:
    evaluate_stage("R0")

    # R1: reinitialize classification head once.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    r1_head = {
        name: parameter.detach().clone()
        for name, parameter in model.head.named_parameters()
    }

    evaluate_stage("R1")

    # R2: retain R1 head, add fc_norm and Block 23.
    with torch.no_grad():
        model.fc_norm.reset_parameters()

        for module in model.blocks[23].modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()

    for name, parameter in model.head.named_parameters():
        assert torch.equal(parameter, r1_head[name])

    assert torch.equal(
        model.fc_norm.weight,
        torch.ones_like(model.fc_norm.weight)
    )

    assert torch.equal(
        model.fc_norm.bias,
        torch.zeros_like(model.fc_norm.bias)
    )

    print("\nCUMULATIVE PARAMETER CHECK")
    print("R1 Head retained in R2: PASS")
    print("Block 23 reinitialization executed: PASS")
    print("fc_norm reset: PASS")

    evaluate_stage("R2")

finally:
    hook.remove()
    captured.clear()

    del model
    gc.collect()
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# 5. Compare R0, R1 and R2
# ------------------------------------------------------------

records = []

for i, image_id in enumerate(PILOT_IDS):
    base_logits, base_cam, _ = outputs["R0"][i]
    r1_logits, r1_cam, _ = outputs["R1"][i]
    r2_logits, r2_cam, r2_raw_max = outputs["R2"][i]

    assert base_cam is not None

    r0_r1 = similarity(base_cam, r1_cam)
    r0_r2 = similarity(base_cam, r2_cam)
    r1_r2 = similarity(r1_cam, r2_cam)

    record = {
        "image_id": image_id,
        "r1_degenerate": r1_cam is None,
        "r2_degenerate": r2_cam is None,
        "r0_r1_spearman": r0_r1["spearman"],
        "r0_r2_spearman": r0_r2["spearman"],
        "r1_r2_spearman": r1_r2["spearman"],
        "r0_r1_ssim": r0_r1["ssim"],
        "r0_r2_ssim": r0_r2["ssim"],
        "r1_r2_ssim": r1_r2["ssim"],
        "r0_r1_pearson": r0_r1["pearson"],
        "r0_r2_pearson": r0_r2["pearson"],
        "r1_r2_pearson": r1_r2["pearson"],
        "r2_raw_max": r2_raw_max,
        "r0_r2_logit_l2": float(
            np.linalg.norm(r2_logits - base_logits)
        )
    }

    records.append(record)

results = pd.DataFrame(records)

print("\nPILOT SUMMARY")
print("Images:", len(results))
print("Distinct source-risk groups:", 8)
print("Seed:", SEED)
print("R1 degenerate CAMs:", int(results["r1_degenerate"].sum()))
print("R2 degenerate CAMs:", int(results["r2_degenerate"].sum()))

for column in [
    "r0_r1_spearman",
    "r0_r2_spearman",
    "r1_r2_spearman",
    "r0_r1_ssim",
    "r0_r2_ssim",
    "r1_r2_ssim",
    "r0_r1_pearson",
    "r0_r2_pearson",
    "r1_r2_pearson"
]:
    print(
        f"Mean {column}:",
        round(float(results[column].mean()), 6)
    )

print(
    "Mean R0-R2 logit L2:",
    round(float(results["r0_r2_logit_l2"].mean()), 6)
)

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Randomization: Cumulative R1 to R2")
print("Output files created: False")
print("Validation images decoded:", len(inputs))
print("Test images accessed: 0")

print("\nSTEP 10.21 EXECUTION COMPLETE")
print("Pilot interpretation: PENDING REVIEW")


# %% [cell 359] Experiment 10 - Step 10.22
# ============================================================
# Experiment 10 - Step 10.22
# Meta MAE R1-to-R2 Cumulative Grad-CAM Pilot
# Validation only; no research artifact writes
# ============================================================

from pathlib import Path
import gc
import hashlib
import json

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import timm

from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.layers import trunc_normal_

ROOT = DR_RESEARCH

CHECKPOINT_PATH = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1/best_model.pt"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

PROTOCOL_PATH = (
    ROOT / "experiment10_results"
    / "Experiment10_Randomization_Protocol_V1.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
)

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available()

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

assert sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA256

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert protocol["randomization"]["stages"]["R1"] == ["head"]
assert set(protocol["randomization"]["stages"]["R2"]) == {
    "head", "fc_norm", "blocks.23"
}
assert SEED in protocol["randomization"]["randomization_seeds"]

print("EXPERIMENT 10.22 - META MAE R1 TO R2 PILOT")
print("Seed:", SEED)
print("Protocol integrity: PASS")

# ============================================================
# 1. Verify frozen validation cohort
# ============================================================

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    flag = str(
        cohort_lookup.loc[
            image_id, "primary_ma_candidate"
        ]
    ).strip().lower()

    assert flag in {"true", "1"}

assert (
    validation.loc[PILOT_IDS, "risk_group_id"].nunique() == 8
)

print("\nCOHORT")
print("Pilot images:", len(PILOT_IDS))
print("Distinct source-risk groups:", 8)
print("Validation-only cohort: PASS")

# ============================================================
# 2. Prepare the original image preprocessing
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

inputs = []

for image_id in PILOT_IDS:
    row = validation.loc[image_id]

    local_path = Path(row["local_path"])
    source_path = Path(row["source_path"])

    image_path = (
        local_path if local_path.is_file()
        else source_path
    )

    assert image_path.is_file()

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    inputs.append(tensor)

assert len(inputs) == 8

# ============================================================
# 3. Load independent Meta MAE model
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

assert checkpoint["epoch"] == 24
assert len(checkpoint["model_state_dict"]) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

load_result = model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

del checkpoint
gc.collect()

print("\nMODEL")
print("Meta MAE checkpoint epoch: 24")
print("Strict checkpoint loading: PASS")
print("Parameters frozen: PASS")

# ============================================================
# 4. Original Grad-CAM method
# ============================================================

captured = {}

def activation_hook(module, inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation

hook = model.blocks[-1].register_forward_hook(
    activation_hook
)

def calculate_cam(tensor, grade):

    captured.clear()

    with torch.enable_grad():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(tensor)

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]

        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2),
        keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12
        cam /= float(cam.max())

        assert np.isfinite(cam).all()

    captured.clear()

    return output_logits, cam, raw_max

# ============================================================
# 5. Spatial similarity metrics
# ============================================================

def similarity(first, second):

    if first is None or second is None:
        return {
            "spearman": np.nan,
            "ssim": np.nan,
            "pearson": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    centered_a = a - a.mean()
    centered_b = b - b.mean()

    denominator = (
        np.linalg.norm(centered_a)
        * np.linalg.norm(centered_b)
    )

    pearson = (
        float(
            np.dot(centered_a, centered_b)
            / denominator
        )
        if denominator > 1e-12
        else np.nan
    )

    spearman = float(spearmanr(a, b).statistic)

    ssim = float(
        structural_similarity(
            first,
            second,
            data_range=1.0
        )
    )

    return {
        "spearman": spearman,
        "ssim": ssim,
        "pearson": pearson
    }

# ============================================================
# 6. Evaluate R0, R1, and R2 sequentially
# ============================================================

outputs = {
    "R0": [],
    "R1": [],
    "R2": []
}

def evaluate_stage(stage_name):

    for tensor, grade in zip(inputs, PILOT_GRADES):
        logits, cam, raw_max = calculate_cam(
            tensor.to(DEVICE),
            grade
        )

        outputs[stage_name].append(
            (logits, cam, raw_max)
        )

try:

    # R0: Original trained model.
    evaluate_stage("R0")

    # R1: Reinitialize the classification head.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(
            model.head.weight,
            std=0.02
        )
        nn.init.zeros_(model.head.bias)

    r1_head = {
        name: parameter.detach().clone()
        for name, parameter in model.head.named_parameters()
    }

    evaluate_stage("R1")

    # R2: Retain the R1 head and additionally
    # reinitialize fc_norm and Transformer Block 23.
    with torch.no_grad():

        model.fc_norm.reset_parameters()

        for module in model.blocks[23].modules():

            if isinstance(module, nn.Linear):
                trunc_normal_(
                    module.weight,
                    std=0.02
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()

    for name, parameter in model.head.named_parameters():
        assert torch.equal(
            parameter,
            r1_head[name]
        )

    assert torch.equal(
        model.fc_norm.weight,
        torch.ones_like(model.fc_norm.weight)
    )

    assert torch.equal(
        model.fc_norm.bias,
        torch.zeros_like(model.fc_norm.bias)
    )

    print("\nCUMULATIVE PARAMETER CHECK")
    print("R1 Head retained in R2: PASS")
    print("Block 23 reinitialization executed: PASS")
    print("fc_norm reset: PASS")

    evaluate_stage("R2")

finally:

    hook.remove()
    captured.clear()

    del model
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Calculate R0-R1, R0-R2, and R1-R2 similarities
# ============================================================

records = []

for i, image_id in enumerate(PILOT_IDS):

    base_logits, base_cam, _ = outputs["R0"][i]
    r1_logits, r1_cam, _ = outputs["R1"][i]
    r2_logits, r2_cam, r2_raw_max = outputs["R2"][i]

    assert base_cam is not None

    r0_r1 = similarity(base_cam, r1_cam)
    r0_r2 = similarity(base_cam, r2_cam)
    r1_r2 = similarity(r1_cam, r2_cam)

    records.append({
        "image_id": image_id,
        "r1_degenerate": r1_cam is None,
        "r2_degenerate": r2_cam is None,
        "r0_r1_spearman": r0_r1["spearman"],
        "r0_r2_spearman": r0_r2["spearman"],
        "r1_r2_spearman": r1_r2["spearman"],
        "r0_r1_ssim": r0_r1["ssim"],
        "r0_r2_ssim": r0_r2["ssim"],
        "r1_r2_ssim": r1_r2["ssim"],
        "r0_r1_pearson": r0_r1["pearson"],
        "r0_r2_pearson": r0_r2["pearson"],
        "r1_r2_pearson": r1_r2["pearson"],
        "r2_raw_max": r2_raw_max,
        "r0_r2_logit_l2": float(
            np.linalg.norm(r2_logits - base_logits)
        )
    })

results = pd.DataFrame(records)

assert len(results) == 8
assert results["image_id"].is_unique

print("\nPILOT SUMMARY")
print("Images:", len(results))
print("Distinct source-risk groups:", 8)
print("Seed:", SEED)

print(
    "R1 degenerate CAMs:",
    int(results["r1_degenerate"].sum())
)

print(
    "R2 degenerate CAMs:",
    int(results["r2_degenerate"].sum())
)

for column in [
    "r0_r1_spearman",
    "r0_r2_spearman",
    "r1_r2_spearman",
    "r0_r1_ssim",
    "r0_r2_ssim",
    "r1_r2_ssim",
    "r0_r1_pearson",
    "r0_r2_pearson",
    "r1_r2_pearson"
]:
    print(
        f"Mean {column}:",
        round(float(results[column].mean()), 6)
    )

print(
    "Mean R0-R2 logit L2:",
    round(float(results["r0_r2_logit_l2"].mean()), 6)
)

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Randomization: Cumulative R1 to R2")
print("Output files created: False")
print("Validation images decoded:", len(inputs))
print("Test images accessed: 0")

print("\nSTEP 10.22 EXECUTION COMPLETE")
print("Pilot interpretation: PENDING REVIEW")


# %% [cell 360] Experiment 10 - Step 10.23
# ============================================================
# Experiment 10 - Step 10.23
# R2 Parameter Isolation and Cumulative Integrity Audit
# Both models; seed 1042; no images or artifact writes
# ============================================================

from pathlib import Path
import gc
import hashlib

import torch
import torch.nn as nn
import timm
from timm.layers import trunc_normal_

ROOT = DR_RESEARCH

CHECKPOINTS = {
    "RETFound": (
        ROOT / "experiment04_results"
        / "retfound_vit_large_v1/best_model.pt"
    ),
    "Meta_MAE": (
        ROOT / "experiment07_results"
        / "meta_mae_vit_large_v1/best_model.pt"
    )
}

EXPECTED_HASHES = {
    "RETFound": (
        "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
    ),
    "Meta_MAE": (
        "d670c42bc5a18a394a4e1780918de04522915a42a7f62675f4fd6d89c4f8b312"
    )
}

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available()

def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

def snapshot(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

def changed_keys(before, after):
    return {
        name
        for name in before
        if not torch.equal(before[name], after[name])
    }

R1_KEYS = {
    "head.weight",
    "head.bias"
}

R2_NEW_KEYS = {
    "fc_norm.weight",
    "fc_norm.bias"
}

BLOCK23_KEYS = set()

audit_results = {}

print("EXPERIMENT 10.23 - R2 PARAMETER ISOLATION AUDIT")

for model_name, checkpoint_path in CHECKPOINTS.items():

    print(f"\nMODEL: {model_name}")

    assert checkpoint_path.is_file()
    assert file_hash(checkpoint_path) == EXPECTED_HASHES[model_name]

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=5,
        global_pool="avg",
        device="meta",
        weight_init="skip"
    )

    model = model.to_empty(device=DEVICE)

    loaded = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    assert not loaded.missing_keys
    assert not loaded.unexpected_keys

    model.eval()
    model.requires_grad_(False)

    original = snapshot(model)

    assert len(original) == 296

    block_keys = {
        name
        for name in original
        if name.startswith("blocks.23.")
    }

    assert len(block_keys) == 12

    expected_r2 = R1_KEYS | R2_NEW_KEYS | block_keys

    assert len(expected_r2) == 16

    # R1: Reinitialize the classification head.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    r1 = snapshot(model)

    changes_r1 = changed_keys(original, r1)

    assert changes_r1 <= R1_KEYS
    assert "head.weight" in changes_r1

    # R2: Retain R1 head and reset final normalization
    # and the last transformer block.
    with torch.no_grad():

        model.fc_norm.reset_parameters()

        for module in model.blocks[23].modules():

            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()

    r2 = snapshot(model)

    changes_r1_r2 = changed_keys(r1, r2)
    changes_r0_r2 = changed_keys(original, r2)

    # Verify that the randomized head remains identical.
    for name in R1_KEYS:
        assert torch.equal(r1[name], r2[name])

    # Verify that no unauthorized parameter changed.
    assert changes_r1_r2 <= (R2_NEW_KEYS | block_keys)
    assert changes_r0_r2 <= expected_r2

    # Confirm that every eligible parameter remains finite.
    for name, tensor in r2.items():
        assert torch.isfinite(tensor).all(), name

    # Verify final normalization reset values.
    assert torch.equal(
        r2["fc_norm.weight"],
        torch.ones_like(r2["fc_norm.weight"])
    )

    assert torch.equal(
        r2["fc_norm.bias"],
        torch.zeros_like(r2["fc_norm.bias"])
    )

    # Confirm that all six Block 23 linear tensors changed.
    block_linear_weights = {
        name
        for name in block_keys
        if name.endswith(".weight")
        and (
            ".attn.qkv." in name
            or ".attn.proj." in name
            or ".mlp.fc1." in name
            or ".mlp.fc2." in name
        )
    }

    assert len(block_linear_weights) == 4
    assert block_linear_weights <= changes_r1_r2

    untouched = set(original) - expected_r2
    assert len(untouched) == 280

    for name in untouched:
        assert torch.equal(original[name], r2[name]), name

    audit_results[model_name] = {
        "total_tensors": len(original),
        "r1_changed": len(changes_r1),
        "r1_to_r2_changed": len(changes_r1_r2),
        "r0_to_r2_changed": len(changes_r0_r2),
        "untouched_verified": len(untouched),
        "head_retained": True,
        "unauthorized_changes": 0
    }

    print("Checkpoint SHA256: PASS")
    print("Total parameter tensors:", len(original))
    print("R1 changed tensors:", len(changes_r1))
    print("R1-to-R2 changed tensors:", len(changes_r1_r2))
    print("R0-to-R2 changed tensors:", len(changes_r0_r2))
    print("Untouched tensors verified:", len(untouched))
    print("R1 head retained: PASS")
    print("Block 23 linear reinitialization: PASS")
    print("fc_norm reset: PASS")
    print("Unauthorized changes: 0")
    print("Finite parameters: PASS")

    del model, checkpoint
    del original, r1, r2
    gc.collect()
    torch.cuda.empty_cache()

print("\nCROSS-MODEL SUMMARY")

for model_name, result in audit_results.items():
    print(f"\n{model_name}")
    for key, value in result.items():
        print(f"{key}: {value}")

print("\nRESEARCH INTEGRITY")
print("Original checkpoints modified: False")
print("Output files created: False")
print("Images decoded: 0")
print("Model inference performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.23 PASS")


# %% [cell 361] Experiment 10 - Step 10.24

# ============================================================
# Experiment 10 - Step 10.24
# RETFound Cumulative R0-R3 Grad-CAM Pilot
# Seed 1042; eight validation images; read-only artifacts
# ============================================================

from pathlib import Path
import gc
import hashlib
import json

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

ROOT = DR_RESEARCH

INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
CHECKPOINT_PATH = ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"
REFERENCE_PATH = ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"
PROTOCOL_PATH = ROOT / "experiment10_results/Experiment10_Randomization_Protocol_V1.json"

EXPECTED_INDEX_HASH = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)
EXPECTED_CHECKPOINT_HASH = (
    "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
)
EXPECTED_PROTOCOL_HASH = (
    "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
)

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available(), "CUDA is required."

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path in [
    INDEX_PATH, COHORT_PATH, CHECKPOINT_PATH,
    REFERENCE_PATH, PROTOCOL_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_HASH
assert sha256_file(CHECKPOINT_PATH) == EXPECTED_CHECKPOINT_HASH
assert sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_HASH

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

stages = protocol["randomization"]["stages"]

assert set(stages["R1"]) == {"head"}
assert set(stages["R2"]) == {"head", "fc_norm", "blocks.23"}
assert set(stages["R3"]) == {
    "head", "fc_norm",
    "blocks.18", "blocks.19", "blocks.20",
    "blocks.21", "blocks.22", "blocks.23"
}
assert SEED in protocol["randomization"]["randomization_seeds"]

print("EXPERIMENT 10.24 - RETFOUND R0-R3 PILOT")
print("Seed:", SEED)
print("Frozen protocol: PASS")
print("Checkpoint and index hashes: PASS")

# ------------------------------------------------------------
# 1. Verify and prepare the frozen pilot cohort
# ------------------------------------------------------------

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)
cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    flag = str(
        cohort_lookup.loc[image_id, "primary_ma_candidate"]
    ).strip().lower()

    assert flag in {"true", "1"}

assert validation.loc[PILOT_IDS, "risk_group_id"].nunique() == 8

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = [
        str(value) for value in archive["image_ids"]
    ]
    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: i for i, image_id in enumerate(reference_ids)
    }

    references = {
        image_id: archive["retfound_cam"][
            positions[image_id]
        ].astype(np.float32).copy()
        for image_id in PILOT_IDS
    }

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

inputs = []

for image_id in PILOT_IDS:
    row = validation.loc[image_id]

    local_path = Path(row["local_path"])
    source_path = Path(row["source_path"])

    image_path = (
        local_path if local_path.is_file() else source_path
    )

    assert image_path.is_file()

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    inputs.append(tensor)

print("\nCOHORT")
print("Pilot images:", len(inputs))
print("Distinct source-risk groups:", 8)
print("Reference CAMs:", len(references))
print("Validation-only cohort: PASS")

# ------------------------------------------------------------
# 2. Construct model and preserve original reference tensors
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

loaded = model.load_state_dict(source_state, strict=True)
assert not loaded.missing_keys
assert not loaded.unexpected_keys

model.eval()
model.requires_grad_(False)

all_names = set(model.state_dict())

r1_names = {
    "head.weight",
    "head.bias"
}

r2_new_names = {
    "fc_norm.weight",
    "fc_norm.bias"
} | {
    name for name in all_names
    if name.startswith("blocks.23.")
}

r3_new_names = {
    name for name in all_names
    if any(
        name.startswith(f"blocks.{i}.")
        for i in range(18, 23)
    )
}

r2_names = r1_names | r2_new_names
r3_names = r2_names | r3_new_names

assert len(r1_names) == 2
assert len(r2_new_names) == 14
assert len(r3_new_names) == 60
assert len(r2_names) == 16
assert len(r3_names) == 76
assert len(all_names - r3_names) == 220

print("\nMODEL")
print("Strict checkpoint loading: PASS")
print("Parameter tensors:", len(all_names))
print("R3 newly eligible tensors:", len(r3_new_names))
print("R3 cumulative eligible tensors:", len(r3_names))

# ------------------------------------------------------------
# 3. Register the same Grad-CAM hook as previous pilots
# ------------------------------------------------------------

captured = {}

def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation

hook = model.blocks[23].register_forward_hook(activation_hook)

def calculate_cam(input_tensor, true_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(input_tensor)

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, true_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2),
        keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        resized_max = float(cam.max())
        assert resized_max > 1e-12

        cam /= resized_max
        assert np.isfinite(cam).all()

    captured.clear()
    return output_logits, cam, raw_max

def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - a.mean()
    bc = b - b.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12 else np.nan
    )

    spearman = float(spearmanr(a, b).statistic)

    ssim = float(
        structural_similarity(
            first, second, data_range=1.0
        )
    )

    mae = float(np.mean(np.abs(a - b)))

    return {
        "pearson": pearson,
        "spearman": spearman,
        "ssim": ssim,
        "mae": mae
    }

# ------------------------------------------------------------
# 4. Controlled cumulative randomization
# ------------------------------------------------------------

outputs = {
    "R0": [],
    "R1": [],
    "R2": [],
    "R3": []
}

def evaluate_stage(stage):
    for tensor, grade in zip(inputs, PILOT_GRADES):
        logits, cam, raw_max = calculate_cam(
            tensor.to(DEVICE), grade
        )
        outputs[stage].append((logits, cam, raw_max))

def reinitialize_block(block):
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()

def capture_subset(names):
    state = model.state_dict()
    return {
        name: state[name].detach().cpu().clone()
        for name in names
    }

try:
    # R0: original trained checkpoint.
    evaluate_stage("R0")

    baseline_similarities = []
    for image_id, (_, cam, _) in zip(PILOT_IDS, outputs["R0"]):
        assert cam is not None
        baseline_similarities.append(
            compare_cams(cam, references[image_id])["pearson"]
        )

    assert min(baseline_similarities) > 0.999

    print("\nBASELINE REPRODUCTION")
    print(
        "Minimum saved-reference Pearson:",
        round(min(baseline_similarities), 6)
    )
    print("Baseline CAM reproduction: PASS")

    # R1: initialize classification head.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    r1_snapshot = capture_subset(r1_names)
    evaluate_stage("R1")

    # R2: retain R1; initialize fc_norm and block 23.
    with torch.no_grad():
        model.fc_norm.reset_parameters()

    reinitialize_block(model.blocks[23])

    r2_snapshot = capture_subset(r2_names)

    for name in r1_names:
        assert torch.equal(
            r1_snapshot[name], r2_snapshot[name]
        ), name

    evaluate_stage("R2")

    # R3: retain R2; initialize blocks 18 through 22.
    r3_previous = capture_subset(r3_new_names)

    for block_index in range(18, 23):
        reinitialize_block(model.blocks[block_index])

    r3_snapshot = capture_subset(r2_names | r3_new_names)

    for name in r2_names:
        assert torch.equal(
            r2_snapshot[name], r3_snapshot[name]
        ), name

    changed_new = {
        name for name in r3_new_names
        if not torch.equal(
            r3_previous[name], r3_snapshot[name]
        )
    }

    # Check the complete 296-tensor state against the
    # original checkpoint, including unauthorized changes.
    state_after_r3 = model.state_dict()

    unauthorized = []
    changed_total = []

    for name, tensor in state_after_r3.items():
        current = tensor.detach().cpu()
        original = source_state[name]

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, original):
            changed_total.append(name)
            if name not in r3_names:
                unauthorized.append(name)

    assert not unauthorized, (
        f"Unauthorized changes: {unauthorized}"
    )

    assert len(changed_new) == 60, (
        f"Unexpected R3 newly changed count: {len(changed_new)}"
    )

    assert len(changed_total) == 76, (
        f"Unexpected cumulative changed count: {len(changed_total)}"
    )

    print("\nR3 PARAMETER INTEGRITY")
    print("R1 Head retained through R3: PASS")
    print("R2 state retained through R3: PASS")
    print("R3 newly changed tensors:", len(changed_new))
    print("R3 cumulative changed tensors:", len(changed_total))
    print("Original tensors still untouched:", 220)
    print("Unauthorized tensor changes:", len(unauthorized))
    print("All tensor values finite: PASS")

    evaluate_stage("R3")

finally:
    hook.remove()
    captured.clear()
    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# 5. Compare spatial attribution across stages
# ------------------------------------------------------------

records = []

for i, image_id in enumerate(PILOT_IDS):
    r0_logits, r0_cam, _ = outputs["R0"][i]
    r1_logits, r1_cam, _ = outputs["R1"][i]
    r2_logits, r2_cam, _ = outputs["R2"][i]
    r3_logits, r3_cam, r3_raw_max = outputs["R3"][i]

    comparisons = {
        "r0_r1": compare_cams(r0_cam, r1_cam),
        "r0_r2": compare_cams(r0_cam, r2_cam),
        "r0_r3": compare_cams(r0_cam, r3_cam),
        "r1_r2": compare_cams(r1_cam, r2_cam),
        "r2_r3": compare_cams(r2_cam, r3_cam)
    }

    record = {
        "image_id": image_id,
        "true_grade": PILOT_GRADES[i],
        "r1_degenerate": r1_cam is None,
        "r2_degenerate": r2_cam is None,
        "r3_degenerate": r3_cam is None,
        "r3_raw_max": r3_raw_max,
        "r0_r3_logit_l2": float(
            np.linalg.norm(r3_logits - r0_logits)
        ),
        "r2_r3_logit_l2": float(
            np.linalg.norm(r3_logits - r2_logits)
        )
    }

    for label, metrics in comparisons.items():
        for metric, value in metrics.items():
            record[f"{label}_{metric}"] = value

    records.append(record)

results = pd.DataFrame(records)

assert len(results) == 8
assert results["image_id"].is_unique

print("\nPILOT SUMMARY")
print("Model: RETFound")
print("Seed:", SEED)
print("Images:", len(results))

for stage in ["r1", "r2", "r3"]:
    print(
        f"{stage.upper()} degenerate CAMs:",
        int(results[f"{stage}_degenerate"].sum())
    )

for label in [
    "r0_r1", "r0_r2", "r0_r3",
    "r1_r2", "r2_r3"
]:
    print(f"\n{label.upper()} SPATIAL COMPARISON")

    valid = results[
        [
            f"{label}_pearson",
            f"{label}_spearman",
            f"{label}_ssim",
            f"{label}_mae"
        ]
    ].dropna()

    print("Valid pairs:", len(valid))

    if len(valid):
        for metric in ["pearson", "spearman", "ssim", "mae"]:
            column = f"{label}_{metric}"
            print(
                f"Mean {metric}:",
                round(float(valid[column].mean()), 6)
            )

print("\nLOGIT CHANGES")
print(
    "Mean R0-R3 logit L2:",
    round(float(results["r0_r3_logit_l2"].mean()), 6)
)
print(
    "Mean R2-R3 logit L2:",
    round(float(results["r2_r3_logit_l2"].mean()), 6)
)

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Original CAM files modified: False")
print("New research files created: False")
print("Randomization: Cumulative R1-R3")
print("Validation images decoded:", len(inputs))
print("Training performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.24 EXECUTION COMPLETE")
print("R3 pilot interpretation: PENDING REVIEW")


# %% [cell 362] Experiment 10 - Step 10.25
# ============================================================
# Experiment 10 - Step 10.25
# Meta MAE Cumulative R0-R3 Grad-CAM Pilot
# Seed 1042; eight validation images; read-only
# ============================================================

from pathlib import Path
import gc
import hashlib
import json

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

ROOT = DR_RESEARCH

INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"

CHECKPOINT_PATH = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1/best_model.pt"
)

REFERENCE_PATH = (
    ROOT / "experiment07_results/gradcam"
    / "Meta_MAE_MA_CAM_Maps_V1.npz"
)

PROTOCOL_PATH = (
    ROOT / "experiment10_results"
    / "Experiment10_Randomization_Protocol_V1.json"
)

EXPECTED_INDEX_HASH = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

EXPECTED_CHECKPOINT_HASH = (
    "d670c42bc5a18a394a4e1780918de04522915a42a7f62675f4fd6d89c4f8b312"
)

EXPECTED_PROTOCOL_HASH = (
    "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
)

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


print("EXPERIMENT 10.25 - META MAE R0-R3 PILOT")
print("Seed:", SEED)

# ============================================================
# 1. Verify frozen files and randomization protocol
# ============================================================

for path in [
    INDEX_PATH,
    COHORT_PATH,
    CHECKPOINT_PATH,
    REFERENCE_PATH,
    PROTOCOL_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_HASH
assert sha256_file(CHECKPOINT_PATH) == EXPECTED_CHECKPOINT_HASH
assert sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_HASH

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

stages = protocol["randomization"]["stages"]

assert set(stages["R1"]) == {"head"}

assert set(stages["R2"]) == {
    "head",
    "fc_norm",
    "blocks.23"
}

assert set(stages["R3"]) == {
    "head",
    "fc_norm",
    "blocks.18",
    "blocks.19",
    "blocks.20",
    "blocks.21",
    "blocks.22",
    "blocks.23"
}

assert SEED in protocol["randomization"]["randomization_seeds"]

print("\nFILE INTEGRITY")
print("Frozen index SHA256: PASS")
print("Meta MAE checkpoint SHA256: PASS")
print("Frozen protocol SHA256: PASS")

# ============================================================
# 2. Verify the exact pilot cohort
# ============================================================

index = pd.read_csv(
    INDEX_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str
    }
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276
assert len(PILOT_IDS) == 8
assert len(set(PILOT_IDS)) == 8

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):

    assert image_id in validation.index
    assert image_id in cohort_lookup.index

    assert int(
        validation.loc[image_id, "dr_grade"]
    ) == grade

    candidate_flag = str(
        cohort_lookup.loc[
            image_id,
            "primary_ma_candidate"
        ]
    ).strip().lower()

    assert candidate_flag in {"true", "1"}

assert (
    validation.loc[
        PILOT_IDS,
        "risk_group_id"
    ].nunique() == 8
)

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:

    assert "image_ids" in archive.files
    assert "meta_mae_cam" in archive.files

    reference_ids = [
        str(value)
        for value in archive["image_ids"]
    ]

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: i
        for i, image_id in enumerate(reference_ids)
    }

    assert set(PILOT_IDS).issubset(positions)

    references = {
        image_id: archive["meta_mae_cam"][
            positions[image_id]
        ].astype(np.float32).copy()
        for image_id in PILOT_IDS
    }

for reference in references.values():
    assert reference.shape == (224, 224)
    assert np.isfinite(reference).all()

print("\nCOHORT")
print("Pilot images:", len(PILOT_IDS))
print("Distinct source-risk groups:", 8)
print("Reference CAMs:", len(references))
print("Validation-only cohort: PASS")

# ============================================================
# 3. Reconstruct original image preprocessing
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

inputs = []

for image_id in PILOT_IDS:

    row = validation.loc[image_id]

    local_path = Path(row["local_path"])
    source_path = Path(row["source_path"])

    image_path = (
        local_path
        if local_path.is_file()
        else source_path
    )

    assert image_path.is_file(), image_id

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    inputs.append(tensor)

assert len(inputs) == 8

# ============================================================
# 4. Construct Meta MAE model
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

assert checkpoint["epoch"] == 24

source_state = checkpoint["model_state_dict"]

assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

loaded = model.load_state_dict(
    source_state,
    strict=True
)

assert not loaded.missing_keys
assert not loaded.unexpected_keys

model.eval()
model.requires_grad_(False)

all_names = set(model.state_dict())

r1_names = {
    "head.weight",
    "head.bias"
}

r2_new_names = {
    "fc_norm.weight",
    "fc_norm.bias"
} | {
    name
    for name in all_names
    if name.startswith("blocks.23.")
}

r3_new_names = {
    name
    for name in all_names
    if any(
        name.startswith(f"blocks.{i}.")
        for i in range(18, 23)
    )
}

r2_names = r1_names | r2_new_names
r3_names = r2_names | r3_new_names

assert len(all_names) == 296
assert len(r1_names) == 2
assert len(r2_new_names) == 14
assert len(r3_new_names) == 60
assert len(r2_names) == 16
assert len(r3_names) == 76
assert len(all_names - r3_names) == 220

print("\nMODEL")
print("Checkpoint epoch:", 24)
print("Strict checkpoint loading: PASS")
print("Parameter tensors:", len(all_names))
print("R3 newly eligible tensors:", len(r3_new_names))
print("R3 cumulative eligible tensors:", len(r3_names))
print("Parameters frozen: PASS")

# ============================================================
# 5. Register Grad-CAM hook
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):

    activation = (
        output.detach()
        .requires_grad_(True)
    )

    captured["activation"] = activation

    return activation


hook = model.blocks[23].register_forward_hook(
    activation_hook
)


def calculate_cam(input_tensor, true_grade):

    captured.clear()

    with torch.enable_grad():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(input_tensor)

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]

        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, true_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach()
        .float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach()
        .float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2),
        keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:

        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        resized_max = float(cam.max())

        assert resized_max > 1e-12

        cam /= resized_max

        assert np.isfinite(cam).all()

    captured.clear()

    return output_logits, cam, raw_max


# ============================================================
# 6. CAM similarity calculations
# ============================================================

def compare_cams(first, second):

    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    centered_a = a - a.mean()
    centered_b = b - b.mean()

    denominator = (
        np.linalg.norm(centered_a)
        * np.linalg.norm(centered_b)
    )

    pearson = (
        float(
            np.dot(centered_a, centered_b)
            / denominator
        )
        if denominator > 1e-12
        else np.nan
    )

    spearman = float(
        spearmanr(a, b).statistic
    )

    ssim = float(
        structural_similarity(
            first,
            second,
            data_range=1.0
        )
    )

    mae = float(
        np.mean(np.abs(a - b))
    )

    return {
        "pearson": pearson,
        "spearman": spearman,
        "ssim": ssim,
        "mae": mae
    }


# ============================================================
# 7. Parameter initialization and snapshots
# ============================================================

def reinitialize_block(block):

    with torch.no_grad():

        for module in block.modules():

            if isinstance(module, nn.Linear):

                trunc_normal_(
                    module.weight,
                    std=0.02
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()


def capture_subset(names):

    state = model.state_dict()

    return {
        name: state[name].detach().cpu().clone()
        for name in names
    }


outputs = {
    "R0": [],
    "R1": [],
    "R2": [],
    "R3": []
}


def evaluate_stage(stage_name):

    for tensor, grade in zip(
        inputs,
        PILOT_GRADES
    ):

        logits, cam, raw_max = calculate_cam(
            tensor.to(DEVICE),
            grade
        )

        outputs[stage_name].append(
            (logits, cam, raw_max)
        )


# ============================================================
# 8. Execute cumulative R0-R3 randomization
# ============================================================

try:

    # R0: Original trained Meta MAE model.
    evaluate_stage("R0")

    baseline_pearsons = []

    for image_id, (_, cam, _) in zip(
        PILOT_IDS,
        outputs["R0"]
    ):

        assert cam is not None

        result = compare_cams(
            cam,
            references[image_id]
        )

        baseline_pearsons.append(
            result["pearson"]
        )

    assert min(baseline_pearsons) > 0.999

    print("\nBASELINE REPRODUCTION")
    print(
        "Minimum saved-reference Pearson:",
        round(min(baseline_pearsons), 6)
    )
    print("Baseline CAM reproduction: PASS")

    # R1: Reinitialize classification head.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():

        trunc_normal_(
            model.head.weight,
            std=0.02
        )

        nn.init.zeros_(model.head.bias)

    r1_snapshot = capture_subset(r1_names)

    evaluate_stage("R1")

    # R2: Preserve R1 and reinitialize fc_norm and block 23.
    with torch.no_grad():
        model.fc_norm.reset_parameters()

    reinitialize_block(model.blocks[23])

    r2_snapshot = capture_subset(r2_names)

    for name in r1_names:

        assert torch.equal(
            r1_snapshot[name],
            r2_snapshot[name]
        ), name

    evaluate_stage("R2")

    # R3: Preserve R2 and reinitialize blocks 18-22.
    r3_previous = capture_subset(r3_new_names)

    for block_index in range(18, 23):
        reinitialize_block(
            model.blocks[block_index]
        )

    r3_snapshot = capture_subset(
        r2_names | r3_new_names
    )

    for name in r2_names:

        assert torch.equal(
            r2_snapshot[name],
            r3_snapshot[name]
        ), name

    changed_new = {
        name
        for name in r3_new_names
        if not torch.equal(
            r3_previous[name],
            r3_snapshot[name]
        )
    }

    # Compare all model tensors with the original checkpoint.
    state_after_r3 = model.state_dict()

    unauthorized = []
    changed_total = []

    for name, tensor in state_after_r3.items():

        current = tensor.detach().cpu()
        original = source_state[name]

        assert torch.isfinite(
            current
        ).all().item(), name

        if not torch.equal(
            current,
            original
        ):

            changed_total.append(name)

            if name not in r3_names:
                unauthorized.append(name)

    assert not unauthorized, (
        f"Unauthorized tensor changes: {unauthorized}"
    )

    assert len(changed_new) == 60, (
        f"Unexpected R3 changed count: {len(changed_new)}"
    )

    assert len(changed_total) == 76, (
        f"Unexpected cumulative count: {len(changed_total)}"
    )

    print("\nR3 PARAMETER INTEGRITY")
    print("R1 Head retained through R3: PASS")
    print("R2 state retained through R3: PASS")
    print("R3 newly changed tensors:", len(changed_new))
    print("R3 cumulative changed tensors:", len(changed_total))
    print("Original tensors still untouched:", 220)
    print("Unauthorized tensor changes:", len(unauthorized))
    print("All tensor values finite: PASS")

    evaluate_stage("R3")

finally:

    hook.remove()
    captured.clear()

    del model
    del checkpoint
    del source_state

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 9. Calculate R0-R3 spatial similarities
# ============================================================

records = []

for i, image_id in enumerate(PILOT_IDS):

    r0_logits, r0_cam, _ = outputs["R0"][i]
    r1_logits, r1_cam, _ = outputs["R1"][i]
    r2_logits, r2_cam, _ = outputs["R2"][i]
    r3_logits, r3_cam, r3_raw_max = outputs["R3"][i]

    comparisons = {
        "r0_r1": compare_cams(r0_cam, r1_cam),
        "r0_r2": compare_cams(r0_cam, r2_cam),
        "r0_r3": compare_cams(r0_cam, r3_cam),
        "r1_r2": compare_cams(r1_cam, r2_cam),
        "r2_r3": compare_cams(r2_cam, r3_cam)
    }

    record = {
        "image_id": image_id,
        "true_grade": PILOT_GRADES[i],
        "r1_degenerate": r1_cam is None,
        "r2_degenerate": r2_cam is None,
        "r3_degenerate": r3_cam is None,
        "r3_raw_max": r3_raw_max,
        "r0_r3_logit_l2": float(
            np.linalg.norm(
                r3_logits - r0_logits
            )
        ),
        "r2_r3_logit_l2": float(
            np.linalg.norm(
                r3_logits - r2_logits
            )
        )
    }

    for label, metrics in comparisons.items():

        for metric, value in metrics.items():
            record[f"{label}_{metric}"] = value

    records.append(record)

results = pd.DataFrame(records)

assert len(results) == 8
assert results["image_id"].is_unique

# ============================================================
# 10. Report observations without discarding degeneracies
# ============================================================

print("\nPILOT SUMMARY")
print("Model: Meta MAE")
print("Seed:", SEED)
print("Images:", len(results))

for stage in ["r1", "r2", "r3"]:

    print(
        f"{stage.upper()} degenerate CAMs:",
        int(
            results[f"{stage}_degenerate"].sum()
        )
    )

for label in [
    "r0_r1",
    "r0_r2",
    "r0_r3",
    "r1_r2",
    "r2_r3"
]:

    print(
        f"\n{label.upper()} SPATIAL COMPARISON"
    )

    columns = [
        f"{label}_pearson",
        f"{label}_spearman",
        f"{label}_ssim",
        f"{label}_mae"
    ]

    valid = results[columns].dropna()

    print("Valid pairs:", len(valid))

    if not valid.empty:

        for metric in [
            "pearson",
            "spearman",
            "ssim",
            "mae"
        ]:

            column = f"{label}_{metric}"

            print(
                f"Mean {metric}:",
                round(
                    float(
                        valid[column].mean()
                    ),
                    6
                )
            )

print("\nLOGIT CHANGES")

print(
    "Mean R0-R3 logit L2:",
    round(
        float(
            results["r0_r3_logit_l2"].mean()
        ),
        6
    )
)

print(
    "Mean R2-R3 logit L2:",
    round(
        float(
            results["r2_r3_logit_l2"].mean()
        ),
        6
    )
)

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Original CAM files modified: False")
print("New research files created: False")
print("Randomization: Cumulative R1-R3")
print("Validation images decoded:", len(inputs))
print("Training performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.25 EXECUTION COMPLETE")
print("R3 pilot interpretation: PENDING REVIEW")


# %% [cell 363] Experiment 10 - Step 10.26
# ============================================================
# Experiment 10 - Step 10.26
# RETFound Cumulative R0-R4 Grad-CAM Pilot
# Seed 1042; frozen validation pilot; read-only artifacts
# ============================================================

from pathlib import Path
import gc
import hashlib
import json

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

ROOT = DR_RESEARCH

INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"

CHECKPOINT_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1/best_model.pt"
)

REFERENCE_PATH = (
    ROOT / "experiment05_results/gradcam"
    / "RETFound_MA_CAM_Maps_V1.npz"
)

PROTOCOL_PATH = (
    ROOT / "experiment10_results"
    / "Experiment10_Randomization_Protocol_V1.json"
)

EXPECTED_INDEX_HASH = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

EXPECTED_CHECKPOINT_HASH = (
    "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
)

EXPECTED_PROTOCOL_HASH = (
    "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
)

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


print("EXPERIMENT 10.26 - RETFOUND R0-R4 PILOT")
print("Seed:", SEED)

# ------------------------------------------------------------
# 1. Verify frozen artifacts and protocol
# ------------------------------------------------------------

for path in [
    INDEX_PATH,
    COHORT_PATH,
    CHECKPOINT_PATH,
    REFERENCE_PATH,
    PROTOCOL_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_HASH
assert sha256_file(CHECKPOINT_PATH) == EXPECTED_CHECKPOINT_HASH
assert sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_HASH

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

stages = protocol["randomization"]["stages"]

expected_stages = {
    "R1": {"head"},
    "R2": {"head", "fc_norm", "blocks.23"},
    "R3": {
        "head", "fc_norm",
        *{f"blocks.{i}" for i in range(18, 24)}
    },
    "R4": {
        "head", "fc_norm",
        *{f"blocks.{i}" for i in range(12, 24)}
    }
}

for stage_name, expected in expected_stages.items():
    assert set(stages[stage_name]) == expected, stage_name

assert SEED in protocol["randomization"]["randomization_seeds"]

print("\nFILE INTEGRITY")
print("Frozen index SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Frozen protocol SHA256: PASS")

# ------------------------------------------------------------
# 2. Validate cohort and load original CAM references
# ------------------------------------------------------------

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276
assert len(PILOT_IDS) == 8
assert len(set(PILOT_IDS)) == 8

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    candidate_flag = str(
        cohort_lookup.loc[image_id, "primary_ma_candidate"]
    ).strip().lower()

    assert candidate_flag in {"true", "1"}

assert validation.loc[
    PILOT_IDS, "risk_group_id"
].nunique() == 8

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = [
        str(value) for value in archive["image_ids"]
    ]

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: i
        for i, image_id in enumerate(reference_ids)
    }

    references = {
        image_id: archive["retfound_cam"][
            positions[image_id]
        ].astype(np.float32).copy()
        for image_id in PILOT_IDS
    }

for cam in references.values():
    assert cam.shape == (224, 224)
    assert np.isfinite(cam).all()

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

inputs = []

for image_id in PILOT_IDS:
    row = validation.loc[image_id]

    local_path = Path(row["local_path"])
    source_path = Path(row["source_path"])

    image_path = (
        local_path if local_path.is_file()
        else source_path
    )

    assert image_path.is_file(), image_id

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    inputs.append(tensor)

print("\nCOHORT")
print("Pilot images:", len(inputs))
print("Distinct source-risk groups:", 8)
print("Reference CAMs:", len(references))
print("Validation-only cohort: PASS")

# ------------------------------------------------------------
# 3. Load the original trained model
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]

assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

load_result = model.load_state_dict(
    source_state,
    strict=True
)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

all_names = set(model.state_dict())

r1_names = {"head.weight", "head.bias"}

r2_new_names = {
    "fc_norm.weight",
    "fc_norm.bias"
} | {
    name for name in all_names
    if name.startswith("blocks.23.")
}

r3_new_names = {
    name for name in all_names
    if any(
        name.startswith(f"blocks.{i}.")
        for i in range(18, 23)
    )
}

r4_new_names = {
    name for name in all_names
    if any(
        name.startswith(f"blocks.{i}.")
        for i in range(12, 18)
    )
}

r2_names = r1_names | r2_new_names
r3_names = r2_names | r3_new_names
r4_names = r3_names | r4_new_names

assert len(all_names) == 296
assert len(r1_names) == 2
assert len(r2_new_names) == 14
assert len(r3_new_names) == 60
assert len(r4_new_names) == 72
assert len(r2_names) == 16
assert len(r3_names) == 76
assert len(r4_names) == 148
assert len(all_names - r4_names) == 148

print("\nMODEL")
print("Strict checkpoint loading: PASS")
print("Total parameter tensors:", len(all_names))
print("R4 newly eligible tensors:", len(r4_new_names))
print("R4 cumulative eligible tensors:", len(r4_names))
print("R4 untouched tensors:", len(all_names - r4_names))

# ------------------------------------------------------------
# 4. Original Grad-CAM calculation
# ------------------------------------------------------------

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(
    activation_hook
)


def calculate_cam(input_tensor, target_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(input_tensor)

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, target_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2),
        keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        resized_max = float(cam.max())
        assert resized_max > 1e-12

        cam /= resized_max
        assert np.isfinite(cam).all()

    captured.clear()
    return output_logits, cam, raw_max


# ------------------------------------------------------------
# 5. CAM spatial similarity metrics
# ------------------------------------------------------------

def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    centered_a = a - a.mean()
    centered_b = b - b.mean()

    denominator = (
        np.linalg.norm(centered_a)
        * np.linalg.norm(centered_b)
    )

    pearson = (
        float(
            np.dot(centered_a, centered_b)
            / denominator
        )
        if denominator > 1e-12
        else np.nan
    )

    spearman = float(spearmanr(a, b).statistic)

    ssim = float(
        structural_similarity(
            first,
            second,
            data_range=1.0
        )
    )

    mae = float(np.mean(np.abs(a - b)))

    return {
        "pearson": pearson,
        "spearman": spearman,
        "ssim": ssim,
        "mae": mae
    }


# ------------------------------------------------------------
# 6. Initialization and parameter-state audit helpers
# ------------------------------------------------------------

def reinitialize_block(block):
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()


def capture_subset(names):
    state = model.state_dict()

    return {
        name: state[name].detach().cpu().clone()
        for name in names
    }


outputs = {
    "R0": [],
    "R1": [],
    "R2": [],
    "R3": [],
    "R4": []
}


def evaluate_stage(stage_name):
    for tensor, grade in zip(inputs, PILOT_GRADES):
        logits, cam, raw_max = calculate_cam(
            tensor.to(DEVICE),
            grade
        )

        outputs[stage_name].append(
            (logits, cam, raw_max)
        )


# ------------------------------------------------------------
# 7. Execute cumulative R0-R4 randomization
# ------------------------------------------------------------

try:
    # R0: Original trained model.
    evaluate_stage("R0")

    baseline_pearsons = []

    for image_id, (_, cam, _) in zip(
        PILOT_IDS,
        outputs["R0"]
    ):
        assert cam is not None

        baseline_pearsons.append(
            compare_cams(
                cam, references[image_id]
            )["pearson"]
        )

    assert min(baseline_pearsons) > 0.999

    print("\nBASELINE REPRODUCTION")
    print(
        "Minimum saved-reference Pearson:",
        round(min(baseline_pearsons), 6)
    )
    print("Baseline CAM reproduction: PASS")

    # R1: Reinitialize classification head once.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    r1_snapshot = capture_subset(r1_names)
    evaluate_stage("R1")

    # R2: Retain R1; reset fc_norm and block 23.
    with torch.no_grad():
        model.fc_norm.reset_parameters()

    reinitialize_block(model.blocks[23])

    r2_snapshot = capture_subset(r2_names)

    for name in r1_names:
        assert torch.equal(
            r1_snapshot[name], r2_snapshot[name]
        ), name

    evaluate_stage("R2")

    # R3: Retain R2; reinitialize blocks 18-22.
    for block_index in range(18, 23):
        reinitialize_block(model.blocks[block_index])

    r3_snapshot = capture_subset(r3_names)

    for name in r2_names:
        assert torch.equal(
            r2_snapshot[name], r3_snapshot[name]
        ), name

    evaluate_stage("R3")

    # R4: Retain R3; reinitialize blocks 12-17.
    r4_previous = capture_subset(r4_new_names)

    for block_index in range(12, 18):
        reinitialize_block(model.blocks[block_index])

    r4_snapshot = capture_subset(r4_names)

    for name in r3_names:
        assert torch.equal(
            r3_snapshot[name],
            r4_snapshot[name]
        ), name

    changed_new = {
        name for name in r4_new_names
        if not torch.equal(
            r4_previous[name],
            r4_snapshot[name]
        )
    }

    # Audit all model tensors against the frozen checkpoint.
    changed_total = []
    unauthorized = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()
        original = source_state[name]

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, original):
            changed_total.append(name)

            if name not in r4_names:
                unauthorized.append(name)

    assert not unauthorized, (
        f"Unauthorized changes: {unauthorized}"
    )

    assert len(changed_new) == 72, (
        f"Unexpected R4 new changes: {len(changed_new)}"
    )

    assert len(changed_total) == 148, (
        f"Unexpected R4 cumulative changes: {len(changed_total)}"
    )

    print("\nR4 PARAMETER INTEGRITY")
    print("R1 Head retained through R4: PASS")
    print("R2 state retained through R4: PASS")
    print("R3 state retained through R4: PASS")
    print("R4 newly changed tensors:", len(changed_new))
    print("R4 cumulative changed tensors:", len(changed_total))
    print("Original tensors still untouched:", 148)
    print("Unauthorized tensor changes:", len(unauthorized))
    print("All tensor values finite: PASS")

    evaluate_stage("R4")

finally:
    hook.remove()
    captured.clear()

    del model
    del checkpoint
    del source_state

    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------------------
# 8. Compare all cumulative stages
# ------------------------------------------------------------

comparison_pairs = [
    ("R0", "R1"),
    ("R0", "R2"),
    ("R0", "R3"),
    ("R0", "R4"),
    ("R1", "R2"),
    ("R2", "R3"),
    ("R3", "R4")
]

records = []

for i, image_id in enumerate(PILOT_IDS):
    record = {
        "image_id": image_id,
        "true_grade": PILOT_GRADES[i]
    }

    for stage in ["R1", "R2", "R3", "R4"]:
        logits, cam, raw_max = outputs[stage][i]

        record[f"{stage.lower()}_degenerate"] = cam is None
        record[f"{stage.lower()}_raw_max"] = raw_max

    r0_logits = outputs["R0"][i][0]
    r3_logits = outputs["R3"][i][0]
    r4_logits = outputs["R4"][i][0]

    record["r0_r4_logit_l2"] = float(
        np.linalg.norm(r4_logits - r0_logits)
    )

    record["r3_r4_logit_l2"] = float(
        np.linalg.norm(r4_logits - r3_logits)
    )

    for first_stage, second_stage in comparison_pairs:
        first_cam = outputs[first_stage][i][1]
        second_cam = outputs[second_stage][i][1]

        metrics = compare_cams(first_cam, second_cam)

        prefix = (
            first_stage.lower()
            + "_"
            + second_stage.lower()
        )

        for metric, value in metrics.items():
            record[f"{prefix}_{metric}"] = value

    records.append(record)

results = pd.DataFrame(records)

assert len(results) == 8
assert results["image_id"].is_unique

print("\nPILOT SUMMARY")
print("Model: RETFound")
print("Seed:", SEED)
print("Images:", len(results))

for stage in ["r1", "r2", "r3", "r4"]:
    print(
        f"{stage.upper()} degenerate CAMs:",
        int(results[f"{stage}_degenerate"].sum())
    )

for first_stage, second_stage in comparison_pairs:
    prefix = (
        first_stage.lower()
        + "_"
        + second_stage.lower()
    )

    print(f"\n{prefix.upper()} SPATIAL COMPARISON")

    columns = [
        f"{prefix}_pearson",
        f"{prefix}_spearman",
        f"{prefix}_ssim",
        f"{prefix}_mae"
    ]

    valid = results[columns].dropna()

    print("Valid pairs:", len(valid))

    if not valid.empty:
        for metric in ["pearson", "spearman", "ssim", "mae"]:
            print(
                f"Mean {metric}:",
                round(
                    float(
                        valid[f"{prefix}_{metric}"].mean()
                    ),
                    6
                )
            )

print("\nLOGIT CHANGES")
print(
    "Mean R0-R4 logit L2:",
    round(float(results["r0_r4_logit_l2"].mean()), 6)
)
print(
    "Mean R3-R4 logit L2:",
    round(float(results["r3_r4_logit_l2"].mean()), 6)
)

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Original CAM files modified: False")
print("New research files created: False")
print("Randomization: Cumulative R1-R4")
print("Validation images decoded:", len(inputs))
print("Training performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.26 EXECUTION COMPLETE")
print("R4 pilot interpretation: PENDING REVIEW")


# %% [cell 364] Experiment 10 - Step 10.27

# ============================================================
# Experiment 10 - Step 10.27
# Meta MAE Cumulative R0-R4 Grad-CAM Pilot
# Seed 1042; eight validation images; read-only artifacts
# ============================================================

from pathlib import Path
import gc
import hashlib
import json

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

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
)

CHECKPOINT_PATH = (
    ROOT / "experiment07_results"
    / "meta_mae_vit_large_v1/best_model.pt"
)

REFERENCE_PATH = (
    ROOT / "experiment07_results/gradcam"
    / "Meta_MAE_MA_CAM_Maps_V1.npz"
)

PROTOCOL_PATH = (
    ROOT / "experiment10_results"
    / "Experiment10_Randomization_Protocol_V1.json"
)

EXPECTED_INDEX_HASH = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

EXPECTED_CHECKPOINT_HASH = (
    "d670c42bc5a18a394a4e1780918de04522915a42a7f62675f4fd6d89c4f8b312"
)

EXPECTED_PROTOCOL_HASH = (
    "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
)

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


print("EXPERIMENT 10.27 - META MAE R0-R4 PILOT")
print("Seed:", SEED)

# ============================================================
# 1. Verify frozen artifacts and protocol
# ============================================================

for path in [
    INDEX_PATH,
    COHORT_PATH,
    CHECKPOINT_PATH,
    REFERENCE_PATH,
    PROTOCOL_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_HASH
assert sha256_file(CHECKPOINT_PATH) == EXPECTED_CHECKPOINT_HASH
assert sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_HASH

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

stages = protocol["randomization"]["stages"]

expected_stages = {
    "R1": {"head"},
    "R2": {"head", "fc_norm", "blocks.23"},
    "R3": {
        "head", "fc_norm",
        *{f"blocks.{i}" for i in range(18, 24)}
    },
    "R4": {
        "head", "fc_norm",
        *{f"blocks.{i}" for i in range(12, 24)}
    }
}

for stage_name, expected in expected_stages.items():
    assert set(stages[stage_name]) == expected, stage_name

assert SEED in protocol["randomization"]["randomization_seeds"]

print("\nFILE INTEGRITY")
print("Frozen index SHA256: PASS")
print("Meta MAE checkpoint SHA256: PASS")
print("Frozen protocol SHA256: PASS")

# ============================================================
# 2. Validate the frozen validation pilot
# ============================================================

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276
assert len(PILOT_IDS) == 8
assert len(set(PILOT_IDS)) == 8

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    candidate_flag = str(
        cohort_lookup.loc[image_id, "primary_ma_candidate"]
    ).strip().lower()

    assert candidate_flag in {"true", "1"}

assert (
    validation.loc[PILOT_IDS, "risk_group_id"].nunique() == 8
)

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    assert "image_ids" in archive.files
    assert "meta_mae_cam" in archive.files

    reference_ids = [
        str(value) for value in archive["image_ids"]
    ]

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: i
        for i, image_id in enumerate(reference_ids)
    }

    assert set(PILOT_IDS).issubset(positions)

    references = {
        image_id: archive["meta_mae_cam"][
            positions[image_id]
        ].astype(np.float32).copy()
        for image_id in PILOT_IDS
    }

for cam in references.values():
    assert cam.shape == (224, 224)
    assert np.isfinite(cam).all()

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

inputs = []

for image_id in PILOT_IDS:
    row = validation.loc[image_id]

    local_path = Path(row["local_path"])
    source_path = Path(row["source_path"])

    image_path = (
        local_path if local_path.is_file()
        else source_path
    )

    assert image_path.is_file(), image_id

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    inputs.append(tensor)

print("\nCOHORT")
print("Pilot images:", len(inputs))
print("Distinct source-risk groups:", 8)
print("Reference CAMs:", len(references))
print("Validation-only cohort: PASS")

# ============================================================
# 3. Load original Meta MAE checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

assert checkpoint["epoch"] == 24

source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

load_result = model.load_state_dict(
    source_state,
    strict=True
)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

all_names = set(model.state_dict())

r1_names = {"head.weight", "head.bias"}

r2_new_names = {
    "fc_norm.weight",
    "fc_norm.bias"
} | {
    name for name in all_names
    if name.startswith("blocks.23.")
}

r3_new_names = {
    name for name in all_names
    if any(
        name.startswith(f"blocks.{i}.")
        for i in range(18, 23)
    )
}

r4_new_names = {
    name for name in all_names
    if any(
        name.startswith(f"blocks.{i}.")
        for i in range(12, 18)
    )
}

r2_names = r1_names | r2_new_names
r3_names = r2_names | r3_new_names
r4_names = r3_names | r4_new_names

assert len(all_names) == 296
assert len(r1_names) == 2
assert len(r2_new_names) == 14
assert len(r3_new_names) == 60
assert len(r4_new_names) == 72
assert len(r2_names) == 16
assert len(r3_names) == 76
assert len(r4_names) == 148
assert len(all_names - r4_names) == 148

print("\nMODEL")
print("Checkpoint epoch:", 24)
print("Strict checkpoint loading: PASS")
print("Total parameter tensors:", len(all_names))
print("R4 newly eligible tensors:", len(r4_new_names))
print("R4 cumulative eligible tensors:", len(r4_names))
print("R4 untouched tensors:", len(all_names - r4_names))
print("Parameters frozen: PASS")

# ============================================================
# 4. Original Grad-CAM calculation
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(
    activation_hook
)


def calculate_cam(input_tensor, target_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(input_tensor)

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, target_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2),
        keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        resized_max = float(cam.max())
        assert resized_max > 1e-12

        cam /= resized_max
        assert np.isfinite(cam).all()

    captured.clear()
    return output_logits, cam, raw_max


# ============================================================
# 5. CAM spatial similarity metrics
# ============================================================

def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    centered_a = a - a.mean()
    centered_b = b - b.mean()

    denominator = (
        np.linalg.norm(centered_a)
        * np.linalg.norm(centered_b)
    )

    pearson = (
        float(
            np.dot(centered_a, centered_b)
            / denominator
        )
        if denominator > 1e-12
        else np.nan
    )

    spearman = float(spearmanr(a, b).statistic)

    ssim = float(
        structural_similarity(
            first,
            second,
            data_range=1.0
        )
    )

    mae = float(np.mean(np.abs(a - b)))

    return {
        "pearson": pearson,
        "spearman": spearman,
        "ssim": ssim,
        "mae": mae
    }


# ============================================================
# 6. Randomization and parameter audit helpers
# ============================================================

def reinitialize_block(block):
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(
                    module.weight,
                    std=0.02
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()


def capture_subset(names):
    state = model.state_dict()

    return {
        name: state[name].detach().cpu().clone()
        for name in names
    }


outputs = {
    "R0": [],
    "R1": [],
    "R2": [],
    "R3": [],
    "R4": []
}


def evaluate_stage(stage_name):
    for tensor, grade in zip(inputs, PILOT_GRADES):
        logits, cam, raw_max = calculate_cam(
            tensor.to(DEVICE),
            grade
        )

        outputs[stage_name].append(
            (logits, cam, raw_max)
        )


# ============================================================
# 7. Execute cumulative R0-R4 randomization
# ============================================================

try:
    # R0: Original trained model.
    evaluate_stage("R0")

    baseline_pearsons = []

    for image_id, (_, cam, _) in zip(
        PILOT_IDS,
        outputs["R0"]
    ):
        assert cam is not None

        baseline_pearsons.append(
            compare_cams(
                cam,
                references[image_id]
            )["pearson"]
        )

    assert min(baseline_pearsons) > 0.999

    print("\nBASELINE REPRODUCTION")
    print(
        "Minimum saved-reference Pearson:",
        round(min(baseline_pearsons), 6)
    )
    print("Baseline CAM reproduction: PASS")

    # R1: Reinitialize classification head.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    r1_snapshot = capture_subset(r1_names)
    evaluate_stage("R1")

    # R2: Preserve R1; reinitialize fc_norm and block 23.
    with torch.no_grad():
        model.fc_norm.reset_parameters()

    reinitialize_block(model.blocks[23])

    r2_snapshot = capture_subset(r2_names)

    for name in r1_names:
        assert torch.equal(
            r1_snapshot[name],
            r2_snapshot[name]
        ), name

    evaluate_stage("R2")

    # R3: Preserve R2; reinitialize blocks 18-22.
    for block_index in range(18, 23):
        reinitialize_block(model.blocks[block_index])

    r3_snapshot = capture_subset(r3_names)

    for name in r2_names:
        assert torch.equal(
            r2_snapshot[name],
            r3_snapshot[name]
        ), name

    evaluate_stage("R3")

    # R4: Preserve R3; reinitialize blocks 12-17.
    r4_previous = capture_subset(r4_new_names)

    for block_index in range(12, 18):
        reinitialize_block(model.blocks[block_index])

    r4_snapshot = capture_subset(r4_names)

    for name in r3_names:
        assert torch.equal(
            r3_snapshot[name],
            r4_snapshot[name]
        ), name

    changed_new = {
        name for name in r4_new_names
        if not torch.equal(
            r4_previous[name],
            r4_snapshot[name]
        )
    }

    changed_total = []
    unauthorized = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()
        original = source_state[name]

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, original):
            changed_total.append(name)

            if name not in r4_names:
                unauthorized.append(name)

    assert not unauthorized, (
        f"Unauthorized changes: {unauthorized}"
    )

    assert len(changed_new) == 72, (
        f"Unexpected R4 new changes: {len(changed_new)}"
    )

    assert len(changed_total) == 148, (
        f"Unexpected R4 cumulative changes: {len(changed_total)}"
    )

    print("\nR4 PARAMETER INTEGRITY")
    print("R1 Head retained through R4: PASS")
    print("R2 state retained through R4: PASS")
    print("R3 state retained through R4: PASS")
    print("R4 newly changed tensors:", len(changed_new))
    print("R4 cumulative changed tensors:", len(changed_total))
    print("Original tensors still untouched:", 148)
    print("Unauthorized tensor changes:", len(unauthorized))
    print("All tensor values finite: PASS")

    evaluate_stage("R4")

finally:
    hook.remove()
    captured.clear()

    del model
    del checkpoint
    del source_state

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 8. Compare attribution across cumulative stages
# ============================================================

comparison_pairs = [
    ("R0", "R1"),
    ("R0", "R2"),
    ("R0", "R3"),
    ("R0", "R4"),
    ("R1", "R2"),
    ("R2", "R3"),
    ("R3", "R4")
]

records = []

for i, image_id in enumerate(PILOT_IDS):
    record = {
        "image_id": image_id,
        "true_grade": PILOT_GRADES[i]
    }

    for stage in ["R1", "R2", "R3", "R4"]:
        logits, cam, raw_max = outputs[stage][i]

        record[f"{stage.lower()}_degenerate"] = cam is None
        record[f"{stage.lower()}_raw_max"] = raw_max

    r0_logits = outputs["R0"][i][0]
    r3_logits = outputs["R3"][i][0]
    r4_logits = outputs["R4"][i][0]

    record["r0_r4_logit_l2"] = float(
        np.linalg.norm(r4_logits - r0_logits)
    )

    record["r3_r4_logit_l2"] = float(
        np.linalg.norm(r4_logits - r3_logits)
    )

    for first_stage, second_stage in comparison_pairs:
        first_cam = outputs[first_stage][i][1]
        second_cam = outputs[second_stage][i][1]

        metrics = compare_cams(first_cam, second_cam)

        prefix = (
            first_stage.lower()
            + "_"
            + second_stage.lower()
        )

        for metric, value in metrics.items():
            record[f"{prefix}_{metric}"] = value

    records.append(record)

results = pd.DataFrame(records)

assert len(results) == 8
assert results["image_id"].is_unique

# ============================================================
# 9. Report full pilot results
# ============================================================

print("\nPILOT SUMMARY")
print("Model: Meta MAE")
print("Seed:", SEED)
print("Images:", len(results))

for stage in ["r1", "r2", "r3", "r4"]:
    print(
        f"{stage.upper()} degenerate CAMs:",
        int(results[f"{stage}_degenerate"].sum())
    )

for first_stage, second_stage in comparison_pairs:
    prefix = (
        first_stage.lower()
        + "_"
        + second_stage.lower()
    )

    print(f"\n{prefix.upper()} SPATIAL COMPARISON")

    columns = [
        f"{prefix}_pearson",
        f"{prefix}_spearman",
        f"{prefix}_ssim",
        f"{prefix}_mae"
    ]

    valid = results[columns].dropna()

    print("Valid pairs:", len(valid))

    if not valid.empty:
        for metric in ["pearson", "spearman", "ssim", "mae"]:
            print(
                f"Mean {metric}:",
                round(
                    float(
                        valid[f"{prefix}_{metric}"].mean()
                    ),
                    6
                )
            )

print("\nLOGIT CHANGES")
print(
    "Mean R0-R4 logit L2:",
    round(
        float(results["r0_r4_logit_l2"].mean()),
        6
    )
)
print(
    "Mean R3-R4 logit L2:",
    round(
        float(results["r3_r4_logit_l2"].mean()),
        6
    )
)

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Original CAM files modified: False")
print("New research files created: False")
print("Randomization: Cumulative R1-R4")
print("Validation images decoded:", len(inputs))
print("Training performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.27 EXECUTION COMPLETE")
print("R4 pilot interpretation: PENDING REVIEW")


# %% [cell 365] Experiment 10 - Step 10.28

# ============================================================
# Experiment 10 - Step 10.28
# RETFound Cumulative R0-R5 Grad-CAM Pilot
# Seed 1042; eight validation images; read-only artifacts
# ============================================================

from pathlib import Path
import gc
import hashlib
import json

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

ROOT = DR_RESEARCH

INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"

CHECKPOINT_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1/best_model.pt"
)

REFERENCE_PATH = (
    ROOT / "experiment05_results/gradcam"
    / "RETFound_MA_CAM_Maps_V1.npz"
)

PROTOCOL_PATH = (
    ROOT / "experiment10_results"
    / "Experiment10_Randomization_Protocol_V1.json"
)

EXPECTED_INDEX_HASH = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

EXPECTED_CHECKPOINT_HASH = (
    "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
)

EXPECTED_PROTOCOL_HASH = (
    "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
)

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


print("EXPERIMENT 10.28 - RETFOUND R0-R5 PILOT")
print("Seed:", SEED)

# ============================================================
# 1. Verify frozen files and protocol
# ============================================================

for path in [
    INDEX_PATH,
    COHORT_PATH,
    CHECKPOINT_PATH,
    REFERENCE_PATH,
    PROTOCOL_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_HASH
assert sha256_file(CHECKPOINT_PATH) == EXPECTED_CHECKPOINT_HASH
assert sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_HASH

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

stages = protocol["randomization"]["stages"]

expected_stages = {
    "R1": {"head"},
    "R2": {"head", "fc_norm", "blocks.23"},
    "R3": {"head", "fc_norm"} | {
        f"blocks.{i}" for i in range(18, 24)
    },
    "R4": {"head", "fc_norm"} | {
        f"blocks.{i}" for i in range(12, 24)
    },
    "R5": {"head", "fc_norm"} | {
        f"blocks.{i}" for i in range(24)
    }
}

for stage, expected in expected_stages.items():
    assert set(stages[stage]) == expected, stage

assert SEED in protocol["randomization"]["randomization_seeds"]

print("\nFILE INTEGRITY")
print("Frozen index SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Frozen protocol SHA256: PASS")

# ============================================================
# 2. Verify validation cohort
# ============================================================

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276
assert len(PILOT_IDS) == 8
assert len(set(PILOT_IDS)) == 8

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    flag = str(
        cohort_lookup.loc[image_id, "primary_ma_candidate"]
    ).strip().lower()

    assert flag in {"true", "1"}

assert validation.loc[
    PILOT_IDS, "risk_group_id"
].nunique() == 8

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = [
        str(value) for value in archive["image_ids"]
    ]
    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: i for i, image_id in enumerate(reference_ids)
    }

    references = {
        image_id: archive["retfound_cam"][
            positions[image_id]
        ].astype(np.float32).copy()
        for image_id in PILOT_IDS
    }

for cam in references.values():
    assert cam.shape == (224, 224)
    assert np.isfinite(cam).all()

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

inputs = []

for image_id in PILOT_IDS:
    row = validation.loc[image_id]
    local = Path(row["local_path"])
    source = Path(row["source_path"])

    image_path = local if local.is_file() else source
    assert image_path.is_file(), image_id

    with Image.open(image_path) as image:
        inputs.append(
            preprocess(image.convert("RGB")).unsqueeze(0)
        )

print("\nCOHORT")
print("Pilot images:", len(inputs))
print("Distinct source-risk groups:", 8)
print("Reference CAMs:", len(references))
print("Validation-only cohort: PASS")

# ============================================================
# 3. Load original RETFound model
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

load_result = model.load_state_dict(
    source_state,
    strict=True
)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

all_names = set(model.state_dict())

r1_names = {"head.weight", "head.bias"}

r2_new = {
    "fc_norm.weight", "fc_norm.bias"
} | {
    name for name in all_names
    if name.startswith("blocks.23.")
}

r3_new = {
    name for name in all_names
    if any(
        name.startswith(f"blocks.{i}.")
        for i in range(18, 23)
    )
}

r4_new = {
    name for name in all_names
    if any(
        name.startswith(f"blocks.{i}.")
        for i in range(12, 18)
    )
}

r5_new = {
    name for name in all_names
    if any(
        name.startswith(f"blocks.{i}.")
        for i in range(12)
    )
}

r2_names = r1_names | r2_new
r3_names = r2_names | r3_new
r4_names = r3_names | r4_new
r5_names = r4_names | r5_new

assert len(all_names) == 296
assert len(r2_names) == 16
assert len(r3_names) == 76
assert len(r4_names) == 148
assert len(r5_new) == 144
assert len(r5_names) == 292

untouched_names = all_names - r5_names

assert untouched_names == {
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "cls_token",
    "pos_embed"
}

print("\nMODEL")
print("Strict checkpoint loading: PASS")
print("Total parameter tensors:", len(all_names))
print("R5 newly eligible tensors:", len(r5_new))
print("R5 cumulative eligible tensors:", len(r5_names))
print("R5 untouched tensors:", len(untouched_names))

# ============================================================
# 4. Grad-CAM computation
# ============================================================

captured = {}

def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation

hook = model.blocks[23].register_forward_hook(
    activation_hook
)

def calculate_cam(tensor, grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(tensor)

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2),
        keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12
        cam /= float(cam.max())

        assert np.isfinite(cam).all()

    captured.clear()

    return output_logits, cam, raw_max

def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - a.mean()
    bc = b - b.mean()

    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12 else np.nan
    )

    spearman = float(spearmanr(a, b).statistic)
    ssim = float(
        structural_similarity(
            first, second, data_range=1.0
        )
    )
    mae = float(np.mean(np.abs(a - b)))

    return {
        "pearson": pearson,
        "spearman": spearman,
        "ssim": ssim,
        "mae": mae
    }

# ============================================================
# 5. Controlled randomization helpers
# ============================================================

def reinitialize_block(block):
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()

def capture_subset(names):
    state = model.state_dict()
    return {
        name: state[name].detach().cpu().clone()
        for name in names
    }

outputs = {
    stage: []
    for stage in ["R0", "R1", "R2", "R3", "R4", "R5"]
}

def evaluate_stage(stage):
    for tensor, grade in zip(inputs, PILOT_GRADES):
        logits, cam, raw_max = calculate_cam(
            tensor.to(DEVICE), grade
        )
        outputs[stage].append((logits, cam, raw_max))

# ============================================================
# 6. Execute cumulative R0-R5 randomization
# ============================================================

try:
    evaluate_stage("R0")

    baseline_pearsons = []

    for image_id, (_, cam, _) in zip(
        PILOT_IDS, outputs["R0"]
    ):
        assert cam is not None

        baseline_pearsons.append(
            compare_cams(
                cam, references[image_id]
            )["pearson"]
        )

    assert min(baseline_pearsons) > 0.999

    print("\nBASELINE REPRODUCTION")
    print(
        "Minimum saved-reference Pearson:",
        round(min(baseline_pearsons), 6)
    )
    print("Baseline CAM reproduction: PASS")

    # R1: Classification head.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    r1_snapshot = capture_subset(r1_names)
    evaluate_stage("R1")

    # R2: Add fc_norm and block 23.
    with torch.no_grad():
        model.fc_norm.reset_parameters()

    reinitialize_block(model.blocks[23])

    r2_snapshot = capture_subset(r2_names)

    for name in r1_names:
        assert torch.equal(
            r1_snapshot[name], r2_snapshot[name]
        ), name

    evaluate_stage("R2")

    # R3: Add blocks 18-22.
    for block_index in range(18, 23):
        reinitialize_block(model.blocks[block_index])

    r3_snapshot = capture_subset(r3_names)

    for name in r2_names:
        assert torch.equal(
            r2_snapshot[name], r3_snapshot[name]
        ), name

    evaluate_stage("R3")

    # R4: Add blocks 12-17.
    for block_index in range(12, 18):
        reinitialize_block(model.blocks[block_index])

    r4_snapshot = capture_subset(r4_names)

    for name in r3_names:
        assert torch.equal(
            r3_snapshot[name], r4_snapshot[name]
        ), name

    evaluate_stage("R4")

    # R5: Preserve R4 and add blocks 0-11.
    r5_previous = capture_subset(r5_new)

    for block_index in range(12):
        reinitialize_block(model.blocks[block_index])

    r5_snapshot = capture_subset(r5_names)

    for name in r4_names:
        assert torch.equal(
            r4_snapshot[name], r5_snapshot[name]
        ), name

    changed_new = {
        name for name in r5_new
        if not torch.equal(
            r5_previous[name], r5_snapshot[name]
        )
    }

    unauthorized = []
    changed_total = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()
        original = source_state[name]

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, original):
            changed_total.append(name)

            if name not in r5_names:
                unauthorized.append(name)

    assert len(changed_new) == 144, (
        f"Unexpected R5 new changes: {len(changed_new)}"
    )

    assert len(changed_total) == 292, (
        f"Unexpected R5 cumulative changes: {len(changed_total)}"
    )

    assert len(unauthorized) == 0, unauthorized

    for name in untouched_names:
        assert torch.equal(
            model.state_dict()[name].detach().cpu(),
            source_state[name]
        ), name

    print("\nR5 PARAMETER INTEGRITY")
    print("R1-R4 state retained in R5: PASS")
    print("R5 newly changed tensors:", len(changed_new))
    print("R5 cumulative changed tensors:", len(changed_total))
    print("Original tensors still untouched:", len(untouched_names))
    print("Unauthorized tensor changes:", len(unauthorized))
    print("All tensor values finite: PASS")

    evaluate_stage("R5")

finally:
    hook.remove()
    captured.clear()

    del model
    del checkpoint
    del source_state

    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Calculate stage-wise comparisons
# ============================================================

comparison_pairs = [
    ("R0", "R1"),
    ("R0", "R2"),
    ("R0", "R3"),
    ("R0", "R4"),
    ("R0", "R5"),
    ("R1", "R2"),
    ("R2", "R3"),
    ("R3", "R4"),
    ("R4", "R5")
]

records = []

for i, image_id in enumerate(PILOT_IDS):
    record = {
        "image_id": image_id,
        "true_grade": PILOT_GRADES[i]
    }

    for stage in ["R1", "R2", "R3", "R4", "R5"]:
        _, cam, raw_max = outputs[stage][i]

        record[f"{stage.lower()}_degenerate"] = cam is None
        record[f"{stage.lower()}_raw_max"] = raw_max

    r0_logits = outputs["R0"][i][0]
    r4_logits = outputs["R4"][i][0]
    r5_logits = outputs["R5"][i][0]

    record["r0_r5_logit_l2"] = float(
        np.linalg.norm(r5_logits - r0_logits)
    )

    record["r4_r5_logit_l2"] = float(
        np.linalg.norm(r5_logits - r4_logits)
    )

    for first_stage, second_stage in comparison_pairs:
        first_cam = outputs[first_stage][i][1]
        second_cam = outputs[second_stage][i][1]

        metrics = compare_cams(first_cam, second_cam)

        prefix = (
            first_stage.lower()
            + "_"
            + second_stage.lower()
        )

        for metric, value in metrics.items():
            record[f"{prefix}_{metric}"] = value

    records.append(record)

results = pd.DataFrame(records)

assert len(results) == 8
assert results["image_id"].is_unique

# ============================================================
# 8. Report all results
# ============================================================

print("\nPILOT SUMMARY")
print("Model: RETFound")
print("Seed:", SEED)
print("Images:", len(results))

for stage in ["r1", "r2", "r3", "r4", "r5"]:
    print(
        f"{stage.upper()} degenerate CAMs:",
        int(results[f"{stage}_degenerate"].sum())
    )

for first_stage, second_stage in comparison_pairs:
    prefix = (
        first_stage.lower()
        + "_"
        + second_stage.lower()
    )

    print(f"\n{prefix.upper()} SPATIAL COMPARISON")

    columns = [
        f"{prefix}_pearson",
        f"{prefix}_spearman",
        f"{prefix}_ssim",
        f"{prefix}_mae"
    ]

    valid = results[columns].dropna()

    print("Valid pairs:", len(valid))

    if len(valid):
        for metric in ["pearson", "spearman", "ssim", "mae"]:
            print(
                f"Mean {metric}:",
                round(
                    float(valid[f"{prefix}_{metric}"].mean()),
                    6
                )
            )

print("\nLOGIT CHANGES")
print(
    "Mean R0-R5 logit L2:",
    round(float(results["r0_r5_logit_l2"].mean()), 6)
)
print(
    "Mean R4-R5 logit L2:",
    round(float(results["r4_r5_logit_l2"].mean()), 6)
)

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Original CAM files modified: False")
print("New research files created: False")
print("Randomization: Cumulative R1-R5")
print("Validation images decoded:", len(inputs))
print("Training performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.28 EXECUTION COMPLETE")
print("R5 pilot interpretation: PENDING REVIEW")


# %% [cell 366] Experiment 10 - Step 10.29
# ============================================================
# Experiment 10 - Step 10.29
# Meta MAE Cumulative R0-R5 Grad-CAM Pilot
# Seed 1042; eight validation images; read-only artifacts
# ============================================================

from pathlib import Path
import gc
import hashlib
import json

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

ROOT = DR_RESEARCH

INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
CHECKPOINT_PATH = ROOT / "experiment07_results/meta_mae_vit_large_v1/best_model.pt"
REFERENCE_PATH = ROOT / "experiment07_results/gradcam/Meta_MAE_MA_CAM_Maps_V1.npz"
PROTOCOL_PATH = ROOT / "experiment10_results/Experiment10_Randomization_Protocol_V1.json"

EXPECTED_HASHES = {
    INDEX_PATH: "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    CHECKPOINT_PATH: "d670c42bc5a18a394a4e1780918de04522915a42a7f62675f4fd6d89c4f8b312",
    PROTOCOL_PATH: "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
}

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]
SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available(), "CUDA is required."

print("EXPERIMENT 10.29 - META MAE R0-R5 PILOT")
print("Seed:", SEED)

# ============================================================
# 1. Frozen file and protocol integrity
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path in [INDEX_PATH, COHORT_PATH, CHECKPOINT_PATH,
             REFERENCE_PATH, PROTOCOL_PATH]:
    assert path.is_file(), f"Missing file: {path}"

for path, expected in EXPECTED_HASHES.items():
    assert sha256_file(path) == expected, f"Hash mismatch: {path}"

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

expected_stages = {
    "R1": {"head"},
    "R2": {"head", "fc_norm", "blocks.23"},
    "R3": {"head", "fc_norm"} | {f"blocks.{i}" for i in range(18, 24)},
    "R4": {"head", "fc_norm"} | {f"blocks.{i}" for i in range(12, 24)},
    "R5": {"head", "fc_norm"} | {f"blocks.{i}" for i in range(24)}
}

for stage, components in expected_stages.items():
    assert set(protocol["randomization"]["stages"][stage]) == components

assert SEED in protocol["randomization"]["randomization_seeds"]

print("\nFILE INTEGRITY")
print("Frozen index SHA256: PASS")
print("Meta MAE checkpoint SHA256: PASS")
print("Frozen protocol SHA256: PASS")

# ============================================================
# 2. Validate pilot cohort and load reference CAMs
# ============================================================

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)
cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[index["split"].eq("val")].set_index("image_id")
cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276
assert len(PILOT_IDS) == 8
assert len(set(PILOT_IDS)) == 8

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    flag = str(
        cohort_lookup.loc[image_id, "primary_ma_candidate"]
    ).strip().lower()

    assert flag in {"true", "1"}

assert validation.loc[PILOT_IDS, "risk_group_id"].nunique() == 8

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    assert {"image_ids", "meta_mae_cam"} <= set(archive.files)

    reference_ids = list(map(str, archive["image_ids"]))
    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {image_id: i for i, image_id in enumerate(reference_ids)}
    assert set(PILOT_IDS).issubset(positions)

    references = {
        image_id: archive["meta_mae_cam"][positions[image_id]]
        .astype(np.float32).copy()
        for image_id in PILOT_IDS
    }

for cam in references.values():
    assert cam.shape == (224, 224)
    assert np.isfinite(cam).all()

preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

inputs = []

for image_id in PILOT_IDS:
    row = validation.loc[image_id]
    local_path = Path(row["local_path"])
    source_path = Path(row["source_path"])
    image_path = local_path if local_path.is_file() else source_path

    assert image_path.is_file(), image_id

    with Image.open(image_path) as image:
        inputs.append(preprocess(image.convert("RGB")).unsqueeze(0))

print("\nCOHORT")
print("Pilot images:", len(inputs))
print("Distinct source-risk groups:", 8)
print("Reference CAMs:", len(references))
print("Validation-only cohort: PASS")

# ============================================================
# 3. Load independent Meta MAE model
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

assert checkpoint["epoch"] == 24
source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)
load_result = model.load_state_dict(source_state, strict=True)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

all_names = set(model.state_dict())

def names_for_blocks(start, stop):
    return {
        name for name in all_names
        if any(name.startswith(f"blocks.{i}.") for i in range(start, stop))
    }

r1_names = {"head.weight", "head.bias"}

r2_new = {"fc_norm.weight", "fc_norm.bias"} | names_for_blocks(23, 24)
r3_new = names_for_blocks(18, 23)
r4_new = names_for_blocks(12, 18)
r5_new = names_for_blocks(0, 12)

r2_names = r1_names | r2_new
r3_names = r2_names | r3_new
r4_names = r3_names | r4_new
r5_names = r4_names | r5_new

assert len(all_names) == 296
assert len(r1_names) == 2
assert len(r2_names) == 16
assert len(r3_names) == 76
assert len(r4_names) == 148
assert len(r5_new) == 144
assert len(r5_names) == 292

untouched_names = all_names - r5_names

assert untouched_names == {
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "cls_token",
    "pos_embed"
}

print("\nMODEL")
print("Checkpoint epoch:", 24)
print("Strict checkpoint loading: PASS")
print("Total parameter tensors:", len(all_names))
print("R5 newly eligible tensors:", len(r5_new))
print("R5 cumulative eligible tensors:", len(r5_names))
print("R5 untouched tensors:", len(untouched_names))

# ============================================================
# 4. Grad-CAM reconstruction
# ============================================================

captured = {}

def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation

hook = model.blocks[23].register_forward_hook(activation_hook)

def calculate_cam(tensor, grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tensor)

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :].detach().float()
        .transpose(0, 1).reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :].detach().float()
        .transpose(0, 1).reshape(1024, 14, 14)
    )

    weights = gradients.mean(dim=(1, 2), keepdim=True)

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()
    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw, (224, 224), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12
        cam /= float(cam.max())
        assert np.isfinite(cam).all()

    captured.clear()
    return output_logits, cam, raw_max

# ============================================================
# 5. CAM spatial comparison
# ============================================================

def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - a.mean()
    bc = b - b.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12 else np.nan
    )

    spearman = float(spearmanr(a, b).statistic)

    ssim = float(
        structural_similarity(first, second, data_range=1.0)
    )

    mae = float(np.mean(np.abs(a - b)))

    return {
        "pearson": pearson,
        "spearman": spearman,
        "ssim": ssim,
        "mae": mae
    }

# ============================================================
# 6. Randomization and parameter audit helpers
# ============================================================

def reinitialize_block(block):
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()

def capture_subset(names):
    state = model.state_dict()
    return {
        name: state[name].detach().cpu().clone()
        for name in names
    }

outputs = {
    stage: []
    for stage in ["R0", "R1", "R2", "R3", "R4", "R5"]
}

def evaluate_stage(stage):
    for tensor, grade in zip(inputs, PILOT_GRADES):
        logits, cam, raw_max = calculate_cam(
            tensor.to(DEVICE), grade
        )
        outputs[stage].append((logits, cam, raw_max))

# ============================================================
# 7. Execute cumulative R0-R5 randomization
# ============================================================

try:
    # R0: Original trained Meta MAE model.
    evaluate_stage("R0")

    baseline_pearsons = []

    for image_id, (_, cam, _) in zip(PILOT_IDS, outputs["R0"]):
        assert cam is not None
        baseline_pearsons.append(
            compare_cams(cam, references[image_id])["pearson"]
        )

    assert min(baseline_pearsons) > 0.999

    print("\nBASELINE REPRODUCTION")
    print(
        "Minimum saved-reference Pearson:",
        round(min(baseline_pearsons), 6)
    )
    print("Baseline CAM reproduction: PASS")

    # R1: Classification head.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    r1_snapshot = capture_subset(r1_names)
    evaluate_stage("R1")

    # R2: Add fc_norm and block 23.
    with torch.no_grad():
        model.fc_norm.reset_parameters()

    reinitialize_block(model.blocks[23])

    r2_snapshot = capture_subset(r2_names)

    for name in r1_names:
        assert torch.equal(r1_snapshot[name], r2_snapshot[name]), name

    evaluate_stage("R2")

    # R3: Add blocks 18-22.
    for block_index in range(18, 23):
        reinitialize_block(model.blocks[block_index])

    r3_snapshot = capture_subset(r3_names)

    for name in r2_names:
        assert torch.equal(r2_snapshot[name], r3_snapshot[name]), name

    evaluate_stage("R3")

    # R4: Add blocks 12-17.
    for block_index in range(12, 18):
        reinitialize_block(model.blocks[block_index])

    r4_snapshot = capture_subset(r4_names)

    for name in r3_names:
        assert torch.equal(r3_snapshot[name], r4_snapshot[name]), name

    evaluate_stage("R4")

    # R5: Preserve R4, reinitialize blocks 0-11.
    r5_previous = capture_subset(r5_new)

    for block_index in range(12):
        reinitialize_block(model.blocks[block_index])

    r5_snapshot = capture_subset(r5_names)

    for name in r4_names:
        assert torch.equal(r4_snapshot[name], r5_snapshot[name]), name

    changed_new = {
        name for name in r5_new
        if not torch.equal(r5_previous[name], r5_snapshot[name])
    }

    changed_total = []
    unauthorized = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()
        original = source_state[name]

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, original):
            changed_total.append(name)

            if name not in r5_names:
                unauthorized.append(name)

    assert len(changed_new) == 144, (
        f"Unexpected R5 new changes: {len(changed_new)}"
    )

    assert len(changed_total) == 292, (
        f"Unexpected R5 cumulative changes: {len(changed_total)}"
    )

    assert not unauthorized, unauthorized

    for name in untouched_names:
        assert torch.equal(
            model.state_dict()[name].detach().cpu(),
            source_state[name]
        ), name

    print("\nR5 PARAMETER INTEGRITY")
    print("R1-R4 state retained in R5: PASS")
    print("R5 newly changed tensors:", len(changed_new))
    print("R5 cumulative changed tensors:", len(changed_total))
    print("Original tensors still untouched:", len(untouched_names))
    print("Unauthorized tensor changes:", len(unauthorized))
    print("All tensor values finite: PASS")

    evaluate_stage("R5")

finally:
    hook.remove()
    captured.clear()

    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 8. Calculate comparisons across all stages
# ============================================================

comparison_pairs = [
    ("R0", "R1"),
    ("R0", "R2"),
    ("R0", "R3"),
    ("R0", "R4"),
    ("R0", "R5"),
    ("R1", "R2"),
    ("R2", "R3"),
    ("R3", "R4"),
    ("R4", "R5")
]

records = []

for i, image_id in enumerate(PILOT_IDS):
    record = {
        "image_id": image_id,
        "true_grade": PILOT_GRADES[i]
    }

    for stage in ["R1", "R2", "R3", "R4", "R5"]:
        _, cam, raw_max = outputs[stage][i]
        record[f"{stage.lower()}_degenerate"] = cam is None
        record[f"{stage.lower()}_raw_max"] = raw_max

    r0_logits = outputs["R0"][i][0]
    r4_logits = outputs["R4"][i][0]
    r5_logits = outputs["R5"][i][0]

    record["r0_r5_logit_l2"] = float(
        np.linalg.norm(r5_logits - r0_logits)
    )

    record["r4_r5_logit_l2"] = float(
        np.linalg.norm(r5_logits - r4_logits)
    )

    for first_stage, second_stage in comparison_pairs:
        first_cam = outputs[first_stage][i][1]
        second_cam = outputs[second_stage][i][1]

        metrics = compare_cams(first_cam, second_cam)
        prefix = f"{first_stage.lower()}_{second_stage.lower()}"

        for metric, value in metrics.items():
            record[f"{prefix}_{metric}"] = value

    records.append(record)

results = pd.DataFrame(records)

assert len(results) == 8
assert results["image_id"].is_unique

# ============================================================
# 9. Report full results
# ============================================================

print("\nPILOT SUMMARY")
print("Model: Meta MAE")
print("Seed:", SEED)
print("Images:", len(results))

for stage in ["r1", "r2", "r3", "r4", "r5"]:
    print(
        f"{stage.upper()} degenerate CAMs:",
        int(results[f"{stage}_degenerate"].sum())
    )

for first_stage, second_stage in comparison_pairs:
    prefix = f"{first_stage.lower()}_{second_stage.lower()}"

    print(f"\n{prefix.upper()} SPATIAL COMPARISON")

    columns = [
        f"{prefix}_pearson",
        f"{prefix}_spearman",
        f"{prefix}_ssim",
        f"{prefix}_mae"
    ]

    valid = results[columns].dropna()
    print("Valid pairs:", len(valid))

    if not valid.empty:
        for metric in ["pearson", "spearman", "ssim", "mae"]:
            print(
                f"Mean {metric}:",
                round(float(valid[f"{prefix}_{metric}"].mean()), 6)
            )

print("\nLOGIT CHANGES")
print(
    "Mean R0-R5 logit L2:",
    round(float(results["r0_r5_logit_l2"].mean()), 6)
)
print(
    "Mean R4-R5 logit L2:",
    round(float(results["r4_r5_logit_l2"].mean()), 6)
)

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Original CAM files modified: False")
print("New research files created: False")
print("Randomization: Cumulative R1-R5")
print("Validation images decoded:", len(inputs))
print("Training performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.29 EXECUTION COMPLETE")
print("R5 pilot interpretation: PENDING REVIEW")


# %% [cell 367] Experiment 10 - Step 10.30
# ============================================================
# Experiment 10 - Step 10.30
# RETFound R5-to-R6 Full-Model Randomization Pilot
# Seed 1042; eight validation images; read-only artifacts
# ============================================================

from pathlib import Path
import gc
import hashlib
import json

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

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
)

CHECKPOINT_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1/best_model.pt"
)

REFERENCE_PATH = (
    ROOT / "experiment05_results/gradcam"
    / "RETFound_MA_CAM_Maps_V1.npz"
)

PROTOCOL_PATH = (
    ROOT / "experiment10_results"
    / "Experiment10_Randomization_Protocol_V1.json"
)

EXPECTED_HASHES = {
    INDEX_PATH: (
        "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
    ),
    CHECKPOINT_PATH: (
        "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
    ),
    PROTOCOL_PATH: (
        "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
    )
}

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


print("EXPERIMENT 10.30 - RETFOUND R5 TO R6 PILOT")
print("Seed:", SEED)

# ============================================================
# 1. Verify frozen artifacts and protocol
# ============================================================

for path in [
    INDEX_PATH,
    COHORT_PATH,
    CHECKPOINT_PATH,
    REFERENCE_PATH,
    PROTOCOL_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

for path, expected_hash in EXPECTED_HASHES.items():
    assert sha256_file(path) == expected_hash, (
        f"Hash mismatch: {path}"
    )

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

stages = protocol["randomization"]["stages"]

expected_r5 = {"head", "fc_norm"} | {
    f"blocks.{i}" for i in range(24)
}

expected_r6 = expected_r5 | {
    "patch_embed",
    "cls_token",
    "pos_embed"
}

assert set(stages["R5"]) == expected_r5
assert set(stages["R6"]) == expected_r6
assert SEED in protocol["randomization"]["randomization_seeds"]

print("\nFILE INTEGRITY")
print("Frozen index SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Frozen protocol SHA256: PASS")

# ============================================================
# 2. Verify frozen pilot and reference CAMs
# ============================================================

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276
assert len(PILOT_IDS) == 8
assert len(set(PILOT_IDS)) == 8

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    flag = str(
        cohort_lookup.loc[image_id, "primary_ma_candidate"]
    ).strip().lower()

    assert flag in {"true", "1"}

assert (
    validation.loc[PILOT_IDS, "risk_group_id"].nunique()
    == 8
)

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: i
        for i, image_id in enumerate(reference_ids)
    }

    references = {
        image_id: archive["retfound_cam"][
            positions[image_id]
        ].astype(np.float32).copy()
        for image_id in PILOT_IDS
    }

for cam in references.values():
    assert cam.shape == (224, 224)
    assert np.isfinite(cam).all()

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

inputs = []

for image_id in PILOT_IDS:
    row = validation.loc[image_id]

    local_path = Path(row["local_path"])
    source_path = Path(row["source_path"])

    image_path = (
        local_path if local_path.is_file()
        else source_path
    )

    assert image_path.is_file(), image_id

    with Image.open(image_path) as image:
        inputs.append(
            preprocess(image.convert("RGB")).unsqueeze(0)
        )

print("\nCOHORT")
print("Validation pilot images:", len(inputs))
print("Distinct source-risk groups:", 8)
print("Reference CAMs:", len(references))
print("Test set accessed: False")

# ============================================================
# 3. Load original RETFound model
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]

assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

load_result = model.load_state_dict(
    source_state,
    strict=True
)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

all_names = set(model.state_dict())

r6_new_names = {
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "cls_token",
    "pos_embed"
}

r5_names = all_names - r6_new_names

assert len(all_names) == 296
assert len(r5_names) == 292
assert len(r6_new_names) == 4

print("\nMODEL")
print("Strict checkpoint loading: PASS")
print("Total parameter tensors:", len(all_names))
print("R5 eligible tensors:", len(r5_names))
print("R6 newly eligible tensors:", len(r6_new_names))
print("R6 cumulative eligible tensors:", len(all_names))

# ============================================================
# 4. Grad-CAM computation
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(
    activation_hook
)


def calculate_cam(tensor, grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(tensor)

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2),
        keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12

        cam /= float(cam.max())
        assert np.isfinite(cam).all()

    captured.clear()

    return output_logits, cam, raw_max


def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - a.mean()
    bc = b - b.mean()

    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12
        else np.nan
    )

    spearman = float(spearmanr(a, b).statistic)

    ssim = float(
        structural_similarity(
            first,
            second,
            data_range=1.0
        )
    )

    mae = float(np.mean(np.abs(a - b)))

    return {
        "pearson": pearson,
        "spearman": spearman,
        "ssim": ssim,
        "mae": mae
    }


# ============================================================
# 5. Controlled initialization helpers
# ============================================================

def reinitialize_block(block):
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()


def evaluate_stage():
    stage_outputs = []

    for tensor, grade in zip(inputs, PILOT_GRADES):
        logits, cam, raw_max = calculate_cam(
            tensor.to(DEVICE), grade
        )

        stage_outputs.append(
            (logits, cam, raw_max)
        )

    return stage_outputs


# ============================================================
# 6. Evaluate R0 and cumulatively initialize R1-R5
# ============================================================

try:
    r0_outputs = evaluate_stage()

    baseline_pearsons = []

    for image_id, (_, cam, _) in zip(
        PILOT_IDS, r0_outputs
    ):
        assert cam is not None

        baseline_pearsons.append(
            compare_cams(
                cam, references[image_id]
            )["pearson"]
        )

    assert min(baseline_pearsons) > 0.999

    print("\nBASELINE REPRODUCTION")
    print(
        "Minimum saved-reference Pearson:",
        round(min(baseline_pearsons), 6)
    )
    print("Baseline CAM reproduction: PASS")

    # R1: Classification head.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    # R2: Final normalization and Block 23.
    with torch.no_grad():
        model.fc_norm.reset_parameters()

    reinitialize_block(model.blocks[23])

    # R3: Blocks 18-22.
    for block_index in range(18, 23):
        reinitialize_block(model.blocks[block_index])

    # R4: Blocks 12-17.
    for block_index in range(12, 18):
        reinitialize_block(model.blocks[block_index])

    # R5: Blocks 0-11.
    for block_index in range(12):
        reinitialize_block(model.blocks[block_index])

    # Verify the complete R5 boundary state.
    r5_changed = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, source_state[name]):
            r5_changed.append(name)

    assert set(r5_changed) == r5_names

    # Store exact R5 tensor fingerprints to audit R6 retention.
    # Fingerprints are kept in memory and are not written to disk.
    r5_fingerprints = {}

    for name in sorted(r5_names):
        tensor = model.state_dict()[name].detach().cpu()
        r5_fingerprints[name] = hashlib.sha256(
            tensor.contiguous().numpy().tobytes()
        ).hexdigest()

    r6_before = {
        name: model.state_dict()[name].detach().cpu().clone()
        for name in r6_new_names
    }

    print("\nR5 BOUNDARY INTEGRITY")
    print("R5 changed tensors:", len(r5_changed))
    print("R5 untouched tensors:", len(r6_new_names))
    print("R5 frozen-state verification: PASS")

    r5_outputs = evaluate_stage()

    # R6: Reinitialize the remaining four tensors.
    # Keep the random generator sequence cumulative.
    with torch.no_grad():
        model.patch_embed.proj.reset_parameters()
        nn.init.normal_(model.cls_token, std=1e-6)
        trunc_normal_(model.pos_embed, std=0.02)

    r6_after = {
        name: model.state_dict()[name].detach().cpu().clone()
        for name in r6_new_names
    }

    changed_r6 = {
        name for name in r6_new_names
        if not torch.equal(
            r6_before[name], r6_after[name]
        )
    }

    assert changed_r6 == r6_new_names

    # Independently verify preservation of all 292 R5 tensors.
    retained_count = 0

    for name in sorted(r5_names):
        tensor = model.state_dict()[name].detach().cpu()
        fingerprint = hashlib.sha256(
            tensor.contiguous().numpy().tobytes()
        ).hexdigest()

        assert fingerprint == r5_fingerprints[name], name
        retained_count += 1

    # Verify that all 296 tensors now differ from the original.
    changed_total = []
    nonfinite = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()

        if not torch.isfinite(current).all().item():
            nonfinite.append(name)

        if not torch.equal(current, source_state[name]):
            changed_total.append(name)

    assert not nonfinite, nonfinite
    assert len(changed_total) == 296

    print("\nR6 PARAMETER INTEGRITY")
    print("R5 tensors retained:", retained_count)
    print("R6 newly changed tensors:", len(changed_r6))
    print("R6 cumulative changed tensors:", len(changed_total))
    print("Unauthorized tensor changes: 0")
    print("All tensor values finite: PASS")

    r6_outputs = evaluate_stage()

finally:
    hook.remove()
    captured.clear()

    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Compare R0, R5, and R6
# ============================================================

records = []

for i, image_id in enumerate(PILOT_IDS):
    r0_logits, r0_cam, _ = r0_outputs[i]
    r5_logits, r5_cam, r5_raw_max = r5_outputs[i]
    r6_logits, r6_cam, r6_raw_max = r6_outputs[i]

    r0_r5 = compare_cams(r0_cam, r5_cam)
    r0_r6 = compare_cams(r0_cam, r6_cam)
    r5_r6 = compare_cams(r5_cam, r6_cam)

    record = {
        "image_id": image_id,
        "true_grade": PILOT_GRADES[i],
        "r5_degenerate": r5_cam is None,
        "r6_degenerate": r6_cam is None,
        "r5_raw_max": r5_raw_max,
        "r6_raw_max": r6_raw_max,
        "r0_r5_logit_l2": float(
            np.linalg.norm(r5_logits - r0_logits)
        ),
        "r0_r6_logit_l2": float(
            np.linalg.norm(r6_logits - r0_logits)
        ),
        "r5_r6_logit_l2": float(
            np.linalg.norm(r6_logits - r5_logits)
        )
    }

    for prefix, metrics in [
        ("r0_r5", r0_r5),
        ("r0_r6", r0_r6),
        ("r5_r6", r5_r6)
    ]:
        for metric, value in metrics.items():
            record[f"{prefix}_{metric}"] = value

    records.append(record)

results = pd.DataFrame(records)

assert len(results) == 8
assert results["image_id"].is_unique

# ============================================================
# 8. Final pilot reporting
# ============================================================

print("\nPILOT SUMMARY")
print("Model: RETFound")
print("Seed:", SEED)
print("Images:", len(results))

print(
    "R5 degenerate CAMs:",
    int(results["r5_degenerate"].sum())
)

print(
    "R6 degenerate CAMs:",
    int(results["r6_degenerate"].sum())
)

for prefix in ["r0_r5", "r0_r6", "r5_r6"]:
    columns = [
        f"{prefix}_pearson",
        f"{prefix}_spearman",
        f"{prefix}_ssim",
        f"{prefix}_mae"
    ]

    valid = results[columns].dropna()

    print(f"\n{prefix.upper()} SPATIAL COMPARISON")
    print("Valid pairs:", len(valid))

    if len(valid):
        for metric in ["pearson", "spearman", "ssim", "mae"]:
            column = f"{prefix}_{metric}"

            print(
                f"Mean {metric}:",
                round(float(valid[column].mean()), 6)
            )

print("\nLOGIT CHANGES")

for column in [
    "r0_r5_logit_l2",
    "r0_r6_logit_l2",
    "r5_r6_logit_l2"
]:
    print(
        f"Mean {column}:",
        round(float(results[column].mean()), 6)
    )

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Original CAM files modified: False")
print("New research files created: False")
print("Randomization: Cumulative R1-R6")
print("Validation images decoded:", len(inputs))
print("Training performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.30 EXECUTION COMPLETE")
print("R6 pilot interpretation: PENDING REVIEW")


# %% [cell 368] Experiment 10 - Step 10.31
# ============================================================
# Experiment 10 - Step 10.31
# Meta MAE R5-to-R6 Full-Model Randomization Pilot
# Seed 1042; validation only; no artifact writes
# ============================================================

from pathlib import Path
import gc
import hashlib
import json

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

ROOT = DR_RESEARCH

INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
CHECKPOINT_PATH = ROOT / "experiment07_results/meta_mae_vit_large_v1/best_model.pt"
REFERENCE_PATH = ROOT / "experiment07_results/gradcam/Meta_MAE_MA_CAM_Maps_V1.npz"
PROTOCOL_PATH = ROOT / "experiment10_results/Experiment10_Randomization_Protocol_V1.json"

EXPECTED_HASHES = {
    INDEX_PATH: "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    CHECKPOINT_PATH: "d670c42bc5a18a394a4e1780918de04522915a42a7f62675f4fd6d89c4f8b312",
    PROTOCOL_PATH: "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
}

PILOT_IDS = [
    "1301_1", "0421_1", "1190_1", "0732_1",
    "0001_2", "0011_3", "0032_3", "0043_3"
]
PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

SEED = 1042
DEVICE = torch.device("cuda")
assert torch.cuda.is_available()

print("EXPERIMENT 10.31 - META MAE R5 TO R6 PILOT")
print("Seed:", SEED)

# ------------------------------------------------------------
# 1. Frozen artifact and protocol verification
# ------------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path in [INDEX_PATH, COHORT_PATH, CHECKPOINT_PATH,
             REFERENCE_PATH, PROTOCOL_PATH]:
    assert path.is_file(), f"Missing file: {path}"

for path, expected in EXPECTED_HASHES.items():
    assert sha256_file(path) == expected, f"SHA256 mismatch: {path}"

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

expected_r5 = {"head", "fc_norm"} | {
    f"blocks.{i}" for i in range(24)
}
expected_r6 = expected_r5 | {
    "patch_embed", "cls_token", "pos_embed"
}

assert set(protocol["randomization"]["stages"]["R5"]) == expected_r5
assert set(protocol["randomization"]["stages"]["R6"]) == expected_r6
assert SEED in protocol["randomization"]["randomization_seeds"]
assert protocol["evaluation"]["test_set_access"] is False

print("\nFILE INTEGRITY")
print("Frozen index SHA256: PASS")
print("Meta MAE checkpoint SHA256: PASS")
print("Frozen protocol SHA256: PASS")

# ------------------------------------------------------------
# 2. Verify frozen pilot and reference CAMs
# ------------------------------------------------------------

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)
cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[index["split"].eq("val")].set_index("image_id")
cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276
assert len(set(PILOT_IDS)) == 8

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    flag = str(
        cohort_lookup.loc[image_id, "primary_ma_candidate"]
    ).strip().lower()
    assert flag in {"true", "1"}

assert validation.loc[PILOT_IDS, "risk_group_id"].nunique() == 8

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))
    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {image_id: i for i, image_id in enumerate(reference_ids)}
    assert set(PILOT_IDS).issubset(positions)

    references = {
        image_id: archive["meta_mae_cam"][positions[image_id]]
        .astype(np.float32).copy()
        for image_id in PILOT_IDS
    }

for cam in references.values():
    assert cam.shape == (224, 224)
    assert np.isfinite(cam).all()

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

inputs = []

for image_id in PILOT_IDS:
    row = validation.loc[image_id]
    local_path = Path(row["local_path"])
    source_path = Path(row["source_path"])
    image_path = local_path if local_path.is_file() else source_path

    assert image_path.is_file(), image_id

    with Image.open(image_path) as image:
        inputs.append(preprocess(image.convert("RGB")).unsqueeze(0))

print("\nCOHORT")
print("Validation pilot images:", len(inputs))
print("Distinct source-risk groups:", 8)
print("Reference CAMs:", len(references))
print("Test set accessed: False")

# ------------------------------------------------------------
# 3. Load original model
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

assert checkpoint["epoch"] == 24

source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)
loaded = model.load_state_dict(source_state, strict=True)

assert not loaded.missing_keys
assert not loaded.unexpected_keys

model.eval()
model.requires_grad_(False)

r6_new_names = {
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "cls_token",
    "pos_embed"
}

all_names = set(model.state_dict())
r5_names = all_names - r6_new_names

assert len(all_names) == 296
assert len(r5_names) == 292
assert len(r6_new_names) == 4

print("\nMODEL")
print("Checkpoint epoch: 24")
print("Strict checkpoint loading: PASS")
print("Total parameter tensors:", len(all_names))
print("R5 eligible tensors:", len(r5_names))
print("R6 newly eligible tensors:", len(r6_new_names))

# ------------------------------------------------------------
# 4. Original Grad-CAM implementation
# ------------------------------------------------------------

captured = {}

def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation

hook = model.blocks[23].register_forward_hook(activation_hook)

def calculate_cam(tensor, grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(tensor)

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :].detach().float()
        .transpose(0, 1).reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :].detach().float()
        .transpose(0, 1).reshape(1024, 14, 14)
    )

    weights = gradients.mean(dim=(1, 2), keepdim=True)

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()
    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12
        cam /= float(cam.max())
        assert np.isfinite(cam).all()

    captured.clear()
    return output_logits, cam, raw_max

def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - a.mean()
    bc = b - b.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12 else np.nan
    )

    return {
        "pearson": pearson,
        "spearman": float(spearmanr(a, b).statistic),
        "ssim": float(
            structural_similarity(first, second, data_range=1.0)
        ),
        "mae": float(np.mean(np.abs(a - b)))
    }

# ------------------------------------------------------------
# 5. Cumulative randomization helpers
# ------------------------------------------------------------

def reinitialize_block(block):
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()

def evaluate_stage():
    stage_outputs = []

    for tensor, grade in zip(inputs, PILOT_GRADES):
        logits, cam, raw_max = calculate_cam(
            tensor.to(DEVICE), grade
        )
        stage_outputs.append((logits, cam, raw_max))

    return stage_outputs

# ------------------------------------------------------------
# 6. Execute R0 -> R5 -> R6
# ------------------------------------------------------------

try:
    r0_outputs = evaluate_stage()

    baseline_pearsons = []

    for image_id, (_, cam, _) in zip(PILOT_IDS, r0_outputs):
        assert cam is not None
        baseline_pearsons.append(
            compare_cams(cam, references[image_id])["pearson"]
        )

    assert min(baseline_pearsons) > 0.999

    print("\nBASELINE REPRODUCTION")
    print(
        "Minimum saved-reference Pearson:",
        round(min(baseline_pearsons), 6)
    )
    print("Baseline CAM reproduction: PASS")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # R1: Classification head.
    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    # R2: Final normalization and Block 23.
    with torch.no_grad():
        model.fc_norm.reset_parameters()
    reinitialize_block(model.blocks[23])

    # R3: Blocks 18-22.
    for block_index in range(18, 23):
        reinitialize_block(model.blocks[block_index])

    # R4: Blocks 12-17.
    for block_index in range(12, 18):
        reinitialize_block(model.blocks[block_index])

    # R5: Blocks 0-11.
    for block_index in range(12):
        reinitialize_block(model.blocks[block_index])

    # Audit the complete R5 boundary.
    r5_changed = set()

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()
        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, source_state[name]):
            r5_changed.add(name)

    assert r5_changed == r5_names

    # Store exact R5 fingerprints without duplicating all
    # parameter tensors or writing files.
    r5_fingerprints = {}

    for name in sorted(r5_names):
        tensor = model.state_dict()[name].detach().cpu()
        r5_fingerprints[name] = hashlib.sha256(
            tensor.contiguous().numpy().tobytes()
        ).hexdigest()

    r6_before = {
        name: model.state_dict()[name].detach().cpu().clone()
        for name in r6_new_names
    }

    print("\nR5 BOUNDARY INTEGRITY")
    print("R5 changed tensors:", len(r5_changed))
    print("R5 untouched tensors:", len(r6_new_names))
    print("R5 frozen-state verification: PASS")

    r5_outputs = evaluate_stage()

    # R6: Reinitialize patch embedding, CLS, and positions.
    # Do not reset the random seed at the R6 boundary.
    with torch.no_grad():
        model.patch_embed.proj.reset_parameters()
        nn.init.normal_(model.cls_token, std=1e-6)
        trunc_normal_(model.pos_embed, std=0.02)

    r6_after = {
        name: model.state_dict()[name].detach().cpu().clone()
        for name in r6_new_names
    }

    changed_r6 = {
        name for name in r6_new_names
        if not torch.equal(
            r6_before[name], r6_after[name]
        )
    }

    assert changed_r6 == r6_new_names

    retained_count = 0

    for name in sorted(r5_names):
        tensor = model.state_dict()[name].detach().cpu()
        fingerprint = hashlib.sha256(
            tensor.contiguous().numpy().tobytes()
        ).hexdigest()

        assert fingerprint == r5_fingerprints[name], name
        retained_count += 1

    changed_total = set()
    nonfinite = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()

        if not torch.isfinite(current).all().item():
            nonfinite.append(name)

        if not torch.equal(current, source_state[name]):
            changed_total.add(name)

    assert not nonfinite, nonfinite
    assert changed_total == all_names

    print("\nR6 PARAMETER INTEGRITY")
    print("R5 tensors retained:", retained_count)
    print("R6 newly changed tensors:", len(changed_r6))
    print("R6 cumulative changed tensors:", len(changed_total))
    print("Unauthorized tensor changes: 0")
    print("All tensor values finite: PASS")

    r6_outputs = evaluate_stage()

finally:
    hook.remove()
    captured.clear()
    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# 7. Report R0-R5, R0-R6, R5-R6 comparisons
# ------------------------------------------------------------

records = []

for i, image_id in enumerate(PILOT_IDS):
    r0_logits, r0_cam, _ = r0_outputs[i]
    r5_logits, r5_cam, r5_raw_max = r5_outputs[i]
    r6_logits, r6_cam, r6_raw_max = r6_outputs[i]

    comparisons = {
        "r0_r5": compare_cams(r0_cam, r5_cam),
        "r0_r6": compare_cams(r0_cam, r6_cam),
        "r5_r6": compare_cams(r5_cam, r6_cam)
    }

    record = {
        "image_id": image_id,
        "true_grade": PILOT_GRADES[i],
        "r5_degenerate": r5_cam is None,
        "r6_degenerate": r6_cam is None,
        "r5_raw_max": r5_raw_max,
        "r6_raw_max": r6_raw_max,
        "r0_r5_logit_l2": float(
            np.linalg.norm(r5_logits - r0_logits)
        ),
        "r0_r6_logit_l2": float(
            np.linalg.norm(r6_logits - r0_logits)
        ),
        "r5_r6_logit_l2": float(
            np.linalg.norm(r6_logits - r5_logits)
        )
    }

    for prefix, metrics in comparisons.items():
        for metric, value in metrics.items():
            record[f"{prefix}_{metric}"] = value

    records.append(record)

results = pd.DataFrame(records)

assert len(results) == 8
assert results["image_id"].is_unique

print("\nPILOT SUMMARY")
print("Model: Meta MAE")
print("Seed:", SEED)
print("Images:", len(results))
print(
    "R5 degenerate CAMs:",
    int(results["r5_degenerate"].sum())
)
print(
    "R6 degenerate CAMs:",
    int(results["r6_degenerate"].sum())
)

for prefix in ["r0_r5", "r0_r6", "r5_r6"]:
    columns = [
        f"{prefix}_pearson",
        f"{prefix}_spearman",
        f"{prefix}_ssim",
        f"{prefix}_mae"
    ]

    valid = results[columns].dropna()

    print(f"\n{prefix.upper()} SPATIAL COMPARISON")
    print("Valid pairs:", len(valid))

    if not valid.empty:
        for metric in ["pearson", "spearman", "ssim", "mae"]:
            print(
                f"Mean {metric}:",
                round(float(valid[f"{prefix}_{metric}"].mean()), 6)
            )

print("\nLOGIT CHANGES")

for column in [
    "r0_r5_logit_l2",
    "r0_r6_logit_l2",
    "r5_r6_logit_l2"
]:
    print(
        f"Mean {column}:",
        round(float(results[column].mean()), 6)
    )

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nRESEARCH INTEGRITY")
print("Original checkpoint modified: False")
print("Original CAM files modified: False")
print("New research files created: False")
print("Randomization: Cumulative R1-R6")
print("Validation images decoded:", len(inputs))
print("Training performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.31 EXECUTION COMPLETE")
print("R6 pilot interpretation: PENDING REVIEW")


# %% [cell 369] Experiment 10 - Step 10.32
# ============================================================
# Experiment 10 - Step 10.32
# Cross-Model R6 Randomized Parameter Identity Audit
# Seed 1042; no images; no artifact writes
# ============================================================

from pathlib import Path
import gc
import hashlib

import torch
import torch.nn as nn
import timm
from timm.layers import trunc_normal_

ROOT = DR_RESEARCH

CHECKPOINTS = {
    "RETFound": ROOT / (
        "experiment04_results/"
        "retfound_vit_large_v1/best_model.pt"
    ),
    "Meta_MAE": ROOT / (
        "experiment07_results/"
        "meta_mae_vit_large_v1/best_model.pt"
    )
}

EXPECTED_HASHES = {
    "RETFound": (
        "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
    ),
    "Meta_MAE": (
        "d670c42bc5a18a394a4e1780918de04522915a42a7f62675f4fd6d89c4f8b312"
    )
}

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available(), "CUDA is required."


def file_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def tensor_sha256(tensor):
    # Include shape and dtype in the fingerprint.
    value = tensor.detach().cpu().contiguous()

    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.numpy().tobytes())

    return digest.hexdigest()


def reinitialize_block(block):
    with torch.no_grad():

        for module in block.modules():

            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()


def audit_one_model(model_name, checkpoint_path):

    assert checkpoint_path.is_file()

    assert (
        file_sha256(checkpoint_path)
        == EXPECTED_HASHES[model_name]
    ), f"Checkpoint hash mismatch: {model_name}"

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    source_state = checkpoint["model_state_dict"]

    assert len(source_state) == 296

    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=5,
        global_pool="avg",
        device="meta",
        weight_init="skip"
    )

    model = model.to_empty(device=DEVICE)

    loaded = model.load_state_dict(
        source_state,
        strict=True
    )

    assert not loaded.missing_keys
    assert not loaded.unexpected_keys

    model.eval()
    model.requires_grad_(False)

    # Capture a few original tensors to verify that
    # the source checkpoints are distinct.
    original_fingerprints = {
        name: tensor_sha256(source_state[name])
        for name in [
            "head.weight",
            "blocks.0.attn.qkv.weight",
            "patch_embed.proj.weight"
        ]
    }

    # Reset the random generator at the start of each
    # model's cumulative randomization sequence.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # R1: Classification head.
    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    # R2: Final normalization and Block 23.
    with torch.no_grad():
        model.fc_norm.reset_parameters()

    reinitialize_block(model.blocks[23])

    # R3: Blocks 18-22.
    for block_index in range(18, 23):
        reinitialize_block(model.blocks[block_index])

    # R4: Blocks 12-17.
    for block_index in range(12, 18):
        reinitialize_block(model.blocks[block_index])

    # R5: Blocks 0-11.
    for block_index in range(12):
        reinitialize_block(model.blocks[block_index])

    r5_fingerprints = {
        name: tensor_sha256(tensor)
        for name, tensor in model.state_dict().items()
    }

    # R6: Patch embedding, CLS token, and position embedding.
    # Do not reset the random seed here.
    with torch.no_grad():
        model.patch_embed.proj.reset_parameters()
        nn.init.normal_(model.cls_token, std=1e-6)
        trunc_normal_(model.pos_embed, std=0.02)

    final_fingerprints = {}
    changed_from_original = []
    changed_from_r5 = []
    nonfinite = []

    for name, tensor in model.state_dict().items():

        if not torch.isfinite(tensor).all().item():
            nonfinite.append(name)

        fingerprint = tensor_sha256(tensor)
        final_fingerprints[name] = fingerprint

        if fingerprint != tensor_sha256(source_state[name]):
            changed_from_original.append(name)

        if fingerprint != r5_fingerprints[name]:
            changed_from_r5.append(name)

    expected_r6_new = {
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "cls_token",
        "pos_embed"
    }

    assert not nonfinite, nonfinite

    assert len(changed_from_original) == 296

    assert set(changed_from_r5) == expected_r6_new

    print(f"\nMODEL: {model_name}")
    print("Original checkpoint integrity: PASS")
    print("Strict checkpoint loading: PASS")
    print("Total R6 tensors:", len(final_fingerprints))
    print("Changed from original:", len(changed_from_original))
    print("Changed from R5:", len(changed_from_r5))
    print("Nonfinite tensors:", len(nonfinite))

    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

    return original_fingerprints, final_fingerprints


print("EXPERIMENT 10.32 - CROSS-MODEL R6 IDENTITY AUDIT")
print("Seed:", SEED)

originals = {}
randomized = {}

for name, path in CHECKPOINTS.items():
    originals[name], randomized[name] = audit_one_model(
        name,
        path
    )

original_differences = [
    name
    for name in originals["RETFound"]
    if originals["RETFound"][name] != originals["Meta_MAE"][name]
]

assert len(original_differences) > 0, (
    "Original checkpoint distinction not established."
)

retfound_names = set(randomized["RETFound"])
meta_names = set(randomized["Meta_MAE"])

assert retfound_names == meta_names
assert len(retfound_names) == 296

matching = [
    name
    for name in sorted(retfound_names)
    if randomized["RETFound"][name] == randomized["Meta_MAE"][name]
]

different = [
    name
    for name in sorted(retfound_names)
    if randomized["RETFound"][name] != randomized["Meta_MAE"][name]
]

print("\nCROSS-MODEL COMPARISON")
print("Distinct original checkpoint tensors verified:",
      len(original_differences))
print("R6 tensors compared:", len(retfound_names))
print("R6 identical tensors:", len(matching))
print("R6 different tensors:", len(different))

if different:
    print("First differing R6 tensor names:")
    for name in different[:20]:
        print(name)
    print("R6 parameter identity: NOT ESTABLISHED")
else:
    print("R6 parameter identity: PASS")
    print("Interpretation: Shared deterministic random-network control")

print("\nRESEARCH INTEGRITY")
print("Original checkpoints modified: False")
print("Images decoded: 0")
print("Model inference performed: False")
print("New research files created: False")
print("Test images accessed: 0")

print("\nSTEP 10.32 EXECUTION COMPLETE")


# %% [cell 370] Experiment 10 - Step 10.33

# ============================================================
# Experiment 10 - Step 10.33
# Cross-Model R6 Functional Identity Audit
# Eight validation images; seed 1042; no file writes
# ============================================================

from pathlib import Path
import gc
import hashlib
import json

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

from PIL import Image
from scipy.stats import pearsonr
from timm.layers import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = DR_RESEARCH

INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
PROTOCOL_PATH = ROOT / "experiment10_results/Experiment10_Randomization_Protocol_V1.json"

CHECKPOINTS = {
    "RETFound": ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt",
    "Meta_MAE": ROOT / "experiment07_results/meta_mae_vit_large_v1/best_model.pt"
}

EXPECTED_HASHES = {
    INDEX_PATH: "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    PROTOCOL_PATH: "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda",
    CHECKPOINTS["RETFound"]: "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8",
    CHECKPOINTS["Meta_MAE"]: "d670c42bc5a18a394a4e1780918de04522915a42a7f62675f4fd6d89c4f8b312"
}

PILOT_IDS = [
    "1301_1", "0421_1", "1190_1", "0732_1",
    "0001_2", "0011_3", "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]

SEED = 1042
DEVICE = torch.device("cuda")

assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


print("EXPERIMENT 10.33 - CROSS-MODEL R6 FUNCTIONAL IDENTITY")
print("Seed:", SEED)

# ============================================================
# 1. Verify frozen inputs and protocol
# ============================================================

for path, expected in EXPECTED_HASHES.items():
    assert path.is_file(), str(path)
    assert sha256_file(path) == expected, str(path)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

expected_r6 = (
    {"head", "fc_norm", "patch_embed", "cls_token", "pos_embed"}
    | {f"blocks.{i}" for i in range(24)}
)

assert set(
    protocol["randomization"]["stages"]["R6"]
) == expected_r6

assert SEED in protocol["randomization"]["randomization_seeds"]

print("Frozen file integrity: PASS")
print("R6 protocol integrity: PASS")

# ============================================================
# 2. Load only the frozen validation pilot images
# ============================================================

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276
assert validation.loc[PILOT_IDS, "risk_group_id"].nunique() == 8

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

pilot_inputs = []

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert int(validation.loc[image_id, "dr_grade"]) == grade
    assert image_id in cohort_lookup.index

    candidate = str(
        cohort_lookup.loc[
            image_id, "primary_ma_candidate"
        ]
    ).strip().lower()

    assert candidate in {"true", "1"}

    row = validation.loc[image_id]

    local_path = Path(row["local_path"])
    source_path = Path(row["source_path"])

    image_path = (
        local_path if local_path.is_file() else source_path
    )

    assert image_path.is_file(), image_id

    with Image.open(image_path) as image:
        pilot_inputs.append(
            preprocess(image.convert("RGB")).unsqueeze(0)
        )

assert len(pilot_inputs) == 8

print("\nCOHORT")
print("Validation images:", len(pilot_inputs))
print("Distinct source-risk groups:", 8)
print("Test images accessed: 0")

# ============================================================
# 3. Reproduce the exact cumulative randomization
# ============================================================

def reinitialize_block(block):
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()


def apply_r1_to_r6(model):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # R1
    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    # R2
    with torch.no_grad():
        model.fc_norm.reset_parameters()

    reinitialize_block(model.blocks[23])

    # R3
    for i in range(18, 23):
        reinitialize_block(model.blocks[i])

    # R4
    for i in range(12, 18):
        reinitialize_block(model.blocks[i])

    # R5
    for i in range(12):
        reinitialize_block(model.blocks[i])

    # R6
    with torch.no_grad():
        model.patch_embed.proj.reset_parameters()
        nn.init.normal_(model.cls_token, std=1e-6)
        trunc_normal_(model.pos_embed, std=0.02)


# ============================================================
# 4. Reconstruct R6 Grad-CAM
# ============================================================

def evaluate_r6(model_name):
    checkpoint = torch.load(
        CHECKPOINTS[model_name],
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    source_state = checkpoint["model_state_dict"]
    assert len(source_state) == 296

    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=5,
        global_pool="avg",
        device="meta",
        weight_init="skip"
    )

    model = model.to_empty(device=DEVICE)
    model.load_state_dict(source_state, strict=True)

    model.eval()
    model.requires_grad_(False)

    apply_r1_to_r6(model)

    # Confirm complete reinitialization.
    changed_count = 0

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, source_state[name]):
            changed_count += 1

    assert changed_count == 296

    captured = {}

    def hook_function(module, inputs, output):
        activation = output.detach().requires_grad_(True)
        captured["activation"] = activation
        return activation

    hook = model.blocks[23].register_forward_hook(
        hook_function
    )

    results = {}

    try:
        for image_id, grade, cpu_tensor in zip(
            PILOT_IDS, PILOT_GRADES, pilot_inputs
        ):
            captured.clear()

            with torch.enable_grad():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16
                ):
                    logits = model(cpu_tensor.to(DEVICE))

                activation = captured["activation"]

                gradient = torch.autograd.grad(
                    logits[0, grade],
                    activation
                )[0]

            features = (
                activation[0, 1:, :].detach().float()
                .transpose(0, 1).reshape(1024, 14, 14)
            )

            gradients = (
                gradient[0, 1:, :].detach().float()
                .transpose(0, 1).reshape(1024, 14, 14)
            )

            weights = gradients.mean(
                dim=(1, 2), keepdim=True
            )

            raw = torch.relu(
                (weights * features).sum(dim=0)
            ).detach().cpu().numpy()

            assert np.isfinite(raw).all()

            raw_max = float(raw.max())

            if raw_max <= 1e-12:
                cam = None
            else:
                cam = cv2.resize(
                    raw,
                    (224, 224),
                    interpolation=cv2.INTER_LINEAR
                ).astype(np.float32)

                assert float(cam.max()) > 1e-12
                cam /= float(cam.max())

            results[image_id] = {
                "logits": (
                    logits[0].detach().float()
                    .cpu().numpy().copy()
                ),
                "cam": cam,
                "raw_max": raw_max
            }

            captured.clear()

    finally:
        hook.remove()
        captured.clear()
        del model, checkpoint, source_state
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"{model_name} R6 parameter reinitialization: PASS"
    )

    return results


retfound = evaluate_r6("RETFound")
meta_mae = evaluate_r6("Meta_MAE")

# ============================================================
# 5. Compare exact outputs for the same eight inputs
# ============================================================

rows = []

for image_id in PILOT_IDS:
    a = retfound[image_id]
    b = meta_mae[image_id]

    assert a["cam"] is not None
    assert b["cam"] is not None

    logits_max_abs = float(
        np.max(np.abs(a["logits"] - b["logits"]))
    )

    cam_max_abs = float(
        np.max(np.abs(a["cam"] - b["cam"]))
    )

    cam_mae = float(
        np.mean(np.abs(a["cam"] - b["cam"]))
    )

    cam_pearson = float(
        pearsonr(
            a["cam"].ravel(),
            b["cam"].ravel()
        ).statistic
    )

    raw_max_abs = abs(
        a["raw_max"] - b["raw_max"]
    )

    rows.append({
        "image_id": image_id,
        "logits_max_abs": logits_max_abs,
        "cam_max_abs": cam_max_abs,
        "cam_mae": cam_mae,
        "cam_pearson": cam_pearson,
        "raw_max_abs": raw_max_abs
    })

results = pd.DataFrame(rows)

assert len(results) == 8

print("\nCROSS-MODEL R6 FUNCTIONAL COMPARISON")
print(results.to_string(index=False))

print("\nSUMMARY")
print(
    "Maximum logit absolute difference:",
    float(results["logits_max_abs"].max())
)
print(
    "Maximum CAM absolute difference:",
    float(results["cam_max_abs"].max())
)
print(
    "Minimum CAM Pearson:",
    float(results["cam_pearson"].min())
)
print(
    "Maximum raw_max difference:",
    float(results["raw_max_abs"].max())
)

# Same parameters and inputs should lead to identical
# outputs or differences within strict numeric tolerance.
functional_match = bool(
    (results["logits_max_abs"] <= 1e-4).all()
    and (results["cam_max_abs"] <= 1e-3).all()
    and (results["cam_pearson"] >= 0.99999).all()
)

print(
    "\nR6 FUNCTIONAL IDENTITY:",
    "PASS" if functional_match else "REQUIRES REVIEW"
)

print("\nRESEARCH INTEGRITY")
print("Original checkpoints modified: False")
print("New research files created: False")
print("Validation images decoded:", len(pilot_inputs))
print("Test images accessed: 0")

print("\nSTEP 10.33 EXECUTION COMPLETE")


# %% [cell 371] Experiment 10 - Step 10.34
# ============================================================
# Experiment 10 - Step 10.34
# Freeze and Archive Audited Pilot Results
# No model inference; no images decoded; no Test access
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

ROOT = DR_RESEARCH
OUTPUT_DIR = ROOT / "experiment10_results"

PROTOCOL_PATH = (
    OUTPUT_DIR / "Experiment10_Randomization_Protocol_V1.json"
)

INDEX_PATH = (
    ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
)

CSV_PATH = OUTPUT_DIR / "Experiment10_Pilot_Results_V1.csv"
JSON_PATH = OUTPUT_DIR / "Experiment10_Pilot_Audit_V1.json"

EXPECTED_PROTOCOL_HASH = (
    "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
)

EXPECTED_INDEX_HASH = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

CHECKPOINT_HASHES = {
    "RETFound": (
        "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
    ),
    "Meta_MAE": (
        "d670c42bc5a18a394a4e1780918de04522915a42a7f62675f4fd6d89c4f8b312"
    )
}

PILOT_IDS = [
    "1301_1", "0421_1", "1190_1", "0732_1",
    "0001_2", "0011_3", "0032_3", "0043_3"
]

PILOT_GRADES = [1, 1, 2, 2, 3, 3, 4, 4]
SEED = 1042

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

print("EXPERIMENT 10.34 - PILOT ARCHIVE")

assert sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_HASH
assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_HASH

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert SEED in protocol["randomization"]["randomization_seeds"]

# Verify that the eight pilot images still belong to
# the frozen validation MA cohort.
index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].set_index("image_id")

cohort_lookup = cohort.set_index("image_id")

assert len(validation) == 276

for image_id, grade in zip(PILOT_IDS, PILOT_GRADES):
    assert image_id in validation.index
    assert image_id in cohort_lookup.index
    assert int(validation.loc[image_id, "dr_grade"]) == grade

    candidate = str(
        cohort_lookup.loc[
            image_id, "primary_ma_candidate"
        ]
    ).strip().lower()

    assert candidate in {"true", "1"}

assert validation.loc[PILOT_IDS, "risk_group_id"].nunique() == 8

# Values transcribed from completed, user-reported
# Experiment 10 Colab pilot outputs.
#
# Column order:
# comparison, pearson, spearman, ssim, mae

pilot_metrics = {
    "RETFound": [
        ("R0_R1", -0.022987, -0.009630, 0.185765, 0.211084),
        ("R0_R2", -0.008887,  0.007123, 0.186887, 0.206823),
        ("R0_R3", -0.049853, -0.015893, 0.178457, 0.205549),
        ("R0_R4", -0.081639, -0.051029, 0.154025, 0.211357),
        ("R0_R5", -0.128839, -0.126098, 0.174168, 0.208770),
        ("R0_R6",  0.113007,  0.176534, 0.232338, 0.179320),
        ("R1_R2",  0.884031,  0.875511, 0.745877, 0.060694),
        ("R2_R3",  0.744368,  0.729998, 0.580035, 0.082870),
        ("R3_R4",  0.836401,  0.802258, 0.676960, 0.063883),
        ("R4_R5",  0.264870,  0.264766, 0.313027, 0.134444),
        ("R5_R6", -0.007604, -0.045508, 0.223257, 0.156051)
    ],
    "Meta_MAE": [
        ("R0_R1", -0.015180, -0.033644, 0.176300, 0.213216),
        ("R0_R2",  0.045294,  0.008062, 0.196539, 0.205846),
        ("R0_R3",  0.019817,  0.037884, 0.197673, 0.204857),
        ("R0_R4",  0.161772,  0.134062, 0.208801, 0.194051),
        ("R0_R5", -0.001387,  0.070180, 0.220781, 0.212890),
        ("R0_R6", -0.029324,  0.068321, 0.223042, 0.191931),
        ("R1_R2",  0.894521,  0.890763, 0.765307, 0.055809),
        ("R2_R3",  0.686267,  0.666433, 0.545542, 0.094181),
        ("R3_R4",  0.723111,  0.682734, 0.575403, 0.087774),
        ("R4_R5", -0.011196,  0.012710, 0.232932, 0.195958),
        ("R5_R6",  0.026525,  0.070644, 0.237098, 0.176365)
    ]
}

rows = []

for model_name, comparisons in pilot_metrics.items():
    for comparison, pearson, spearman, ssim, mae in comparisons:
        rows.append({
            "model": model_name,
            "seed": SEED,
            "comparison": comparison,
            "n_images": 8,
            "valid_pairs": 8,
            "pearson_mean": pearson,
            "spearman_mean": spearman,
            "ssim_mean": ssim,
            "mae_mean": mae,
            "source_type": "transcribed_pilot_stdout"
        })

results = pd.DataFrame(rows)

assert len(results) == 22
assert not results.duplicated(
    subset=["model", "seed", "comparison"]
).any()

assert results["valid_pairs"].eq(8).all()
assert results["n_images"].eq(8).all()

assert results[
    ["pearson_mean", "spearman_mean", "ssim_mean", "mae_mean"]
].notna().all().all()

# Checks derived from reported parameter audits.
expected_cumulative_tensors = {
    "R0": 0,
    "R1": 2,
    "R2": 16,
    "R3": 76,
    "R4": 148,
    "R5": 292,
    "R6": 296
}

expected_new_tensors = {
    "R1": 2,
    "R2": 14,
    "R3": 60,
    "R4": 72,
    "R5": 144,
    "R6": 4
}

assert sum(expected_new_tensors.values()) == 296

audit = {
    "experiment": "Experiment 10",
    "record_type": "pilot_audit_archive",
    "archive_version": "V1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_provenance": (
        "Numerical values manually transcribed from previously "
        "completed and reviewed Google Colab stdout. "
        "This archive does not independently recompute CAMs."
    ),
    "analysis_status": "pilot_only_not_inferential",
    "seed": SEED,
    "models": ["RETFound", "Meta_MAE"],
    "pilot_image_ids": PILOT_IDS,
    "pilot_true_grades": PILOT_GRADES,
    "pilot_image_count": 8,
    "distinct_source_risk_group_count": 8,
    "validation_cohort_size": 276,
    "ma_candidate_cohort_size_reported": 156,
    "protocol_sha256": EXPECTED_PROTOCOL_HASH,
    "index_sha256": EXPECTED_INDEX_HASH,
    "checkpoint_sha256": CHECKPOINT_HASHES,
    "cumulative_parameter_tensor_counts": expected_cumulative_tensors,
    "new_parameter_tensor_counts": expected_new_tensors,
    "pilot_degenerate_cam_counts": {
        "RETFound": {
            stage: 0 for stage in expected_cumulative_tensors
        },
        "Meta_MAE": {
            stage: 0 for stage in expected_cumulative_tensors
        }
    },
    "r6_cross_model_audit": {
        "identical_parameter_tensors": 296,
        "total_parameter_tensors": 296,
        "maximum_logit_absolute_difference": 0.0,
        "maximum_cam_absolute_difference": 0.0,
        "maximum_raw_max_difference": 0.0,
        "minimum_reported_cam_pearson": 0.9999986886978149,
        "functional_identity": "PASS",
        "control_type": "shared_deterministic_random_network"
    },
    "limitations": [
        "Eight-image pilot only.",
        "Single randomization seed.",
        "Means were transcribed from reported stdout.",
        "Source-risk groups are not confirmed patient identities.",
        "No statistical significance or confidence intervals.",
        "No inference about MA localization superiority.",
        "R6 is a shared random-network control, not two independent replicates.",
        "Test set remains sealed."
    ],
    "metrics_csv_filename": CSV_PATH.name
}

# Never silently overwrite a previously frozen archive.
assert not CSV_PATH.exists(), (
    f"Archive already exists: {CSV_PATH}"
)
assert not JSON_PATH.exists(), (
    f"Archive already exists: {JSON_PATH}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use exclusive-create mode to avoid accidental overwrite.
with CSV_PATH.open("x", encoding="utf-8", newline="") as handle:
    results.to_csv(handle, index=False, float_format="%.6f")

csv_hash = sha256_file(CSV_PATH)
audit["metrics_csv_sha256"] = csv_hash

with JSON_PATH.open("x", encoding="utf-8") as handle:
    json.dump(audit, handle, indent=2, ensure_ascii=False)

json_hash = sha256_file(JSON_PATH)

# Verify archive round-trip.
reloaded_results = pd.read_csv(CSV_PATH)

with JSON_PATH.open("r", encoding="utf-8") as handle:
    reloaded_audit = json.load(handle)

assert len(reloaded_results) == 22
assert reloaded_audit["metrics_csv_sha256"] == csv_hash
assert reloaded_audit["pilot_image_ids"] == PILOT_IDS
assert reloaded_audit["r6_cross_model_audit"]["functional_identity"] == "PASS"

print("\nARCHIVE SUMMARY")
print("Models:", len(pilot_metrics))
print("Randomization seed:", SEED)
print("Pilot images:", len(PILOT_IDS))
print("Metric comparison rows:", len(results))
print("Cross-model R6 identity: PASS")
print("Archive round-trip verification: PASS")

print("\nCSV")
print("Path:", CSV_PATH)
print("SHA256:", csv_hash)

print("\nJSON")
print("Path:", JSON_PATH)
print("SHA256:", json_hash)

print("\nMETRIC PREVIEW")
print(
    results[
        results["comparison"].isin(["R4_R5", "R5_R6"])
    ].to_string(index=False)
)

print("\nRESEARCH INTEGRITY")
print("Original protocol modified: False")
print("Original checkpoints modified: False")
print("Original CAM files modified: False")
print("Validation images decoded: 0")
print("Model inference performed: False")
print("Test images accessed: 0")
print("New archive files created: 2")

print("\nSTEP 10.34 EXECUTION COMPLETE")
print("Pilot results archived: PASS")


# %% [cell 372] identical to cell 373, omitted here


# %% [cell 373] Experiment 10 - Step 10.35
# ============================================================
# Experiment 10 - Step 10.35
# Full Validation MA Cohort Preflight Audit
# Read-only; no image decoding; no model inference
# ============================================================

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
)

RET_CAM_PATH = (
    ROOT / "experiment05_results/gradcam/"
    "RETFound_MA_CAM_Maps_V1.npz"
)

META_CAM_PATH = (
    ROOT / "experiment07_results/gradcam/"
    "Meta_MAE_MA_CAM_Maps_V1.npz"
)

PROTOCOL_PATH = (
    ROOT / "experiment10_results/"
    "Experiment10_Randomization_Protocol_V1.json"
)

PILOT_CSV_PATH = (
    ROOT / "experiment10_results/"
    "Experiment10_Pilot_Results_V1.csv"
)

PILOT_JSON_PATH = (
    ROOT / "experiment10_results/"
    "Experiment10_Pilot_Audit_V1.json"
)

EXPECTED_HASHES = {
    INDEX_PATH: (
        "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
    ),
    PROTOCOL_PATH: (
        "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
    ),
    PILOT_CSV_PATH: (
        "be90c0959f35c3cb9d17d35639925ff0562a64b00ada30183f6c299581524280"
    ),
    PILOT_JSON_PATH: (
        "624cf6688ee918199f726eaebecd210ade51d779632eed72e520eb0d35aaf304"
    )
}

PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]

SEEDS = [1042, 2042, 3042]


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalise_binary(series):
    values = series.astype(str).str.strip().str.lower()

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False
    }

    unknown = sorted(set(values) - set(mapping))

    assert not unknown, (
        f"Unexpected MA candidate flags: {unknown}"
    )

    return values.map(mapping).astype(bool)


print("EXPERIMENT 10.35 - VALIDATION COHORT PREFLIGHT")

# ============================================================
# 1. Frozen artifact integrity
# ============================================================

required_files = [
    INDEX_PATH,
    COHORT_PATH,
    RET_CAM_PATH,
    META_CAM_PATH,
    PROTOCOL_PATH,
    PILOT_CSV_PATH,
    PILOT_JSON_PATH
]

for path in required_files:
    assert path.is_file(), f"Missing file: {path}"

for path, expected_hash in EXPECTED_HASHES.items():
    assert sha256_file(path) == expected_hash, (
        f"SHA256 mismatch: {path}"
    )

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

with PILOT_JSON_PATH.open("r", encoding="utf-8") as handle:
    pilot_audit = json.load(handle)

assert set(SEEDS) == set(
    protocol["randomization"]["randomization_seeds"]
)

assert pilot_audit["pilot_image_ids"] == PILOT_IDS

assert pilot_audit["metrics_csv_sha256"] == sha256_file(
    PILOT_CSV_PATH
)

assert (
    pilot_audit["r6_cross_model_audit"]["functional_identity"]
    == "PASS"
)

print("\nFROZEN ARTIFACTS")
print("Training index SHA256: PASS")
print("Randomization protocol SHA256: PASS")
print("Pilot CSV SHA256: PASS")
print("Pilot JSON SHA256: PASS")
print("Pilot archive consistency: PASS")
print("Randomization seeds:", SEEDS)

# ============================================================
# 2. Verify frozen index and validation cohort
# ============================================================

index = pd.read_csv(
    INDEX_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str
    }
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert len(index) == 1842
assert index["image_id"].is_unique
assert not index["image_id"].isna().any()

assert index["split"].value_counts().to_dict() == {
    "train": 1290,
    "val": 276,
    "test": 276
}

assert len(cohort) == 276
assert cohort["image_id"].is_unique
assert not cohort["image_id"].isna().any()

validation = index.loc[
    index["split"].eq("val")
].copy()

assert set(cohort["image_id"]) == set(
    validation["image_id"]
)

cohort["primary_ma_candidate_checked"] = normalise_binary(
    cohort["primary_ma_candidate"]
)

selected = cohort.loc[
    cohort["primary_ma_candidate_checked"]
].copy()

assert len(selected) == 156
assert selected["image_id"].is_unique

selected_ids = set(selected["image_id"])

assert set(PILOT_IDS).issubset(selected_ids)

validation_lookup = validation.set_index("image_id")

# Verify labels and risk group fields.
assert "dr_grade" in validation.columns
assert "true_grade" in cohort.columns
assert "risk_group_id" in validation.columns

selected["index_grade"] = selected["image_id"].map(
    validation_lookup["dr_grade"]
)

assert selected["index_grade"].notna().all()
assert selected["true_grade"].notna().all()

grade_mismatches = selected.loc[
    pd.to_numeric(
        selected["true_grade"],
        errors="raise"
    ).astype(int)
    !=
    pd.to_numeric(
        selected["index_grade"],
        errors="raise"
    ).astype(int)
]

assert grade_mismatches.empty, (
    "Cohort/index grade mismatch detected."
)

selected["risk_group_id"] = selected["image_id"].map(
    validation_lookup["risk_group_id"]
)

assert selected["risk_group_id"].notna().all()
assert (
    selected["risk_group_id"].astype(str).str.strip() != ""
).all()

print("\nVALIDATION COHORT")
print("All indexed images:", len(index))
print("Frozen validation images:", len(validation))
print("Validation cohort rows:", len(cohort))
print("Selected MA candidates:", len(selected))
print("Selected candidate ID duplicates: 0")
print("Index/cohort grade mismatches: 0")
print("Pilot image IDs retained: 8/8")

# ============================================================
# 3. Verify source-risk grouping
# ============================================================

group_sizes = (
    selected.groupby("risk_group_id")
    .size()
    .sort_values(ascending=False)
)

repeated_groups = group_sizes.loc[group_sizes > 1]

print("\nSOURCE-RISK GROUPS")
print("Unique source-risk groups:", len(group_sizes))
print("Groups containing multiple images:", len(repeated_groups))
print("Maximum images in one group:", int(group_sizes.max()))

if not repeated_groups.empty:
    print("\nMULTI-IMAGE GROUPS")
    print(repeated_groups.head(20).to_string())
else:
    print("All selected source-risk groups are single-image groups.")

print(
    "Grouping interpretation:",
    "Source-risk groups are not verified patient IDs."
)

# ============================================================
# 4. Verify both original CAM archives
# ============================================================

def inspect_cam_archive(path, cam_key):
    with np.load(path, allow_pickle=False) as archive:
        assert "image_ids" in archive.files
        assert cam_key in archive.files

        ids = [str(value) for value in archive["image_ids"]]

        assert len(ids) == 156
        assert len(set(ids)) == 156

        cams = archive[cam_key]

        assert cams.shape == (156, 224, 224)
        assert np.isfinite(cams).all()

        minimum = float(cams.min())
        maximum = float(cams.max())

        assert minimum >= -1e-5
        assert maximum <= 1.0 + 1e-5

        zero_maps = int(
            np.sum(
                np.max(cams, axis=(1, 2)) <= 1e-12
            )
        )

    return {
        "ids": ids,
        "minimum": minimum,
        "maximum": maximum,
        "zero_maps": zero_maps
    }


ret_cam = inspect_cam_archive(
    RET_CAM_PATH,
    "retfound_cam"
)

meta_cam = inspect_cam_archive(
    META_CAM_PATH,
    "meta_mae_cam"
)

ret_ids = set(ret_cam["ids"])
meta_ids = set(meta_cam["ids"])

assert ret_ids == selected_ids, (
    "RETFound CAM IDs do not match selected cohort."
)

assert meta_ids == selected_ids, (
    "Meta MAE CAM IDs do not match selected cohort."
)

assert ret_ids == meta_ids

print("\nCAM ARCHIVE ALIGNMENT")
print("RETFound original CAMs:", len(ret_cam["ids"]))
print("Meta MAE original CAMs:", len(meta_cam["ids"]))
print("Selected MA candidate IDs:", len(selected_ids))
print("Cross-model ID set alignment: PASS")
print("Frozen cohort ID set alignment: PASS")
print("RETFound zero-valued maps:", ret_cam["zero_maps"])
print("Meta MAE zero-valued maps:", meta_cam["zero_maps"])
print(
    "RETFound CAM value range:",
    ret_cam["minimum"],
    ret_cam["maximum"]
)
print(
    "Meta MAE CAM value range:",
    meta_cam["minimum"],
    meta_cam["maximum"]
)

# ============================================================
# 5. Check image availability without decoding
# ============================================================

missing_images = []
selected_paths = {}

for image_id in sorted(selected_ids):
    row = validation_lookup.loc[image_id]

    local_value = row["local_path"]
    source_value = row["source_path"]

    local_path = (
        Path(str(local_value))
        if pd.notna(local_value)
        else None
    )

    source_path = (
        Path(str(source_value))
        if pd.notna(source_value)
        else None
    )

    chosen = None

    if local_path is not None and local_path.is_file():
        chosen = local_path

    elif source_path is not None and source_path.is_file():
        chosen = source_path

    if chosen is None:
        missing_images.append(image_id)
    else:
        selected_paths[image_id] = chosen

print("\nIMAGE PATH AVAILABILITY")
print("Required MA candidate images:", len(selected_ids))
print("Accessible image paths:", len(selected_paths))
print("Missing image paths:", len(missing_images))

if missing_images:
    print("Missing image IDs:", missing_images[:30])

assert not missing_images, (
    "One or more validation images are inaccessible."
)

# ============================================================
# 6. Summarize grade and experimental workload
# ============================================================

grade_counts = (
    selected["index_grade"]
    .astype(int)
    .value_counts()
    .sort_index()
)

assert set(grade_counts.index).issubset(
    {0, 1, 2, 3, 4}
)

print("\nDR GRADE DISTRIBUTION")
print(grade_counts.to_string())

n_images = len(selected)
n_seeds = len(SEEDS)
n_stages = 7

# A shared R6 is evaluated once per seed and image.
# R0 can also be reused across seeds.
randomized_nonshared_stages = 5

randomized_model_image_evaluations = (
    2 * n_images * n_seeds * randomized_nonshared_stages
)

shared_r6_evaluations = (
    n_images * n_seeds
)

original_model_evaluations = (
    2 * n_images
)

total_evaluations = (
    randomized_model_image_evaluations
    + shared_r6_evaluations
    + original_model_evaluations
)

print("\nPLANNED FULL VALIDATION WORKLOAD")
print("Models:", 2)
print("Randomization seeds:", n_seeds)
print("Images per model:", n_images)
print("Stages R0-R6:", n_stages)
print(
    "Nonshared randomized stage-image evaluations:",
    randomized_model_image_evaluations
)
print(
    "Shared R6 stage-image evaluations:",
    shared_r6_evaluations
)
print(
    "Original R0 stage-image evaluations:",
    original_model_evaluations
)
print(
    "Total distinct stage-image evaluations:",
    total_evaluations
)
print(
    "Note: Workload is a planning count, not an execution."
)

# ============================================================
# 7. Final research integrity checks
# ============================================================

assert len(selected_ids) == 156
assert len(selected_paths) == 156
assert ret_ids == meta_ids == selected_ids
assert len(SEEDS) == 3

print("\nRESEARCH INTEGRITY")
print("Frozen protocol modified: False")
print("Original checkpoints modified: False")
print("Original CAM files modified: False")
print("New research files created: False")
print("Images decoded: 0")
print("Model inference performed: False")
print("Test image paths examined: 0")
print("Test images accessed: 0")

print("\nSTEP 10.35 EXECUTION COMPLETE")
print("Full validation input readiness: PASS")


# %% [cell 374] Experiment 10 - Step 10.36
# ============================================================
# Experiment 10 - Step 10.36
# Freeze Full Validation Execution Manifest V1
# Read-only source artifacts; no image decoding or inference
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
OUTPUT_DIR = ROOT / "experiment10_results"

INDEX_PATH = (
    ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
)

RET_CAM_PATH = (
    ROOT / "experiment05_results/gradcam/"
    "RETFound_MA_CAM_Maps_V1.npz"
)

META_CAM_PATH = (
    ROOT / "experiment07_results/gradcam/"
    "Meta_MAE_MA_CAM_Maps_V1.npz"
)

RET_CHECKPOINT = (
    ROOT / "experiment04_results/"
    "retfound_vit_large_v1/best_model.pt"
)

META_CHECKPOINT = (
    ROOT / "experiment07_results/"
    "meta_mae_vit_large_v1/best_model.pt"
)

PROTOCOL_PATH = (
    OUTPUT_DIR / "Experiment10_Randomization_Protocol_V1.json"
)

PILOT_CSV_PATH = (
    OUTPUT_DIR / "Experiment10_Pilot_Results_V1.csv"
)

PILOT_JSON_PATH = (
    OUTPUT_DIR / "Experiment10_Pilot_Audit_V1.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR / "Experiment10_Full_Validation_Manifest_V1.csv"
)

PLAN_PATH = (
    OUTPUT_DIR / "Experiment10_Full_Execution_Plan_V1.json"
)

EXPECTED_HASHES = {
    INDEX_PATH: (
        "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
    ),
    RET_CHECKPOINT: (
        "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
    ),
    META_CHECKPOINT: (
        "d670c42bc5a18a394a4e1780918de04522915a42a7f62675f4fd6d89c4f8b312"
    ),
    PROTOCOL_PATH: (
        "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
    ),
    PILOT_CSV_PATH: (
        "be90c0959f35c3cb9d17d35639925ff0562a64b00ada30183f6c299581524280"
    ),
    PILOT_JSON_PATH: (
        "624cf6688ee918199f726eaebecd210ade51d779632eed72e520eb0d35aaf304"
    )
}

EXPECTED_SEEDS = [1042, 2042, 3042]

EXPECTED_PILOT_IDS = [
    "1301_1", "0421_1",
    "1190_1", "0732_1",
    "0001_2", "0011_3",
    "0032_3", "0043_3"
]


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parse_binary(series):
    values = series.astype(str).str.strip().str.lower()

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False
    }

    unexpected = sorted(set(values) - set(mapping))
    assert not unexpected, unexpected

    return values.map(mapping).astype(bool)


print("EXPERIMENT 10.36 - EXECUTION MANIFEST FREEZE")

# ============================================================
# 1. Protect the existing frozen research artifacts
# ============================================================

for path, expected in EXPECTED_HASHES.items():
    assert path.is_file(), f"Missing file: {path}"
    assert sha256_file(path) == expected, (
        f"SHA256 mismatch: {path}"
    )

for path in [COHORT_PATH, RET_CAM_PATH, META_CAM_PATH]:
    assert path.is_file(), f"Missing input: {path}"

assert not MANIFEST_PATH.exists(), (
    f"Manifest already exists: {MANIFEST_PATH}"
)

assert not PLAN_PATH.exists(), (
    f"Execution plan already exists: {PLAN_PATH}"
)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

with PILOT_JSON_PATH.open("r", encoding="utf-8") as handle:
    pilot_audit = json.load(handle)

assert set(
    protocol["randomization"]["randomization_seeds"]
) == set(EXPECTED_SEEDS)

assert pilot_audit["pilot_image_ids"] == EXPECTED_PILOT_IDS

assert (
    pilot_audit["r6_cross_model_audit"]["functional_identity"]
    == "PASS"
)

# ============================================================
# 2. Reconstruct the 156-image frozen validation cohort
# ============================================================

index = pd.read_csv(
    INDEX_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str
    }
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert len(index) == 1842
assert index["image_id"].is_unique

assert len(cohort) == 276
assert cohort["image_id"].is_unique

validation = index.loc[
    index["split"].eq("val")
].copy()

assert len(validation) == 276

assert set(validation["image_id"]) == set(
    cohort["image_id"]
)

cohort["candidate_flag"] = parse_binary(
    cohort["primary_ma_candidate"]
)

selected = cohort.loc[
    cohort["candidate_flag"]
].copy()

assert len(selected) == 156
assert selected["image_id"].is_unique

lookup = validation.set_index("image_id")

assert set(selected["image_id"]).issubset(
    set(lookup.index)
)

# ============================================================
# 3. Check original CAM image identifiers
# ============================================================

def read_cam_ids(path, cam_key):
    with np.load(path, allow_pickle=False) as archive:
        assert {"image_ids", cam_key}.issubset(
            set(archive.files)
        )

        ids = [
            str(item)
            for item in archive["image_ids"]
        ]

        assert len(ids) == 156
        assert len(set(ids)) == 156
        assert archive[cam_key].shape == (156, 224, 224)

    return ids


ret_ids = read_cam_ids(
    RET_CAM_PATH,
    "retfound_cam"
)

meta_ids = read_cam_ids(
    META_CAM_PATH,
    "meta_mae_cam"
)

selected_ids = set(selected["image_id"])

assert selected_ids == set(ret_ids)
assert selected_ids == set(meta_ids)

# ============================================================
# 4. Build a deterministic 156-row execution manifest
# ============================================================

records = []

for image_id in sorted(selected_ids):

    index_row = lookup.loc[image_id]

    cohort_row = selected.loc[
        selected["image_id"].eq(image_id)
    ].iloc[0]

    grade = int(index_row["dr_grade"])

    assert grade == int(cohort_row["true_grade"])
    assert grade in {1, 2, 3, 4}

    group_id = str(index_row["risk_group_id"])

    assert group_id.strip() not in {"", "nan", "None"}

    local_value = index_row["local_path"]
    source_value = index_row["source_path"]

    local_path = (
        Path(str(local_value))
        if pd.notna(local_value)
        else None
    )

    source_path = (
        Path(str(source_value))
        if pd.notna(source_value)
        else None
    )

    chosen = None

    if local_path is not None and local_path.is_file():
        chosen = local_path

    elif source_path is not None and source_path.is_file():
        chosen = source_path

    assert chosen is not None, image_id

    assert pd.notna(index_row["sha256"])
    image_sha256 = str(index_row["sha256"]).strip()

    assert len(image_sha256) == 64

    records.append({
        "image_id": image_id,
        "true_grade": grade,
        "risk_group_id": group_id,
        "split": "val",
        "primary_ma_candidate": True,
        "image_sha256_from_frozen_index": image_sha256,
        "resolved_image_path": str(chosen),
        "retfound_reference_cam": True,
        "meta_mae_reference_cam": True
    })


manifest = pd.DataFrame(records)

assert len(manifest) == 156
assert manifest["image_id"].is_unique

assert manifest["risk_group_id"].nunique() == 134

group_sizes = manifest.groupby(
    "risk_group_id"
).size()

assert int((group_sizes > 1).sum()) == 16
assert int(group_sizes.max()) == 3

assert manifest["true_grade"].value_counts().to_dict() == {
    1: 28,
    2: 50,
    3: 62,
    4: 16
}

assert set(EXPECTED_PILOT_IDS).issubset(
    set(manifest["image_id"])
)

# ============================================================
# 5. Prepare execution rules and source provenance
# ============================================================

source_hashes = {
    "index": sha256_file(INDEX_PATH),
    "validation_cohort": sha256_file(COHORT_PATH),
    "retfound_reference_cams": sha256_file(RET_CAM_PATH),
    "meta_mae_reference_cams": sha256_file(META_CAM_PATH),
    "retfound_checkpoint": sha256_file(RET_CHECKPOINT),
    "meta_mae_checkpoint": sha256_file(META_CHECKPOINT),
    "randomization_protocol": sha256_file(PROTOCOL_PATH),
    "pilot_metrics_archive": sha256_file(PILOT_CSV_PATH),
    "pilot_audit_archive": sha256_file(PILOT_JSON_PATH)
}

execution_plan = {
    "experiment": "Experiment 10",
    "version": "V1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "execution_manifest_frozen",
    "dataset": "FGADR",
    "split": "val",
    "test_access_permitted": False,
    "image_count": 156,
    "source_risk_group_count": 134,
    "multi_image_risk_group_count": 16,
    "risk_group_semantics": (
        "Source-risk grouping, not verified patient identity."
    ),
    "models": [
        "RETFound",
        "Meta_MAE"
    ],
    "architecture": "vit_large_patch16_224",
    "target_layer": "blocks.23",
    "target_logit": "true_dr_grade",
    "randomization_seeds": EXPECTED_SEEDS,
    "stages": [
        "R0", "R1", "R2", "R3",
        "R4", "R5", "R6"
    ],
    "cumulative_parameter_tensor_counts": {
        "R0": 0,
        "R1": 2,
        "R2": 16,
        "R3": 76,
        "R4": 148,
        "R5": 292,
        "R6": 296
    },
    "execution_rules": {
        "r0": (
            "Original model CAM is seed-independent; "
            "evaluate once per model per image."
        ),
        "r1_to_r5": (
            "Evaluate independently for each model and seed "
            "while maintaining the frozen cumulative initialization order."
        ),
        "r6": (
            "Shared deterministic random network. "
            "Evaluate once per seed per image and reuse "
            "the same raw result for both model comparisons."
        ),
        "stage_randomization": (
            "Reset RNG at the beginning of each model-seed run, "
            "not between cumulative randomization stages."
        ),
        "degenerate_cam": (
            "Record explicitly; exclude only from metrics "
            "that require a nondegenerate CAM."
        ),
        "aggregation": (
            "Compute image-level metrics before aggregating; "
            "retain the model, seed, stage, image ID and source-risk group."
        ),
        "inference_statistics": (
            "Paired image comparisons with source-risk-group "
            "cluster bootstrap; seeds are repeated randomization runs."
        ),
        "test_set": "Sealed; no access."
    },
    "source_hashes": source_hashes,
    "manifest_filename": MANIFEST_PATH.name,
    "provenance": (
        "Generated directly from the frozen validation index "
        "and MA candidate cohort. Original CAM archives "
        "were checked for identifier alignment."
    ),
    "limitations": [
        "Source-risk groups are not verified patient identifiers.",
        "The manifest uses index-recorded image SHA256 values; "
        "image file bytes are not rehashed in this step.",
        "No CAMs or image-level randomization metrics are "
        "generated by this manifest freeze."
    ]
}

# ============================================================
# 6. Save new files without overwriting existing artifacts
# ============================================================

assert OUTPUT_DIR.is_dir()

with MANIFEST_PATH.open(
    "x",
    encoding="utf-8",
    newline=""
) as handle:

    manifest.to_csv(
        handle,
        index=False
    )

manifest_sha256 = sha256_file(MANIFEST_PATH)

execution_plan["manifest_sha256"] = manifest_sha256

with PLAN_PATH.open(
    "x",
    encoding="utf-8"
) as handle:

    json.dump(
        execution_plan,
        handle,
        indent=2,
        ensure_ascii=False
    )

plan_sha256 = sha256_file(PLAN_PATH)

# ============================================================
# 7. Verify saved files and source invariants
# ============================================================

reloaded_manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str
    }
)

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    reloaded_plan = json.load(handle)

assert len(reloaded_manifest) == 156
assert reloaded_manifest["image_id"].is_unique

assert reloaded_manifest[
    "risk_group_id"
].nunique() == 134

assert reloaded_plan["manifest_sha256"] == manifest_sha256

assert reloaded_plan["randomization_seeds"] == EXPECTED_SEEDS

assert (
    reloaded_plan["execution_rules"]["test_set"]
    == "Sealed; no access."
)

print("\nMANIFEST SUMMARY")
print("Selected images:", len(reloaded_manifest))
print(
    "Source-risk groups:",
    reloaded_manifest["risk_group_id"].nunique()
)
print("Multi-image risk groups:", int((group_sizes > 1).sum()))
print("Image ID duplicates:", 0)
print("Grade mismatches:", 0)
print("Cross-model CAM ID alignment: PASS")
print("Randomization seeds:", EXPECTED_SEEDS)

print("\nDR GRADE DISTRIBUTION")
print(
    reloaded_manifest["true_grade"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nFROZEN MANIFEST")
print("Path:", MANIFEST_PATH)
print("SHA256:", manifest_sha256)

print("\nFROZEN EXECUTION PLAN")
print("Path:", PLAN_PATH)
print("SHA256:", plan_sha256)

print("\nROUND-TRIP VERIFICATION")
print("Manifest reload: PASS")
print("Execution plan reload: PASS")
print("Source hash provenance: PASS")
print("R6 shared-control rule recorded: PASS")

print("\nRESEARCH INTEGRITY")
print("Existing research files modified: False")
print("Images decoded: 0")
print("Model inference performed: False")
print("Test images accessed: 0")
print("New execution files created: 2")

print("\nSTEP 10.36 EXECUTION COMPLETE")
print("Full validation execution manifest: FROZEN")


# %% [cell 375] Experiment 10 - Step 10.37
# ============================================================
# Experiment 10 - Step 10.37
# RETFound R1 Production Batch 001
# Seed 1042 | First 16 frozen validation images
# Atomic archive creation | No test-set access
# ============================================================

from pathlib import Path
import gc
import hashlib
import io
import json
import os
import tempfile

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

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"
PLAN_PATH = EXP10 / "Experiment10_Full_Execution_Plan_V1.json"
PROTOCOL_PATH = EXP10 / "Experiment10_Randomization_Protocol_V1.json"

INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"

CHECKPOINT_PATH = (
    ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"
)

REFERENCE_PATH = (
    ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"
)

OUTPUT_DIR = EXP10 / "full_validation_batches"

BATCH_NAME = "RETFound_Seed1042_R1_Batch001"
NPZ_PATH = OUTPUT_DIR / f"{BATCH_NAME}.npz"
CSV_PATH = OUTPUT_DIR / f"{BATCH_NAME}.csv"

EXPECTED_HASHES = {
    MANIFEST_PATH: "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5",
    PLAN_PATH: "da52de5513de274a0b91c99f7060049a1f77db4c6dc2a788263607e6d8168590",
    PROTOCOL_PATH: "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda",
    INDEX_PATH: "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    CHECKPOINT_PATH: "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
}

SEED = 1042
BATCH_START = 0
BATCH_END = 16
DEVICE = torch.device("cuda")

assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - a.mean()
    bc = b - b.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12
        else np.nan
    )

    spearman = float(spearmanr(a, b).statistic)

    ssim = float(
        structural_similarity(first, second, data_range=1.0)
    )

    mae = float(np.mean(np.abs(a - b)))

    return {
        "pearson": pearson,
        "spearman": spearman,
        "ssim": ssim,
        "mae": mae
    }


def atomic_create_bytes(destination, data):
    """
    Publish a complete file without overwriting an existing one.
    The temporary file is created in the destination directory.
    """
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".experiment10_tmp_",
        dir=str(destination.parent)
    )

    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        # Atomic exclusive publication on the same filesystem.
        os.link(temporary_path, destination)

    finally:
        temporary_path.unlink(missing_ok=True)


print("EXPERIMENT 10.37 - RETFOUND R1 PRODUCTION BATCH 001")
print("Seed:", SEED)
print("Batch interval:", BATCH_START, BATCH_END)

# ============================================================
# 1. Verify immutable research inputs
# ============================================================

for path, expected in EXPECTED_HASHES.items():
    assert path.is_file(), f"Missing input: {path}"
    assert sha256_file(path) == expected, (
        f"SHA256 mismatch: {path}"
    )

assert REFERENCE_PATH.is_file()
assert COHORT_PATH.is_file()

# Never overwrite a previously published batch.
assert not NPZ_PATH.exists(), (
    f"NPZ output already exists: {NPZ_PATH}"
)
assert not CSV_PATH.exists(), (
    f"CSV output already exists: {CSV_PATH}"
)

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    plan = json.load(handle)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert plan["manifest_sha256"] == EXPECTED_HASHES[MANIFEST_PATH]
assert plan["image_count"] == 156
assert plan["source_risk_group_count"] == 134
assert plan["test_access_permitted"] is False

assert plan["randomization_seeds"] == [1042, 2042, 3042]
assert SEED in protocol["randomization"]["randomization_seeds"]
assert set(protocol["randomization"]["stages"]["R1"]) == {"head"}

print("\nFROZEN INPUT INTEGRITY")
print("Manifest SHA256: PASS")
print("Execution plan SHA256: PASS")
print("Randomization protocol SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Frozen index SHA256: PASS")

# ============================================================
# 2. Select exactly 16 images from the frozen manifest
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str
    }
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique
assert manifest["split"].eq("val").all()
assert manifest["image_id"].tolist() == sorted(
    manifest["image_id"].tolist()
)

batch = manifest.iloc[BATCH_START:BATCH_END].copy()

assert len(batch) == 16
assert batch["image_id"].is_unique
assert batch["true_grade"].isin([1, 2, 3, 4]).all()

image_ids = batch["image_id"].tolist()
grades = batch["true_grade"].astype(int).tolist()
risk_groups = batch["risk_group_id"].astype(str).tolist()

assert all(
    value.strip() not in {"", "nan", "None"}
    for value in risk_groups
)

# Confirm each image remains in the frozen candidate cohort.
cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

candidate_values = (
    cohort.set_index("image_id")
    .loc[image_ids, "primary_ma_candidate"]
    .astype(str).str.strip().str.lower()
)

assert candidate_values.isin(["true", "1"]).all()

# Load original CAMs by image ID rather than array position.
with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    reference_lookup = {
        image_id: i
        for i, image_id in enumerate(reference_ids)
    }

    assert set(image_ids).issubset(reference_lookup)

    reference_cams = np.stack([
        archive["retfound_cam"][reference_lookup[image_id]]
        .astype(np.float32)
        for image_id in image_ids
    ])

assert reference_cams.shape == (16, 224, 224)
assert np.isfinite(reference_cams).all()

print("\nBATCH COHORT")
print("Images:", len(image_ids))
print("Distinct source-risk groups:", len(set(risk_groups)))
print("Validation-only selection: PASS")
print("Original CAM ID alignment: PASS")
print("First image ID:", image_ids[0])
print("Last image ID:", image_ids[-1])

# ============================================================
# 3. Decode only the selected validation images
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensors = []

for _, row in batch.iterrows():
    image_id = row["image_id"]
    image_path = Path(row["resolved_image_path"])

    assert image_path.is_file(), image_id

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    assert tensor.shape == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all().item()

    input_tensors.append(tensor)

print("Images decoded:", len(input_tensors))

# ============================================================
# 4. Load the frozen RETFound model
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]

assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

loaded = model.load_state_dict(
    source_state,
    strict=True
)

assert not loaded.missing_keys
assert not loaded.unexpected_keys

model.eval()
model.requires_grad_(False)

assert len(model.state_dict()) == 296

# ============================================================
# 5. Use the frozen Grad-CAM implementation
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(
    activation_hook
)


def calculate_cam(cpu_tensor, true_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(cpu_tensor.to(DEVICE))

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, true_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2), keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12

        cam /= float(cam.max())
        assert np.isfinite(cam).all()

    captured.clear()

    return output_logits, cam, raw_max

# ============================================================
# 6. Reproduce R0, then execute seed-1042 R1
# ============================================================

r0_logits = []
r0_cams = []
r1_logits = []
r1_cams = []
r1_raw_maxima = []

original_head_weight = model.head.weight.detach().clone()
original_head_bias = model.head.bias.detach().clone()

try:
    # R0 must agree with the existing original CAM archive.
    baseline_pearsons = []
    baseline_maes = []

    for i, (tensor, grade) in enumerate(
        zip(input_tensors, grades)
    ):
        logits, cam, raw_max = calculate_cam(
            tensor, grade
        )

        assert cam is not None, image_ids[i]

        metrics = compare_cams(
            cam,
            reference_cams[i]
        )

        baseline_pearsons.append(metrics["pearson"])
        baseline_maes.append(metrics["mae"])

        r0_logits.append(logits)
        r0_cams.append(cam)

    # Apply precisely the same R1 initialization as pilot.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    assert not torch.equal(
        model.head.weight,
        original_head_weight
    )

    assert torch.equal(
        model.head.bias,
        torch.zeros_like(model.head.bias)
    )

    # Verify that every non-head tensor remains unchanged.
    changed_names = []

    for name, parameter in model.state_dict().items():
        original = source_state[name]
        current = parameter.detach().cpu()

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, original):
            changed_names.append(name)

    assert set(changed_names) == {
        "head.weight",
        "head.bias"
    }

    print("\nR1 PARAMETER INTEGRITY")
    print("Changed tensors:", len(changed_names))
    print("Expected changed tensors:", 2)
    print("Unauthorized changes:", 0)
    print("Original checkpoint modified: False")

    print("\nR0 REPRODUCTION")
    print(
        "Minimum reference CAM Pearson:",
        round(float(np.min(baseline_pearsons)), 6)
    )
    print(
        "Mean reference CAM MAE:",
        round(float(np.mean(baseline_maes)), 8)
    )

    assert np.isfinite(baseline_pearsons).all()
    assert min(baseline_pearsons) > 0.999

    # R1: Same randomized head for all 16 images.
    for tensor, grade in zip(input_tensors, grades):
        logits, cam, raw_max = calculate_cam(
            tensor, grade
        )

        r1_logits.append(logits)
        r1_cams.append(cam)
        r1_raw_maxima.append(raw_max)

finally:
    hook.remove()
    captured.clear()

    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Build image-level results
# ============================================================

rows = []
stored_r1_cams = []

for i, image_id in enumerate(image_ids):
    metrics = compare_cams(
        r0_cams[i],
        r1_cams[i]
    )

    record = {
        "model": "RETFound",
        "seed": SEED,
        "stage": "R1",
        "batch": "001",
        "image_id": image_id,
        "true_grade": grades[i],
        "risk_group_id": risk_groups[i],
        "degenerate_cam": r1_cams[i] is None,
        "raw_max": r1_raw_maxima[i],
        "r0_r1_pearson": metrics["pearson"],
        "r0_r1_spearman": metrics["spearman"],
        "r0_r1_ssim": metrics["ssim"],
        "r0_r1_mae": metrics["mae"],
        "r0_r1_logit_l2": float(
            np.linalg.norm(
                r1_logits[i] - r0_logits[i]
            )
        )
    }

    rows.append(record)

    if r1_cams[i] is None:
        stored_r1_cams.append(
            np.zeros((224, 224), dtype=np.float32)
        )
    else:
        stored_r1_cams.append(
            r1_cams[i].astype(np.float32)
        )

results = pd.DataFrame(rows)

r1_cam_array = np.stack(stored_r1_cams)
r0_logit_array = np.stack(r0_logits)
r1_logit_array = np.stack(r1_logits)

assert len(results) == 16
assert results["image_id"].is_unique
assert r1_cam_array.shape == (16, 224, 224)
assert r0_logit_array.shape == (16, 5)
assert r1_logit_array.shape == (16, 5)

assert np.isfinite(r1_cam_array).all()
assert np.isfinite(r0_logit_array).all()
assert np.isfinite(r1_logit_array).all()

# ============================================================
# 8. Atomically publish original image-level results
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert not NPZ_PATH.exists()
assert not CSV_PATH.exists()

# Store complete CAM arrays and paired logits.
npz_buffer = io.BytesIO()

np.savez_compressed(
    npz_buffer,
    image_ids=np.asarray(image_ids, dtype=str),
    true_grades=np.asarray(grades, dtype=np.int64),
    risk_group_ids=np.asarray(risk_groups, dtype=str),
    seed=np.asarray(SEED),
    stage=np.asarray("R1"),
    model=np.asarray("RETFound"),
    r0_cam=np.stack(r0_cams).astype(np.float32),
    r1_cam=r1_cam_array,
    r0_logits=r0_logit_array.astype(np.float32),
    r1_logits=r1_logit_array.astype(np.float32),
    r1_raw_max=np.asarray(r1_raw_maxima, dtype=np.float64),
    r1_degenerate=results["degenerate_cam"].to_numpy(dtype=bool),
    manifest_sha256=np.asarray(
        EXPECTED_HASHES[MANIFEST_PATH]
    ),
    execution_plan_sha256=np.asarray(
        EXPECTED_HASHES[PLAN_PATH]
    )
)

npz_bytes = npz_buffer.getvalue()
csv_bytes = results.to_csv(
    index=False,
    float_format="%.10g"
).encode("utf-8")

atomic_create_bytes(NPZ_PATH, npz_bytes)
atomic_create_bytes(CSV_PATH, csv_bytes)

# ============================================================
# 9. Verify saved results by reading both files
# ============================================================

with np.load(NPZ_PATH, allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert archive["r0_cam"].shape == (16, 224, 224)
    assert archive["r1_cam"].shape == (16, 224, 224)

    assert np.array_equal(
        archive["r1_cam"],
        r1_cam_array
    )

    assert np.array_equal(
        archive["r1_logits"],
        r1_logit_array.astype(np.float32)
    )

reloaded_csv = pd.read_csv(
    CSV_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(reloaded_csv) == 16
assert reloaded_csv["image_id"].tolist() == image_ids
assert reloaded_csv["seed"].eq(SEED).all()
assert reloaded_csv["stage"].eq("R1").all()

npz_hash = sha256_file(NPZ_PATH)
csv_hash = sha256_file(CSV_PATH)

# ============================================================
# 10. Final production batch report
# ============================================================

print("\nBATCH RESULTS")
print("Model: RETFound")
print("Seed:", SEED)
print("Stage: R1")
print("Images evaluated:", len(results))
print(
    "Degenerate CAMs:",
    int(results["degenerate_cam"].sum())
)

for metric in [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]:
    print(
        f"Mean {metric}:",
        round(float(results[metric].mean()), 6)
    )

print("\nPER-IMAGE METRIC PREVIEW")
print(
    results[
        [
            "image_id",
            "true_grade",
            "risk_group_id",
            "degenerate_cam",
            "r0_r1_pearson",
            "r0_r1_spearman",
            "r0_r1_ssim",
            "r0_r1_mae"
        ]
    ].to_string(index=False)
)

print("\nARCHIVE INTEGRITY")
print("NPZ path:", NPZ_PATH)
print("NPZ SHA256:", npz_hash)
print("CSV path:", CSV_PATH)
print("CSV SHA256:", csv_hash)
print("Archive round-trip: PASS")
print("Previous batches overwritten: False")

print("\nRESEARCH INTEGRITY")
print("Frozen inputs modified: False")
print("Validation images decoded:", len(input_tensors))
print("Test images accessed: 0")
print("New batch files created: 2")

print("\nSTEP 10.37 EXECUTION COMPLETE")
print("Production batch 001: PENDING REVIEW")


# %% [cell 376] Experiment 10 - Step 10.37B

# ============================================================
# Experiment 10 - Step 10.37B
# Recover Production Batch 001 Archive
# No GPU inference; no image decoding; no file overwrite
# ============================================================

from pathlib import Path
import hashlib
import io

import numpy as np
import pandas as pd

print("EXPERIMENT 10.37B - BATCH 001 ARCHIVE RECOVERY")

# ============================================================
# 1. Verify that the previous cell's results remain in memory
# ============================================================

required_variables = [
    "npz_bytes",
    "csv_bytes",
    "NPZ_PATH",
    "CSV_PATH",
    "image_ids",
    "grades",
    "risk_groups",
    "results",
    "r1_cam_array",
    "r0_logit_array",
    "r1_logit_array"
]

missing = [
    name for name in required_variables
    if name not in globals()
]

if missing:
    raise RuntimeError(
        "Previous computation results are no longer available "
        f"in memory: {missing}. "
        "Do not recreate files from estimated values."
    )

assert isinstance(npz_bytes, bytes)
assert isinstance(csv_bytes, bytes)

assert len(image_ids) == 16
assert len(set(image_ids)) == 16
assert len(grades) == 16
assert len(risk_groups) == 16

assert len(results) == 16
assert results["image_id"].tolist() == image_ids
assert r1_cam_array.shape == (16, 224, 224)
assert r0_logit_array.shape == (16, 5)
assert r1_logit_array.shape == (16, 5)

assert np.isfinite(r1_cam_array).all()
assert np.isfinite(r0_logit_array).all()
assert np.isfinite(r1_logit_array).all()

expected_npz_name = "RETFound_Seed1042_R1_Batch001.npz"
expected_csv_name = "RETFound_Seed1042_R1_Batch001.csv"

NPZ_PATH = Path(NPZ_PATH)
CSV_PATH = Path(CSV_PATH)

assert NPZ_PATH.name == expected_npz_name
assert CSV_PATH.name == expected_csv_name
assert NPZ_PATH.parent == CSV_PATH.parent

expected_npz_sha256 = hashlib.sha256(
    npz_bytes
).hexdigest()

expected_csv_sha256 = hashlib.sha256(
    csv_bytes
).hexdigest()

print("\nIN-MEMORY RECOVERY")
print("Images:", len(image_ids))
print("NPZ bytes:", len(npz_bytes))
print("CSV bytes:", len(csv_bytes))
print("Memory arrays: PASS")

# ============================================================
# 2. Validate both serialized archives before writing
# ============================================================

with np.load(
    io.BytesIO(npz_bytes),
    allow_pickle=False
) as archive:

    assert list(map(str, archive["image_ids"])) == image_ids
    assert archive["r0_cam"].shape == (16, 224, 224)
    assert archive["r1_cam"].shape == (16, 224, 224)

    assert np.array_equal(
        archive["r1_cam"],
        r1_cam_array
    )

    assert np.array_equal(
        archive["r0_logits"],
        r0_logit_array.astype(np.float32)
    )

    assert np.array_equal(
        archive["r1_logits"],
        r1_logit_array.astype(np.float32)
    )

    assert str(archive["model"].item()) == "RETFound"
    assert str(archive["stage"].item()) == "R1"
    assert int(archive["seed"].item()) == 1042

memory_csv = pd.read_csv(
    io.BytesIO(csv_bytes),
    dtype={
        "image_id": str,
        "risk_group_id": str
    }
)

assert len(memory_csv) == 16
assert memory_csv["image_id"].tolist() == image_ids
assert memory_csv["seed"].eq(1042).all()
assert memory_csv["stage"].eq("R1").all()
assert memory_csv["model"].eq("RETFound").all()

print("Serialized NPZ validation: PASS")
print("Serialized CSV validation: PASS")

# ============================================================
# 3. Google Drive-compatible exclusive file creation
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def publish_without_overwrite(path, payload, expected_hash):
    """
    Create a new file exclusively.

    If the file already exists, accept it only when its
    contents exactly match the prepared in-memory payload.
    Never overwrite an existing file.
    """

    if path.exists():
        actual_hash = sha256_file(path)

        assert actual_hash == expected_hash, (
            f"Existing file differs from memory: {path}"
        )

        return "EXISTING_VERIFIED"

    # Exclusive creation is supported by the Drive mount.
    # Unlike os.link(), this does not require hard-link support.
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()

    except FileExistsError:
        actual_hash = sha256_file(path)

        assert actual_hash == expected_hash, (
            f"File appeared with unexpected content: {path}"
        )

        return "EXISTING_VERIFIED"

    actual_hash = sha256_file(path)

    assert actual_hash == expected_hash, (
        f"Post-write SHA256 mismatch: {path}"
    )

    return "CREATED_VERIFIED"


assert NPZ_PATH.parent.is_dir(), (
    "Expected production output directory is missing."
)

npz_status = publish_without_overwrite(
    NPZ_PATH,
    npz_bytes,
    expected_npz_sha256
)

csv_status = publish_without_overwrite(
    CSV_PATH,
    csv_bytes,
    expected_csv_sha256
)

# ============================================================
# 4. Verify saved data through independent file reads
# ============================================================

with np.load(
    NPZ_PATH,
    allow_pickle=False
) as archive:

    assert list(map(str, archive["image_ids"])) == image_ids

    assert archive["r0_cam"].shape == (16, 224, 224)
    assert archive["r1_cam"].shape == (16, 224, 224)

    assert np.array_equal(
        archive["r1_cam"],
        r1_cam_array
    )

    assert np.array_equal(
        archive["r0_logits"],
        r0_logit_array.astype(np.float32)
    )

    assert np.array_equal(
        archive["r1_logits"],
        r1_logit_array.astype(np.float32)
    )

    assert int(archive["seed"].item()) == 1042
    assert str(archive["stage"].item()) == "R1"

saved_csv = pd.read_csv(
    CSV_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str
    }
)

assert len(saved_csv) == 16

assert saved_csv["image_id"].tolist() == image_ids
assert saved_csv["seed"].eq(1042).all()
assert saved_csv["stage"].eq("R1").all()
assert saved_csv["model"].eq("RETFound").all()

npz_hash = sha256_file(NPZ_PATH)
csv_hash = sha256_file(CSV_PATH)

assert npz_hash == expected_npz_sha256
assert csv_hash == expected_csv_sha256

# ============================================================
# 5. Report recovered production results
# ============================================================

print("\nBATCH 001 RESULTS")
print("Model: RETFound")
print("Seed: 1042")
print("Stage: R1")
print("Images evaluated:", len(saved_csv))

print(
    "Degenerate CAMs:",
    int(saved_csv["degenerate_cam"].sum())
)

for column in [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]:
    print(
        f"Mean {column}:",
        round(
            float(saved_csv[column].mean()),
            6
        )
    )

print("\nARCHIVE INTEGRITY")

print("NPZ status:", npz_status)
print("NPZ path:", NPZ_PATH)
print("NPZ SHA256:", npz_hash)

print("CSV status:", csv_status)
print("CSV path:", CSV_PATH)
print("CSV SHA256:", csv_hash)

print("Archive round-trip: PASS")
print("Existing outputs overwritten: False")

print("\nRESEARCH INTEGRITY")
print("Model inference performed: False")
print("Images decoded in recovery: 0")
print("Original research files modified: False")
print("Test images accessed: 0")

print("\nSTEP 10.37B EXECUTION COMPLETE")
print("Production batch 001 archive: PENDING REVIEW")


# %% [cell 377] Experiment 10 - Step 10.38

# ============================================================
# Experiment 10 - Step 10.38
# RETFound R1 Production Batch 002
# Seed 1042 | Frozen manifest rows 16:32
# Exclusive archive creation | No test-set access
# ============================================================

from pathlib import Path
import gc
import hashlib
import io
import json

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

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"
OUTPUT_DIR = EXP10 / "full_validation_batches"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"
PLAN_PATH = EXP10 / "Experiment10_Full_Execution_Plan_V1.json"
PROTOCOL_PATH = EXP10 / "Experiment10_Randomization_Protocol_V1.json"
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"

CHECKPOINT_PATH = (
    ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"
)

REFERENCE_PATH = (
    ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"
)

BATCH_NAME = "RETFound_Seed1042_R1_Batch002"
NPZ_PATH = OUTPUT_DIR / f"{BATCH_NAME}.npz"
CSV_PATH = OUTPUT_DIR / f"{BATCH_NAME}.csv"

PREVIOUS_NPZ = (
    OUTPUT_DIR / "RETFound_Seed1042_R1_Batch001.npz"
)
PREVIOUS_CSV = (
    OUTPUT_DIR / "RETFound_Seed1042_R1_Batch001.csv"
)

EXPECTED_HASHES = {
    MANIFEST_PATH:
        "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5",
    PLAN_PATH:
        "da52de5513de274a0b91c99f7060049a1f77db4c6dc2a788263607e6d8168590",
    PROTOCOL_PATH:
        "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda",
    INDEX_PATH:
        "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    CHECKPOINT_PATH:
        "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8",
    PREVIOUS_NPZ:
        "4f71b19be85c52bcdce8f0d628fbc34d0171f9b91170204fa1c0b4d6b6a28c43",
    PREVIOUS_CSV:
        "529f23beed5a47c67b0a545ce32330ddcec1973ee93eb2d8ea315c88ab7d0918"
}

SEED = 1042
BATCH_START = 16
BATCH_END = 32
DEVICE = torch.device("cuda")

assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - a.mean()
    bc = b - b.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12 else np.nan
    )

    return {
        "pearson": pearson,
        "spearman": float(spearmanr(a, b).statistic),
        "ssim": float(
            structural_similarity(first, second, data_range=1.0)
        ),
        "mae": float(np.mean(np.abs(a - b)))
    }


def publish_exclusively(path, data):
    """
    Create a new file only if it does not exist.
    This method does not claim atomic commit on Google Drive.
    """
    if path.exists():
        raise FileExistsError(
            f"Existing batch output requires review: {path}"
        )

    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()

    expected_hash = hashlib.sha256(data).hexdigest()
    actual_hash = sha256_file(path)

    assert actual_hash == expected_hash, (
        f"Post-write hash mismatch: {path}"
    )

    return actual_hash


print("EXPERIMENT 10.38 - RETFOUND R1 PRODUCTION BATCH 002")
print("Seed:", SEED)
print("Batch interval:", BATCH_START, BATCH_END)

# ============================================================
# 1. Verify frozen inputs and previously completed batch
# ============================================================

for path, expected_hash in EXPECTED_HASHES.items():
    assert path.is_file(), f"Missing file: {path}"
    assert sha256_file(path) == expected_hash, (
        f"SHA256 mismatch: {path}"
    )

assert COHORT_PATH.is_file()
assert REFERENCE_PATH.is_file()

assert not NPZ_PATH.exists(), (
    f"Batch 002 NPZ already exists: {NPZ_PATH}"
)
assert not CSV_PATH.exists(), (
    f"Batch 002 CSV already exists: {CSV_PATH}"
)

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    plan = json.load(handle)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert plan["manifest_sha256"] == EXPECTED_HASHES[MANIFEST_PATH]
assert plan["image_count"] == 156
assert plan["test_access_permitted"] is False
assert plan["randomization_seeds"] == [1042, 2042, 3042]

assert SEED in protocol["randomization"]["randomization_seeds"]
assert set(protocol["randomization"]["stages"]["R1"]) == {"head"}

print("\nFROZEN INPUT INTEGRITY")
print("Manifest SHA256: PASS")
print("Execution plan SHA256: PASS")
print("Randomization protocol SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Batch 001 NPZ SHA256: PASS")
print("Batch 001 CSV SHA256: PASS")

# ============================================================
# 2. Select the exact 16-image production batch
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique
assert manifest["split"].eq("val").all()
assert manifest["image_id"].tolist() == sorted(
    manifest["image_id"].tolist()
)

batch = manifest.iloc[BATCH_START:BATCH_END].copy()

assert len(batch) == 16
assert batch["image_id"].is_unique
assert batch["true_grade"].isin([1, 2, 3, 4]).all()

image_ids = batch["image_id"].tolist()
grades = batch["true_grade"].astype(int).tolist()
risk_groups = batch["risk_group_id"].astype(str).tolist()

# Verify non-overlap with the already archived first batch.
with np.load(PREVIOUS_NPZ, allow_pickle=False) as previous:
    previous_ids = list(map(str, previous["image_ids"]))
    assert len(previous_ids) == 16
    assert len(set(previous_ids)) == 16
    assert previous_ids == manifest.iloc[:16]["image_id"].tolist()

assert set(image_ids).isdisjoint(previous_ids)

cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})

assert cohort["image_id"].is_unique

candidate_flags = (
    cohort.set_index("image_id")
    .loc[image_ids, "primary_ma_candidate"]
    .astype(str)
    .str.strip()
    .str.lower()
)

assert candidate_flags.isin(["true", "1"]).all()

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    reference_positions = {
        image_id: i for i, image_id in enumerate(reference_ids)
    }

    assert set(image_ids).issubset(reference_positions)

    reference_cams = np.stack([
        archive["retfound_cam"][reference_positions[image_id]]
        .astype(np.float32)
        for image_id in image_ids
    ])

assert reference_cams.shape == (16, 224, 224)
assert np.isfinite(reference_cams).all()

print("\nBATCH COHORT")
print("Images:", len(image_ids))
print("Distinct source-risk groups:", len(set(risk_groups)))
print("Batch 001 image overlap:", 0)
print("Validation-only selection: PASS")
print("Original CAM ID alignment: PASS")
print("First image ID:", image_ids[0])
print("Last image ID:", image_ids[-1])

# ============================================================
# 3. Decode only the selected validation images
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensors = []

for _, row in batch.iterrows():
    image_path = Path(row["resolved_image_path"])
    assert image_path.is_file(), row["image_id"]

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    assert tensor.shape == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all().item()

    input_tensors.append(tensor)

print("Images decoded:", len(input_tensors))

# ============================================================
# 4. Load frozen RETFound checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

loaded = model.load_state_dict(
    source_state,
    strict=True
)

assert not loaded.missing_keys
assert not loaded.unexpected_keys

model.eval()
model.requires_grad_(False)

# ============================================================
# 5. Reconstruct the frozen Grad-CAM method
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(
    activation_hook
)


def calculate_cam(cpu_tensor, true_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(cpu_tensor.to(DEVICE))

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, true_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2), keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12
        cam /= float(cam.max())

        assert np.isfinite(cam).all()

    captured.clear()
    return output_logits, cam, raw_max


# ============================================================
# 6. Calculate R0 and R1 using one fixed randomized head
# ============================================================

r0_logits = []
r0_cams = []
r1_logits = []
r1_cams = []
r1_raw_maxima = []

try:
    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, _ = calculate_cam(cpu_tensor, grade)

        assert cam is not None

        r0_logits.append(logits)
        r0_cams.append(cam)

    baseline_metrics = [
        compare_cams(cam, reference)
        for cam, reference in zip(r0_cams, reference_cams)
    ]

    baseline_pearsons = [
        item["pearson"] for item in baseline_metrics
    ]

    baseline_maes = [
        item["mae"] for item in baseline_metrics
    ]

    assert np.isfinite(baseline_pearsons).all()
    assert min(baseline_pearsons) > 0.999

    print("\nR0 REPRODUCTION")
    print(
        "Minimum reference CAM Pearson:",
        round(float(np.min(baseline_pearsons)), 6)
    )
    print(
        "Mean reference CAM MAE:",
        round(float(np.mean(baseline_maes)), 8)
    )
    print("Baseline CAM reproduction: PASS")

    # Keep the same initialization sequence as Batch 001.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    changed_names = []

    for name, parameter in model.state_dict().items():
        current = parameter.detach().cpu()

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, source_state[name]):
            changed_names.append(name)

    assert set(changed_names) == {
        "head.weight", "head.bias"
    }

    print("\nR1 PARAMETER INTEGRITY")
    print("Changed tensors:", len(changed_names))
    print("Unauthorized changes:", 0)
    print("Seed:", SEED)

    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, raw_max = calculate_cam(
            cpu_tensor, grade
        )

        r1_logits.append(logits)
        r1_cams.append(cam)
        r1_raw_maxima.append(raw_max)

finally:
    hook.remove()
    captured.clear()

    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Build complete image-level batch records
# ============================================================

rows = []
stored_cams = []

for i, image_id in enumerate(image_ids):
    metrics = compare_cams(
        r0_cams[i],
        r1_cams[i]
    )

    rows.append({
        "model": "RETFound",
        "seed": SEED,
        "stage": "R1",
        "batch": "002",
        "image_id": image_id,
        "true_grade": grades[i],
        "risk_group_id": risk_groups[i],
        "degenerate_cam": r1_cams[i] is None,
        "raw_max": r1_raw_maxima[i],
        "r0_r1_pearson": metrics["pearson"],
        "r0_r1_spearman": metrics["spearman"],
        "r0_r1_ssim": metrics["ssim"],
        "r0_r1_mae": metrics["mae"],
        "r0_r1_logit_l2": float(
            np.linalg.norm(
                r1_logits[i] - r0_logits[i]
            )
        )
    })

    stored_cams.append(
        np.zeros((224, 224), dtype=np.float32)
        if r1_cams[i] is None
        else r1_cams[i].astype(np.float32)
    )

results = pd.DataFrame(rows)

r0_cam_array = np.stack(r0_cams).astype(np.float32)
r1_cam_array = np.stack(stored_cams).astype(np.float32)

r0_logit_array = np.stack(r0_logits).astype(np.float32)
r1_logit_array = np.stack(r1_logits).astype(np.float32)

assert len(results) == 16
assert results["image_id"].is_unique
assert r0_cam_array.shape == (16, 224, 224)
assert r1_cam_array.shape == (16, 224, 224)
assert r0_logit_array.shape == (16, 5)
assert r1_logit_array.shape == (16, 5)

assert np.isfinite(r0_cam_array).all()
assert np.isfinite(r1_cam_array).all()
assert np.isfinite(r0_logit_array).all()
assert np.isfinite(r1_logit_array).all()

# ============================================================
# 8. Serialize and validate data before publication
# ============================================================

npz_buffer = io.BytesIO()

np.savez_compressed(
    npz_buffer,
    image_ids=np.asarray(image_ids, dtype=str),
    true_grades=np.asarray(grades, dtype=np.int64),
    risk_group_ids=np.asarray(risk_groups, dtype=str),
    model=np.asarray("RETFound"),
    seed=np.asarray(SEED),
    stage=np.asarray("R1"),
    r0_cam=r0_cam_array,
    r1_cam=r1_cam_array,
    r0_logits=r0_logit_array,
    r1_logits=r1_logit_array,
    r1_raw_max=np.asarray(r1_raw_maxima, dtype=np.float64),
    r1_degenerate=results["degenerate_cam"].to_numpy(dtype=bool),
    manifest_sha256=np.asarray(EXPECTED_HASHES[MANIFEST_PATH]),
    execution_plan_sha256=np.asarray(EXPECTED_HASHES[PLAN_PATH])
)

npz_bytes = npz_buffer.getvalue()

csv_bytes = results.to_csv(
    index=False,
    float_format="%.10g"
).encode("utf-8")

with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)

prepared_csv = pd.read_csv(
    io.BytesIO(csv_bytes),
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(prepared_csv) == 16
assert prepared_csv["image_id"].tolist() == image_ids

# ============================================================
# 9. Publish on Google Drive without hard links
# ============================================================

assert OUTPUT_DIR.is_dir()
assert not NPZ_PATH.exists()
assert not CSV_PATH.exists()

# Keep npz_bytes and csv_bytes in memory for recovery if
# Google Drive raises an exception during publication.
npz_hash = publish_exclusively(NPZ_PATH, npz_bytes)
csv_hash = publish_exclusively(CSV_PATH, csv_bytes)

# ============================================================
# 10. Verify published NPZ and CSV independently
# ============================================================

with np.load(NPZ_PATH, allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert archive["r0_cam"].shape == (16, 224, 224)
    assert archive["r1_cam"].shape == (16, 224, 224)

    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)

    assert np.array_equal(
        archive["r0_logits"], r0_logit_array
    )

    assert np.array_equal(
        archive["r1_logits"], r1_logit_array
    )

    assert int(archive["seed"].item()) == 1042
    assert str(archive["stage"].item()) == "R1"
    assert str(archive["model"].item()) == "RETFound"

saved_csv = pd.read_csv(
    CSV_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(saved_csv) == 16
assert saved_csv["image_id"].tolist() == image_ids
assert saved_csv["seed"].eq(SEED).all()
assert saved_csv["stage"].eq("R1").all()
assert saved_csv["model"].eq("RETFound").all()

assert sha256_file(NPZ_PATH) == npz_hash
assert sha256_file(CSV_PATH) == csv_hash

# ============================================================
# 11. Report completed production batch
# ============================================================

print("\nBATCH RESULTS")
print("Model: RETFound")
print("Seed:", SEED)
print("Stage: R1")
print("Batch: 002")
print("Images evaluated:", len(saved_csv))

print(
    "Degenerate CAMs:",
    int(saved_csv["degenerate_cam"].sum())
)

for column in [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]:
    print(
        f"Mean {column}:",
        round(float(saved_csv[column].mean()), 6)
    )

print("\nPER-IMAGE RESULTS")
print(
    saved_csv[
        [
            "image_id",
            "true_grade",
            "risk_group_id",
            "degenerate_cam",
            "r0_r1_pearson",
            "r0_r1_spearman",
            "r0_r1_ssim",
            "r0_r1_mae"
        ]
    ].to_string(index=False)
)

print("\nARCHIVE INTEGRITY")
print("NPZ path:", NPZ_PATH)
print("NPZ SHA256:", npz_hash)
print("CSV path:", CSV_PATH)
print("CSV SHA256:", csv_hash)
print("Batch 001 hashes unchanged: PASS")
print("Batch 002 archive round-trip: PASS")
print("Previous outputs overwritten: False")

print("\nRESEARCH INTEGRITY")
print("Frozen inputs modified: False")
print("Validation images decoded:", len(input_tensors))
print("Test images accessed: 0")
print("New batch files created: 2")

print("\nSTEP 10.38 EXECUTION COMPLETE")
print("Production batch 002: PENDING REVIEW")


# %% [cell 378] Experiment 10 - Step 10.39

# ============================================================
# Experiment 10 - Step 10.39
# RETFound R1 Production Batch 003
# Seed 1042 | Frozen validation manifest rows 32:48
# Google Drive-compatible exclusive publication
# ============================================================

from pathlib import Path
import gc
import hashlib
import io
import json

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

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"
OUTPUT_DIR = EXP10 / "full_validation_batches"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"
PLAN_PATH = EXP10 / "Experiment10_Full_Execution_Plan_V1.json"
PROTOCOL_PATH = EXP10 / "Experiment10_Randomization_Protocol_V1.json"
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
CHECKPOINT_PATH = ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"
REFERENCE_PATH = ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"

SEED = 1042
BATCH_START = 32
BATCH_END = 48
BATCH_NUMBER = "003"
BATCH_NAME = f"RETFound_Seed{SEED}_R1_Batch{BATCH_NUMBER}"

NPZ_PATH = OUTPUT_DIR / f"{BATCH_NAME}.npz"
CSV_PATH = OUTPUT_DIR / f"{BATCH_NAME}.csv"

EXPECTED_HASHES = {
    MANIFEST_PATH: "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5",
    PLAN_PATH: "da52de5513de274a0b91c99f7060049a1f77db4c6dc2a788263607e6d8168590",
    PROTOCOL_PATH: "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda",
    INDEX_PATH: "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    CHECKPOINT_PATH: "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8",
    OUTPUT_DIR / "RETFound_Seed1042_R1_Batch001.npz":
        "4f71b19be85c52bcdce8f0d628fbc34d0171f9b91170204fa1c0b4d6b6a28c43",
    OUTPUT_DIR / "RETFound_Seed1042_R1_Batch001.csv":
        "529f23beed5a47c67b0a545ce32330ddcec1973ee93eb2d8ea315c88ab7d0918",
    OUTPUT_DIR / "RETFound_Seed1042_R1_Batch002.npz":
        "946cc0bd797bf43f8d857cd5618ab4976aefbcbad96e0d6a2606365a30c1deab",
    OUTPUT_DIR / "RETFound_Seed1042_R1_Batch002.csv":
        "ca444fde5920a09f5fc64479d09d0919945d741cdeeb24ffac5ca8341f88089a"
}

DEVICE = torch.device("cuda")
assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - np.mean(a)
    bc = b - np.mean(b)

    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12 else np.nan
    )

    return {
        "pearson": pearson,
        "spearman": float(spearmanr(a, b).statistic),
        "ssim": float(
            structural_similarity(first, second, data_range=1.0)
        ),
        "mae": float(np.mean(np.abs(a - b)))
    }


def publish_exclusively(path, payload):
    if path.exists():
        raise FileExistsError(
            f"Existing output requires integrity review: {path}"
        )

    expected_hash = hashlib.sha256(payload).hexdigest()

    # Keep payload in memory if Google Drive publication fails.
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()

    actual_hash = sha256_file(path)

    assert actual_hash == expected_hash, (
        f"Post-publication SHA256 mismatch: {path}"
    )

    return actual_hash


print("EXPERIMENT 10.39 - RETFOUND R1 PRODUCTION BATCH 003")
print("Seed:", SEED)
print("Batch interval:", BATCH_START, BATCH_END)

# ============================================================
# 1. Verify frozen inputs and both previous batches
# ============================================================

for path, expected_hash in EXPECTED_HASHES.items():
    assert path.is_file(), f"Missing file: {path}"
    assert sha256_file(path) == expected_hash, (
        f"SHA256 mismatch: {path}"
    )

assert COHORT_PATH.is_file()
assert REFERENCE_PATH.is_file()
assert OUTPUT_DIR.is_dir()

assert not NPZ_PATH.exists(), f"Output already exists: {NPZ_PATH}"
assert not CSV_PATH.exists(), f"Output already exists: {CSV_PATH}"

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    plan = json.load(handle)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert plan["manifest_sha256"] == EXPECTED_HASHES[MANIFEST_PATH]
assert plan["image_count"] == 156
assert plan["test_access_permitted"] is False
assert plan["randomization_seeds"] == [1042, 2042, 3042]

assert set(protocol["randomization"]["stages"]["R1"]) == {"head"}
assert SEED in protocol["randomization"]["randomization_seeds"]

print("\nFROZEN INPUT INTEGRITY")
print("Manifest SHA256: PASS")
print("Execution plan SHA256: PASS")
print("Randomization protocol SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Batch 001 NPZ/CSV SHA256: PASS")
print("Batch 002 NPZ/CSV SHA256: PASS")

# ============================================================
# 2. Resolve the exact frozen batch
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique
assert manifest["split"].eq("val").all()

assert manifest["image_id"].tolist() == sorted(
    manifest["image_id"].tolist()
)

batch = manifest.iloc[BATCH_START:BATCH_END].copy()

assert len(batch) == 16
assert batch["image_id"].is_unique
assert batch["true_grade"].isin([1, 2, 3, 4]).all()

image_ids = batch["image_id"].tolist()
grades = batch["true_grade"].astype(int).tolist()
risk_groups = batch["risk_group_id"].astype(str).tolist()

assert all(
    group.strip() not in {"", "nan", "None"}
    for group in risk_groups
)

previous_ids = []

for previous_number, start, stop in [
    ("001", 0, 16),
    ("002", 16, 32)
]:
    previous_path = (
        OUTPUT_DIR
        / f"RETFound_Seed1042_R1_Batch{previous_number}.npz"
    )

    with np.load(previous_path, allow_pickle=False) as archive:
        ids = list(map(str, archive["image_ids"]))
        assert ids == manifest.iloc[start:stop]["image_id"].tolist()
        previous_ids.extend(ids)

assert len(previous_ids) == 32
assert len(set(previous_ids)) == 32
assert set(image_ids).isdisjoint(previous_ids)

cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})

assert cohort["image_id"].is_unique

candidate_flags = (
    cohort.set_index("image_id")
    .loc[image_ids, "primary_ma_candidate"]
    .astype(str)
    .str.strip()
    .str.lower()
)

assert candidate_flags.isin(["true", "1"]).all()

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    lookup = {
        image_id: i for i, image_id in enumerate(reference_ids)
    }

    assert set(image_ids).issubset(lookup)

    reference_cams = np.stack([
        archive["retfound_cam"][lookup[image_id]]
        .astype(np.float32)
        for image_id in image_ids
    ])

assert reference_cams.shape == (16, 224, 224)
assert np.isfinite(reference_cams).all()

print("\nBATCH COHORT")
print("Images:", len(image_ids))
print("Distinct source-risk groups:", len(set(risk_groups)))
print("Overlap with previous batches:", 0)
print("Validation-only selection: PASS")
print("Original CAM ID alignment: PASS")
print("First image ID:", image_ids[0])
print("Last image ID:", image_ids[-1])

# ============================================================
# 3. Decode only the selected validation images
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensors = []

for _, row in batch.iterrows():
    image_path = Path(row["resolved_image_path"])
    assert image_path.is_file(), row["image_id"]

    with Image.open(image_path) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0)

    assert tensor.shape == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all().item()

    input_tensors.append(tensor)

print("Images decoded:", len(input_tensors))

# ============================================================
# 4. Load original RETFound model
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)
load_result = model.load_state_dict(source_state, strict=True)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

# ============================================================
# 5. Reconstruct frozen Grad-CAM
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(activation_hook)


def calculate_cam(cpu_tensor, true_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(cpu_tensor.to(DEVICE))

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, true_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :].detach().float()
        .transpose(0, 1).reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :].detach().float()
        .transpose(0, 1).reshape(1024, 14, 14)
    )

    weights = gradients.mean(dim=(1, 2), keepdim=True)

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()
    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12
        cam /= float(cam.max())
        assert np.isfinite(cam).all()

    captured.clear()

    return output_logits, cam, raw_max


# ============================================================
# 6. Calculate R0 and R1
# ============================================================

r0_logits = []
r0_cams = []
r1_logits = []
r1_cams = []
r1_raw_maxima = []

try:
    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, _ = calculate_cam(cpu_tensor, grade)

        assert cam is not None

        r0_logits.append(logits)
        r0_cams.append(cam)

    baseline_metrics = [
        compare_cams(cam, reference)
        for cam, reference in zip(r0_cams, reference_cams)
    ]

    baseline_pearsons = [
        item["pearson"] for item in baseline_metrics
    ]

    baseline_maes = [
        item["mae"] for item in baseline_metrics
    ]

    assert np.isfinite(baseline_pearsons).all()
    assert min(baseline_pearsons) > 0.999

    print("\nR0 REPRODUCTION")
    print(
        "Minimum reference CAM Pearson:",
        round(float(np.min(baseline_pearsons)), 6)
    )
    print(
        "Mean reference CAM MAE:",
        round(float(np.mean(baseline_maes)), 8)
    )
    print("Baseline CAM reproduction: PASS")

    # Reset RNG once, exactly as in previous R1 batches.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    changed_names = []

    for name, parameter in model.state_dict().items():
        current = parameter.detach().cpu()

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, source_state[name]):
            changed_names.append(name)

    assert set(changed_names) == {
        "head.weight",
        "head.bias"
    }

    print("\nR1 PARAMETER INTEGRITY")
    print("Changed tensors:", len(changed_names))
    print("Unauthorized changes:", 0)
    print("Seed:", SEED)

    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, raw_max = calculate_cam(cpu_tensor, grade)

        r1_logits.append(logits)
        r1_cams.append(cam)
        r1_raw_maxima.append(raw_max)

finally:
    hook.remove()
    captured.clear()

    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Assemble original image-level results
# ============================================================

rows = []
stored_cams = []

for i, image_id in enumerate(image_ids):
    metrics = compare_cams(r0_cams[i], r1_cams[i])

    rows.append({
        "model": "RETFound",
        "seed": SEED,
        "stage": "R1",
        "batch": BATCH_NUMBER,
        "image_id": image_id,
        "true_grade": grades[i],
        "risk_group_id": risk_groups[i],
        "degenerate_cam": r1_cams[i] is None,
        "raw_max": r1_raw_maxima[i],
        "r0_r1_pearson": metrics["pearson"],
        "r0_r1_spearman": metrics["spearman"],
        "r0_r1_ssim": metrics["ssim"],
        "r0_r1_mae": metrics["mae"],
        "r0_r1_logit_l2": float(
            np.linalg.norm(r1_logits[i] - r0_logits[i])
        )
    })

    stored_cams.append(
        np.zeros((224, 224), dtype=np.float32)
        if r1_cams[i] is None
        else r1_cams[i].astype(np.float32)
    )

results = pd.DataFrame(rows)

r0_cam_array = np.stack(r0_cams).astype(np.float32)
r1_cam_array = np.stack(stored_cams).astype(np.float32)
r0_logit_array = np.stack(r0_logits).astype(np.float32)
r1_logit_array = np.stack(r1_logits).astype(np.float32)

assert len(results) == 16
assert results["image_id"].is_unique
assert r0_cam_array.shape == (16, 224, 224)
assert r1_cam_array.shape == (16, 224, 224)
assert r0_logit_array.shape == (16, 5)
assert r1_logit_array.shape == (16, 5)

for array in [
    r0_cam_array, r1_cam_array,
    r0_logit_array, r1_logit_array
]:
    assert np.isfinite(array).all()

# ============================================================
# 8. Serialize and validate archives in memory
# ============================================================

npz_buffer = io.BytesIO()

np.savez_compressed(
    npz_buffer,
    image_ids=np.asarray(image_ids, dtype=str),
    true_grades=np.asarray(grades, dtype=np.int64),
    risk_group_ids=np.asarray(risk_groups, dtype=str),
    model=np.asarray("RETFound"),
    seed=np.asarray(SEED),
    stage=np.asarray("R1"),
    r0_cam=r0_cam_array,
    r1_cam=r1_cam_array,
    r0_logits=r0_logit_array,
    r1_logits=r1_logit_array,
    r1_raw_max=np.asarray(r1_raw_maxima, dtype=np.float64),
    r1_degenerate=results["degenerate_cam"].to_numpy(dtype=bool),
    manifest_sha256=np.asarray(EXPECTED_HASHES[MANIFEST_PATH]),
    execution_plan_sha256=np.asarray(EXPECTED_HASHES[PLAN_PATH])
)

npz_bytes = npz_buffer.getvalue()

csv_bytes = results.to_csv(
    index=False,
    float_format="%.10g"
).encode("utf-8")

with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)
    assert str(archive["model"].item()) == "RETFound"
    assert str(archive["stage"].item()) == "R1"
    assert int(archive["seed"].item()) == SEED

prepared_csv = pd.read_csv(
    io.BytesIO(csv_bytes),
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(prepared_csv) == 16
assert prepared_csv["image_id"].tolist() == image_ids
assert prepared_csv["seed"].eq(SEED).all()

print("\nIN-MEMORY ARCHIVE VERIFICATION")
print("NPZ serialization: PASS")
print("CSV serialization: PASS")

# ============================================================
# 9. Save without overwriting existing files
# ============================================================

assert not NPZ_PATH.exists()
assert not CSV_PATH.exists()

# These byte buffers remain in memory if publication fails.
npz_hash = publish_exclusively(NPZ_PATH, npz_bytes)
csv_hash = publish_exclusively(CSV_PATH, csv_bytes)

# ============================================================
# 10. Independently verify published files
# ============================================================

with np.load(NPZ_PATH, allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)

saved_csv = pd.read_csv(
    CSV_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(saved_csv) == 16
assert saved_csv["image_id"].tolist() == image_ids
assert saved_csv["seed"].eq(SEED).all()
assert saved_csv["stage"].eq("R1").all()
assert saved_csv["model"].eq("RETFound").all()

assert sha256_file(NPZ_PATH) == npz_hash
assert sha256_file(CSV_PATH) == csv_hash

# ============================================================
# 11. Final report
# ============================================================

print("\nBATCH RESULTS")
print("Model: RETFound")
print("Seed:", SEED)
print("Stage: R1")
print("Batch:", BATCH_NUMBER)
print("Images evaluated:", len(saved_csv))
print(
    "Degenerate CAMs:",
    int(saved_csv["degenerate_cam"].sum())
)

for column in [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]:
    print(
        f"Mean {column}:",
        round(float(saved_csv[column].mean()), 6)
    )

print("\nPER-IMAGE RESULTS")
print(
    saved_csv[
        [
            "image_id",
            "true_grade",
            "risk_group_id",
            "degenerate_cam",
            "r0_r1_pearson",
            "r0_r1_spearman",
            "r0_r1_ssim",
            "r0_r1_mae"
        ]
    ].to_string(index=False)
)

print("\nARCHIVE INTEGRITY")
print("NPZ path:", NPZ_PATH)
print("NPZ SHA256:", npz_hash)
print("CSV path:", CSV_PATH)
print("CSV SHA256:", csv_hash)
print("Previous batch hashes unchanged: PASS")
print("Batch 003 archive round-trip: PASS")
print("Existing files overwritten: False")

print("\nRESEARCH INTEGRITY")
print("Frozen input files modified: False")
print("Validation images decoded:", len(input_tensors))
print("Test images accessed: 0")
print("New batch files created: 2")

print("\nSTEP 10.39 EXECUTION COMPLETE")
print("Production batch 003: PENDING REVIEW")


# %% [cell 379] Experiment 10 - Step 10.40

# ============================================================
# Experiment 10 - Step 10.40
# RETFound R1 Production Batch 004
# Seed 1042 | Frozen validation manifest rows 48:64
# Read-only input artifacts | Exclusive archive publication
# ============================================================

from pathlib import Path
import gc
import hashlib
import io
import json

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

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"
OUTPUT_DIR = EXP10 / "full_validation_batches"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"
PLAN_PATH = EXP10 / "Experiment10_Full_Execution_Plan_V1.json"
PROTOCOL_PATH = EXP10 / "Experiment10_Randomization_Protocol_V1.json"
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"

CHECKPOINT_PATH = (
    ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"
)

REFERENCE_PATH = (
    ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"
)

SEED = 1042
BATCH_START = 48
BATCH_END = 64
BATCH_NUMBER = "004"

BATCH_NAME = f"RETFound_Seed{SEED}_R1_Batch{BATCH_NUMBER}"
NPZ_PATH = OUTPUT_DIR / f"{BATCH_NAME}.npz"
CSV_PATH = OUTPUT_DIR / f"{BATCH_NAME}.csv"

EXPECTED_HASHES = {
    MANIFEST_PATH:
        "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5",
    PLAN_PATH:
        "da52de5513de274a0b91c99f7060049a1f77db4c6dc2a788263607e6d8168590",
    PROTOCOL_PATH:
        "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda",
    INDEX_PATH:
        "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    CHECKPOINT_PATH:
        "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8",
    OUTPUT_DIR / "RETFound_Seed1042_R1_Batch001.npz":
        "4f71b19be85c52bcdce8f0d628fbc34d0171f9b91170204fa1c0b4d6b6a28c43",
    OUTPUT_DIR / "RETFound_Seed1042_R1_Batch001.csv":
        "529f23beed5a47c67b0a545ce32330ddcec1973ee93eb2d8ea315c88ab7d0918",
    OUTPUT_DIR / "RETFound_Seed1042_R1_Batch002.npz":
        "946cc0bd797bf43f8d857cd5618ab4976aefbcbad96e0d6a2606365a30c1deab",
    OUTPUT_DIR / "RETFound_Seed1042_R1_Batch002.csv":
        "ca444fde5920a09f5fc64479d09d0919945d741cdeeb24ffac5ca8341f88089a",
    OUTPUT_DIR / "RETFound_Seed1042_R1_Batch003.npz":
        "ebb83b4f68c13e6bcb2f079290c7846707ff3e6f76cdacfa264a38beab9ff9c6",
    OUTPUT_DIR / "RETFound_Seed1042_R1_Batch003.csv":
        "4a53c0c75cb040d8309413d934599f11515cd10e1498d636735f8cf87d60772f"
}

DEVICE = torch.device("cuda")
assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - a.mean()
    bc = b - b.mean()

    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12
        else np.nan
    )

    return {
        "pearson": pearson,
        "spearman": float(spearmanr(a, b).statistic),
        "ssim": float(
            structural_similarity(
                first, second, data_range=1.0
            )
        ),
        "mae": float(np.mean(np.abs(a - b)))
    }


def publish_exclusively(path, payload):
    if path.exists():
        raise FileExistsError(
            f"Existing output requires review: {path}"
        )

    expected_hash = hashlib.sha256(payload).hexdigest()

    # Exclusive creation avoids overwriting existing Drive files.
    # A post-write hash check detects incomplete publication.
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()

    actual_hash = sha256_file(path)

    assert actual_hash == expected_hash, (
        f"Post-write SHA256 mismatch: {path}"
    )

    return actual_hash


print("EXPERIMENT 10.40 - RETFOUND R1 PRODUCTION BATCH 004")
print("Seed:", SEED)
print("Batch interval:", BATCH_START, BATCH_END)

# ============================================================
# 1. Verify immutable inputs and all previous batches
# ============================================================

for path, expected_hash in EXPECTED_HASHES.items():
    assert path.is_file(), f"Missing file: {path}"
    assert sha256_file(path) == expected_hash, (
        f"SHA256 mismatch: {path}"
    )

assert COHORT_PATH.is_file()
assert REFERENCE_PATH.is_file()
assert OUTPUT_DIR.is_dir()

assert not NPZ_PATH.exists(), (
    f"Output already exists: {NPZ_PATH}"
)
assert not CSV_PATH.exists(), (
    f"Output already exists: {CSV_PATH}"
)

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    plan = json.load(handle)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert plan["manifest_sha256"] == EXPECTED_HASHES[MANIFEST_PATH]
assert plan["image_count"] == 156
assert plan["source_risk_group_count"] == 134
assert plan["test_access_permitted"] is False
assert plan["randomization_seeds"] == [1042, 2042, 3042]

assert set(protocol["randomization"]["stages"]["R1"]) == {"head"}
assert SEED in protocol["randomization"]["randomization_seeds"]

print("\nFROZEN INPUT INTEGRITY")
print("Manifest SHA256: PASS")
print("Execution plan SHA256: PASS")
print("Randomization protocol SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Batch 001 NPZ/CSV SHA256: PASS")
print("Batch 002 NPZ/CSV SHA256: PASS")
print("Batch 003 NPZ/CSV SHA256: PASS")

# ============================================================
# 2. Resolve exactly 16 frozen validation images
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique
assert manifest["split"].eq("val").all()
assert manifest["image_id"].tolist() == sorted(
    manifest["image_id"].tolist()
)

batch = manifest.iloc[BATCH_START:BATCH_END].copy()

assert len(batch) == 16
assert batch["image_id"].is_unique
assert batch["true_grade"].isin([1, 2, 3, 4]).all()

image_ids = batch["image_id"].tolist()
grades = batch["true_grade"].astype(int).tolist()
risk_groups = batch["risk_group_id"].astype(str).tolist()

assert all(
    group.strip() not in {"", "nan", "None"}
    for group in risk_groups
)

previous_ids = []
previous_groups = []

for number, start, stop in [
    ("001", 0, 16),
    ("002", 16, 32),
    ("003", 32, 48)
]:
    previous_path = (
        OUTPUT_DIR /
        f"RETFound_Seed1042_R1_Batch{number}.npz"
    )

    with np.load(previous_path, allow_pickle=False) as archive:
        old_ids = list(map(str, archive["image_ids"]))
        old_groups = list(map(str, archive["risk_group_ids"]))

        assert old_ids == (
            manifest.iloc[start:stop]["image_id"].tolist()
        )

        assert old_groups == (
            manifest.iloc[start:stop]["risk_group_id"]
            .astype(str).tolist()
        )

        previous_ids.extend(old_ids)
        previous_groups.extend(old_groups)

assert len(previous_ids) == 48
assert len(set(previous_ids)) == 48
assert set(image_ids).isdisjoint(set(previous_ids))

overlapping_risk_groups = sorted(
    set(risk_groups) & set(previous_groups)
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert cohort["image_id"].is_unique

candidate_flags = (
    cohort.set_index("image_id")
    .loc[image_ids, "primary_ma_candidate"]
    .astype(str)
    .str.strip()
    .str.lower()
)

assert candidate_flags.isin(["true", "1"]).all()

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    reference_positions = {
        image_id: i
        for i, image_id in enumerate(reference_ids)
    }

    assert set(image_ids).issubset(reference_positions)

    reference_cams = np.stack([
        archive["retfound_cam"][
            reference_positions[image_id]
        ].astype(np.float32)
        for image_id in image_ids
    ])

assert reference_cams.shape == (16, 224, 224)
assert np.isfinite(reference_cams).all()

print("\nBATCH COHORT")
print("Images:", len(image_ids))
print("Distinct source-risk groups:", len(set(risk_groups)))
print("Image ID overlap with previous batches:", 0)
print(
    "Risk groups overlapping previous batches:",
    len(overlapping_risk_groups)
)
print("Validation-only selection: PASS")
print("Original CAM ID alignment: PASS")
print("First image ID:", image_ids[0])
print("Last image ID:", image_ids[-1])

if overlapping_risk_groups:
    print("Previously observed risk groups:")
    for group in overlapping_risk_groups:
        print(group)

# ============================================================
# 3. Decode only Batch 004 images
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensors = []

for _, row in batch.iterrows():
    image_path = Path(row["resolved_image_path"])

    assert image_path.is_file(), row["image_id"]

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    assert tensor.shape == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all().item()

    input_tensors.append(tensor)

print("Images decoded:", len(input_tensors))

# ============================================================
# 4. Load the original RETFound checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]

assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

loaded = model.load_state_dict(
    source_state, strict=True
)

assert not loaded.missing_keys
assert not loaded.unexpected_keys

model.eval()
model.requires_grad_(False)

# ============================================================
# 5. Reconstruct the validated Grad-CAM procedure
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(
    activation_hook
)


def calculate_cam(cpu_tensor, true_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(cpu_tensor.to(DEVICE))

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, true_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2), keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12

        cam /= float(cam.max())
        assert np.isfinite(cam).all()

    captured.clear()

    return output_logits, cam, raw_max


# ============================================================
# 6. Reproduce R0 and initialize R1 with seed 1042
# ============================================================

r0_logits = []
r0_cams = []
r1_logits = []
r1_cams = []
r1_raw_maxima = []

try:
    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, _ = calculate_cam(cpu_tensor, grade)

        assert cam is not None

        r0_logits.append(logits)
        r0_cams.append(cam)

    baseline_metrics = [
        compare_cams(cam, reference)
        for cam, reference in zip(r0_cams, reference_cams)
    ]

    baseline_pearsons = [
        entry["pearson"]
        for entry in baseline_metrics
    ]

    baseline_maes = [
        entry["mae"]
        for entry in baseline_metrics
    ]

    assert np.isfinite(baseline_pearsons).all()
    assert min(baseline_pearsons) > 0.999

    print("\nR0 REPRODUCTION")
    print(
        "Minimum reference CAM Pearson:",
        round(float(np.min(baseline_pearsons)), 6)
    )
    print(
        "Mean reference CAM MAE:",
        round(float(np.mean(baseline_maes)), 8)
    )
    print("Baseline CAM reproduction: PASS")

    # R1: Reset the RNG once and reinitialize only the head.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    changed_names = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, source_state[name]):
            changed_names.append(name)

    assert set(changed_names) == {
        "head.weight",
        "head.bias"
    }

    print("\nR1 PARAMETER INTEGRITY")
    print("Changed tensors:", len(changed_names))
    print("Expected changed tensors: 2")
    print("Unauthorized changes: 0")
    print("Seed:", SEED)

    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, raw_max = calculate_cam(
            cpu_tensor, grade
        )

        r1_logits.append(logits)
        r1_cams.append(cam)
        r1_raw_maxima.append(raw_max)

finally:
    hook.remove()
    captured.clear()

    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Build image-level results
# ============================================================

rows = []
stored_r1_cams = []

for i, image_id in enumerate(image_ids):
    metrics = compare_cams(
        r0_cams[i],
        r1_cams[i]
    )

    rows.append({
        "model": "RETFound",
        "seed": SEED,
        "stage": "R1",
        "batch": BATCH_NUMBER,
        "image_id": image_id,
        "true_grade": grades[i],
        "risk_group_id": risk_groups[i],
        "degenerate_cam": r1_cams[i] is None,
        "raw_max": r1_raw_maxima[i],
        "r0_r1_pearson": metrics["pearson"],
        "r0_r1_spearman": metrics["spearman"],
        "r0_r1_ssim": metrics["ssim"],
        "r0_r1_mae": metrics["mae"],
        "r0_r1_logit_l2": float(
            np.linalg.norm(
                r1_logits[i] - r0_logits[i]
            )
        )
    })

    stored_r1_cams.append(
        np.zeros((224, 224), dtype=np.float32)
        if r1_cams[i] is None
        else r1_cams[i].astype(np.float32)
    )

results = pd.DataFrame(rows)

r0_cam_array = np.stack(r0_cams).astype(np.float32)
r1_cam_array = np.stack(stored_r1_cams).astype(np.float32)
r0_logit_array = np.stack(r0_logits).astype(np.float32)
r1_logit_array = np.stack(r1_logits).astype(np.float32)

assert len(results) == 16
assert results["image_id"].is_unique
assert r0_cam_array.shape == (16, 224, 224)
assert r1_cam_array.shape == (16, 224, 224)
assert r0_logit_array.shape == (16, 5)
assert r1_logit_array.shape == (16, 5)

for values in [
    r0_cam_array,
    r1_cam_array,
    r0_logit_array,
    r1_logit_array
]:
    assert np.isfinite(values).all()

# ============================================================
# 8. Serialize data and validate in memory
# ============================================================

npz_buffer = io.BytesIO()

np.savez_compressed(
    npz_buffer,
    image_ids=np.asarray(image_ids, dtype=str),
    true_grades=np.asarray(grades, dtype=np.int64),
    risk_group_ids=np.asarray(risk_groups, dtype=str),
    model=np.asarray("RETFound"),
    seed=np.asarray(SEED),
    stage=np.asarray("R1"),
    r0_cam=r0_cam_array,
    r1_cam=r1_cam_array,
    r0_logits=r0_logit_array,
    r1_logits=r1_logit_array,
    r1_raw_max=np.asarray(r1_raw_maxima, dtype=np.float64),
    r1_degenerate=results["degenerate_cam"].to_numpy(dtype=bool),
    manifest_sha256=np.asarray(EXPECTED_HASHES[MANIFEST_PATH]),
    execution_plan_sha256=np.asarray(EXPECTED_HASHES[PLAN_PATH])
)

npz_bytes = npz_buffer.getvalue()

csv_bytes = results.to_csv(
    index=False,
    float_format="%.10g"
).encode("utf-8")

with np.load(
    io.BytesIO(npz_bytes),
    allow_pickle=False
) as archive:

    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)

    assert str(archive["model"].item()) == "RETFound"
    assert str(archive["stage"].item()) == "R1"
    assert int(archive["seed"].item()) == SEED

prepared_csv = pd.read_csv(
    io.BytesIO(csv_bytes),
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(prepared_csv) == 16
assert prepared_csv["image_id"].tolist() == image_ids
assert prepared_csv["seed"].eq(SEED).all()
assert prepared_csv["stage"].eq("R1").all()

print("\nIN-MEMORY ARCHIVE VERIFICATION")
print("NPZ serialization: PASS")
print("CSV serialization: PASS")

# ============================================================
# 9. Publish without overwriting existing research files
# ============================================================

assert not NPZ_PATH.exists()
assert not CSV_PATH.exists()

# Retain both byte buffers in memory for recovery if needed.
npz_hash = publish_exclusively(NPZ_PATH, npz_bytes)
csv_hash = publish_exclusively(CSV_PATH, csv_bytes)

# ============================================================
# 10. Independently verify saved files
# ============================================================

with np.load(NPZ_PATH, allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)

saved_csv = pd.read_csv(
    CSV_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(saved_csv) == 16
assert saved_csv["image_id"].tolist() == image_ids
assert saved_csv["seed"].eq(SEED).all()
assert saved_csv["stage"].eq("R1").all()
assert saved_csv["model"].eq("RETFound").all()

assert sha256_file(NPZ_PATH) == npz_hash
assert sha256_file(CSV_PATH) == csv_hash

# ============================================================
# 11. Final production report
# ============================================================

print("\nBATCH RESULTS")
print("Model: RETFound")
print("Seed:", SEED)
print("Stage: R1")
print("Batch:", BATCH_NUMBER)
print("Images evaluated:", len(saved_csv))
print(
    "Degenerate CAMs:",
    int(saved_csv["degenerate_cam"].sum())
)

for column in [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]:
    print(
        f"Mean {column}:",
        round(float(saved_csv[column].mean()), 6)
    )

print("\nPER-IMAGE RESULTS")
print(
    saved_csv[
        [
            "image_id",
            "true_grade",
            "risk_group_id",
            "degenerate_cam",
            "r0_r1_pearson",
            "r0_r1_spearman",
            "r0_r1_ssim",
            "r0_r1_mae"
        ]
    ].to_string(index=False)
)

print("\nARCHIVE INTEGRITY")
print("NPZ path:", NPZ_PATH)
print("NPZ SHA256:", npz_hash)
print("CSV path:", CSV_PATH)
print("CSV SHA256:", csv_hash)
print("Batch 001-003 SHA256 unchanged: PASS")
print("Batch 004 archive round-trip: PASS")
print("Existing files overwritten: False")

print("\nRESEARCH INTEGRITY")
print("Frozen input files modified: False")
print("Validation images decoded:", len(input_tensors))
print("Test images accessed: 0")
print("New batch files created: 2")

print("\nSTEP 10.40 EXECUTION COMPLETE")
print("Production batch 004: PENDING REVIEW")


# %% [cell 380] Experiment 10 - Step 10.41

# ============================================================
# Experiment 10 - Step 10.41
# RETFound R1 Production Batch 005
# Seed 1042 | Frozen validation manifest rows 64:80
# Exclusive publication | No test-set access
# ============================================================

from pathlib import Path
import gc
import hashlib
import io
import json

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

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"
OUTPUT_DIR = EXP10 / "full_validation_batches"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"
PLAN_PATH = EXP10 / "Experiment10_Full_Execution_Plan_V1.json"
PROTOCOL_PATH = EXP10 / "Experiment10_Randomization_Protocol_V1.json"
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"

CHECKPOINT_PATH = (
    ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"
)

REFERENCE_PATH = (
    ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"
)

SEED = 1042
BATCH_START = 64
BATCH_END = 80
BATCH_NUMBER = "005"

BATCH_NAME = f"RETFound_Seed{SEED}_R1_Batch{BATCH_NUMBER}"
NPZ_PATH = OUTPUT_DIR / f"{BATCH_NAME}.npz"
CSV_PATH = OUTPUT_DIR / f"{BATCH_NAME}.csv"

EXPECTED_HASHES = {
    MANIFEST_PATH:
        "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5",
    PLAN_PATH:
        "da52de5513de274a0b91c99f7060049a1f77db4c6dc2a788263607e6d8168590",
    PROTOCOL_PATH:
        "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda",
    INDEX_PATH:
        "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    CHECKPOINT_PATH:
        "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8",
}

PREVIOUS_HASHES = {
    "001": (
        "4f71b19be85c52bcdce8f0d628fbc34d0171f9b91170204fa1c0b4d6b6a28c43",
        "529f23beed5a47c67b0a545ce32330ddcec1973ee93eb2d8ea315c88ab7d0918"
    ),
    "002": (
        "946cc0bd797bf43f8d857cd5618ab4976aefbcbad96e0d6a2606365a30c1deab",
        "ca444fde5920a09f5fc64479d09d0919945d741cdeeb24ffac5ca8341f88089a"
    ),
    "003": (
        "ebb83b4f68c13e6bcb2f079290c7846707ff3e6f76cdacfa264a38beab9ff9c6",
        "4a53c0c75cb040d8309413d934599f11515cd10e1498d636735f8cf87d60772f"
    ),
    "004": (
        "5a64e2f7ab7c515c1acdcbbc8bfaa4aaa509344a0ad2f99a8e1430b7dc176e62",
        "8d176e808a8979c750f7b060d0070b60e4d8cf7d956ed8fd2bf2cc63b54dc717"
    )
}

DEVICE = torch.device("cuda")
assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - a.mean()
    bc = b - b.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12
        else np.nan
    )

    return {
        "pearson": pearson,
        "spearman": float(spearmanr(a, b).statistic),
        "ssim": float(
            structural_similarity(
                first, second, data_range=1.0
            )
        ),
        "mae": float(np.mean(np.abs(a - b)))
    }


def publish_exclusively(path, payload):
    if path.exists():
        raise FileExistsError(
            f"Existing output requires review: {path}"
        )

    expected_hash = hashlib.sha256(payload).hexdigest()

    # Google Drive supports exclusive file creation.
    # The SHA256 check verifies complete publication.
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()

    actual_hash = sha256_file(path)

    assert actual_hash == expected_hash, (
        f"Post-publication SHA256 mismatch: {path}"
    )

    return actual_hash


print("EXPERIMENT 10.41 - RETFOUND R1 PRODUCTION BATCH 005")
print("Seed:", SEED)
print("Batch interval:", BATCH_START, BATCH_END)

# ============================================================
# 1. Verify frozen inputs and Batch 001-004
# ============================================================

for path, expected_hash in EXPECTED_HASHES.items():
    assert path.is_file(), f"Missing input: {path}"
    assert sha256_file(path) == expected_hash, (
        f"SHA256 mismatch: {path}"
    )

for number, (npz_expected, csv_expected) in PREVIOUS_HASHES.items():
    previous_npz = (
        OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.npz"
    )
    previous_csv = (
        OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.csv"
    )

    assert previous_npz.is_file()
    assert previous_csv.is_file()

    assert sha256_file(previous_npz) == npz_expected
    assert sha256_file(previous_csv) == csv_expected

assert COHORT_PATH.is_file()
assert REFERENCE_PATH.is_file()
assert OUTPUT_DIR.is_dir()

assert not NPZ_PATH.exists(), f"Output already exists: {NPZ_PATH}"
assert not CSV_PATH.exists(), f"Output already exists: {CSV_PATH}"

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    plan = json.load(handle)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert plan["manifest_sha256"] == EXPECTED_HASHES[MANIFEST_PATH]
assert plan["image_count"] == 156
assert plan["source_risk_group_count"] == 134
assert plan["test_access_permitted"] is False
assert plan["randomization_seeds"] == [1042, 2042, 3042]

assert SEED in protocol["randomization"]["randomization_seeds"]
assert set(protocol["randomization"]["stages"]["R1"]) == {"head"}

print("\nFROZEN INPUT INTEGRITY")
print("Manifest SHA256: PASS")
print("Execution plan SHA256: PASS")
print("Randomization protocol SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Batch 001-004 NPZ/CSV SHA256: PASS")

# ============================================================
# 2. Select and validate Batch 005
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique
assert manifest["split"].eq("val").all()
assert manifest["image_id"].tolist() == sorted(
    manifest["image_id"].tolist()
)

batch = manifest.iloc[BATCH_START:BATCH_END].copy()

assert len(batch) == 16
assert batch["image_id"].is_unique
assert batch["true_grade"].isin([1, 2, 3, 4]).all()

image_ids = batch["image_id"].tolist()
grades = batch["true_grade"].astype(int).tolist()
risk_groups = batch["risk_group_id"].astype(str).tolist()

assert all(
    value.strip() not in {"", "nan", "None"}
    for value in risk_groups
)

previous_ids = []
previous_groups = []

for index_number, number in enumerate(PREVIOUS_HASHES, start=0):
    path = (
        OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.npz"
    )

    with np.load(path, allow_pickle=False) as archive:
        ids = list(map(str, archive["image_ids"]))
        groups = list(map(str, archive["risk_group_ids"]))

        expected_rows = manifest.iloc[
            index_number * 16:(index_number + 1) * 16
        ]

        assert ids == expected_rows["image_id"].tolist()
        assert groups == expected_rows["risk_group_id"].tolist()

        previous_ids.extend(ids)
        previous_groups.extend(groups)

assert len(previous_ids) == 64
assert len(set(previous_ids)) == 64
assert set(image_ids).isdisjoint(set(previous_ids))

overlapping_groups = sorted(
    set(risk_groups) & set(previous_groups)
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert cohort["image_id"].is_unique

candidate_flags = (
    cohort.set_index("image_id")
    .loc[image_ids, "primary_ma_candidate"]
    .astype(str)
    .str.strip()
    .str.lower()
)

assert candidate_flags.isin(["true", "1"]).all()

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: i for i, image_id in enumerate(reference_ids)
    }

    assert set(image_ids).issubset(positions)

    reference_cams = np.stack([
        archive["retfound_cam"][positions[image_id]]
        .astype(np.float32)
        for image_id in image_ids
    ])

assert reference_cams.shape == (16, 224, 224)
assert np.isfinite(reference_cams).all()

print("\nBATCH COHORT")
print("Images:", len(image_ids))
print("Distinct source-risk groups:", len(set(risk_groups)))
print("Image ID overlap with previous batches:", 0)
print("Risk groups overlapping previous batches:", len(overlapping_groups))
print("Validation-only selection: PASS")
print("Original CAM ID alignment: PASS")
print("First image ID:", image_ids[0])
print("Last image ID:", image_ids[-1])

if overlapping_groups:
    print("Previously observed risk groups:")
    for group in overlapping_groups:
        print(group)

# ============================================================
# 3. Decode only 16 validation images
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensors = []

for _, row in batch.iterrows():
    image_path = Path(row["resolved_image_path"])
    assert image_path.is_file(), row["image_id"]

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    assert tensor.shape == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all().item()

    input_tensors.append(tensor)

print("Images decoded:", len(input_tensors))

# ============================================================
# 4. Load the original RETFound checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)
loaded = model.load_state_dict(source_state, strict=True)

assert not loaded.missing_keys
assert not loaded.unexpected_keys

model.eval()
model.requires_grad_(False)

# ============================================================
# 5. Use validated Grad-CAM implementation
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(
    activation_hook
)


def calculate_cam(cpu_tensor, true_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(cpu_tensor.to(DEVICE))

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, true_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :].detach().float()
        .transpose(0, 1).reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :].detach().float()
        .transpose(0, 1).reshape(1024, 14, 14)
    )

    weights = gradients.mean(dim=(1, 2), keepdim=True)

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12

        cam /= float(cam.max())
        assert np.isfinite(cam).all()

    captured.clear()
    return output_logits, cam, raw_max


# ============================================================
# 6. Evaluate R0 and R1
# ============================================================

r0_logits = []
r0_cams = []
r1_logits = []
r1_cams = []
r1_raw_maxima = []

try:
    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, _ = calculate_cam(cpu_tensor, grade)

        assert cam is not None

        r0_logits.append(logits)
        r0_cams.append(cam)

    baseline_metrics = [
        compare_cams(cam, reference)
        for cam, reference in zip(r0_cams, reference_cams)
    ]

    baseline_pearsons = [
        metric["pearson"] for metric in baseline_metrics
    ]

    baseline_maes = [
        metric["mae"] for metric in baseline_metrics
    ]

    assert np.isfinite(baseline_pearsons).all()
    assert min(baseline_pearsons) > 0.999

    print("\nR0 REPRODUCTION")
    print(
        "Minimum reference CAM Pearson:",
        round(float(min(baseline_pearsons)), 6)
    )
    print(
        "Mean reference CAM MAE:",
        round(float(np.mean(baseline_maes)), 8)
    )
    print("Baseline CAM reproduction: PASS")

    # Reset RNG exactly once before R1 initialization.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    changed_names = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, source_state[name]):
            changed_names.append(name)

    assert set(changed_names) == {
        "head.weight",
        "head.bias"
    }

    print("\nR1 PARAMETER INTEGRITY")
    print("Changed tensors:", len(changed_names))
    print("Expected changed tensors: 2")
    print("Unauthorized changes: 0")
    print("Seed:", SEED)

    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, raw_max = calculate_cam(
            cpu_tensor, grade
        )

        r1_logits.append(logits)
        r1_cams.append(cam)
        r1_raw_maxima.append(raw_max)

finally:
    hook.remove()
    captured.clear()

    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Build image-level results
# ============================================================

rows = []
stored_r1_cams = []

for i, image_id in enumerate(image_ids):
    metrics = compare_cams(r0_cams[i], r1_cams[i])

    rows.append({
        "model": "RETFound",
        "seed": SEED,
        "stage": "R1",
        "batch": BATCH_NUMBER,
        "image_id": image_id,
        "true_grade": grades[i],
        "risk_group_id": risk_groups[i],
        "degenerate_cam": r1_cams[i] is None,
        "raw_max": r1_raw_maxima[i],
        "r0_r1_pearson": metrics["pearson"],
        "r0_r1_spearman": metrics["spearman"],
        "r0_r1_ssim": metrics["ssim"],
        "r0_r1_mae": metrics["mae"],
        "r0_r1_logit_l2": float(
            np.linalg.norm(r1_logits[i] - r0_logits[i])
        )
    })

    stored_r1_cams.append(
        np.zeros((224, 224), dtype=np.float32)
        if r1_cams[i] is None
        else r1_cams[i].astype(np.float32)
    )

results = pd.DataFrame(rows)

r0_cam_array = np.stack(r0_cams).astype(np.float32)
r1_cam_array = np.stack(stored_r1_cams).astype(np.float32)
r0_logit_array = np.stack(r0_logits).astype(np.float32)
r1_logit_array = np.stack(r1_logits).astype(np.float32)

assert len(results) == 16
assert results["image_id"].is_unique
assert r0_cam_array.shape == (16, 224, 224)
assert r1_cam_array.shape == (16, 224, 224)
assert r0_logit_array.shape == (16, 5)
assert r1_logit_array.shape == (16, 5)

for array in [
    r0_cam_array,
    r1_cam_array,
    r0_logit_array,
    r1_logit_array
]:
    assert np.isfinite(array).all()

# ============================================================
# 8. Serialize and verify both archives in memory
# ============================================================

npz_buffer = io.BytesIO()

np.savez_compressed(
    npz_buffer,
    image_ids=np.asarray(image_ids, dtype=str),
    true_grades=np.asarray(grades, dtype=np.int64),
    risk_group_ids=np.asarray(risk_groups, dtype=str),
    model=np.asarray("RETFound"),
    seed=np.asarray(SEED),
    stage=np.asarray("R1"),
    r0_cam=r0_cam_array,
    r1_cam=r1_cam_array,
    r0_logits=r0_logit_array,
    r1_logits=r1_logit_array,
    r1_raw_max=np.asarray(r1_raw_maxima, dtype=np.float64),
    r1_degenerate=results["degenerate_cam"].to_numpy(dtype=bool),
    manifest_sha256=np.asarray(EXPECTED_HASHES[MANIFEST_PATH]),
    execution_plan_sha256=np.asarray(EXPECTED_HASHES[PLAN_PATH])
)

npz_bytes = npz_buffer.getvalue()

csv_bytes = results.to_csv(
    index=False,
    float_format="%.10g"
).encode("utf-8")

with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)
    assert str(archive["model"].item()) == "RETFound"
    assert str(archive["stage"].item()) == "R1"
    assert int(archive["seed"].item()) == SEED

prepared_csv = pd.read_csv(
    io.BytesIO(csv_bytes),
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(prepared_csv) == 16
assert prepared_csv["image_id"].tolist() == image_ids
assert prepared_csv["seed"].eq(SEED).all()
assert prepared_csv["stage"].eq("R1").all()
assert prepared_csv["model"].eq("RETFound").all()

print("\nIN-MEMORY ARCHIVE VERIFICATION")
print("NPZ serialization: PASS")
print("CSV serialization: PASS")

# ============================================================
# 9. Publish both new files exclusively
# ============================================================

assert OUTPUT_DIR.is_dir()
assert not NPZ_PATH.exists()
assert not CSV_PATH.exists()

# Keep both byte buffers available for recovery if needed.
npz_hash = publish_exclusively(NPZ_PATH, npz_bytes)
csv_hash = publish_exclusively(CSV_PATH, csv_bytes)

# ============================================================
# 10. Independently verify the published files
# ============================================================

with np.load(NPZ_PATH, allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)

saved_csv = pd.read_csv(
    CSV_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(saved_csv) == 16
assert saved_csv["image_id"].tolist() == image_ids
assert saved_csv["seed"].eq(SEED).all()
assert saved_csv["stage"].eq("R1").all()
assert saved_csv["model"].eq("RETFound").all()

assert sha256_file(NPZ_PATH) == npz_hash
assert sha256_file(CSV_PATH) == csv_hash

# ============================================================
# 11. Final production report
# ============================================================

print("\nBATCH RESULTS")
print("Model: RETFound")
print("Seed:", SEED)
print("Stage: R1")
print("Batch:", BATCH_NUMBER)
print("Images evaluated:", len(saved_csv))

print(
    "Degenerate CAMs:",
    int(saved_csv["degenerate_cam"].sum())
)

for column in [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]:
    print(
        f"Mean {column}:",
        round(float(saved_csv[column].mean()), 6)
    )

print("\nPER-IMAGE RESULTS")
print(
    saved_csv[
        [
            "image_id",
            "true_grade",
            "risk_group_id",
            "degenerate_cam",
            "r0_r1_pearson",
            "r0_r1_spearman",
            "r0_r1_ssim",
            "r0_r1_mae"
        ]
    ].to_string(index=False)
)

print("\nARCHIVE INTEGRITY")
print("NPZ path:", NPZ_PATH)
print("NPZ SHA256:", npz_hash)
print("CSV path:", CSV_PATH)
print("CSV SHA256:", csv_hash)
print("Batch 001-004 hashes unchanged: PASS")
print("Batch 005 archive round-trip: PASS")
print("Existing files overwritten: False")

print("\nRESEARCH INTEGRITY")
print("Frozen input files modified: False")
print("Validation images decoded:", len(input_tensors))
print("Test images accessed: 0")
print("New batch files created: 2")

print("\nSTEP 10.41 EXECUTION COMPLETE")
print("Production batch 005: PENDING REVIEW")


# %% [cell 381] Experiment 10 - Step 10.42

# ============================================================
# Experiment 10 - Step 10.42
# RETFound R1 Production Batch 006
# Seed 1042 | Frozen validation manifest rows 80:96
# Read-only source artifacts | Exclusive archive publication
# ============================================================

from pathlib import Path
import gc
import hashlib
import io
import json

import cv2
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from timm.layers import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"
OUTPUT_DIR = EXP10 / "full_validation_batches"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"
PLAN_PATH = EXP10 / "Experiment10_Full_Execution_Plan_V1.json"
PROTOCOL_PATH = EXP10 / "Experiment10_Randomization_Protocol_V1.json"
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
CHECKPOINT_PATH = ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"
REFERENCE_PATH = ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"

SEED = 1042
BATCH_START = 80
BATCH_END = 96
BATCH_NUMBER = "006"
BATCH_NAME = f"RETFound_Seed{SEED}_R1_Batch{BATCH_NUMBER}"

NPZ_PATH = OUTPUT_DIR / f"{BATCH_NAME}.npz"
CSV_PATH = OUTPUT_DIR / f"{BATCH_NAME}.csv"

EXPECTED_HASHES = {
    MANIFEST_PATH:
        "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5",
    PLAN_PATH:
        "da52de5513de274a0b91c99f7060049a1f77db4c6dc2a788263607e6d8168590",
    PROTOCOL_PATH:
        "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda",
    INDEX_PATH:
        "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    CHECKPOINT_PATH:
        "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
}

PREVIOUS_HASHES = {
    "001": (
        "4f71b19be85c52bcdce8f0d628fbc34d0171f9b91170204fa1c0b4d6b6a28c43",
        "529f23beed5a47c67b0a545ce32330ddcec1973ee93eb2d8ea315c88ab7d0918"
    ),
    "002": (
        "946cc0bd797bf43f8d857cd5618ab4976aefbcbad96e0d6a2606365a30c1deab",
        "ca444fde5920a09f5fc64479d09d0919945d741cdeeb24ffac5ca8341f88089a"
    ),
    "003": (
        "ebb83b4f68c13e6bcb2f079290c7846707ff3e6f76cdacfa264a38beab9ff9c6",
        "4a53c0c75cb040d8309413d934599f11515cd10e1498d636735f8cf87d60772f"
    ),
    "004": (
        "5a64e2f7ab7c515c1acdcbbc8bfaa4aaa509344a0ad2f99a8e1430b7dc176e62",
        "8d176e808a8979c750f7b060d0070b60e4d8cf7d956ed8fd2bf2cc63b54dc717"
    ),
    "005": (
        "f21a22e43dce2863ca64fe80066bc81e3a1368982b1076b602fa79506bd64a1b",
        "e1313fce88897aa043c5d5a2a0cba4b9445c408e4cea4bc5114e1393b27ddfe5"
    )
}

DEVICE = torch.device("cuda")
assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - a.mean()
    bc = b - b.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12 else np.nan
    )

    return {
        "pearson": pearson,
        "spearman": float(spearmanr(a, b).statistic),
        "ssim": float(
            structural_similarity(
                first, second, data_range=1.0
            )
        ),
        "mae": float(np.mean(np.abs(a - b)))
    }


def publish_exclusively(path, payload):
    if path.exists():
        raise FileExistsError(
            f"Existing output requires review: {path}"
        )

    expected_hash = hashlib.sha256(payload).hexdigest()

    # Drive-compatible exclusive creation; no overwrite.
    # If interrupted, inspect the file before any retry.
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()

    actual_hash = sha256_file(path)
    assert actual_hash == expected_hash, (
        f"Post-write SHA256 mismatch: {path}"
    )

    return actual_hash


print("EXPERIMENT 10.42 - RETFOUND R1 PRODUCTION BATCH 006")
print("Seed:", SEED)
print("Batch interval:", BATCH_START, BATCH_END)

# ============================================================
# 1. Verify frozen sources and five archived batches
# ============================================================

for path, expected in EXPECTED_HASHES.items():
    assert path.is_file(), f"Missing input: {path}"
    assert sha256_file(path) == expected, (
        f"SHA256 mismatch: {path}"
    )

for number, (npz_expected, csv_expected) in PREVIOUS_HASHES.items():
    old_npz = OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.npz"
    old_csv = OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.csv"

    assert old_npz.is_file() and old_csv.is_file()
    assert sha256_file(old_npz) == npz_expected
    assert sha256_file(old_csv) == csv_expected

assert COHORT_PATH.is_file()
assert REFERENCE_PATH.is_file()
assert OUTPUT_DIR.is_dir()
assert not NPZ_PATH.exists()
assert not CSV_PATH.exists()

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    plan = json.load(handle)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert plan["manifest_sha256"] == EXPECTED_HASHES[MANIFEST_PATH]
assert plan["image_count"] == 156
assert plan["source_risk_group_count"] == 134
assert plan["test_access_permitted"] is False
assert plan["randomization_seeds"] == [1042, 2042, 3042]

assert SEED in protocol["randomization"]["randomization_seeds"]
assert set(protocol["randomization"]["stages"]["R1"]) == {"head"}

print("\nFROZEN INPUT INTEGRITY")
print("Manifest SHA256: PASS")
print("Execution plan SHA256: PASS")
print("Randomization protocol SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Batch 001-005 NPZ/CSV SHA256: PASS")

# ============================================================
# 2. Validate the next 16-image cohort
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique
assert manifest["split"].eq("val").all()
assert manifest["image_id"].tolist() == sorted(
    manifest["image_id"].tolist()
)

batch = manifest.iloc[BATCH_START:BATCH_END].copy()

assert len(batch) == 16
assert batch["image_id"].is_unique
assert batch["true_grade"].isin([1, 2, 3, 4]).all()

image_ids = batch["image_id"].tolist()
grades = batch["true_grade"].astype(int).tolist()
risk_groups = batch["risk_group_id"].astype(str).tolist()

assert all(
    group.strip() not in {"", "nan", "None"}
    for group in risk_groups
)

previous_ids = []
previous_groups = []

for index_number, number in enumerate(PREVIOUS_HASHES):
    old_npz = (
        OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.npz"
    )

    with np.load(old_npz, allow_pickle=False) as archive:
        ids = list(map(str, archive["image_ids"]))
        groups = list(map(str, archive["risk_group_ids"]))

        previous_slice = manifest.iloc[
            index_number * 16:(index_number + 1) * 16
        ]

        assert ids == previous_slice["image_id"].tolist()
        assert groups == previous_slice["risk_group_id"].tolist()

        previous_ids.extend(ids)
        previous_groups.extend(groups)

assert len(previous_ids) == 80
assert len(set(previous_ids)) == 80
assert set(image_ids).isdisjoint(previous_ids)

overlapping_groups = sorted(
    set(risk_groups) & set(previous_groups)
)

cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})
assert cohort["image_id"].is_unique

candidate_flags = (
    cohort.set_index("image_id")
    .loc[image_ids, "primary_ma_candidate"]
    .astype(str).str.strip().str.lower()
)

assert candidate_flags.isin(["true", "1"]).all()

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: i for i, image_id in enumerate(reference_ids)
    }

    assert set(image_ids).issubset(positions)

    reference_cams = np.stack([
        archive["retfound_cam"][positions[image_id]]
        .astype(np.float32)
        for image_id in image_ids
    ])

assert reference_cams.shape == (16, 224, 224)
assert np.isfinite(reference_cams).all()

print("\nBATCH COHORT")
print("Images:", len(image_ids))
print("Distinct source-risk groups:", len(set(risk_groups)))
print("Image ID overlap with previous batches: 0")
print("Risk groups overlapping previous batches:", len(overlapping_groups))
print("Validation-only selection: PASS")
print("Original CAM ID alignment: PASS")
print("First image ID:", image_ids[0])
print("Last image ID:", image_ids[-1])

if overlapping_groups:
    print("Previously observed risk groups:")
    for group in overlapping_groups:
        print(group)

# ============================================================
# 3. Decode only Batch 006 validation images
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensors = []

for _, row in batch.iterrows():
    image_path = Path(row["resolved_image_path"])
    assert image_path.is_file(), row["image_id"]

    with Image.open(image_path) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0)

    assert tensor.shape == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all().item()
    input_tensors.append(tensor)

print("Images decoded:", len(input_tensors))

# ============================================================
# 4. Load frozen RETFound checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)
loaded = model.load_state_dict(source_state, strict=True)

assert not loaded.missing_keys
assert not loaded.unexpected_keys

model.eval()
model.requires_grad_(False)

# ============================================================
# 5. Reconstruct frozen Grad-CAM
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(activation_hook)


def calculate_cam(cpu_tensor, true_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(cpu_tensor.to(DEVICE))

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, true_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :].detach().float()
        .transpose(0, 1).reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :].detach().float()
        .transpose(0, 1).reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2), keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()
    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12
        cam /= float(cam.max())
        assert np.isfinite(cam).all()

    captured.clear()
    return output_logits, cam, raw_max


# ============================================================
# 6. Reproduce R0, then initialize and evaluate R1
# ============================================================

r0_logits = []
r0_cams = []
r1_logits = []
r1_cams = []
r1_raw_maxima = []

try:
    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, _ = calculate_cam(cpu_tensor, grade)

        assert cam is not None
        r0_logits.append(logits)
        r0_cams.append(cam)

    baseline_metrics = [
        compare_cams(cam, reference)
        for cam, reference in zip(r0_cams, reference_cams)
    ]

    baseline_pearsons = [
        metric["pearson"] for metric in baseline_metrics
    ]

    baseline_maes = [
        metric["mae"] for metric in baseline_metrics
    ]

    assert np.isfinite(baseline_pearsons).all()
    assert min(baseline_pearsons) > 0.999

    print("\nR0 REPRODUCTION")
    print(
        "Minimum reference CAM Pearson:",
        round(float(min(baseline_pearsons)), 6)
    )
    print(
        "Mean reference CAM MAE:",
        round(float(np.mean(baseline_maes)), 8)
    )
    print("Baseline CAM reproduction: PASS")

    # Apply the same R1 initialization for every production batch.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        torch.nn.init.zeros_(model.head.bias)

    changed_names = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()
        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, source_state[name]):
            changed_names.append(name)

    assert set(changed_names) == {
        "head.weight", "head.bias"
    }

    print("\nR1 PARAMETER INTEGRITY")
    print("Changed tensors:", len(changed_names))
    print("Expected changed tensors: 2")
    print("Unauthorized changes: 0")
    print("Seed:", SEED)

    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, raw_max = calculate_cam(cpu_tensor, grade)
        r1_logits.append(logits)
        r1_cams.append(cam)
        r1_raw_maxima.append(raw_max)

finally:
    hook.remove()
    captured.clear()
    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Assemble image-level results
# ============================================================

rows = []
stored_r1_cams = []

for i, image_id in enumerate(image_ids):
    metrics = compare_cams(r0_cams[i], r1_cams[i])

    rows.append({
        "model": "RETFound",
        "seed": SEED,
        "stage": "R1",
        "batch": BATCH_NUMBER,
        "image_id": image_id,
        "true_grade": grades[i],
        "risk_group_id": risk_groups[i],
        "degenerate_cam": r1_cams[i] is None,
        "raw_max": r1_raw_maxima[i],
        "r0_r1_pearson": metrics["pearson"],
        "r0_r1_spearman": metrics["spearman"],
        "r0_r1_ssim": metrics["ssim"],
        "r0_r1_mae": metrics["mae"],
        "r0_r1_logit_l2": float(
            np.linalg.norm(r1_logits[i] - r0_logits[i])
        )
    })

    stored_r1_cams.append(
        np.zeros((224, 224), dtype=np.float32)
        if r1_cams[i] is None
        else r1_cams[i].astype(np.float32)
    )

results = pd.DataFrame(rows)

r0_cam_array = np.stack(r0_cams).astype(np.float32)
r1_cam_array = np.stack(stored_r1_cams).astype(np.float32)
r0_logit_array = np.stack(r0_logits).astype(np.float32)
r1_logit_array = np.stack(r1_logits).astype(np.float32)

assert len(results) == 16
assert results["image_id"].is_unique
assert r0_cam_array.shape == (16, 224, 224)
assert r1_cam_array.shape == (16, 224, 224)
assert r0_logit_array.shape == (16, 5)
assert r1_logit_array.shape == (16, 5)

for array in [
    r0_cam_array, r1_cam_array,
    r0_logit_array, r1_logit_array
]:
    assert np.isfinite(array).all()

# ============================================================
# 8. Serialize and verify the archives in memory
# ============================================================

npz_buffer = io.BytesIO()

np.savez_compressed(
    npz_buffer,
    image_ids=np.asarray(image_ids, dtype=str),
    true_grades=np.asarray(grades, dtype=np.int64),
    risk_group_ids=np.asarray(risk_groups, dtype=str),
    model=np.asarray("RETFound"),
    seed=np.asarray(SEED),
    stage=np.asarray("R1"),
    r0_cam=r0_cam_array,
    r1_cam=r1_cam_array,
    r0_logits=r0_logit_array,
    r1_logits=r1_logit_array,
    r1_raw_max=np.asarray(r1_raw_maxima, dtype=np.float64),
    r1_degenerate=results["degenerate_cam"].to_numpy(dtype=bool),
    manifest_sha256=np.asarray(EXPECTED_HASHES[MANIFEST_PATH]),
    execution_plan_sha256=np.asarray(EXPECTED_HASHES[PLAN_PATH])
)

npz_bytes = npz_buffer.getvalue()

csv_bytes = results.to_csv(
    index=False,
    float_format="%.10g"
).encode("utf-8")

with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)
    assert str(archive["model"].item()) == "RETFound"
    assert str(archive["stage"].item()) == "R1"
    assert int(archive["seed"].item()) == SEED

prepared_csv = pd.read_csv(
    io.BytesIO(csv_bytes),
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(prepared_csv) == 16
assert prepared_csv["image_id"].tolist() == image_ids
assert prepared_csv["seed"].eq(SEED).all()
assert prepared_csv["stage"].eq("R1").all()
assert prepared_csv["model"].eq("RETFound").all()

print("\nIN-MEMORY ARCHIVE VERIFICATION")
print("NPZ serialization: PASS")
print("CSV serialization: PASS")

# ============================================================
# 9. Publish new NPZ and CSV without overwriting
# ============================================================

assert OUTPUT_DIR.is_dir()
assert not NPZ_PATH.exists()
assert not CSV_PATH.exists()

# Both byte buffers remain available for recovery if needed.
npz_hash = publish_exclusively(NPZ_PATH, npz_bytes)
csv_hash = publish_exclusively(CSV_PATH, csv_bytes)

# ============================================================
# 10. Independently reload and validate saved files
# ============================================================

with np.load(NPZ_PATH, allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)
    assert int(archive["seed"].item()) == SEED
    assert str(archive["stage"].item()) == "R1"

saved_csv = pd.read_csv(
    CSV_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(saved_csv) == 16
assert saved_csv["image_id"].tolist() == image_ids
assert saved_csv["seed"].eq(SEED).all()
assert saved_csv["stage"].eq("R1").all()
assert saved_csv["model"].eq("RETFound").all()

assert sha256_file(NPZ_PATH) == npz_hash
assert sha256_file(CSV_PATH) == csv_hash

# ============================================================
# 11. Final report
# ============================================================

print("\nBATCH RESULTS")
print("Model: RETFound")
print("Seed:", SEED)
print("Stage: R1")
print("Batch:", BATCH_NUMBER)
print("Images evaluated:", len(saved_csv))
print(
    "Degenerate CAMs:",
    int(saved_csv["degenerate_cam"].sum())
)

for column in [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]:
    print(
        f"Mean {column}:",
        round(float(saved_csv[column].mean()), 6)
    )

print("\nPER-IMAGE RESULTS")
print(
    saved_csv[
        [
            "image_id", "true_grade", "risk_group_id",
            "degenerate_cam", "r0_r1_pearson",
            "r0_r1_spearman", "r0_r1_ssim", "r0_r1_mae"
        ]
    ].to_string(index=False)
)

print("\nARCHIVE INTEGRITY")
print("NPZ path:", NPZ_PATH)
print("NPZ SHA256:", npz_hash)
print("CSV path:", CSV_PATH)
print("CSV SHA256:", csv_hash)
print("Batch 001-005 hashes unchanged: PASS")
print("Batch 006 archive round-trip: PASS")
print("Existing files overwritten: False")

print("\nRESEARCH INTEGRITY")
print("Frozen input files modified: False")
print("Validation images decoded:", len(input_tensors))
print("Test images accessed: 0")
print("New batch files created: 2")

print("\nSTEP 10.42 EXECUTION COMPLETE")
print("Production batch 006: PENDING REVIEW")


# %% [cell 382] Experiment 10 - Step 10.43

# ============================================================
# Experiment 10 - Step 10.43
# RETFound R1 Production Batch 007
# Seed 1042 | Validation manifest rows 96:112
# Read-only inputs | No test-set access
# ============================================================

from pathlib import Path
import gc
import hashlib
import io
import json

import cv2
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from timm.layers import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"
OUTPUT_DIR = EXP10 / "full_validation_batches"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"
PLAN_PATH = EXP10 / "Experiment10_Full_Execution_Plan_V1.json"
PROTOCOL_PATH = EXP10 / "Experiment10_Randomization_Protocol_V1.json"
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
CHECKPOINT_PATH = ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"
REFERENCE_PATH = ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"

SEED = 1042
BATCH_START = 96
BATCH_END = 112
BATCH_NUMBER = "007"
BATCH_NAME = f"RETFound_Seed{SEED}_R1_Batch{BATCH_NUMBER}"

NPZ_PATH = OUTPUT_DIR / f"{BATCH_NAME}.npz"
CSV_PATH = OUTPUT_DIR / f"{BATCH_NAME}.csv"

EXPECTED_HASHES = {
    MANIFEST_PATH: "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5",
    PLAN_PATH: "da52de5513de274a0b91c99f7060049a1f77db4c6dc2a788263607e6d8168590",
    PROTOCOL_PATH: "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda",
    INDEX_PATH: "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    CHECKPOINT_PATH: "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
}

PREVIOUS_HASHES = {
    "001": (
        "4f71b19be85c52bcdce8f0d628fbc34d0171f9b91170204fa1c0b4d6b6a28c43",
        "529f23beed5a47c67b0a545ce32330ddcec1973ee93eb2d8ea315c88ab7d0918"
    ),
    "002": (
        "946cc0bd797bf43f8d857cd5618ab4976aefbcbad96e0d6a2606365a30c1deab",
        "ca444fde5920a09f5fc64479d09d0919945d741cdeeb24ffac5ca8341f88089a"
    ),
    "003": (
        "ebb83b4f68c13e6bcb2f079290c7846707ff3e6f76cdacfa264a38beab9ff9c6",
        "4a53c0c75cb040d8309413d934599f11515cd10e1498d636735f8cf87d60772f"
    ),
    "004": (
        "5a64e2f7ab7c515c1acdcbbc8bfaa4aaa509344a0ad2f99a8e1430b7dc176e62",
        "8d176e808a8979c750f7b060d0070b60e4d8cf7d956ed8fd2bf2cc63b54dc717"
    ),
    "005": (
        "f21a22e43dce2863ca64fe80066bc81e3a1368982b1076b602fa79506bd64a1b",
        "e1313fce88897aa043c5d5a2a0cba4b9445c408e4cea4bc5114e1393b27ddfe5"
    ),
    "006": (
        "341661bb0f5eb18980296e8e0e068b7c8ca25f25bbcdd9f2e21162b4681e61e6",
        "f492ec4f5f2bbe198ba4dddea640bb1b9fac3d5f68ecefed2dee44e9046619f5"
    )
}

DEVICE = torch.device("cuda")
assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - a.mean()
    bc = b - b.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12
        else np.nan
    )

    return {
        "pearson": pearson,
        "spearman": float(spearmanr(a, b).statistic),
        "ssim": float(
            structural_similarity(first, second, data_range=1.0)
        ),
        "mae": float(np.mean(np.abs(a - b)))
    }


def publish_exclusively(path, payload):
    if path.exists():
        raise FileExistsError(
            f"Existing output requires review: {path}"
        )

    expected_hash = hashlib.sha256(payload).hexdigest()

    # Exclusive creation prevents overwriting prior results.
    # Publication is accepted only after SHA256 verification.
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()

    actual_hash = sha256_file(path)
    assert actual_hash == expected_hash, (
        f"Post-write SHA256 mismatch: {path}"
    )

    return actual_hash


print("EXPERIMENT 10.43 - RETFOUND R1 PRODUCTION BATCH 007")
print("Seed:", SEED)
print("Batch interval:", BATCH_START, BATCH_END)

# ============================================================
# 1. Verify frozen inputs and all previous batch hashes
# ============================================================

for path, expected_hash in EXPECTED_HASHES.items():
    assert path.is_file(), f"Missing input: {path}"
    assert sha256_file(path) == expected_hash, (
        f"SHA256 mismatch: {path}"
    )

for number, (expected_npz, expected_csv) in PREVIOUS_HASHES.items():
    old_npz = OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.npz"
    old_csv = OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.csv"

    assert old_npz.is_file() and old_csv.is_file()
    assert sha256_file(old_npz) == expected_npz
    assert sha256_file(old_csv) == expected_csv

assert COHORT_PATH.is_file()
assert REFERENCE_PATH.is_file()
assert OUTPUT_DIR.is_dir()
assert not NPZ_PATH.exists()
assert not CSV_PATH.exists()

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    plan = json.load(handle)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert plan["manifest_sha256"] == EXPECTED_HASHES[MANIFEST_PATH]
assert plan["image_count"] == 156
assert plan["source_risk_group_count"] == 134
assert plan["test_access_permitted"] is False
assert plan["randomization_seeds"] == [1042, 2042, 3042]
assert SEED in protocol["randomization"]["randomization_seeds"]
assert set(protocol["randomization"]["stages"]["R1"]) == {"head"}

print("\nFROZEN INPUT INTEGRITY")
print("Manifest SHA256: PASS")
print("Execution plan SHA256: PASS")
print("Randomization protocol SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Batch 001-006 NPZ/CSV SHA256: PASS")

# ============================================================
# 2. Resolve Batch 007 from the frozen manifest
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique
assert manifest["split"].eq("val").all()
assert manifest["image_id"].tolist() == sorted(
    manifest["image_id"].tolist()
)

batch = manifest.iloc[BATCH_START:BATCH_END].copy()

assert len(batch) == 16
assert batch["image_id"].is_unique
assert batch["true_grade"].isin([1, 2, 3, 4]).all()

image_ids = batch["image_id"].tolist()
grades = batch["true_grade"].astype(int).tolist()
risk_groups = batch["risk_group_id"].astype(str).tolist()

assert all(
    value.strip() not in {"", "nan", "None"}
    for value in risk_groups
)

previous_ids = []
previous_groups = []

for batch_index, number in enumerate(PREVIOUS_HASHES):
    old_npz = OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.npz"

    with np.load(old_npz, allow_pickle=False) as archive:
        old_ids = list(map(str, archive["image_ids"]))
        old_groups = list(map(str, archive["risk_group_ids"]))

        expected_slice = manifest.iloc[
            batch_index * 16:(batch_index + 1) * 16
        ]

        assert old_ids == expected_slice["image_id"].tolist()
        assert old_groups == expected_slice["risk_group_id"].tolist()

        previous_ids.extend(old_ids)
        previous_groups.extend(old_groups)

assert len(previous_ids) == 96
assert len(set(previous_ids)) == 96
assert set(image_ids).isdisjoint(set(previous_ids))

overlapping_groups = sorted(
    set(risk_groups) & set(previous_groups)
)

cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})
assert cohort["image_id"].is_unique

candidate_flags = (
    cohort.set_index("image_id")
    .loc[image_ids, "primary_ma_candidate"]
    .astype(str)
    .str.strip()
    .str.lower()
)

assert candidate_flags.isin(["true", "1"]).all()

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: index
        for index, image_id in enumerate(reference_ids)
    }

    assert set(image_ids).issubset(positions)

    reference_cams = np.stack([
        archive["retfound_cam"][positions[image_id]]
        .astype(np.float32)
        for image_id in image_ids
    ])

assert reference_cams.shape == (16, 224, 224)
assert np.isfinite(reference_cams).all()

print("\nBATCH COHORT")
print("Images:", len(image_ids))
print("Distinct source-risk groups:", len(set(risk_groups)))
print("Image ID overlap with previous batches: 0")
print("Risk groups overlapping previous batches:", len(overlapping_groups))
print("Validation-only selection: PASS")
print("Original CAM ID alignment: PASS")
print("First image ID:", image_ids[0])
print("Last image ID:", image_ids[-1])

if overlapping_groups:
    print("Previously observed risk groups:")
    for group in overlapping_groups:
        print(group)

# ============================================================
# 3. Decode only Batch 007 validation images
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensors = []

for _, row in batch.iterrows():
    image_path = Path(row["resolved_image_path"])
    assert image_path.is_file(), row["image_id"]

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    assert tensor.shape == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all().item()

    input_tensors.append(tensor)

print("Images decoded:", len(input_tensors))

# ============================================================
# 4. Load original RETFound checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)
loaded = model.load_state_dict(source_state, strict=True)

assert not loaded.missing_keys
assert not loaded.unexpected_keys

model.eval()
model.requires_grad_(False)

# ============================================================
# 5. Reconstruct frozen Grad-CAM procedure
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(activation_hook)


def calculate_cam(cpu_tensor, true_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(cpu_tensor.to(DEVICE))

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, true_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1).reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1).reshape(1024, 14, 14)
    )

    weights = gradients.mean(dim=(1, 2), keepdim=True)

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()
    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12
        cam /= float(cam.max())
        assert np.isfinite(cam).all()

    captured.clear()
    return output_logits, cam, raw_max


# ============================================================
# 6. Reproduce R0 and evaluate R1 using seed 1042
# ============================================================

r0_logits = []
r0_cams = []
r1_logits = []
r1_cams = []
r1_raw_maxima = []

try:
    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, _ = calculate_cam(cpu_tensor, grade)

        assert cam is not None
        r0_logits.append(logits)
        r0_cams.append(cam)

    baseline_metrics = [
        compare_cams(cam, reference)
        for cam, reference in zip(r0_cams, reference_cams)
    ]

    baseline_pearsons = [
        item["pearson"] for item in baseline_metrics
    ]

    baseline_maes = [
        item["mae"] for item in baseline_metrics
    ]

    assert np.isfinite(baseline_pearsons).all()
    assert min(baseline_pearsons) > 0.999

    print("\nR0 REPRODUCTION")
    print(
        "Minimum reference CAM Pearson:",
        round(float(min(baseline_pearsons)), 6)
    )
    print(
        "Mean reference CAM MAE:",
        round(float(np.mean(baseline_maes)), 8)
    )
    print("Baseline CAM reproduction: PASS")

    # Reset RNG once before R1 initialization.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        torch.nn.init.zeros_(model.head.bias)

    changed_names = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, source_state[name]):
            changed_names.append(name)

    assert set(changed_names) == {
        "head.weight", "head.bias"
    }

    print("\nR1 PARAMETER INTEGRITY")
    print("Changed tensors:", len(changed_names))
    print("Expected changed tensors: 2")
    print("Unauthorized changes: 0")
    print("Seed:", SEED)

    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, raw_max = calculate_cam(cpu_tensor, grade)
        r1_logits.append(logits)
        r1_cams.append(cam)
        r1_raw_maxima.append(raw_max)

finally:
    hook.remove()
    captured.clear()

    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Build image-level metrics and CAM arrays
# ============================================================

rows = []
stored_r1_cams = []

for i, image_id in enumerate(image_ids):
    metrics = compare_cams(r0_cams[i], r1_cams[i])

    rows.append({
        "model": "RETFound",
        "seed": SEED,
        "stage": "R1",
        "batch": BATCH_NUMBER,
        "image_id": image_id,
        "true_grade": grades[i],
        "risk_group_id": risk_groups[i],
        "degenerate_cam": r1_cams[i] is None,
        "raw_max": r1_raw_maxima[i],
        "r0_r1_pearson": metrics["pearson"],
        "r0_r1_spearman": metrics["spearman"],
        "r0_r1_ssim": metrics["ssim"],
        "r0_r1_mae": metrics["mae"],
        "r0_r1_logit_l2": float(
            np.linalg.norm(r1_logits[i] - r0_logits[i])
        )
    })

    stored_r1_cams.append(
        np.zeros((224, 224), dtype=np.float32)
        if r1_cams[i] is None
        else r1_cams[i].astype(np.float32)
    )

results = pd.DataFrame(rows)

r0_cam_array = np.stack(r0_cams).astype(np.float32)
r1_cam_array = np.stack(stored_r1_cams).astype(np.float32)
r0_logit_array = np.stack(r0_logits).astype(np.float32)
r1_logit_array = np.stack(r1_logits).astype(np.float32)

assert len(results) == 16
assert results["image_id"].is_unique
assert r0_cam_array.shape == (16, 224, 224)
assert r1_cam_array.shape == (16, 224, 224)
assert r0_logit_array.shape == (16, 5)
assert r1_logit_array.shape == (16, 5)

for array in [
    r0_cam_array, r1_cam_array,
    r0_logit_array, r1_logit_array
]:
    assert np.isfinite(array).all()

# ============================================================
# 8. Serialize and verify archives in memory
# ============================================================

npz_buffer = io.BytesIO()

np.savez_compressed(
    npz_buffer,
    image_ids=np.asarray(image_ids, dtype=str),
    true_grades=np.asarray(grades, dtype=np.int64),
    risk_group_ids=np.asarray(risk_groups, dtype=str),
    model=np.asarray("RETFound"),
    seed=np.asarray(SEED),
    stage=np.asarray("R1"),
    r0_cam=r0_cam_array,
    r1_cam=r1_cam_array,
    r0_logits=r0_logit_array,
    r1_logits=r1_logit_array,
    r1_raw_max=np.asarray(r1_raw_maxima, dtype=np.float64),
    r1_degenerate=results["degenerate_cam"].to_numpy(dtype=bool),
    manifest_sha256=np.asarray(EXPECTED_HASHES[MANIFEST_PATH]),
    execution_plan_sha256=np.asarray(EXPECTED_HASHES[PLAN_PATH])
)

npz_bytes = npz_buffer.getvalue()

csv_bytes = results.to_csv(
    index=False,
    float_format="%.10g"
).encode("utf-8")

with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)
    assert str(archive["model"].item()) == "RETFound"
    assert str(archive["stage"].item()) == "R1"
    assert int(archive["seed"].item()) == SEED

prepared_csv = pd.read_csv(
    io.BytesIO(csv_bytes),
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(prepared_csv) == 16
assert prepared_csv["image_id"].tolist() == image_ids
assert prepared_csv["seed"].eq(SEED).all()
assert prepared_csv["stage"].eq("R1").all()
assert prepared_csv["model"].eq("RETFound").all()

print("\nIN-MEMORY ARCHIVE VERIFICATION")
print("NPZ serialization: PASS")
print("CSV serialization: PASS")

# ============================================================
# 9. Publish without overwriting research files
# ============================================================

assert OUTPUT_DIR.is_dir()
assert not NPZ_PATH.exists()
assert not CSV_PATH.exists()

# Preserve byte buffers in memory if Drive publication fails.
npz_hash = publish_exclusively(NPZ_PATH, npz_bytes)
csv_hash = publish_exclusively(CSV_PATH, csv_bytes)

# ============================================================
# 10. Independently reload both published files
# ============================================================

with np.load(NPZ_PATH, allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)
    assert int(archive["seed"].item()) == SEED
    assert str(archive["stage"].item()) == "R1"

saved_csv = pd.read_csv(
    CSV_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(saved_csv) == 16
assert saved_csv["image_id"].tolist() == image_ids
assert saved_csv["seed"].eq(SEED).all()
assert saved_csv["stage"].eq("R1").all()
assert saved_csv["model"].eq("RETFound").all()

assert sha256_file(NPZ_PATH) == npz_hash
assert sha256_file(CSV_PATH) == csv_hash

# ============================================================
# 11. Final batch report
# ============================================================

print("\nBATCH RESULTS")
print("Model: RETFound")
print("Seed:", SEED)
print("Stage: R1")
print("Batch:", BATCH_NUMBER)
print("Images evaluated:", len(saved_csv))
print(
    "Degenerate CAMs:",
    int(saved_csv["degenerate_cam"].sum())
)

for column in [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]:
    print(
        f"Mean {column}:",
        round(float(saved_csv[column].mean()), 6)
    )

print("\nPER-IMAGE RESULTS")
print(
    saved_csv[
        [
            "image_id", "true_grade", "risk_group_id",
            "degenerate_cam", "r0_r1_pearson",
            "r0_r1_spearman", "r0_r1_ssim", "r0_r1_mae"
        ]
    ].to_string(index=False)
)

print("\nARCHIVE INTEGRITY")
print("NPZ path:", NPZ_PATH)
print("NPZ SHA256:", npz_hash)
print("CSV path:", CSV_PATH)
print("CSV SHA256:", csv_hash)
print("Batch 001-006 hashes unchanged: PASS")
print("Batch 007 archive round-trip: PASS")
print("Existing files overwritten: False")

print("\nRESEARCH INTEGRITY")
print("Frozen input files modified: False")
print("Validation images decoded:", len(input_tensors))
print("Test images accessed: 0")
print("New batch files created: 2")

print("\nSTEP 10.43 EXECUTION COMPLETE")
print("Production batch 007: PENDING REVIEW")


# %% [cell 383] identical to cell 384, omitted here


# %% [cell 384] Experiment 10 - Step 10.44

# ============================================================
# Experiment 10 - Step 10.44
# RETFound R1 Production Batch 008
# Seed 1042 | Frozen validation manifest rows 112:128
# Standalone Colab cell | No Test Set access
# ============================================================

from pathlib import Path
import gc
import hashlib
import io
import json

import cv2
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from timm.layers import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"
OUTPUT_DIR = EXP10 / "full_validation_batches"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"
PLAN_PATH = EXP10 / "Experiment10_Full_Execution_Plan_V1.json"
PROTOCOL_PATH = EXP10 / "Experiment10_Randomization_Protocol_V1.json"
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"

CHECKPOINT_PATH = (
    ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"
)
REFERENCE_PATH = (
    ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"
)

SEED = 1042
BATCH_START = 112
BATCH_END = 128
BATCH_NUMBER = "008"
BATCH_NAME = f"RETFound_Seed{SEED}_R1_Batch{BATCH_NUMBER}"

NPZ_PATH = OUTPUT_DIR / f"{BATCH_NAME}.npz"
CSV_PATH = OUTPUT_DIR / f"{BATCH_NAME}.csv"

EXPECTED_HASHES = {
    MANIFEST_PATH:
        "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5",
    PLAN_PATH:
        "da52de5513de274a0b91c99f7060049a1f77db4c6dc2a788263607e6d8168590",
    PROTOCOL_PATH:
        "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda",
    INDEX_PATH:
        "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    CHECKPOINT_PATH:
        "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
}

PREVIOUS_HASHES = {
    "001": (
        "4f71b19be85c52bcdce8f0d628fbc34d0171f9b91170204fa1c0b4d6b6a28c43",
        "529f23beed5a47c67b0a545ce32330ddcec1973ee93eb2d8ea315c88ab7d0918"
    ),
    "002": (
        "946cc0bd797bf43f8d857cd5618ab4976aefbcbad96e0d6a2606365a30c1deab",
        "ca444fde5920a09f5fc64479d09d0919945d741cdeeb24ffac5ca8341f88089a"
    ),
    "003": (
        "ebb83b4f68c13e6bcb2f079290c7846707ff3e6f76cdacfa264a38beab9ff9c6",
        "4a53c0c75cb040d8309413d934599f11515cd10e1498d636735f8cf87d60772f"
    ),
    "004": (
        "5a64e2f7ab7c515c1acdcbbc8bfaa4aaa509344a0ad2f99a8e1430b7dc176e62",
        "8d176e808a8979c750f7b060d0070b60e4d8cf7d956ed8fd2bf2cc63b54dc717"
    ),
    "005": (
        "f21a22e43dce2863ca64fe80066bc81e3a1368982b1076b602fa79506bd64a1b",
        "e1313fce88897aa043c5d5a2a0cba4b9445c408e4cea4bc5114e1393b27ddfe5"
    ),
    "006": (
        "341661bb0f5eb18980296e8e0e068b7c8ca25f25bbcdd9f2e21162b4681e61e6",
        "f492ec4f5f2bbe198ba4dddea640bb1b9fac3d5f68ecefed2dee44e9046619f5"
    ),
    "007": (
        "35ef3b851f6c9e7feb76b3eab1201132c6ea1b9f64135ff50007d76f67eab3a8",
        "8760cc8daf4920d31b9a526a32b9e704d82dfb04f1ae480d72bf740dad75a0b3"
    )
}

DEVICE = torch.device("cuda")
assert torch.cuda.is_available(), "CUDA is required."


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()
    ac = a - a.mean()
    bc = b - b.mean()

    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12 else np.nan
    )

    return {
        "pearson": pearson,
        "spearman": float(spearmanr(a, b).statistic),
        "ssim": float(
            structural_similarity(first, second, data_range=1.0)
        ),
        "mae": float(np.mean(np.abs(a - b)))
    }


def publish_exclusively(path, payload):
    if path.exists():
        raise FileExistsError(
            f"Existing output requires review: {path}"
        )

    expected_hash = hashlib.sha256(payload).hexdigest()

    # Exclusive creation avoids overwriting prior results.
    # A hash and round-trip check validate complete publication.
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()

    actual_hash = sha256_file(path)

    assert actual_hash == expected_hash, (
        f"Post-write SHA256 mismatch: {path}"
    )

    return actual_hash


print("EXPERIMENT 10.44 - RETFOUND R1 PRODUCTION BATCH 008")
print("Seed:", SEED)
print("Batch interval:", BATCH_START, BATCH_END)

# ============================================================
# 1. Verify frozen inputs and all seven previous batches
# ============================================================

for path, expected_hash in EXPECTED_HASHES.items():
    assert path.is_file(), f"Missing input: {path}"
    assert sha256_file(path) == expected_hash, (
        f"SHA256 mismatch: {path}"
    )

for number, (npz_expected, csv_expected) in PREVIOUS_HASHES.items():
    old_npz = OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.npz"
    old_csv = OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.csv"

    assert old_npz.is_file() and old_csv.is_file()
    assert sha256_file(old_npz) == npz_expected
    assert sha256_file(old_csv) == csv_expected

assert COHORT_PATH.is_file()
assert REFERENCE_PATH.is_file()
assert OUTPUT_DIR.is_dir()

assert not NPZ_PATH.exists(), f"Output already exists: {NPZ_PATH}"
assert not CSV_PATH.exists(), f"Output already exists: {CSV_PATH}"

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    plan = json.load(handle)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert plan["manifest_sha256"] == EXPECTED_HASHES[MANIFEST_PATH]
assert plan["image_count"] == 156
assert plan["source_risk_group_count"] == 134
assert plan["test_access_permitted"] is False
assert plan["randomization_seeds"] == [1042, 2042, 3042]

assert SEED in protocol["randomization"]["randomization_seeds"]
assert set(protocol["randomization"]["stages"]["R1"]) == {"head"}

print("\nFROZEN INPUT INTEGRITY")
print("Manifest SHA256: PASS")
print("Execution plan SHA256: PASS")
print("Randomization protocol SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Batch 001-007 NPZ/CSV SHA256: PASS")

# ============================================================
# 2. Validate Batch 008 and exclude previously processed IDs
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique
assert manifest["split"].eq("val").all()
assert manifest["image_id"].tolist() == sorted(
    manifest["image_id"].tolist()
)

batch = manifest.iloc[BATCH_START:BATCH_END].copy()

assert len(batch) == 16
assert batch["image_id"].is_unique
assert batch["true_grade"].isin([1, 2, 3, 4]).all()

image_ids = batch["image_id"].tolist()
grades = batch["true_grade"].astype(int).tolist()
risk_groups = batch["risk_group_id"].astype(str).tolist()

assert all(
    group.strip() not in {"", "nan", "None"}
    for group in risk_groups
)

previous_ids = []
previous_groups = []

for batch_index, number in enumerate(PREVIOUS_HASHES):
    old_npz = (
        OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.npz"
    )

    with np.load(old_npz, allow_pickle=False) as archive:
        old_ids = list(map(str, archive["image_ids"]))
        old_groups = list(map(str, archive["risk_group_ids"]))

        expected_slice = manifest.iloc[
            batch_index * 16:(batch_index + 1) * 16
        ]

        assert old_ids == expected_slice["image_id"].tolist()
        assert old_groups == expected_slice["risk_group_id"].tolist()

        previous_ids.extend(old_ids)
        previous_groups.extend(old_groups)

assert len(previous_ids) == 112
assert len(set(previous_ids)) == 112
assert set(image_ids).isdisjoint(set(previous_ids))

overlapping_groups = sorted(
    set(risk_groups) & set(previous_groups)
)

cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})
assert cohort["image_id"].is_unique

candidate_flags = (
    cohort.set_index("image_id")
    .loc[image_ids, "primary_ma_candidate"]
    .astype(str).str.strip().str.lower()
)

assert candidate_flags.isin(["true", "1"]).all()

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: index
        for index, image_id in enumerate(reference_ids)
    }

    assert set(image_ids).issubset(positions)

    reference_cams = np.stack([
        archive["retfound_cam"][positions[image_id]]
        .astype(np.float32)
        for image_id in image_ids
    ])

assert reference_cams.shape == (16, 224, 224)
assert np.isfinite(reference_cams).all()

print("\nBATCH COHORT")
print("Images:", len(image_ids))
print("Distinct source-risk groups:", len(set(risk_groups)))
print("Image ID overlap with previous batches: 0")
print("Risk groups overlapping previous batches:", len(overlapping_groups))
print("Validation-only selection: PASS")
print("Original CAM ID alignment: PASS")
print("First image ID:", image_ids[0])
print("Last image ID:", image_ids[-1])

if overlapping_groups:
    print("Previously observed risk groups:")
    for group in overlapping_groups:
        print(group)

# ============================================================
# 3. Decode only the selected validation images
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensors = []

for _, row in batch.iterrows():
    image_path = Path(row["resolved_image_path"])

    assert image_path.is_file(), row["image_id"]

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    assert tensor.shape == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all().item()

    input_tensors.append(tensor)

print("Images decoded:", len(input_tensors))

# ============================================================
# 4. Load the original RETFound checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

load_result = model.load_state_dict(
    source_state, strict=True
)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

# ============================================================
# 5. Reconstruct validated Grad-CAM
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(activation_hook)


def calculate_cam(cpu_tensor, true_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(cpu_tensor.to(DEVICE))

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, true_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2), keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()
    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12
        cam /= float(cam.max())

        assert np.isfinite(cam).all()

    captured.clear()
    return output_logits, cam, raw_max


# ============================================================
# 6. Reproduce R0 and evaluate R1 at seed 1042
# ============================================================

r0_logits = []
r0_cams = []
r1_logits = []
r1_cams = []
r1_raw_maxima = []

try:
    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, _ = calculate_cam(cpu_tensor, grade)

        assert cam is not None

        r0_logits.append(logits)
        r0_cams.append(cam)

    baseline_metrics = [
        compare_cams(cam, reference)
        for cam, reference in zip(r0_cams, reference_cams)
    ]

    baseline_pearsons = [
        metric["pearson"] for metric in baseline_metrics
    ]

    baseline_maes = [
        metric["mae"] for metric in baseline_metrics
    ]

    assert np.isfinite(baseline_pearsons).all()
    assert min(baseline_pearsons) > 0.999

    print("\nR0 REPRODUCTION")
    print(
        "Minimum reference CAM Pearson:",
        round(float(min(baseline_pearsons)), 6)
    )
    print(
        "Mean reference CAM MAE:",
        round(float(np.mean(baseline_maes)), 8)
    )
    print("Baseline CAM reproduction: PASS")

    # R1: Reset the RNG once and initialize only the head.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        torch.nn.init.zeros_(model.head.bias)

    changed_names = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, source_state[name]):
            changed_names.append(name)

    assert set(changed_names) == {
        "head.weight",
        "head.bias"
    }

    print("\nR1 PARAMETER INTEGRITY")
    print("Changed tensors:", len(changed_names))
    print("Expected changed tensors: 2")
    print("Unauthorized changes: 0")
    print("Seed:", SEED)

    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, raw_max = calculate_cam(
            cpu_tensor, grade
        )

        r1_logits.append(logits)
        r1_cams.append(cam)
        r1_raw_maxima.append(raw_max)

finally:
    hook.remove()
    captured.clear()

    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Build per-image metrics and arrays
# ============================================================

rows = []
stored_r1_cams = []

for i, image_id in enumerate(image_ids):
    metrics = compare_cams(r0_cams[i], r1_cams[i])

    rows.append({
        "model": "RETFound",
        "seed": SEED,
        "stage": "R1",
        "batch": BATCH_NUMBER,
        "image_id": image_id,
        "true_grade": grades[i],
        "risk_group_id": risk_groups[i],
        "degenerate_cam": r1_cams[i] is None,
        "raw_max": r1_raw_maxima[i],
        "r0_r1_pearson": metrics["pearson"],
        "r0_r1_spearman": metrics["spearman"],
        "r0_r1_ssim": metrics["ssim"],
        "r0_r1_mae": metrics["mae"],
        "r0_r1_logit_l2": float(
            np.linalg.norm(r1_logits[i] - r0_logits[i])
        )
    })

    stored_r1_cams.append(
        np.zeros((224, 224), dtype=np.float32)
        if r1_cams[i] is None
        else r1_cams[i].astype(np.float32)
    )

results = pd.DataFrame(rows)

r0_cam_array = np.stack(r0_cams).astype(np.float32)
r1_cam_array = np.stack(stored_r1_cams).astype(np.float32)
r0_logit_array = np.stack(r0_logits).astype(np.float32)
r1_logit_array = np.stack(r1_logits).astype(np.float32)

assert len(results) == 16
assert results["image_id"].is_unique
assert r0_cam_array.shape == (16, 224, 224)
assert r1_cam_array.shape == (16, 224, 224)
assert r0_logit_array.shape == (16, 5)
assert r1_logit_array.shape == (16, 5)

for array in [
    r0_cam_array,
    r1_cam_array,
    r0_logit_array,
    r1_logit_array
]:
    assert np.isfinite(array).all()

# ============================================================
# 8. Serialize and check archives in memory
# ============================================================

npz_buffer = io.BytesIO()

np.savez_compressed(
    npz_buffer,
    image_ids=np.asarray(image_ids, dtype=str),
    true_grades=np.asarray(grades, dtype=np.int64),
    risk_group_ids=np.asarray(risk_groups, dtype=str),
    model=np.asarray("RETFound"),
    seed=np.asarray(SEED),
    stage=np.asarray("R1"),
    r0_cam=r0_cam_array,
    r1_cam=r1_cam_array,
    r0_logits=r0_logit_array,
    r1_logits=r1_logit_array,
    r1_raw_max=np.asarray(r1_raw_maxima, dtype=np.float64),
    r1_degenerate=results["degenerate_cam"].to_numpy(dtype=bool),
    manifest_sha256=np.asarray(EXPECTED_HASHES[MANIFEST_PATH]),
    execution_plan_sha256=np.asarray(EXPECTED_HASHES[PLAN_PATH])
)

npz_bytes = npz_buffer.getvalue()

csv_bytes = results.to_csv(
    index=False,
    float_format="%.10g"
).encode("utf-8")

with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)
    assert str(archive["model"].item()) == "RETFound"
    assert str(archive["stage"].item()) == "R1"
    assert int(archive["seed"].item()) == SEED

prepared_csv = pd.read_csv(
    io.BytesIO(csv_bytes),
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(prepared_csv) == 16
assert prepared_csv["image_id"].tolist() == image_ids
assert prepared_csv["seed"].eq(SEED).all()
assert prepared_csv["stage"].eq("R1").all()
assert prepared_csv["model"].eq("RETFound").all()

print("\nIN-MEMORY ARCHIVE VERIFICATION")
print("NPZ serialization: PASS")
print("CSV serialization: PASS")

# ============================================================
# 9. Publish files without overwriting earlier batches
# ============================================================

assert OUTPUT_DIR.is_dir()
assert not NPZ_PATH.exists()
assert not CSV_PATH.exists()

# In-memory buffers remain available if Drive publication fails.
npz_hash = publish_exclusively(NPZ_PATH, npz_bytes)
csv_hash = publish_exclusively(CSV_PATH, csv_bytes)

# ============================================================
# 10. Independently verify the saved NPZ and CSV
# ============================================================

with np.load(NPZ_PATH, allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)
    assert int(archive["seed"].item()) == SEED
    assert str(archive["stage"].item()) == "R1"

saved_csv = pd.read_csv(
    CSV_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(saved_csv) == 16
assert saved_csv["image_id"].tolist() == image_ids
assert saved_csv["seed"].eq(SEED).all()
assert saved_csv["stage"].eq("R1").all()
assert saved_csv["model"].eq("RETFound").all()

assert sha256_file(NPZ_PATH) == npz_hash
assert sha256_file(CSV_PATH) == csv_hash

# ============================================================
# 11. Final report
# ============================================================

print("\nBATCH RESULTS")
print("Model: RETFound")
print("Seed:", SEED)
print("Stage: R1")
print("Batch:", BATCH_NUMBER)
print("Images evaluated:", len(saved_csv))
print(
    "Degenerate CAMs:",
    int(saved_csv["degenerate_cam"].sum())
)

for column in [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]:
    print(
        f"Mean {column}:",
        round(float(saved_csv[column].mean()), 6)
    )

print("\nPER-IMAGE RESULTS")
print(
    saved_csv[
        [
            "image_id",
            "true_grade",
            "risk_group_id",
            "degenerate_cam",
            "r0_r1_pearson",
            "r0_r1_spearman",
            "r0_r1_ssim",
            "r0_r1_mae"
        ]
    ].to_string(index=False)
)

print("\nARCHIVE INTEGRITY")
print("NPZ path:", NPZ_PATH)
print("NPZ SHA256:", npz_hash)
print("CSV path:", CSV_PATH)
print("CSV SHA256:", csv_hash)
print("Batch 001-007 hashes unchanged: PASS")
print("Batch 008 archive round-trip: PASS")
print("Existing files overwritten: False")

print("\nRESEARCH INTEGRITY")
print("Frozen input files modified: False")
print("Validation images decoded:", len(input_tensors))
print("Test images accessed: 0")
print("New batch files created: 2")

print("\nSTEP 10.44 EXECUTION COMPLETE")
print("Production batch 008: PENDING REVIEW")


# %% [cell 385] Experiment 10 - Step 10.44A

# ============================================================
# Experiment 10 - Step 10.44A
# Existing Batch 008 Artifact Forensic Audit
# Read-only; no inference; no artifact modification
# ============================================================

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"
OUTPUT_DIR = EXP10 / "full_validation_batches"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"
PLAN_PATH = EXP10 / "Experiment10_Full_Execution_Plan_V1.json"

NPZ_PATH = OUTPUT_DIR / "RETFound_Seed1042_R1_Batch008.npz"
CSV_PATH = OUTPUT_DIR / "RETFound_Seed1042_R1_Batch008.csv"

EXPECTED_MANIFEST_SHA256 = (
    "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5"
)
EXPECTED_PLAN_SHA256 = (
    "da52de5513de274a0b91c99f7060049a1f77db4c6dc2a788263607e6d8168590"
)

BATCH_START = 112
BATCH_END = 128


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_npz(path, expected_ids, expected_grades, expected_groups):
    if not path.is_file():
        print("NPZ status: MISSING")
        return False, None

    print("NPZ size (bytes):", path.stat().st_size)
    print("NPZ SHA256:", sha256_file(path))

    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)

            required = {
                "image_ids",
                "true_grades",
                "risk_group_ids",
                "model",
                "seed",
                "stage",
                "r0_cam",
                "r1_cam",
                "r0_logits",
                "r1_logits",
                "r1_raw_max",
                "r1_degenerate",
                "manifest_sha256",
                "execution_plan_sha256"
            }

            assert required.issubset(keys), sorted(required - keys)

            ids = list(map(str, archive["image_ids"]))
            grades = archive["true_grades"].astype(int).tolist()
            groups = list(map(str, archive["risk_group_ids"]))

            assert ids == expected_ids
            assert grades == expected_grades
            assert groups == expected_groups

            assert str(archive["model"].item()) == "RETFound"
            assert int(archive["seed"].item()) == 1042
            assert str(archive["stage"].item()) == "R1"

            assert (
                str(archive["manifest_sha256"].item())
                == EXPECTED_MANIFEST_SHA256
            )
            assert (
                str(archive["execution_plan_sha256"].item())
                == EXPECTED_PLAN_SHA256
            )

            assert archive["r0_cam"].shape == (16, 224, 224)
            assert archive["r1_cam"].shape == (16, 224, 224)
            assert archive["r0_logits"].shape == (16, 5)
            assert archive["r1_logits"].shape == (16, 5)
            assert archive["r1_raw_max"].shape == (16,)
            assert archive["r1_degenerate"].shape == (16,)

            for key in [
                "r0_cam", "r1_cam",
                "r0_logits", "r1_logits", "r1_raw_max"
            ]:
                assert np.isfinite(archive[key]).all(), key

            degenerate = archive["r1_degenerate"].astype(bool)
            r1_cams = archive["r1_cam"]

            for i in range(16):
                if degenerate[i]:
                    assert np.max(np.abs(r1_cams[i])) <= 1e-12
                else:
                    assert np.max(r1_cams[i]) > 1e-12

            assert (
                (archive["r0_cam"] >= -1e-5).all()
                and (archive["r0_cam"] <= 1.0 + 1e-5).all()
            )
            assert (
                (archive["r1_cam"] >= -1e-5).all()
                and (archive["r1_cam"] <= 1.0 + 1e-5).all()
            )

            npz_summary = {
                "image_ids": ids,
                "true_grades": grades,
                "risk_group_ids": groups,
                "degenerate": degenerate.copy(),
                "raw_max": archive["r1_raw_max"].copy()
            }

        print("NPZ structure and metadata: PASS")
        print("NPZ image alignment: PASS")
        print("NPZ numeric integrity: PASS")
        print("NPZ degenerate CAM count:", int(degenerate.sum()))
        return True, npz_summary

    except Exception as error:
        print("NPZ validation: FAIL")
        print("Error type:", type(error).__name__)
        print("Error:", str(error))
        return False, None


def inspect_csv(path, expected_ids, expected_grades, expected_groups):
    if not path.is_file():
        print("CSV status: MISSING")
        return False, None

    print("CSV size (bytes):", path.stat().st_size)
    print("CSV SHA256:", sha256_file(path))

    try:
        frame = pd.read_csv(
            path,
            dtype={
                "image_id": str,
                "risk_group_id": str,
                "batch": str
            }
        )

        required = {
            "model", "seed", "stage", "batch",
            "image_id", "true_grade", "risk_group_id",
            "degenerate_cam", "raw_max",
            "r0_r1_pearson", "r0_r1_spearman",
            "r0_r1_ssim", "r0_r1_mae",
            "r0_r1_logit_l2"
        }

        assert required.issubset(set(frame.columns))
        assert len(frame) == 16
        assert frame["image_id"].tolist() == expected_ids
        assert frame["true_grade"].astype(int).tolist() == expected_grades
        assert frame["risk_group_id"].tolist() == expected_groups

        assert frame["model"].eq("RETFound").all()
        assert frame["seed"].eq(1042).all()
        assert frame["stage"].eq("R1").all()
        assert frame["batch"].str.zfill(3).eq("008").all()

        flags = frame["degenerate_cam"].astype(str).str.lower()
        assert flags.isin(["true", "false"]).all()

        numeric_cols = [
            "raw_max", "r0_r1_pearson", "r0_r1_spearman",
            "r0_r1_ssim", "r0_r1_mae", "r0_r1_logit_l2"
        ]

        assert np.isfinite(
            frame["raw_max"].to_numpy(dtype=float)
        ).all()

        assert np.isfinite(
            frame["r0_r1_logit_l2"].to_numpy(dtype=float)
        ).all()

        nondegenerate = flags.eq("false")
        assert np.isfinite(
            frame.loc[
                nondegenerate, numeric_cols
            ].to_numpy(dtype=float)
        ).all()

        print("CSV structure and metadata: PASS")
        print("CSV image alignment: PASS")
        print("CSV numeric integrity: PASS")

        return True, frame

    except Exception as error:
        print("CSV validation: FAIL")
        print("Error type:", type(error).__name__)
        print("Error:", str(error))
        return False, None


print("EXPERIMENT 10.44A - BATCH 008 FILE FORENSIC AUDIT")

assert sha256_file(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256
assert sha256_file(PLAN_PATH) == EXPECTED_PLAN_SHA256

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    plan = json.load(handle)

assert plan["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
assert plan["test_access_permitted"] is False

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique

batch = manifest.iloc[BATCH_START:BATCH_END]

expected_ids = batch["image_id"].tolist()
expected_grades = batch["true_grade"].astype(int).tolist()
expected_groups = batch["risk_group_id"].astype(str).tolist()

assert len(expected_ids) == 16

print("\nEXPECTED BATCH")
print("Model: RETFound")
print("Seed: 1042")
print("Stage: R1")
print("Manifest slice: 112:128")
print("Expected images:", len(expected_ids))
print("First image ID:", expected_ids[0])
print("Last image ID:", expected_ids[-1])

print("\nNPZ INSPECTION")
npz_valid, npz_data = inspect_npz(
    NPZ_PATH, expected_ids, expected_grades, expected_groups
)

print("\nCSV INSPECTION")
csv_valid, csv_data = inspect_csv(
    CSV_PATH, expected_ids, expected_grades, expected_groups
)

cross_valid = False

if npz_valid and csv_valid:
    flags_csv = (
        csv_data["degenerate_cam"]
        .astype(str).str.lower().eq("true")
        .to_numpy()
    )

    flag_match = np.array_equal(
        flags_csv, npz_data["degenerate"]
    )

    raw_max_match = np.allclose(
        csv_data["raw_max"].to_numpy(dtype=float),
        npz_data["raw_max"],
        rtol=1e-8,
        atol=1e-10
    )

    cross_valid = bool(flag_match and raw_max_match)

    print("\nCROSS-FILE CONSISTENCY")
    print("Degenerate flags match:", flag_match)
    print("Raw CAM maxima match:", raw_max_match)

print("\nFINAL AUDIT")
print("NPZ valid:", npz_valid)
print("CSV valid:", csv_valid)
print("Cross-file consistency:", cross_valid)

if npz_valid and csv_valid and cross_valid:
    print("Existing Batch 008 structural audit: PASS")
    print(
        "Status: Existing output is structurally consistent; "
        "independent computation provenance remains to be reviewed."
    )
else:
    print("Existing Batch 008 structural audit: REQUIRES REVIEW")
    print(
        "Status: Preserve existing files. "
        "Do not delete or overwrite them."
    )

print("\nRESEARCH INTEGRITY")
print("Files modified: 0")
print("Images decoded: 0")
print("Model inference performed: False")
print("Test images accessed: 0")

print("\nSTEP 10.44A EXECUTION COMPLETE")


# %% [cell 386] Experiment 10 - Step 10.44B

# ============================================================
# Experiment 10 - Step 10.44B
# Batch 008 Independent Metric Recalculation Audit
# Read-only; no model inference; no image decoding
# ============================================================

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"
BATCH_DIR = EXP10 / "full_validation_batches"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"

REFERENCE_PATH = (
    ROOT / "experiment05_results/gradcam/"
    "RETFound_MA_CAM_Maps_V1.npz"
)

NPZ_PATH = BATCH_DIR / "RETFound_Seed1042_R1_Batch008.npz"
CSV_PATH = BATCH_DIR / "RETFound_Seed1042_R1_Batch008.csv"

EXPECTED_HASHES = {
    MANIFEST_PATH:
        "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5",
    NPZ_PATH:
        "7464b02a141c197c0a6b419e5412ea98410bd02e7da04b1c92bd32608b47c0c2",
    CSV_PATH:
        "755a1c3e6b9f751cffe614f74fb63e81e2bf46f3a42b44fbcf6a6db8b7bb61b7"
}

PREVIOUS_NPZ_HASHES = {
    "001": "4f71b19be85c52bcdce8f0d628fbc34d0171f9b91170204fa1c0b4d6b6a28c43",
    "002": "946cc0bd797bf43f8d857cd5618ab4976aefbcbad96e0d6a2606365a30c1deab",
    "003": "ebb83b4f68c13e6bcb2f079290c7846707ff3e6f76cdacfa264a38beab9ff9c6",
    "004": "5a64e2f7ab7c515c1acdcbbc8bfaa4aaa509344a0ad2f99a8e1430b7dc176e62",
    "005": "f21a22e43dce2863ca64fe80066bc81e3a1368982b1076b602fa79506bd64a1b",
    "006": "341661bb0f5eb18980296e8e0e068b7c8ca25f25bbcdd9f2e21162b4681e61e6",
    "007": "35ef3b851f6c9e7feb76b3eab1201132c6ea1b9f64135ff50007d76f67eab3a8"
}

EXPECTED_PILOT_R1_PEARSON = {
    "1301_1": -0.108207,
    "0421_1": -0.065059,
    "1190_1": -0.226063,
    "0732_1": -0.240862
}

BATCH_START = 112
BATCH_END = 128


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def recalculate_metrics(first, second):
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    centered_a = a - a.mean()
    centered_b = b - b.mean()

    denominator = (
        np.linalg.norm(centered_a)
        * np.linalg.norm(centered_b)
    )

    pearson = (
        float(np.dot(centered_a, centered_b) / denominator)
        if denominator > 1e-12
        else np.nan
    )

    spearman = float(spearmanr(a, b).statistic)

    ssim = float(
        structural_similarity(
            first,
            second,
            data_range=1.0
        )
    )

    mae = float(np.mean(np.abs(a - b)))

    return {
        "pearson": pearson,
        "spearman": spearman,
        "ssim": ssim,
        "mae": mae
    }


print("EXPERIMENT 10.44B - BATCH 008 METRIC AUDIT")

# ============================================================
# 1. Verify known file hashes
# ============================================================

for path, expected_hash in EXPECTED_HASHES.items():
    assert path.is_file(), str(path)
    assert sha256_file(path) == expected_hash, str(path)

assert REFERENCE_PATH.is_file()

previous_ids = []

for number, expected_hash in PREVIOUS_NPZ_HASHES.items():
    path = (
        BATCH_DIR
        / f"RETFound_Seed1042_R1_Batch{number}.npz"
    )

    assert path.is_file()
    assert sha256_file(path) == expected_hash

    with np.load(path, allow_pickle=False) as archive:
        previous_ids.extend(
            list(map(str, archive["image_ids"]))
        )

assert len(previous_ids) == 112
assert len(set(previous_ids)) == 112

print("\nFILE INTEGRITY")
print("Manifest SHA256: PASS")
print("Batch 008 NPZ SHA256: PASS")
print("Batch 008 CSV SHA256: PASS")
print("Batch 001-007 NPZ SHA256: PASS")

# ============================================================
# 2. Load frozen manifest and archived raw arrays
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique

batch = manifest.iloc[BATCH_START:BATCH_END]

expected_ids = batch["image_id"].tolist()
expected_grades = batch["true_grade"].astype(int).tolist()
expected_groups = batch["risk_group_id"].astype(str).tolist()

with np.load(NPZ_PATH, allow_pickle=False) as archive:
    image_ids = list(map(str, archive["image_ids"]))
    true_grades = archive["true_grades"].astype(int).tolist()
    risk_groups = list(map(str, archive["risk_group_ids"]))

    assert image_ids == expected_ids
    assert true_grades == expected_grades
    assert risk_groups == expected_groups

    assert str(archive["model"].item()) == "RETFound"
    assert str(archive["stage"].item()) == "R1"
    assert int(archive["seed"].item()) == 1042

    r0_cams = archive["r0_cam"].copy()
    r1_cams = archive["r1_cam"].copy()
    r0_logits = archive["r0_logits"].copy()
    r1_logits = archive["r1_logits"].copy()
    raw_maxima = archive["r1_raw_max"].copy()
    degenerate = archive["r1_degenerate"].astype(bool).copy()

assert r0_cams.shape == (16, 224, 224)
assert r1_cams.shape == (16, 224, 224)
assert r0_logits.shape == (16, 5)
assert r1_logits.shape == (16, 5)

assert not set(image_ids).intersection(previous_ids)

csv = pd.read_csv(
    CSV_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str,
        "batch": str
    }
)

assert csv["image_id"].tolist() == image_ids
assert csv["true_grade"].astype(int).tolist() == true_grades
assert csv["risk_group_id"].tolist() == risk_groups
assert csv["model"].eq("RETFound").all()
assert csv["seed"].eq(1042).all()
assert csv["stage"].eq("R1").all()
assert csv["batch"].str.zfill(3).eq("008").all()

csv_flags = (
    csv["degenerate_cam"]
    .astype(str).str.lower().eq("true")
    .to_numpy()
)

assert np.array_equal(csv_flags, degenerate)

print("\nBATCH ALIGNMENT")
print("Images:", len(image_ids))
print("Image ID overlap with earlier batches:", 0)
print("Frozen manifest alignment: PASS")
print("CSV and NPZ image alignment: PASS")
print("Degenerate CAMs:", int(degenerate.sum()))

# ============================================================
# 3. Recompute CSV metrics from archived raw data
# ============================================================

recomputed_rows = []

for i, image_id in enumerate(image_ids):
    if degenerate[i]:
        metrics = {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }
    else:
        metrics = recalculate_metrics(
            r0_cams[i],
            r1_cams[i]
        )

    logit_l2 = float(
        np.linalg.norm(
            r1_logits[i] - r0_logits[i]
        )
    )

    recomputed_rows.append({
        "image_id": image_id,
        "r0_r1_pearson": metrics["pearson"],
        "r0_r1_spearman": metrics["spearman"],
        "r0_r1_ssim": metrics["ssim"],
        "r0_r1_mae": metrics["mae"],
        "r0_r1_logit_l2": logit_l2
    })

recomputed = pd.DataFrame(recomputed_rows)

metric_columns = [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]

maximum_differences = {}

for column in metric_columns:
    stored = csv[column].to_numpy(dtype=float)
    calculated = recomputed[column].to_numpy(dtype=float)

    assert np.array_equal(
        np.isnan(stored),
        np.isnan(calculated)
    ), f"NaN mismatch: {column}"

    differences = np.abs(stored - calculated)
    finite_differences = differences[np.isfinite(differences)]

    maximum = (
        float(finite_differences.max())
        if len(finite_differences)
        else 0.0
    )

    maximum_differences[column] = maximum

    assert np.allclose(
        stored,
        calculated,
        rtol=1e-6,
        atol=1e-7,
        equal_nan=True
    ), f"Metric mismatch: {column}"

assert np.allclose(
    csv["raw_max"].to_numpy(dtype=float),
    raw_maxima,
    rtol=1e-8,
    atol=1e-10
)

print("\nINDEPENDENT CSV METRIC RECOMPUTATION")

for column, maximum in maximum_differences.items():
    print(
        f"Maximum absolute difference for {column}:",
        maximum
    )

print("All five image-level metrics: PASS")
print("Raw CAM maxima: PASS")

# ============================================================
# 4. Compare stored R0 maps with frozen reference CAMs
# ============================================================

with np.load(
    REFERENCE_PATH,
    allow_pickle=False
) as archive:

    reference_ids = list(map(str, archive["image_ids"]))

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: i
        for i, image_id in enumerate(reference_ids)
    }

    assert set(image_ids).issubset(positions)

    references = np.stack([
        archive["retfound_cam"][positions[image_id]]
        .astype(np.float32)
        for image_id in image_ids
    ])

reference_metrics = [
    recalculate_metrics(r0_cams[i], references[i])
    for i in range(16)
]

reference_pearsons = np.asarray([
    item["pearson"] for item in reference_metrics
])

reference_maes = np.asarray([
    item["mae"] for item in reference_metrics
])

assert np.isfinite(reference_pearsons).all()
assert reference_pearsons.min() > 0.999

print("\nR0 REFERENCE CAM AUDIT")
print(
    "Minimum reference Pearson:",
    round(float(reference_pearsons.min()), 6)
)
print(
    "Mean reference MAE:",
    round(float(reference_maes.mean()), 8)
)
print("Archived R0 reference reproduction: PASS")

# ============================================================
# 5. Compare any shared pilot image against reported pilot
# ============================================================

pilot_matches = 0

print("\nPILOT CROSS-CHECK")

for i, image_id in enumerate(image_ids):
    if image_id not in EXPECTED_PILOT_R1_PEARSON:
        continue

    pilot_matches += 1

    expected = EXPECTED_PILOT_R1_PEARSON[image_id]
    actual = float(recomputed.loc[
        recomputed["image_id"].eq(image_id),
        "r0_r1_pearson"
    ].iloc[0])

    difference = abs(actual - expected)

    assert difference < 1e-4, (
        f"Pilot R1 discrepancy: {image_id}"
    )

    print(
        image_id,
        "reported_pilot=", expected,
        "archived_batch=", round(actual, 6),
        "difference=", round(difference, 8)
    )

print("Matching pilot images in this batch:", pilot_matches)

if pilot_matches:
    print("Pilot cross-check: PASS")
else:
    print("Pilot cross-check: NOT APPLICABLE")

# ============================================================
# 6. Summarize image-level metrics
# ============================================================

print("\nRECOMPUTED BATCH SUMMARY")

for column in metric_columns:
    print(
        f"Mean {column}:",
        round(float(recomputed[column].mean()), 6)
    )

print("\nPER-IMAGE RECOMPUTED METRICS")
print(recomputed.to_string(index=False))

print("\nFINAL AUDIT")
print("NPZ and CSV numeric consistency: PASS")
print("Frozen R0 CAM consistency: PASS")
print("Existing file hashes unchanged: PASS")
print(
    "Computation provenance:",
    "PARTIALLY CORROBORATED; original runtime log not verified"
)
print("Model inference performed: False")
print("Images decoded: 0")
print("Files modified: 0")
print("Test images accessed: 0")

print("\nSTEP 10.44B EXECUTION COMPLETE")
print("Batch 008 independent metric audit: PASS")


# %% [cell 387] Experiment 10 - Step 10.45

# ============================================================
# Experiment 10 - Step 10.45
# RETFound R1 Production Batch 009
# Seed 1042 | Frozen validation manifest rows 128:144
# Standalone execution | Exclusive archive publication
# ============================================================

from pathlib import Path
import gc
import hashlib
import io
import json

import cv2
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from timm.layers import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"
OUTPUT_DIR = EXP10 / "full_validation_batches"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"
PLAN_PATH = EXP10 / "Experiment10_Full_Execution_Plan_V1.json"
PROTOCOL_PATH = EXP10 / "Experiment10_Randomization_Protocol_V1.json"
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
CHECKPOINT_PATH = ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"
REFERENCE_PATH = ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"

SEED = 1042
BATCH_START = 128
BATCH_END = 144
BATCH_NUMBER = "009"
BATCH_NAME = f"RETFound_Seed{SEED}_R1_Batch{BATCH_NUMBER}"

NPZ_PATH = OUTPUT_DIR / f"{BATCH_NAME}.npz"
CSV_PATH = OUTPUT_DIR / f"{BATCH_NAME}.csv"
DEVICE = torch.device("cuda")

EXPECTED_HASHES = {
    MANIFEST_PATH: "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5",
    PLAN_PATH: "da52de5513de274a0b91c99f7060049a1f77db4c6dc2a788263607e6d8168590",
    PROTOCOL_PATH: "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda",
    INDEX_PATH: "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    CHECKPOINT_PATH: "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
}

PREVIOUS_HASHES = {
    "001": (
        "4f71b19be85c52bcdce8f0d628fbc34d0171f9b91170204fa1c0b4d6b6a28c43",
        "529f23beed5a47c67b0a545ce32330ddcec1973ee93eb2d8ea315c88ab7d0918"
    ),
    "002": (
        "946cc0bd797bf43f8d857cd5618ab4976aefbcbad96e0d6a2606365a30c1deab",
        "ca444fde5920a09f5fc64479d09d0919945d741cdeeb24ffac5ca8341f88089a"
    ),
    "003": (
        "ebb83b4f68c13e6bcb2f079290c7846707ff3e6f76cdacfa264a38beab9ff9c6",
        "4a53c0c75cb040d8309413d934599f11515cd10e1498d636735f8cf87d60772f"
    ),
    "004": (
        "5a64e2f7ab7c515c1acdcbbc8bfaa4aaa509344a0ad2f99a8e1430b7dc176e62",
        "8d176e808a8979c750f7b060d0070b60e4d8cf7d956ed8fd2bf2cc63b54dc717"
    ),
    "005": (
        "f21a22e43dce2863ca64fe80066bc81e3a1368982b1076b602fa79506bd64a1b",
        "e1313fce88897aa043c5d5a2a0cba4b9445c408e4cea4bc5114e1393b27ddfe5"
    ),
    "006": (
        "341661bb0f5eb18980296e8e0e068b7c8ca25f25bbcdd9f2e21162b4681e61e6",
        "f492ec4f5f2bbe198ba4dddea640bb1b9fac3d5f68ecefed2dee44e9046619f5"
    ),
    "007": (
        "35ef3b851f6c9e7feb76b3eab1201132c6ea1b9f64135ff50007d76f67eab3a8",
        "8760cc8daf4920d31b9a526a32b9e704d82dfb04f1ae480d72bf740dad75a0b3"
    ),
    "008": (
        "7464b02a141c197c0a6b419e5412ea98410bd02e7da04b1c92bd32608b47c0c2",
        "755a1c3e6b9f751cffe614f74fb63e81e2bf46f3a42b44fbcf6a6db8b7bb61b7"
    )
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    centered_a = a - a.mean()
    centered_b = b - b.mean()
    denominator = (
        np.linalg.norm(centered_a)
        * np.linalg.norm(centered_b)
    )

    pearson = (
        float(np.dot(centered_a, centered_b) / denominator)
        if denominator > 1e-12 else np.nan
    )

    return {
        "pearson": pearson,
        "spearman": float(spearmanr(a, b).statistic),
        "ssim": float(
            structural_similarity(first, second, data_range=1.0)
        ),
        "mae": float(np.mean(np.abs(a - b)))
    }


def publish_exclusively(path, payload):
    if path.exists():
        raise FileExistsError(
            f"Existing output requires review: {path}"
        )

    expected_hash = hashlib.sha256(payload).hexdigest()

    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()

    actual_hash = sha256_file(path)

    assert actual_hash == expected_hash, (
        f"Post-write SHA256 mismatch: {path}"
    )

    return actual_hash


print("EXPERIMENT 10.45 - RETFOUND R1 PRODUCTION BATCH 009")
print("Seed:", SEED)
print("Batch interval:", BATCH_START, BATCH_END)

# ============================================================
# 1. Verify frozen sources and all previous batch archives
# ============================================================

assert torch.cuda.is_available(), "CUDA is required."

for path, expected_hash in EXPECTED_HASHES.items():
    assert path.is_file(), f"Missing frozen input: {path}"
    assert sha256_file(path) == expected_hash, (
        f"Frozen input SHA256 mismatch: {path}"
    )

for number, (expected_npz, expected_csv) in PREVIOUS_HASHES.items():
    old_npz = OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.npz"
    old_csv = OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.csv"

    assert old_npz.is_file() and old_csv.is_file()
    assert sha256_file(old_npz) == expected_npz, str(old_npz)
    assert sha256_file(old_csv) == expected_csv, str(old_csv)

assert REFERENCE_PATH.is_file()
assert COHORT_PATH.is_file()
assert OUTPUT_DIR.is_dir()

assert not NPZ_PATH.exists(), (
    f"Existing output requires forensic audit: {NPZ_PATH}"
)
assert not CSV_PATH.exists(), (
    f"Existing output requires forensic audit: {CSV_PATH}"
)

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    plan = json.load(handle)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert plan["manifest_sha256"] == EXPECTED_HASHES[MANIFEST_PATH]
assert plan["image_count"] == 156
assert plan["source_risk_group_count"] == 134
assert plan["test_access_permitted"] is False
assert plan["randomization_seeds"] == [1042, 2042, 3042]

assert set(protocol["randomization"]["stages"]["R1"]) == {"head"}
assert SEED in protocol["randomization"]["randomization_seeds"]

print("\nFROZEN INPUT INTEGRITY")
print("Manifest SHA256: PASS")
print("Execution plan SHA256: PASS")
print("Randomization protocol SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Batch 001-008 NPZ/CSV SHA256: PASS")

# ============================================================
# 2. Select exactly 16 new validation images
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique
assert manifest["split"].eq("val").all()
assert manifest["image_id"].tolist() == sorted(
    manifest["image_id"].tolist()
)

batch = manifest.iloc[BATCH_START:BATCH_END].copy()

assert len(batch) == 16
assert batch["image_id"].is_unique
assert batch["true_grade"].isin([1, 2, 3, 4]).all()

image_ids = batch["image_id"].tolist()
grades = batch["true_grade"].astype(int).tolist()
risk_groups = batch["risk_group_id"].astype(str).tolist()

assert all(
    value.strip() not in {"", "nan", "None"}
    for value in risk_groups
)

previous_ids = []
previous_groups = []

for batch_index, number in enumerate(PREVIOUS_HASHES):
    previous_path = (
        OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.npz"
    )

    with np.load(previous_path, allow_pickle=False) as archive:
        old_ids = list(map(str, archive["image_ids"]))
        old_groups = list(map(str, archive["risk_group_ids"]))

        expected_slice = manifest.iloc[
            batch_index * 16:(batch_index + 1) * 16
        ]

        assert old_ids == expected_slice["image_id"].tolist()
        assert old_groups == expected_slice["risk_group_id"].tolist()

        previous_ids.extend(old_ids)
        previous_groups.extend(old_groups)

assert len(previous_ids) == 128
assert len(set(previous_ids)) == 128
assert set(image_ids).isdisjoint(previous_ids)

overlapping_groups = sorted(
    set(risk_groups) & set(previous_groups)
)

cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})
assert cohort["image_id"].is_unique

candidate_flags = (
    cohort.set_index("image_id")
    .loc[image_ids, "primary_ma_candidate"]
    .astype(str)
    .str.strip()
    .str.lower()
)

assert candidate_flags.isin(["true", "1"]).all()

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: i for i, image_id in enumerate(reference_ids)
    }

    assert set(image_ids).issubset(positions)

    reference_cams = np.stack([
        archive["retfound_cam"][positions[image_id]]
        .astype(np.float32)
        for image_id in image_ids
    ])

assert reference_cams.shape == (16, 224, 224)
assert np.isfinite(reference_cams).all()

print("\nBATCH COHORT")
print("Images:", len(image_ids))
print("Distinct source-risk groups:", len(set(risk_groups)))
print("Image ID overlap with previous batches: 0")
print("Risk groups overlapping previous batches:", len(overlapping_groups))
print("Validation-only selection: PASS")
print("Original CAM ID alignment: PASS")
print("First image ID:", image_ids[0])
print("Last image ID:", image_ids[-1])

if overlapping_groups:
    print("Previously observed risk groups:")
    for group in overlapping_groups:
        print(group)

# ============================================================
# 3. Decode only Batch 009 validation images
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensors = []

for _, row in batch.iterrows():
    image_path = Path(row["resolved_image_path"])

    assert image_path.is_file(), row["image_id"]

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    assert tensor.shape == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all().item()

    input_tensors.append(tensor)

print("Images decoded:", len(input_tensors))

# ============================================================
# 4. Load the original RETFound checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

load_result = model.load_state_dict(
    source_state,
    strict=True
)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

# ============================================================
# 5. Use the same frozen Grad-CAM method
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(
    activation_hook
)


def calculate_cam(cpu_tensor, true_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(cpu_tensor.to(DEVICE))

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, true_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2), keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()
    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12
        cam /= float(cam.max())

        assert np.isfinite(cam).all()

    captured.clear()
    return output_logits, cam, raw_max


# ============================================================
# 6. Reproduce R0, then apply the frozen R1 initialization
# ============================================================

r0_logits = []
r0_cams = []
r1_logits = []
r1_cams = []
r1_raw_maxima = []

try:
    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, _ = calculate_cam(cpu_tensor, grade)

        assert cam is not None
        r0_logits.append(logits)
        r0_cams.append(cam)

    baseline_metrics = [
        compare_cams(cam, reference)
        for cam, reference in zip(r0_cams, reference_cams)
    ]

    baseline_pearsons = [
        item["pearson"] for item in baseline_metrics
    ]

    baseline_maes = [
        item["mae"] for item in baseline_metrics
    ]

    assert np.isfinite(baseline_pearsons).all()
    assert min(baseline_pearsons) > 0.999

    print("\nR0 REPRODUCTION")
    print(
        "Minimum reference CAM Pearson:",
        round(float(min(baseline_pearsons)), 6)
    )
    print(
        "Mean reference CAM MAE:",
        round(float(np.mean(baseline_maes)), 8)
    )
    print("Baseline CAM reproduction: PASS")

    # Reset the random seed once before R1 head initialization.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        torch.nn.init.zeros_(model.head.bias)

    changed_names = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(current, source_state[name]):
            changed_names.append(name)

    assert set(changed_names) == {
        "head.weight",
        "head.bias"
    }

    print("\nR1 PARAMETER INTEGRITY")
    print("Changed tensors:", len(changed_names))
    print("Expected changed tensors: 2")
    print("Unauthorized changes: 0")
    print("Seed:", SEED)

    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, raw_max = calculate_cam(
            cpu_tensor, grade
        )

        r1_logits.append(logits)
        r1_cams.append(cam)
        r1_raw_maxima.append(raw_max)

finally:
    hook.remove()
    captured.clear()

    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Build image-level metrics and arrays
# ============================================================

rows = []
stored_r1_cams = []

for i, image_id in enumerate(image_ids):
    metrics = compare_cams(r0_cams[i], r1_cams[i])

    rows.append({
        "model": "RETFound",
        "seed": SEED,
        "stage": "R1",
        "batch": BATCH_NUMBER,
        "image_id": image_id,
        "true_grade": grades[i],
        "risk_group_id": risk_groups[i],
        "degenerate_cam": r1_cams[i] is None,
        "raw_max": r1_raw_maxima[i],
        "r0_r1_pearson": metrics["pearson"],
        "r0_r1_spearman": metrics["spearman"],
        "r0_r1_ssim": metrics["ssim"],
        "r0_r1_mae": metrics["mae"],
        "r0_r1_logit_l2": float(
            np.linalg.norm(r1_logits[i] - r0_logits[i])
        )
    })

    stored_r1_cams.append(
        np.zeros((224, 224), dtype=np.float32)
        if r1_cams[i] is None
        else r1_cams[i].astype(np.float32)
    )

results = pd.DataFrame(rows)

r0_cam_array = np.stack(r0_cams).astype(np.float32)
r1_cam_array = np.stack(stored_r1_cams).astype(np.float32)
r0_logit_array = np.stack(r0_logits).astype(np.float32)
r1_logit_array = np.stack(r1_logits).astype(np.float32)

assert len(results) == 16
assert results["image_id"].is_unique
assert r0_cam_array.shape == (16, 224, 224)
assert r1_cam_array.shape == (16, 224, 224)
assert r0_logit_array.shape == (16, 5)
assert r1_logit_array.shape == (16, 5)

for array in [
    r0_cam_array,
    r1_cam_array,
    r0_logit_array,
    r1_logit_array
]:
    assert np.isfinite(array).all()

# ============================================================
# 8. Serialize and verify NPZ and CSV before publication
# ============================================================

npz_buffer = io.BytesIO()

np.savez_compressed(
    npz_buffer,
    image_ids=np.asarray(image_ids, dtype=str),
    true_grades=np.asarray(grades, dtype=np.int64),
    risk_group_ids=np.asarray(risk_groups, dtype=str),
    model=np.asarray("RETFound"),
    seed=np.asarray(SEED),
    stage=np.asarray("R1"),
    r0_cam=r0_cam_array,
    r1_cam=r1_cam_array,
    r0_logits=r0_logit_array,
    r1_logits=r1_logit_array,
    r1_raw_max=np.asarray(r1_raw_maxima, dtype=np.float64),
    r1_degenerate=results["degenerate_cam"].to_numpy(dtype=bool),
    manifest_sha256=np.asarray(EXPECTED_HASHES[MANIFEST_PATH]),
    execution_plan_sha256=np.asarray(EXPECTED_HASHES[PLAN_PATH])
)

npz_bytes = npz_buffer.getvalue()

csv_bytes = results.to_csv(
    index=False,
    float_format="%.10g"
).encode("utf-8")

with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)
    assert int(archive["seed"].item()) == SEED
    assert str(archive["stage"].item()) == "R1"
    assert str(archive["model"].item()) == "RETFound"

prepared_csv = pd.read_csv(
    io.BytesIO(csv_bytes),
    dtype={
        "image_id": str,
        "risk_group_id": str,
        "batch": str
    }
)

assert len(prepared_csv) == 16
assert prepared_csv["image_id"].tolist() == image_ids
assert prepared_csv["seed"].eq(SEED).all()
assert prepared_csv["stage"].eq("R1").all()
assert prepared_csv["batch"].eq(BATCH_NUMBER).all()

print("\nIN-MEMORY ARCHIVE VERIFICATION")
print("NPZ serialization: PASS")
print("CSV serialization: PASS")

# ============================================================
# 9. Publish without overwriting any previous output
# ============================================================

assert not NPZ_PATH.exists()
assert not CSV_PATH.exists()

# Retain serialized bytes in memory for interrupted-save recovery.
npz_hash = publish_exclusively(NPZ_PATH, npz_bytes)
csv_hash = publish_exclusively(CSV_PATH, csv_bytes)

# ============================================================
# 10. Independently reload and verify the published files
# ============================================================

with np.load(NPZ_PATH, allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids
    assert np.array_equal(archive["r0_cam"], r0_cam_array)
    assert np.array_equal(archive["r1_cam"], r1_cam_array)
    assert np.array_equal(archive["r0_logits"], r0_logit_array)
    assert np.array_equal(archive["r1_logits"], r1_logit_array)
    assert int(archive["seed"].item()) == SEED
    assert str(archive["stage"].item()) == "R1"
    assert str(archive["model"].item()) == "RETFound"

saved_csv = pd.read_csv(
    CSV_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str,
        "batch": str
    }
)

assert len(saved_csv) == 16
assert saved_csv["image_id"].tolist() == image_ids
assert saved_csv["seed"].eq(SEED).all()
assert saved_csv["stage"].eq("R1").all()
assert saved_csv["model"].eq("RETFound").all()
assert saved_csv["batch"].eq(BATCH_NUMBER).all()

assert sha256_file(NPZ_PATH) == npz_hash
assert sha256_file(CSV_PATH) == csv_hash

# ============================================================
# 11. Report results and output fingerprints
# ============================================================

print("\nBATCH RESULTS")
print("Model: RETFound")
print("Seed:", SEED)
print("Stage: R1")
print("Batch:", BATCH_NUMBER)
print("Images evaluated:", len(saved_csv))
print(
    "Degenerate CAMs:",
    int(saved_csv["degenerate_cam"].sum())
)

for column in [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]:
    print(
        f"Mean {column}:",
        round(float(saved_csv[column].mean()), 6)
    )

print("\nPER-IMAGE RESULTS")
print(
    saved_csv[
        [
            "image_id",
            "true_grade",
            "risk_group_id",
            "degenerate_cam",
            "r0_r1_pearson",
            "r0_r1_spearman",
            "r0_r1_ssim",
            "r0_r1_mae"
        ]
    ].to_string(index=False)
)

print("\nARCHIVE INTEGRITY")
print("NPZ path:", NPZ_PATH)
print("NPZ SHA256:", npz_hash)
print("CSV path:", CSV_PATH)
print("CSV SHA256:", csv_hash)
print("Batch 001-008 SHA256 verified: PASS")
print("Batch 009 archive round-trip: PASS")
print("Existing files overwritten: False")

print("\nRESEARCH INTEGRITY")
print("Frozen input files modified: False")
print("Validation images decoded:", len(input_tensors))
print("Test images accessed: 0")
print("New batch files created: 2")

print("\nSTEP 10.45 EXECUTION COMPLETE")
print("Production batch 009: PENDING REVIEW")


# %% [cell 388] Experiment 10 - Step 10.46

# ============================================================
# Experiment 10 - Step 10.46
# RETFound R1 Production Batch 010
# Seed 1042 | Final validation manifest rows 144:156
# Standalone Colab cell | Read-only frozen inputs
# ============================================================

from pathlib import Path
import gc
import hashlib
import io
import json

import cv2
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from timm.layers import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"
OUTPUT_DIR = EXP10 / "full_validation_batches"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"
PLAN_PATH = EXP10 / "Experiment10_Full_Execution_Plan_V1.json"
PROTOCOL_PATH = EXP10 / "Experiment10_Randomization_Protocol_V1.json"
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
CHECKPOINT_PATH = (
    ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"
)
REFERENCE_PATH = (
    ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"
)

SEED = 1042
BATCH_START = 144
BATCH_END = 156
EXPECTED_BATCH_SIZE = 12
BATCH_NUMBER = "010"
BATCH_NAME = f"RETFound_Seed{SEED}_R1_Batch{BATCH_NUMBER}"

NPZ_PATH = OUTPUT_DIR / f"{BATCH_NAME}.npz"
CSV_PATH = OUTPUT_DIR / f"{BATCH_NAME}.csv"

EXPECTED_HASHES = {
    MANIFEST_PATH:
        "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5",
    PLAN_PATH:
        "da52de5513de274a0b91c99f7060049a1f77db4c6dc2a788263607e6d8168590",
    PROTOCOL_PATH:
        "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda",
    INDEX_PATH:
        "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3",
    CHECKPOINT_PATH:
        "557902976006b63f3246f6bef594cf119165b105d79dbe1fd0d098b2f7318bf8"
}

PREVIOUS_HASHES = {
    "001": (
        "4f71b19be85c52bcdce8f0d628fbc34d0171f9b91170204fa1c0b4d6b6a28c43",
        "529f23beed5a47c67b0a545ce32330ddcec1973ee93eb2d8ea315c88ab7d0918"
    ),
    "002": (
        "946cc0bd797bf43f8d857cd5618ab4976aefbcbad96e0d6a2606365a30c1deab",
        "ca444fde5920a09f5fc64479d09d0919945d741cdeeb24ffac5ca8341f88089a"
    ),
    "003": (
        "ebb83b4f68c13e6bcb2f079290c7846707ff3e6f76cdacfa264a38beab9ff9c6",
        "4a53c0c75cb040d8309413d934599f11515cd10e1498d636735f8cf87d60772f"
    ),
    "004": (
        "5a64e2f7ab7c515c1acdcbbc8bfaa4aaa509344a0ad2f99a8e1430b7dc176e62",
        "8d176e808a8979c750f7b060d0070b60e4d8cf7d956ed8fd2bf2cc63b54dc717"
    ),
    "005": (
        "f21a22e43dce2863ca64fe80066bc81e3a1368982b1076b602fa79506bd64a1b",
        "e1313fce88897aa043c5d5a2a0cba4b9445c408e4cea4bc5114e1393b27ddfe5"
    ),
    "006": (
        "341661bb0f5eb18980296e8e0e068b7c8ca25f25bbcdd9f2e21162b4681e61e6",
        "f492ec4f5f2bbe198ba4dddea640bb1b9fac3d5f68ecefed2dee44e9046619f5"
    ),
    "007": (
        "35ef3b851f6c9e7feb76b3eab1201132c6ea1b9f64135ff50007d76f67eab3a8",
        "8760cc8daf4920d31b9a526a32b9e704d82dfb04f1ae480d72bf740dad75a0b3"
    ),
    "008": (
        "7464b02a141c197c0a6b419e5412ea98410bd02e7da04b1c92bd32608b47c0c2",
        "755a1c3e6b9f751cffe614f74fb63e81e2bf46f3a42b44fbcf6a6db8b7bb61b7"
    ),
    "009": (
        "793f4bfc32c4b819bb4fa5d9f3ae26b747a5ce2a5bfd969947b7696baeb552f7",
        "e14f90a4608d7c15a2b4cbceb63fb3023fd4322c98f401fa32a0bb26cd80f5fe"
    )
}

DEVICE = torch.device("cuda")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def compare_cams(first, second):
    if first is None or second is None:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "ssim": np.nan,
            "mae": np.nan
        }

    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    ac = a - a.mean()
    bc = b - b.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12 else np.nan
    )

    return {
        "pearson": pearson,
        "spearman": float(spearmanr(a, b).statistic),
        "ssim": float(
            structural_similarity(first, second, data_range=1.0)
        ),
        "mae": float(np.mean(np.abs(a - b)))
    }


def publish_exclusively(path, payload):
    if path.exists():
        raise FileExistsError(
            f"Existing output requires forensic review: {path}"
        )

    expected_hash = hashlib.sha256(payload).hexdigest()

    # Exclusive creation prevents accidental overwrites.
    # An interrupted Google Drive write is not an atomic commit.
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()

    actual_hash = sha256_file(path)

    assert actual_hash == expected_hash, (
        f"Post-write SHA256 mismatch: {path}"
    )

    return actual_hash


print("EXPERIMENT 10.46 - RETFOUND R1 FINAL BATCH 010")
print("Seed:", SEED)
print("Batch interval:", BATCH_START, BATCH_END)

# ============================================================
# 1. Verify frozen inputs and the previous nine batches
# ============================================================

assert torch.cuda.is_available(), "CUDA is required."

for path, expected_hash in EXPECTED_HASHES.items():
    assert path.is_file(), f"Missing frozen input: {path}"
    assert sha256_file(path) == expected_hash, (
        f"Frozen input SHA256 mismatch: {path}"
    )

for number, (expected_npz, expected_csv) in PREVIOUS_HASHES.items():
    old_npz = (
        OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.npz"
    )
    old_csv = (
        OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.csv"
    )

    assert old_npz.is_file() and old_csv.is_file()
    assert sha256_file(old_npz) == expected_npz, str(old_npz)
    assert sha256_file(old_csv) == expected_csv, str(old_csv)

assert REFERENCE_PATH.is_file()
assert COHORT_PATH.is_file()
assert OUTPUT_DIR.is_dir()

assert not NPZ_PATH.exists(), (
    f"Existing output requires forensic review: {NPZ_PATH}"
)
assert not CSV_PATH.exists(), (
    f"Existing output requires forensic review: {CSV_PATH}"
)

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    plan = json.load(handle)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert plan["manifest_sha256"] == EXPECTED_HASHES[MANIFEST_PATH]
assert plan["image_count"] == 156
assert plan["source_risk_group_count"] == 134
assert plan["test_access_permitted"] is False
assert plan["randomization_seeds"] == [1042, 2042, 3042]

assert SEED in protocol["randomization"]["randomization_seeds"]
assert set(protocol["randomization"]["stages"]["R1"]) == {"head"}

print("\nFROZEN INPUT INTEGRITY")
print("Manifest SHA256: PASS")
print("Execution plan SHA256: PASS")
print("Randomization protocol SHA256: PASS")
print("RETFound checkpoint SHA256: PASS")
print("Batch 001-009 NPZ/CSV SHA256: PASS")

# ============================================================
# 2. Verify the exact final 12-image cohort
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique
assert manifest["split"].eq("val").all()

assert manifest["image_id"].tolist() == sorted(
    manifest["image_id"].tolist()
)

batch = manifest.iloc[BATCH_START:BATCH_END].copy()

assert len(batch) == EXPECTED_BATCH_SIZE
assert batch["image_id"].is_unique
assert batch["true_grade"].isin([1, 2, 3, 4]).all()

image_ids = batch["image_id"].tolist()
grades = batch["true_grade"].astype(int).tolist()
risk_groups = batch["risk_group_id"].astype(str).tolist()

assert all(
    value.strip() not in {"", "nan", "None"}
    for value in risk_groups
)

previous_ids = []
previous_groups = []

for batch_index, number in enumerate(PREVIOUS_HASHES):
    previous_path = (
        OUTPUT_DIR / f"RETFound_Seed1042_R1_Batch{number}.npz"
    )

    with np.load(previous_path, allow_pickle=False) as archive:
        old_ids = list(map(str, archive["image_ids"]))
        old_groups = list(map(str, archive["risk_group_ids"]))

        expected_slice = manifest.iloc[
            batch_index * 16:(batch_index + 1) * 16
        ]

        assert old_ids == expected_slice["image_id"].tolist()
        assert old_groups == expected_slice["risk_group_id"].tolist()

        previous_ids.extend(old_ids)
        previous_groups.extend(old_groups)

assert len(previous_ids) == 144
assert len(set(previous_ids)) == 144
assert set(image_ids).isdisjoint(previous_ids)

all_ids = previous_ids + image_ids

assert len(all_ids) == 156
assert len(set(all_ids)) == 156
assert all_ids == manifest["image_id"].tolist()

overlapping_groups = sorted(
    set(risk_groups) & set(previous_groups)
)

all_groups = previous_groups + risk_groups

assert len(all_groups) == 156
assert len(set(all_groups)) == 134

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

assert cohort["image_id"].is_unique

candidate_flags = (
    cohort.set_index("image_id")
    .loc[image_ids, "primary_ma_candidate"]
    .astype(str)
    .str.strip()
    .str.lower()
)

assert candidate_flags.isin(["true", "1"]).all()

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))

    assert len(reference_ids) == 156
    assert len(set(reference_ids)) == 156

    positions = {
        image_id: i
        for i, image_id in enumerate(reference_ids)
    }

    assert set(image_ids).issubset(positions)

    reference_cams = np.stack([
        archive["retfound_cam"][positions[image_id]]
        .astype(np.float32)
        for image_id in image_ids
    ])

assert reference_cams.shape == (
    EXPECTED_BATCH_SIZE, 224, 224
)
assert np.isfinite(reference_cams).all()

print("\nFINAL BATCH COHORT")
print("Batch images:", len(image_ids))
print("Distinct batch source-risk groups:", len(set(risk_groups)))
print("Image ID overlap with earlier batches: 0")
print("Source-risk groups overlapping earlier batches:",
      len(overlapping_groups))
print("Total unique image IDs across all batches:", len(set(all_ids)))
print("Total unique source-risk groups:", len(set(all_groups)))
print("Frozen 156-image manifest coverage: PASS")
print("Validation-only selection: PASS")
print("Original CAM ID alignment: PASS")
print("First image ID:", image_ids[0])
print("Last image ID:", image_ids[-1])

if overlapping_groups:
    print("Previously observed risk groups:")
    for group in overlapping_groups:
        print(group)

# ============================================================
# 3. Decode only the final 12 validation images
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensors = []

for _, row in batch.iterrows():
    image_path = Path(row["resolved_image_path"])

    assert image_path.is_file(), row["image_id"]

    with Image.open(image_path) as image:
        tensor = preprocess(
            image.convert("RGB")
        ).unsqueeze(0)

    assert tensor.shape == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all().item()

    input_tensors.append(tensor)

assert len(input_tensors) == EXPECTED_BATCH_SIZE

print("Images decoded:", len(input_tensors))

# ============================================================
# 4. Load the frozen RETFound checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

source_state = checkpoint["model_state_dict"]
assert len(source_state) == 296

model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg",
    device="meta",
    weight_init="skip"
)

model = model.to_empty(device=DEVICE)

load_result = model.load_state_dict(
    source_state,
    strict=True
)

assert not load_result.missing_keys
assert not load_result.unexpected_keys

model.eval()
model.requires_grad_(False)

# ============================================================
# 5. Reconstruct the validated Grad-CAM calculation
# ============================================================

captured = {}


def activation_hook(module, module_inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation


hook = model.blocks[23].register_forward_hook(
    activation_hook
)


def calculate_cam(cpu_tensor, true_grade):
    captured.clear()

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(cpu_tensor.to(DEVICE))

        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all().item()

        activation = captured["activation"]
        assert activation.shape == (1, 197, 1024)

        gradient = torch.autograd.grad(
            outputs=logits[0, true_grade],
            inputs=activation,
            retain_graph=False,
            create_graph=False
        )[0]

    features = (
        activation[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    gradients = (
        gradient[0, 1:, :]
        .detach().float()
        .transpose(0, 1)
        .reshape(1024, 14, 14)
    )

    weights = gradients.mean(
        dim=(1, 2), keepdim=True
    )

    raw = torch.relu(
        (weights * features).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw).all()

    raw_max = float(raw.max())

    output_logits = (
        logits[0].detach().float().cpu().numpy().copy()
    )

    if raw_max <= 1e-12:
        cam = None
    else:
        cam = cv2.resize(
            raw,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        assert float(cam.max()) > 1e-12

        cam /= float(cam.max())
        assert np.isfinite(cam).all()

    captured.clear()

    return output_logits, cam, raw_max


# ============================================================
# 6. Reproduce R0 and calculate R1 with frozen seed 1042
# ============================================================

r0_logits = []
r0_cams = []
r1_logits = []
r1_cams = []
r1_raw_maxima = []

try:
    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, _ = calculate_cam(
            cpu_tensor, grade
        )

        assert cam is not None

        r0_logits.append(logits)
        r0_cams.append(cam)

    baseline_metrics = [
        compare_cams(cam, reference)
        for cam, reference in zip(
            r0_cams, reference_cams
        )
    ]

    baseline_pearsons = [
        item["pearson"]
        for item in baseline_metrics
    ]

    baseline_maes = [
        item["mae"]
        for item in baseline_metrics
    ]

    assert np.isfinite(baseline_pearsons).all()
    assert min(baseline_pearsons) > 0.999

    print("\nR0 REPRODUCTION")
    print(
        "Minimum reference CAM Pearson:",
        round(float(min(baseline_pearsons)), 6)
    )
    print(
        "Mean reference CAM MAE:",
        round(float(np.mean(baseline_maes)), 8)
    )
    print("Baseline CAM reproduction: PASS")

    # Reinitialize only the R1 classification head.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        trunc_normal_(model.head.weight, std=0.02)
        torch.nn.init.zeros_(model.head.bias)

    changed_names = []

    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu()

        assert torch.isfinite(current).all().item(), name

        if not torch.equal(
            current, source_state[name]
        ):
            changed_names.append(name)

    assert set(changed_names) == {
        "head.weight",
        "head.bias"
    }

    print("\nR1 PARAMETER INTEGRITY")
    print("Changed tensors:", len(changed_names))
    print("Expected changed tensors: 2")
    print("Unauthorized changes: 0")
    print("Seed:", SEED)

    for cpu_tensor, grade in zip(input_tensors, grades):
        logits, cam, raw_max = calculate_cam(
            cpu_tensor, grade
        )

        r1_logits.append(logits)
        r1_cams.append(cam)
        r1_raw_maxima.append(raw_max)

finally:
    hook.remove()
    captured.clear()

    del model, checkpoint, source_state
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 7. Assemble 12 image-level records
# ============================================================

rows = []
stored_r1_cams = []

for i, image_id in enumerate(image_ids):
    metrics = compare_cams(
        r0_cams[i],
        r1_cams[i]
    )

    rows.append({
        "model": "RETFound",
        "seed": SEED,
        "stage": "R1",
        "batch": BATCH_NUMBER,
        "image_id": image_id,
        "true_grade": grades[i],
        "risk_group_id": risk_groups[i],
        "degenerate_cam": r1_cams[i] is None,
        "raw_max": r1_raw_maxima[i],
        "r0_r1_pearson": metrics["pearson"],
        "r0_r1_spearman": metrics["spearman"],
        "r0_r1_ssim": metrics["ssim"],
        "r0_r1_mae": metrics["mae"],
        "r0_r1_logit_l2": float(
            np.linalg.norm(
                r1_logits[i] - r0_logits[i]
            )
        )
    })

    stored_r1_cams.append(
        np.zeros(
            (224, 224), dtype=np.float32
        )
        if r1_cams[i] is None
        else r1_cams[i].astype(np.float32)
    )

results = pd.DataFrame(rows)

r0_cam_array = np.stack(r0_cams).astype(np.float32)
r1_cam_array = np.stack(stored_r1_cams).astype(np.float32)
r0_logit_array = np.stack(r0_logits).astype(np.float32)
r1_logit_array = np.stack(r1_logits).astype(np.float32)

assert len(results) == EXPECTED_BATCH_SIZE
assert results["image_id"].is_unique

assert r0_cam_array.shape == (12, 224, 224)
assert r1_cam_array.shape == (12, 224, 224)
assert r0_logit_array.shape == (12, 5)
assert r1_logit_array.shape == (12, 5)

for array in [
    r0_cam_array,
    r1_cam_array,
    r0_logit_array,
    r1_logit_array
]:
    assert np.isfinite(array).all()

# ============================================================
# 8. Serialize and validate NPZ and CSV in memory
# ============================================================

npz_buffer = io.BytesIO()

np.savez_compressed(
    npz_buffer,
    image_ids=np.asarray(image_ids, dtype=str),
    true_grades=np.asarray(grades, dtype=np.int64),
    risk_group_ids=np.asarray(risk_groups, dtype=str),
    model=np.asarray("RETFound"),
    seed=np.asarray(SEED),
    stage=np.asarray("R1"),
    r0_cam=r0_cam_array,
    r1_cam=r1_cam_array,
    r0_logits=r0_logit_array,
    r1_logits=r1_logit_array,
    r1_raw_max=np.asarray(
        r1_raw_maxima, dtype=np.float64
    ),
    r1_degenerate=results[
        "degenerate_cam"
    ].to_numpy(dtype=bool),
    manifest_sha256=np.asarray(
        EXPECTED_HASHES[MANIFEST_PATH]
    ),
    execution_plan_sha256=np.asarray(
        EXPECTED_HASHES[PLAN_PATH]
    )
)

npz_bytes = npz_buffer.getvalue()

csv_bytes = results.to_csv(
    index=False,
    float_format="%.10g"
).encode("utf-8")

with np.load(
    io.BytesIO(npz_bytes),
    allow_pickle=False
) as archive:

    assert list(map(str, archive["image_ids"])) == image_ids

    assert np.array_equal(
        archive["r0_cam"], r0_cam_array
    )
    assert np.array_equal(
        archive["r1_cam"], r1_cam_array
    )
    assert np.array_equal(
        archive["r0_logits"], r0_logit_array
    )
    assert np.array_equal(
        archive["r1_logits"], r1_logit_array
    )

    assert int(archive["seed"].item()) == SEED
    assert str(archive["stage"].item()) == "R1"
    assert str(archive["model"].item()) == "RETFound"

prepared_csv = pd.read_csv(
    io.BytesIO(csv_bytes),
    dtype={
        "image_id": str,
        "risk_group_id": str,
        "batch": str
    }
)

assert len(prepared_csv) == EXPECTED_BATCH_SIZE
assert prepared_csv["image_id"].tolist() == image_ids
assert prepared_csv["seed"].eq(SEED).all()
assert prepared_csv["stage"].eq("R1").all()
assert prepared_csv["batch"].eq(BATCH_NUMBER).all()

print("\nIN-MEMORY ARCHIVE VERIFICATION")
print("NPZ serialization: PASS")
print("CSV serialization: PASS")

# ============================================================
# 9. Publish Batch 010 without overwriting existing data
# ============================================================

assert OUTPUT_DIR.is_dir()
assert not NPZ_PATH.exists()
assert not CSV_PATH.exists()

# Retain both buffers for interrupted-publication recovery.
npz_hash = publish_exclusively(
    NPZ_PATH, npz_bytes
)
csv_hash = publish_exclusively(
    CSV_PATH, csv_bytes
)

# ============================================================
# 10. Independently verify the saved files
# ============================================================

with np.load(NPZ_PATH, allow_pickle=False) as archive:
    assert list(map(str, archive["image_ids"])) == image_ids

    assert np.array_equal(
        archive["r0_cam"], r0_cam_array
    )
    assert np.array_equal(
        archive["r1_cam"], r1_cam_array
    )
    assert np.array_equal(
        archive["r0_logits"], r0_logit_array
    )
    assert np.array_equal(
        archive["r1_logits"], r1_logit_array
    )

    assert int(archive["seed"].item()) == SEED
    assert str(archive["stage"].item()) == "R1"
    assert str(archive["model"].item()) == "RETFound"

saved_csv = pd.read_csv(
    CSV_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str,
        "batch": str
    }
)

assert len(saved_csv) == EXPECTED_BATCH_SIZE
assert saved_csv["image_id"].tolist() == image_ids
assert saved_csv["seed"].eq(SEED).all()
assert saved_csv["stage"].eq("R1").all()
assert saved_csv["model"].eq("RETFound").all()
assert saved_csv["batch"].eq(BATCH_NUMBER).all()

assert sha256_file(NPZ_PATH) == npz_hash
assert sha256_file(CSV_PATH) == csv_hash

# ============================================================
# 11. Report final production batch
# ============================================================

print("\nBATCH RESULTS")
print("Model: RETFound")
print("Seed:", SEED)
print("Stage: R1")
print("Batch:", BATCH_NUMBER)
print("Images evaluated:", len(saved_csv))

print(
    "Degenerate CAMs:",
    int(saved_csv["degenerate_cam"].sum())
)

for column in [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]:
    print(
        f"Mean {column}:",
        round(
            float(saved_csv[column].mean()), 6
        )
    )

print("\nPER-IMAGE RESULTS")
print(
    saved_csv[
        [
            "image_id",
            "true_grade",
            "risk_group_id",
            "degenerate_cam",
            "r0_r1_pearson",
            "r0_r1_spearman",
            "r0_r1_ssim",
            "r0_r1_mae"
        ]
    ].to_string(index=False)
)

print("\nARCHIVE INTEGRITY")
print("NPZ path:", NPZ_PATH)
print("NPZ SHA256:", npz_hash)
print("CSV path:", CSV_PATH)
print("CSV SHA256:", csv_hash)
print("Batch 001-009 hashes unchanged: PASS")
print("Batch 010 archive round-trip: PASS")
print("Existing files overwritten: False")

print("\nFINAL R1 COVERAGE")
print("Image IDs represented:", len(all_ids))
print("Unique image IDs:", len(set(all_ids)))
print("Source-risk groups:", len(set(all_groups)))
print("Frozen cohort coverage: 156 / 156")
print("All images from Validation: PASS")

print("\nRESEARCH INTEGRITY")
print("Frozen input files modified: False")
print("Validation images decoded:", len(input_tensors))
print("Test images accessed: 0")
print("New batch files created: 2")

print("\nSTEP 10.46 EXECUTION COMPLETE")
print("Production batch 010: PENDING REVIEW")
print(
    "RETFound Seed1042 R1 full coverage: "
    "PENDING FINAL CROSS-BATCH AUDIT"
)


# %% [cell 389] Experiment 10 - Step 10.47

# ============================================================
# Experiment 10 - Step 10.47
# Full Cross-Batch Audit: RETFound / Seed 1042 / R1
# 156 validation images | 10 archived batches
# Read-only | No GPU inference | No image decoding
# ============================================================

from pathlib import Path
from collections import Counter
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

ROOT = DR_RESEARCH
EXP10 = ROOT / "experiment10_results"
BATCH_DIR = EXP10 / "full_validation_batches"

MANIFEST_PATH = EXP10 / "Experiment10_Full_Validation_Manifest_V1.csv"
PLAN_PATH = EXP10 / "Experiment10_Full_Execution_Plan_V1.json"
PROTOCOL_PATH = EXP10 / "Experiment10_Randomization_Protocol_V1.json"
REFERENCE_PATH = ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"

FROZEN_HASHES = {
    MANIFEST_PATH: "73275c9393f740948ec3992dbce73d044e3dcc116c615f4f02b44aeecd1170f5",
    PLAN_PATH: "da52de5513de274a0b91c99f7060049a1f77db4c6dc2a788263607e6d8168590",
    PROTOCOL_PATH: "efe9965c035a6406ce0a95c274f7d712e0c506a4dfdb5bffaa988bd05a1aebda"
}

BATCH_HASHES = {
    "001": ("4f71b19be85c52bcdce8f0d628fbc34d0171f9b91170204fa1c0b4d6b6a28c43", "529f23beed5a47c67b0a545ce32330ddcec1973ee93eb2d8ea315c88ab7d0918"),
    "002": ("946cc0bd797bf43f8d857cd5618ab4976aefbcbad96e0d6a2606365a30c1deab", "ca444fde5920a09f5fc64479d09d0919945d741cdeeb24ffac5ca8341f88089a"),
    "003": ("ebb83b4f68c13e6bcb2f079290c7846707ff3e6f76cdacfa264a38beab9ff9c6", "4a53c0c75cb040d8309413d934599f11515cd10e1498d636735f8cf87d60772f"),
    "004": ("5a64e2f7ab7c515c1acdcbbc8bfaa4aaa509344a0ad2f99a8e1430b7dc176e62", "8d176e808a8979c750f7b060d0070b60e4d8cf7d956ed8fd2bf2cc63b54dc717"),
    "005": ("f21a22e43dce2863ca64fe80066bc81e3a1368982b1076b602fa79506bd64a1b", "e1313fce88897aa043c5d5a2a0cba4b9445c408e4cea4bc5114e1393b27ddfe5"),
    "006": ("341661bb0f5eb18980296e8e0e068b7c8ca25f25bbcdd9f2e21162b4681e61e6", "f492ec4f5f2bbe198ba4dddea640bb1b9fac3d5f68ecefed2dee44e9046619f5"),
    "007": ("35ef3b851f6c9e7feb76b3eab1201132c6ea1b9f64135ff50007d76f67eab3a8", "8760cc8daf4920d31b9a526a32b9e704d82dfb04f1ae480d72bf740dad75a0b3"),
    "008": ("7464b02a141c197c0a6b419e5412ea98410bd02e7da04b1c92bd32608b47c0c2", "755a1c3e6b9f751cffe614f74fb63e81e2bf46f3a42b44fbcf6a6db8b7bb61b7"),
    "009": ("793f4bfc32c4b819bb4fa5d9f3ae26b747a5ce2a5bfd969947b7696baeb552f7", "e14f90a4608d7c15a2b4cbceb63fb3023fd4322c98f401fa32a0bb26cd80f5fe"),
    "010": ("ca813571beedc79b91ba74544f5bdc6d7306cacee15fd158b5f0528831690866", "5cecac643dcc147b530a46a13ba8e8d94d00fa8a0624f458caf2b51c9ec9ea8f")
}

METRIC_COLUMNS = [
    "r0_r1_pearson",
    "r0_r1_spearman",
    "r0_r1_ssim",
    "r0_r1_mae",
    "r0_r1_logit_l2"
]


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def cam_metrics(a, b):
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()

    ac = a64 - a64.mean()
    bc = b64 - b64.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)

    pearson = (
        float(np.dot(ac, bc) / denominator)
        if denominator > 1e-12
        else np.nan
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rank_corr = float(spearmanr(a64, b64).statistic)

    return {
        "r0_r1_pearson": pearson,
        "r0_r1_spearman": rank_corr,
        "r0_r1_ssim": float(
            structural_similarity(a, b, data_range=1.0)
        ),
        "r0_r1_mae": float(np.mean(np.abs(a64 - b64)))
    }


print("EXPERIMENT 10.47 - RETFOUND SEED1042 R1 FULL AUDIT")

# ============================================================
# 1. Verify frozen research artifacts
# ============================================================

for path, expected_hash in FROZEN_HASHES.items():
    assert path.is_file(), f"Missing frozen file: {path}"
    assert sha256_file(path) == expected_hash, (
        f"Frozen file hash mismatch: {path}"
    )

with PLAN_PATH.open("r", encoding="utf-8") as handle:
    plan = json.load(handle)

with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
    protocol = json.load(handle)

assert plan["manifest_sha256"] == FROZEN_HASHES[MANIFEST_PATH]
assert plan["image_count"] == 156
assert plan["source_risk_group_count"] == 134
assert plan["test_access_permitted"] is False
assert 1042 in protocol["randomization"]["randomization_seeds"]
assert set(protocol["randomization"]["stages"]["R1"]) == {"head"}

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(manifest) == 156
assert manifest["image_id"].is_unique
assert manifest["split"].eq("val").all()
assert manifest["image_id"].tolist() == sorted(
    manifest["image_id"].tolist()
)
assert manifest["risk_group_id"].nunique() == 134

print("\nFROZEN INPUT AUDIT")
print("Manifest SHA256: PASS")
print("Execution plan SHA256: PASS")
print("Randomization protocol SHA256: PASS")
print("Frozen validation images:", len(manifest))
print("Frozen source-risk groups:", manifest["risk_group_id"].nunique())

# ============================================================
# 2. Load and verify frozen reference CAMs
# ============================================================

assert REFERENCE_PATH.is_file()

with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
    reference_ids = list(map(str, archive["image_ids"]))
    reference_cams = archive["retfound_cam"].astype(np.float32)

assert len(reference_ids) == 156
assert len(set(reference_ids)) == 156
assert set(reference_ids) == set(manifest["image_id"])

assert reference_cams.shape == (156, 224, 224)
assert np.isfinite(reference_cams).all()

reference_lookup = {
    image_id: index for index, image_id in enumerate(reference_ids)
}

# ============================================================
# 3. Independently audit each archived batch
# ============================================================

all_rows = []
all_ids = []
all_groups = []
r0_reference_pearsons = []
r0_reference_maes = []
maximum_metric_differences = {key: 0.0 for key in METRIC_COLUMNS}
batch_summaries = []

for batch_index, (number, hashes) in enumerate(BATCH_HASHES.items()):
    npz_path = (
        BATCH_DIR
        / f"RETFound_Seed1042_R1_Batch{number}.npz"
    )
    csv_path = (
        BATCH_DIR
        / f"RETFound_Seed1042_R1_Batch{number}.csv"
    )

    assert npz_path.is_file(), str(npz_path)
    assert csv_path.is_file(), str(csv_path)

    assert sha256_file(npz_path) == hashes[0], str(npz_path)
    assert sha256_file(csv_path) == hashes[1], str(csv_path)

    start = batch_index * 16
    stop = min(start + 16, 156)

    frozen_slice = manifest.iloc[start:stop]
    expected_count = stop - start

    assert expected_count == (12 if number == "010" else 16)

    with np.load(npz_path, allow_pickle=False) as archive:
        ids = list(map(str, archive["image_ids"]))
        grades = archive["true_grades"].astype(int).tolist()
        groups = list(map(str, archive["risk_group_ids"]))

        assert str(archive["model"].item()) == "RETFound"
        assert int(archive["seed"].item()) == 1042
        assert str(archive["stage"].item()) == "R1"

        assert str(archive["manifest_sha256"].item()) == (
            FROZEN_HASHES[MANIFEST_PATH]
        )
        assert str(archive["execution_plan_sha256"].item()) == (
            FROZEN_HASHES[PLAN_PATH]
        )

        r0_cam = archive["r0_cam"].copy()
        r1_cam = archive["r1_cam"].copy()
        r0_logits = archive["r0_logits"].copy()
        r1_logits = archive["r1_logits"].copy()
        raw_max = archive["r1_raw_max"].copy()
        degenerate = archive["r1_degenerate"].astype(bool).copy()

    assert len(ids) == expected_count
    assert ids == frozen_slice["image_id"].tolist()
    assert grades == frozen_slice["true_grade"].astype(int).tolist()
    assert groups == frozen_slice["risk_group_id"].astype(str).tolist()

    assert r0_cam.shape == (expected_count, 224, 224)
    assert r1_cam.shape == (expected_count, 224, 224)
    assert r0_logits.shape == (expected_count, 5)
    assert r1_logits.shape == (expected_count, 5)
    assert raw_max.shape == (expected_count,)
    assert degenerate.shape == (expected_count,)

    for array in [r0_cam, r1_cam, r0_logits, r1_logits, raw_max]:
        assert np.isfinite(array).all(), f"Nonfinite array in {number}"

    assert (r0_cam >= -1e-5).all()
    assert (r0_cam <= 1.0 + 1e-5).all()
    assert (r1_cam >= -1e-5).all()
    assert (r1_cam <= 1.0 + 1e-5).all()
    assert (raw_max >= 0).all()

    csv = pd.read_csv(
        csv_path,
        dtype={
            "image_id": str,
            "risk_group_id": str,
            "batch": str
        }
    )

    assert len(csv) == expected_count
    assert csv["image_id"].tolist() == ids
    assert csv["true_grade"].astype(int).tolist() == grades
    assert csv["risk_group_id"].tolist() == groups
    assert csv["model"].eq("RETFound").all()
    assert csv["seed"].eq(1042).all()
    assert csv["stage"].eq("R1").all()
    assert csv["batch"].str.zfill(3).eq(number).all()

    csv_degenerate = (
        csv["degenerate_cam"]
        .astype(str).str.lower().eq("true")
        .to_numpy()
    )

    assert np.array_equal(csv_degenerate, degenerate)

    assert np.allclose(
        csv["raw_max"].to_numpy(dtype=float),
        raw_max,
        rtol=1e-8,
        atol=1e-10
    )

    for i, image_id in enumerate(ids):
        # Preserve degenerate CAMs as missing spatial metrics.
        if degenerate[i]:
            assert np.max(np.abs(r1_cam[i])) <= 1e-12

            calculated = {
                "r0_r1_pearson": np.nan,
                "r0_r1_spearman": np.nan,
                "r0_r1_ssim": np.nan,
                "r0_r1_mae": np.nan
            }
        else:
            assert raw_max[i] > 1e-12
            assert np.max(r1_cam[i]) > 1e-12

            calculated = cam_metrics(
                r0_cam[i],
                r1_cam[i]
            )

        calculated["r0_r1_logit_l2"] = float(
            np.linalg.norm(r1_logits[i] - r0_logits[i])
        )

        for column in METRIC_COLUMNS:
            stored_value = float(csv.iloc[i][column])
            new_value = float(calculated[column])

            assert np.isclose(
                stored_value,
                new_value,
                rtol=1e-6,
                atol=1e-7,
                equal_nan=True
            ), (
                f"Metric mismatch: {number}, "
                f"{image_id}, {column}"
            )

            if np.isfinite(stored_value) and np.isfinite(new_value):
                difference = abs(stored_value - new_value)
                maximum_metric_differences[column] = max(
                    maximum_metric_differences[column],
                    difference
                )

        reference = reference_cams[
            reference_lookup[image_id]
        ]

        reference_metrics = cam_metrics(
            r0_cam[i], reference
        )

        ref_pearson = reference_metrics["r0_r1_pearson"]
        ref_mae = reference_metrics["r0_r1_mae"]

        assert np.isfinite(ref_pearson)
        assert ref_pearson > 0.999

        r0_reference_pearsons.append(ref_pearson)
        r0_reference_maes.append(ref_mae)

        all_rows.append(csv.iloc[i].to_dict())

    all_ids.extend(ids)
    all_groups.extend(groups)

    batch_summaries.append({
        "batch": number,
        "images": expected_count,
        "degenerate": int(degenerate.sum()),
        "mean_pearson": float(csv["r0_r1_pearson"].mean())
    })

    print(
        f"Batch {number}: PASS | "
        f"images={expected_count} | "
        f"degenerate={int(degenerate.sum())}"
    )

# ============================================================
# 4. Verify complete cohort coverage and clustering
# ============================================================

assert len(all_ids) == 156
assert len(set(all_ids)) == 156
assert all_ids == manifest["image_id"].tolist()

assert len(all_groups) == 156
assert all_groups == manifest["risk_group_id"].astype(str).tolist()
assert len(set(all_groups)) == 134

group_counts = Counter(all_groups)

multi_image_groups = {
    group: count
    for group, count in group_counts.items()
    if count > 1
}

assert len(multi_image_groups) == 16
assert max(group_counts.values()) == 3

combined = pd.DataFrame(all_rows)

assert len(combined) == 156
assert combined["image_id"].is_unique
assert combined["model"].eq("RETFound").all()
assert combined["seed"].eq(1042).all()
assert combined["stage"].eq("R1").all()

degenerate_flags = (
    combined["degenerate_cam"]
    .astype(str).str.lower().eq("true")
)

assert combined["true_grade"].astype(int).value_counts().sort_index().to_dict() == {
    1: 28,
    2: 50,
    3: 62,
    4: 16
}

# ============================================================
# 5. Final whole-cohort summary and audit decision
# ============================================================

print("\nFULL COHORT INTEGRITY")
print("Archived batches:", len(BATCH_HASHES))
print("Verified batch files:", len(BATCH_HASHES) * 2)
print("Total image records:", len(combined))
print("Unique image IDs:", combined["image_id"].nunique())
print("Missing image IDs:", len(set(manifest["image_id"]) - set(all_ids)))
print("Duplicate image IDs:", len(all_ids) - len(set(all_ids)))
print("Unique source-risk groups:", len(group_counts))
print("Multi-image source-risk groups:", len(multi_image_groups))
print("Largest source-risk group:", max(group_counts.values()))
print("Degenerate CAMs:", int(degenerate_flags.sum()))

print("\nGRADE DISTRIBUTION")
print(
    combined["true_grade"]
    .astype(int)
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nINDEPENDENT METRIC RECOMPUTATION")
for metric, difference in maximum_metric_differences.items():
    print(f"Maximum absolute difference {metric}: {difference:.12g}")

print("\nR0 REFERENCE REPRODUCTION")
print(
    "Minimum reference Pearson:",
    round(float(np.min(r0_reference_pearsons)), 6)
)
print(
    "Mean reference MAE:",
    round(float(np.mean(r0_reference_maes)), 8)
)

print("\nFULL 156-IMAGE DESCRIPTIVE SUMMARY")
for metric in METRIC_COLUMNS:
    values = pd.to_numeric(combined[metric], errors="coerce")
    print(
        f"{metric}: "
        f"mean={values.mean():.6f}, "
        f"median={values.median():.6f}, "
        f"sd={values.std(ddof=1):.6f}, "
        f"valid_n={values.notna().sum()}"
    )

print("\nBATCH-LEVEL AUDIT")
print(pd.DataFrame(batch_summaries).to_string(index=False))

print("\nFINAL SCIENTIFIC AUDIT")
print("Frozen manifest coverage: PASS")
print("Cross-batch image alignment: PASS")
print("Cross-batch source-risk alignment: PASS")
print("All 20 archive SHA256 values: PASS")
print("All image-level metric checks: PASS")
print("R0 frozen CAM reference check: PASS")
print("Test image access in this cell: 0")
print("Files created or modified: 0")
print("Model inference in this cell: False")
print(
    "Batch 008 provenance limitation: "
    "original runtime log not independently verified"
)
print("External image-byte rehashing in this cell: NOT PERFORMED")
print("Cross-seed or cross-model inference: NOT PERFORMED")

print("\nSTEP 10.47 EXECUTION COMPLETE")
print(
    "RETFound Seed1042 R1 full-cohort archive audit: "
    "PASS - SUBJECT TO PROVENANCE LIMITATIONS"
)
