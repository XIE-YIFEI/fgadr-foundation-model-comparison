# Experiment 03 - training index, data loaders and the EfficientNet-B0 baseline
#
# Source: FGADR_MA_Mask_Audit.ipynb (master notebook, 392 cells), cells 79-87, run step by step in Google Colab (L4 GPU).
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

# %% [cell 79] Experiment 03 - Step 3.1A

from pathlib import Path
import hashlib
import importlib.util
import platform
import pandas as pd
import torch
import cv2

# ============================================================
# Experiment 03 - Step 3.1A
# Model Environment and Frozen Split Preflight
# ============================================================

ROOT = DR_RESEARCH
RESULTS = ROOT / "experiment02_results"

SPLIT_PATH = RESULTS / "FGADR_Split_V2_Frozen.csv"
MANIFEST_PATH = RESULTS / "FGADR_Image_SHA256_Manifest.csv"

EXPECTED_SPLIT_SHA256 = (
    "a3d83242847834ef38c6d1de8c9566556f33792041370807c709a5af4370d49f"
)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

print("EXPERIMENT 03 - ENVIRONMENT PREFLIGHT")
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory (GiB):",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    )

for package in ["timm", "torchvision", "sklearn", "safetensors"]:
    print(f"{package} available:", importlib.util.find_spec(package) is not None)

assert SPLIT_PATH.is_file(), "Frozen split is unavailable."
assert MANIFEST_PATH.is_file(), "Image manifest is unavailable."

actual_sha256 = sha256_file(SPLIT_PATH)
print("\nFrozen split SHA256:", actual_sha256)

assert actual_sha256 == EXPECTED_SPLIT_SHA256, (
    "Frozen split checksum mismatch."
)

split_df = pd.read_csv(SPLIT_PATH, dtype={"image_id": str})
manifest = pd.read_csv(MANIFEST_PATH, dtype={"image_id": str})

merged = split_df.merge(
    manifest[["image_id", "local_path", "source_path"]],
    on="image_id",
    how="left",
    validate="one_to_one",
    indicator=True
)

assert len(merged) == 1842
assert merged["_merge"].eq("both").all()
assert split_df["image_id"].is_unique

local_available = sum(
    Path(path).is_file()
    for path in merged["local_path"]
)

source_available = sum(
    Path(path).is_file()
    for path in merged["source_path"]
)

print("\nDATA PREFLIGHT")
print("Frozen split rows:", len(split_df))
print("Manifest join status: PASS")
print("Local images available:", local_available)
print("Drive source images available:", source_available)
print("Split counts:")
print(split_df["split"].value_counts().to_string())

print("\nSTEP 3.1A COMPLETED")


# %% [cell 80]

from pathlib import Path

pass  # Google Drive mount removed

root = DR_RESEARCH
results = root / "experiment02_results"

split_path = results / "FGADR_Split_V2_Frozen.csv"
manifest_path = results / "FGADR_Image_SHA256_Manifest.csv"

print("\nGOOGLE DRIVE ACCESS CHECK")
print("Project directory:", root.is_dir())
print("Results directory:", results.is_dir())
print("Frozen split available:", split_path.is_file())
print("Image manifest available:", manifest_path.is_file())

assert split_path.is_file(), "Frozen split is still unavailable."
assert manifest_path.is_file(), "Image manifest is still unavailable."

print("\nDRIVE ACCESS RESTORED")


# %% [cell 81]

from pathlib import Path
import pandas as pd

root = DR_RESEARCH
manifest = pd.read_csv(
    root / "experiment02_results" / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

local_available = sum(
    Path(path).is_file()
    for path in manifest["local_path"]
)

source_available = sum(
    Path(path).is_file()
    for path in manifest["source_path"]
)

print("IMAGE CACHE CHECK")
print("Expected images:", len(manifest))
print("Local images available:", local_available)
print("Drive source images available:", source_available)

if local_available == len(manifest):
    print("Local image cache: READY")
elif source_available == len(manifest):
    print("Local image cache: NEEDS RESTORATION")
else:
    print("Image paths: REQUIRE INVESTIGATION")


# %% [cell 82] Experiment 03 - Step 3.1B

from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import hashlib
import shutil
import cv2

# ============================================================
# Experiment 03 - Step 3.1B
# Restore and Verify Local Image Cache
# ============================================================

ROOT = DR_RESEARCH
MANIFEST_PATH = (
    ROOT / "experiment02_results" / "FGADR_Image_SHA256_Manifest.csv"
)

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "sha256": str}
)

assert len(manifest) == 1842
assert manifest["image_id"].is_unique

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()

restored = 0
already_verified = 0
failures = []

for row in tqdm(
    manifest.itertuples(index=False),
    total=len(manifest),
    desc="Restoring training images"
):
    source = Path(row.source_path)
    destination = Path(row.local_path)

    try:
        if not source.is_file():
            raise FileNotFoundError(
                f"Source image missing: {source}"
            )

        if destination.is_file():
            if sha256_file(destination) == row.sha256:
                already_verified += 1
                continue

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copyfile(source, destination)

        if sha256_file(destination) != row.sha256:
            raise RuntimeError(
                f"SHA256 mismatch: {row.image_id}"
            )

        restored += 1

    except Exception as exc:
        failures.append({
            "image_id": row.image_id,
            "error": str(exc)
        })

print("\nSTEP 3.1B - IMAGE CACHE RESTORATION")
print("Expected images:", len(manifest))
print("Restored images:", restored)
print("Previously verified images:", already_verified)
print("Failures:", len(failures))

assert len(failures) == 0
assert restored + already_verified == 1842

# Verify that representative images can be decoded.
sample = manifest.iloc[
    [0, 250, 500, 1000, 1500, 1841]
]

for row in sample.itertuples(index=False):
    image = cv2.imread(
        str(row.local_path),
        cv2.IMREAD_COLOR
    )

    assert image is not None, (
        f"Image decoding failed: {row.image_id}"
    )

    assert image.shape[:2] == (1280, 1280), (
        f"Unexpected image dimensions: {row.image_id}"
    )

print("Sample image decoding: PASS")
print("SHA256 integrity: PASS")
print("\nSTEP 3.1B COMPLETED")


# %% [cell 83] Experiment 03 - Step 3.1C

from pathlib import Path
import pandas as pd
import hashlib
import json
import torch
import cv2

# ============================================================
# Experiment 03 - Step 3.1C
# Frozen Split and Training Data Verification
# ============================================================

ROOT = DR_RESEARCH
RESULTS_DIR = ROOT / "experiment02_results"
EXPERIMENT_DIR = ROOT / "experiment03_results"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_PATH = RESULTS_DIR / "FGADR_Split_V2_Frozen.csv"
MANIFEST_PATH = RESULTS_DIR / "FGADR_Image_SHA256_Manifest.csv"

EXPECTED_SHA256 = (
    "a3d83242847834ef38c6d1de8c9566556f33792041370807c709a5af4370d49f"
)

def calculate_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

# 1. Verify frozen split identity
assert SPLIT_PATH.is_file()
assert MANIFEST_PATH.is_file()

actual_sha256 = calculate_sha256(SPLIT_PATH)
assert actual_sha256 == EXPECTED_SHA256, (
    "Frozen split checksum mismatch."
)

# 2. Load original metadata
split_df = pd.read_csv(
    SPLIT_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "sha256": str}
)

assert len(split_df) == 1842
assert len(manifest) == 1842
assert split_df["image_id"].is_unique
assert manifest["image_id"].is_unique
assert split_df["dr_grade"].notna().all()
assert manifest["dr_grade"].notna().all()

# 3. Verify original labels independently
split_grades = (
    split_df.set_index("image_id")["dr_grade"]
    .astype(int).sort_index()
)

source_grades = (
    manifest.set_index("image_id")["dr_grade"]
    .astype(int).sort_index()
)

assert split_grades.equals(source_grades)

# 4. Join image paths without duplicating label columns
path_columns = [
    "image_id",
    "local_path",
    "source_path",
    "sha256"
]

data = split_df.merge(
    manifest[path_columns],
    on="image_id",
    how="left",
    validate="one_to_one",
    indicator=True
)

assert len(data) == 1842
assert data["_merge"].eq("both").all()
assert data["local_path"].notna().all()
assert data["sha256"].notna().all()

data = data.drop(columns=["_merge"])

data["dr_grade"] = data["dr_grade"].astype(int)

assert set(data["dr_grade"]) == {0, 1, 2, 3, 4}
assert set(data["split"]) == {"train", "val", "test"}

# 5. Verify local image availability
available_images = sum(
    Path(path).is_file()
    for path in data["local_path"]
)

assert available_images == 1842

# 6. Verify risk groups do not cross splits
group_split_counts = data.groupby(
    "risk_group_id"
)["split"].nunique()

assert group_split_counts.max() == 1

group_sets = {
    name: set(
        data.loc[
            data["split"] == name,
            "risk_group_id"
        ]
    )
    for name in ["train", "val", "test"]
}

assert group_sets["train"].isdisjoint(group_sets["val"])
assert group_sets["train"].isdisjoint(group_sets["test"])
assert group_sets["val"].isdisjoint(group_sets["test"])

# 7. Verify expected split distributions
distribution = pd.crosstab(
    data["split"],
    data["dr_grade"]
).reindex(
    index=["train", "val", "test"],
    columns=[0, 1, 2, 3, 4],
    fill_value=0
)

expected_distribution = pd.DataFrame(
    [
        [71, 148, 417, 453, 201],
        [15, 32, 89, 97, 43],
        [15, 32, 89, 97, 43]
    ],
    index=["train", "val", "test"],
    columns=[0, 1, 2, 3, 4]
)

assert distribution.equals(expected_distribution)

# 8. Save verified training index without changing split assignments
OUTPUT_PATH = EXPERIMENT_DIR / "FGADR_Training_Index_V1.csv"

if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"Training index already exists: {OUTPUT_PATH}"
    )

data = data.sort_values("image_id").reset_index(drop=True)
data.to_csv(OUTPUT_PATH, index=False)

training_index_sha256 = calculate_sha256(OUTPUT_PATH)

# 9. Display final verification
print("STEP 3.1C - FROZEN DATA VERIFICATION")
print("Frozen split SHA256:", actual_sha256)
print("Images:", len(data))
print("Available local images:", available_images)
print("Unique image IDs:", data["image_id"].nunique())
print("Risk groups:", data["risk_group_id"].nunique())
print("Cross-split risk groups:", int((group_split_counts > 1).sum()))
print("Original label agreement: PASS")
print("GPU available:", torch.cuda.is_available())

print("\nVERIFIED DR DISTRIBUTION")
print(distribution.to_string())

print("\nDATA QUALITY FLAGS")
for column in [
    "group_grade_conflict",
    "exact_duplicate_label_conflict"
]:
    if column in data.columns:
        print(f"{column}: {int(data[column].sum())}")

print("\nTRAINING INDEX SHA256")
print(training_index_sha256)

print("\nOUTPUT FILE")
print(OUTPUT_PATH)

print("\nSTEP 3.1C COMPLETED")


# %% [cell 84] Experiment 03 - Step 3.2A

from pathlib import Path
import hashlib
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ============================================================
# Experiment 03 - Step 3.2A
# EfficientNet Dataset and DataLoader Smoke Test
# ============================================================

ROOT = DR_RESEARCH
INDEX_PATH = ROOT / "experiment03_results" / "FGADR_Training_Index_V1.csv"

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

assert INDEX_PATH.is_file()
assert file_sha256(INDEX_PATH) == EXPECTED_INDEX_SHA256

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

data = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(data) == 1842
assert data["image_id"].is_unique
assert set(data["dr_grade"].astype(int)) == {0, 1, 2, 3, 4}

# ImageNet normalization, suitable for the initial
# ImageNet-pretrained EfficientNet baseline.
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

train_frame = data[data["split"] == "train"].copy()
val_frame = data[data["split"] == "val"].copy()
test_frame = data[data["split"] == "test"].copy()

assert (
    len(train_frame),
    len(val_frame),
    len(test_frame)
) == (1290, 276, 276)

# Protect all known source-risk groups from crossing splits.
train_groups = set(train_frame["risk_group_id"])
val_groups = set(val_frame["risk_group_id"])
test_groups = set(test_frame["risk_group_id"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

train_dataset = FGADRClassificationDataset(
    train_frame, train_transform
)
val_dataset = FGADRClassificationDataset(
    val_frame, eval_transform
)
test_dataset = FGADRClassificationDataset(
    test_frame, eval_transform
)

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

# ============================================================
# Smoke test: train and validation only
# ============================================================

train_images, train_labels, train_ids = next(
    iter(train_loader)
)

val_images, val_labels, val_ids = next(
    iter(val_loader)
)

assert train_images.shape == (BATCH_SIZE, 3, 224, 224)
assert val_images.shape == (BATCH_SIZE, 3, 224, 224)

assert train_images.dtype == torch.float32
assert val_images.dtype == torch.float32

assert torch.isfinite(train_images).all()
assert torch.isfinite(val_images).all()

assert train_labels.dtype == torch.long
assert val_labels.dtype == torch.long

assert train_labels.min() >= 0
assert train_labels.max() <= 4
assert val_labels.min() >= 0
assert val_labels.max() <= 4

label_lookup = data.set_index("image_id")["dr_grade"].to_dict()

for ids, labels in [
    (train_ids, train_labels),
    (val_ids, val_labels)
]:
    for image_id, label in zip(ids, labels.tolist()):
        assert int(label_lookup[image_id]) == int(label)

# Validation preprocessing must be deterministic.
sample_a = val_dataset[0][0]
sample_b = val_dataset[0][0]
assert torch.equal(sample_a, sample_b)

print("STEP 3.2A - DATALOADER VALIDATION")
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images reserved:", len(test_dataset))
print("Train batch shape:", tuple(train_images.shape))
print("Validation batch shape:", tuple(val_images.shape))
print("Train batch labels:", train_labels.tolist())
print("Validation batch labels:", val_labels.tolist())
print("Image tensor dtype:", str(train_images.dtype))
print("Train tensor finite:", bool(torch.isfinite(train_images).all()))
print("Validation deterministic:", bool(torch.equal(sample_a, sample_b)))
print("Cross-split risk groups: 0")
print("Batch image-label agreement: PASS")
print("Frozen training index SHA256: PASS")

print("\nDATALOADER COUNTS")
print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Reserved test batches:", len(test_loader))

print("\nSTEP 3.2A COMPLETED")


# %% [cell 85] Experiment 03 - Step 3.3A

# ============================================================
# Experiment 03 - Step 3.3A
# EfficientNet-B0 Pretrained Model Smoke Test
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

ROOT = DR_RESEARCH
RESULTS_DIR = ROOT / "experiment03_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 5
LEARNING_RATE = 1e-4

assert torch.cuda.is_available(), "CUDA GPU is required."
assert len(train_loader.dataset) == 1290
assert len(val_loader.dataset) == 276
assert len(test_loader.dataset) == 276

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda")

# ============================================================
# 1. Load pretrained backbone and replace classifier
# ============================================================

print("STEP 3.3A - EFFICIENTNET MODEL SMOKE TEST")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("timm:", timm.__version__)

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=NUM_CLASSES
).to(device)

pretrained_cfg = model.pretrained_cfg

print("\nMODEL CONFIGURATION")
print("Architecture:", MODEL_NAME)
print("Output classes:", model.num_classes)
print("Pretrained source:", pretrained_cfg.get("url", ""))
print("Pretrained input size:", pretrained_cfg.get("input_size"))
print("Pretrained mean:", pretrained_cfg.get("mean"))
print("Pretrained std:", pretrained_cfg.get("std"))

assert model.num_classes == 5

# Check that the pretrained normalization matches Step 3.2A.
cfg_mean = pretrained_cfg.get("mean")
cfg_std = pretrained_cfg.get("std")

assert cfg_mean is not None and cfg_std is not None

assert np.allclose(cfg_mean, MEAN, atol=1e-6), (
    "Model mean differs from DataLoader normalization."
)

assert np.allclose(cfg_std, STD, atol=1e-6), (
    "Model std differs from DataLoader normalization."
)

parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
)

trainable_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Total parameters:", parameter_count)
print("Trainable parameters:", trainable_count)

# ============================================================
# 2. Prepare one training batch
# ============================================================

images, labels, image_ids = next(iter(train_loader))

images = images.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)

assert images.shape == (16, 3, 224, 224)
assert labels.shape == (16,)
assert torch.isfinite(images).all()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# ============================================================
# 3. Test forward and backward propagation
# ============================================================

model.train()
optimizer.zero_grad(set_to_none=True)

logits = model(images)

assert logits.shape == (16, NUM_CLASSES)
assert torch.isfinite(logits).all()

loss = criterion(logits, labels)

assert torch.isfinite(loss)

loss.backward()

classifier_parameters = list(model.get_classifier().parameters())

classifier_gradient_valid = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in classifier_parameters
)

assert classifier_gradient_valid, (
    "Classifier gradient is missing or non-finite."
)

gradient_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
    error_if_nonfinite=True
)

assert torch.isfinite(gradient_norm)

# Confirm the optimizer actually updates the classifier.
parameter_before = classifier_parameters[0].detach().clone()

optimizer.step()

parameter_after = classifier_parameters[0].detach()

parameter_change = float(
    (parameter_after - parameter_before)
    .abs().max().item()
)

assert parameter_change > 0, (
    "Optimizer did not update classifier parameters."
)

print("\nTRAINING BATCH TEST")
print("Input shape:", tuple(images.shape))
print("Logit shape:", tuple(logits.shape))
print("Initial batch loss:", round(float(loss.item()), 6))
print("Gradient norm:", round(float(gradient_norm.item()), 6))
print("Classifier parameter update:", parameter_change)
print("Backward propagation: PASS")
print("Optimizer update: PASS")

# ============================================================
# 4. Test validation forward pass without updating weights
# ============================================================

model.eval()

val_images, val_labels, val_ids = next(iter(val_loader))
val_images = val_images.to(device, non_blocking=True)
val_labels = val_labels.to(device, non_blocking=True)

with torch.inference_mode():
    val_logits = model(val_images)
    val_loss = criterion(val_logits, val_labels)
    val_probabilities = torch.softmax(val_logits, dim=1)

assert val_logits.shape == (16, NUM_CLASSES)
assert torch.isfinite(val_logits).all()
assert torch.isfinite(val_loss)
assert torch.allclose(
    val_probabilities.sum(dim=1),
    torch.ones(16, device=device),
    atol=1e-5
)

print("\nVALIDATION BATCH TEST")
print("Validation loss:", round(float(val_loss.item()), 6))
print("Probability shape:", tuple(val_probabilities.shape))
print("Softmax normalization: PASS")
print("Inference mode: PASS")

# ============================================================
# 5. Save smoke-test metadata (not model weights)
# ============================================================

report = {
    "experiment": "Experiment03_Step3.3A",
    "model": MODEL_NAME,
    "num_classes": NUM_CLASSES,
    "pretrained": True,
    "input_size": 224,
    "seed": SEED,
    "learning_rate_smoke_test": LEARNING_RATE,
    "batch_size": 16,
    "total_parameters": parameter_count,
    "trainable_parameters": trainable_count,
    "torch_version": torch.__version__,
    "timm_version": timm.__version__,
    "pretrained_cfg": {
        "architecture": str(pretrained_cfg.get("architecture")),
        "url": str(pretrained_cfg.get("url")),
        "hf_hub_id": str(pretrained_cfg.get("hf_hub_id")),
        "input_size": pretrained_cfg.get("input_size"),
        "mean": list(cfg_mean),
        "std": list(cfg_std)
    },
    "gradient_valid": bool(classifier_gradient_valid),
    "optimizer_update_valid": parameter_change > 0,
    "test_split_evaluated": False,
    "smoke_test_only": True,
    "created_utc": datetime.now(timezone.utc).isoformat()
}

report_path = RESULTS_DIR / "EfficientNet_B0_Smoke_Test.json"

with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nOUTPUT FILE:", report_path)
print("Test split evaluated: False")
print("Smoke-test weights should not be used for formal training.")

print("\nSTEP 3.3A COMPLETED")


# %% [cell 86] Experiment 03 - Step 3.4A

# ============================================================
# Experiment 03 - Step 3.4A
# EfficientNet-B0 Formal Baseline Training
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
from tqdm.auto import tqdm
import hashlib
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score
)

# ============================================================
# 1. Experiment configuration
# ============================================================

ROOT = DR_RESEARCH

EXPERIMENT_DIR = (
    ROOT / "experiment03_results" / "efficientnet_b0_baseline_v1"
)

INDEX_PATH = (
    ROOT / "experiment03_results" / "FGADR_Training_Index_V1.csv"
)

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

SEED = 42
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 5
EPOCHS = 25
MIN_EPOCHS = 8
PATIENCE = 6

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224

DEVICE = torch.device("cuda")

CHECKPOINT_PATH = EXPERIMENT_DIR / "best_model.pt"
HISTORY_PATH = EXPERIMENT_DIR / "training_history.csv"
CONFIG_PATH = EXPERIMENT_DIR / "training_config.json"

assert torch.cuda.is_available()
assert len(train_loader.dataset) == 1290
assert len(val_loader.dataset) == 276
assert len(test_loader.dataset) == 276

# Prevent accidental reuse or overwriting of an existing run.
assert not EXPERIMENT_DIR.exists(), (
    f"Experiment directory already exists: {EXPERIMENT_DIR}"
)

def calculate_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

assert calculate_sha256(INDEX_PATH) == EXPECTED_INDEX_SHA256

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Strict deterministic behavior may require additional
# environment configuration for some CUDA operations.
# Record determinism settings without claiming full
# bitwise reproducibility.

# ============================================================
# 2. Compute class weights from training labels only
# ============================================================

train_labels_array = train_frame["dr_grade"].astype(int).to_numpy()

class_counts = np.bincount(
    train_labels_array,
    minlength=NUM_CLASSES
)

assert class_counts.tolist() == [
    71, 148, 417, 453, 201
]

# Inverse-square-root weighting avoids excessively large
# weights for the smallest class.
class_weights = 1.0 / np.sqrt(class_counts.astype(np.float64))
class_weights = class_weights / class_weights.mean()

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("STEP 3.4A - EFFICIENTNET FORMAL TRAINING")
print("Training class counts:", class_counts.tolist())
print("Class weights:", np.round(class_weights, 4).tolist())
print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 3. Create a fresh ImageNet-pretrained model
# ============================================================

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

# ============================================================
# 4. Save training configuration
# ============================================================

config = {
    "experiment": "Experiment03_EfficientNet_B0_Baseline_V1",
    "architecture": MODEL_NAME,
    "pretrained": True,
    "pretrained_cfg": {
        "hf_hub_id": str(model.pretrained_cfg.get("hf_hub_id")),
        "url": str(model.pretrained_cfg.get("url")),
        "input_size": list(model.pretrained_cfg.get("input_size")),
        "mean": list(model.pretrained_cfg.get("mean")),
        "std": list(model.pretrained_cfg.get("std"))
    },
    "num_classes": NUM_CLASSES,
    "image_size": IMAGE_SIZE,
    "batch_size": train_loader.batch_size,
    "epochs": EPOCHS,
    "minimum_epochs": MIN_EPOCHS,
    "early_stopping_patience": PATIENCE,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "scheduler": "CosineAnnealingLR",
    "loss": "InverseSqrtWeightedCrossEntropy",
    "class_counts": class_counts.tolist(),
    "class_weights": class_weights.tolist(),
    "checkpoint_selection": "validation_macro_f1",
    "seed": SEED,
    "torch_version": torch.__version__,
    "timm_version": timm.__version__,
    "training_index_sha256": EXPECTED_INDEX_SHA256,
    "test_set_used_for_training": False,
    "created_utc": datetime.now(timezone.utc).isoformat()
}

with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)

# ============================================================
# 5. Training and validation functions
# ============================================================

def run_train_epoch(model, loader):
    model.train()

    total_loss = 0.0
    all_true = []
    all_pred = []

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels, _ in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True
        )

        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = logits.argmax(dim=1)

        all_true.extend(labels.cpu().tolist())
        all_pred.extend(predictions.detach().cpu().tolist())

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(
            all_true, all_pred,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0
        )
    }


def run_validation_epoch(model, loader):
    model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.inference_mode():
        for images, labels, _ in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss.")

            total_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(predictions.cpu().tolist())

    qwk = cohen_kappa_score(
        all_true,
        all_pred,
        labels=list(range(NUM_CLASSES)),
        weights="quadratic"
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(
            all_true, all_pred,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0
        ),
        "qwk": float(qwk),
        "per_class_recall": recall_score(
            all_true, all_pred,
            labels=list(range(NUM_CLASSES)),
            average=None,
            zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            all_true, all_pred,
            labels=list(range(NUM_CLASSES))
        ).tolist()
    }

# ============================================================
# 6. Formal training loop
# ============================================================

history = []
best_macro_f1 = -1.0
best_epoch = -1
epochs_without_improvement = 0

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    train_metrics = run_train_epoch(
        model, train_loader
    )

    val_metrics = run_validation_epoch(
        model, val_loader
    )

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    record = {
        "epoch": epoch,
        "learning_rate": current_lr,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_qwk": val_metrics["qwk"],
        "epoch_seconds": time.time() - epoch_start
    }

    for grade, value in enumerate(val_metrics["per_class_recall"]):
        record[f"val_recall_dr{grade}"] = float(value)

    history.append(record)

    pd.DataFrame(history).to_csv(
        HISTORY_PATH,
        index=False
    )

    improved = val_metrics["macro_f1"] > best_macro_f1 + 1e-6

    if improved:
        best_macro_f1 = val_metrics["macro_f1"]
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_macro_f1,
                "val_qwk": val_metrics["qwk"],
                "val_confusion_matrix": val_metrics["confusion_matrix"],
                "architecture": MODEL_NAME,
                "num_classes": NUM_CLASSES,
                "training_index_sha256": EXPECTED_INDEX_SHA256
            },
            CHECKPOINT_PATH
        )
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Train F1: {train_metrics['macro_f1']:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val F1: {val_metrics['macro_f1']:.4f} | "
        f"Val QWK: {val_metrics['qwk']:.4f} | "
        f"Best: {best_macro_f1:.4f}"
    )

    if (
        epoch >= MIN_EPOCHS
        and epochs_without_improvement >= PATIENCE
    ):
        print(
            f"\nEarly stopping triggered at epoch {epoch}."
        )
        break

# ============================================================
# 7. Training summary
# ============================================================

history_df = pd.DataFrame(history)

assert CHECKPOINT_PATH.is_file()
assert HISTORY_PATH.is_file()
assert len(history_df) >= 1

print("\nEFFICIENTNET TRAINING SUMMARY")
print("Epochs completed:", len(history_df))
print("Best epoch:", best_epoch)
print("Best validation macro-F1:", round(best_macro_f1, 6))
print(
    "Best recorded validation QWK:",
    round(
        float(history_df["val_qwk"].max()),
        6
    )
)
print(
    "Total training time (minutes):",
    round((time.time() - start_time) / 60.0, 2)
)
print("Test dataset evaluated: False")

print("\nLAST 10 EPOCHS")
print(
    history_df[
        [
            "epoch",
            "train_loss",
            "train_macro_f1",
            "val_loss",
            "val_macro_f1",
            "val_qwk"
        ]
    ].tail(10).round(5).to_string(index=False)
)

print("\nOUTPUT FILES")
print(CHECKPOINT_PATH)
print(HISTORY_PATH)
print(CONFIG_PATH)

print("\nSTEP 3.4A COMPLETED")


# %% [cell 87] Experiment 03 - Step 3.4B

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import timm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
    precision_score
)

# ============================================================
# Experiment 03 - Step 3.4B
# Best Checkpoint Validation Audit
# ============================================================

ROOT = DR_RESEARCH
RUN_DIR = ROOT / "experiment03_results" / "efficientnet_b0_baseline_v1"

CHECKPOINT_PATH = RUN_DIR / "best_model.pt"
PREDICTIONS_PATH = RUN_DIR / "validation_predictions.csv"
METRICS_PATH = RUN_DIR / "validation_audit_metrics.json"

INDEX_PATH = ROOT / "experiment03_results" / "FGADR_Training_Index_V1.csv"

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_SHA256

for output in [PREDICTIONS_PATH, METRICS_PATH]:
    if output.exists():
        raise FileExistsError(
            f"Validation audit output already exists: {output}"
        )

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

assert checkpoint["architecture"] == "efficientnet_b0"
assert checkpoint["num_classes"] == 5
assert checkpoint["training_index_sha256"] == EXPECTED_INDEX_SHA256

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=5
)

model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model = model.to(device)
model.eval()

index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})
val_index = index[index["split"] == "val"].copy()

assert len(val_index) == 276
assert len(val_loader.dataset) == 276

metadata = val_index.set_index("image_id")

records = []

with torch.inference_mode():
    for images, labels, image_ids in val_loader:
        logits = model(images.to(device))
        probabilities = torch.softmax(
            logits, dim=1
        ).cpu().numpy()

        predictions = probabilities.argmax(axis=1)

        for i, image_id in enumerate(image_ids):
            assert image_id in metadata.index

            true_grade = int(labels[i].item())

            assert true_grade == int(
                metadata.loc[image_id, "dr_grade"]
            )

            record = {
                "image_id": image_id,
                "true_grade": true_grade,
                "predicted_grade": int(predictions[i]),
                "correct": bool(predictions[i] == true_grade),
                "risk_group_id": metadata.loc[
                    image_id, "risk_group_id"
                ],
                "group_grade_conflict": bool(
                    metadata.loc[image_id, "group_grade_conflict"]
                )
            }

            for grade in range(5):
                record[f"prob_dr{grade}"] = float(
                    probabilities[i, grade]
                )

            records.append(record)

pred = pd.DataFrame(records).sort_values(
    "image_id"
).reset_index(drop=True)

assert len(pred) == 276
assert pred["image_id"].is_unique
assert set(pred["image_id"]) == set(val_index["image_id"])
assert np.allclose(
    pred[[f"prob_dr{i}" for i in range(5)]].sum(axis=1),
    1.0,
    atol=1e-5
)

y_true = pred["true_grade"].to_numpy()
y_pred = pred["predicted_grade"].to_numpy()

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        zero_division=0
    )
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3, 4]
)

macro_f1 = float(
    f1_score(y_true, y_pred, average="macro")
)

qwk = float(
    cohen_kappa_score(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        weights="quadratic"
    )
)

assert abs(macro_f1 - checkpoint["val_macro_f1"]) < 1e-5
assert abs(qwk - checkpoint["val_qwk"]) < 1e-5

# Referable DR: grades 2, 3, and 4.
referable_true = y_true >= 2
referable_pred = y_pred >= 2

binary_cm = confusion_matrix(
    referable_true,
    referable_pred,
    labels=[False, True]
)

tn, fp, fn, tp = binary_cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# This comparison is descriptive, not a causal estimate.
subgroup_records = []

for flag in [False, True]:
    subset = pred[
        pred["group_grade_conflict"] == flag
    ]

    subgroup_records.append({
        "group_conflict": flag,
        "images": len(subset),
        "accuracy": accuracy_score(
            subset["true_grade"],
            subset["predicted_grade"]
        ),
        "macro_f1": f1_score(
            subset["true_grade"],
            subset["predicted_grade"],
            labels=[0, 1, 2, 3, 4],
            average="macro",
            zero_division=0
        )
    })

subgroups = pd.DataFrame(subgroup_records)

metrics = {
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "images": len(pred),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "macro_f1": macro_f1,
    "qwk": qwk,
    "referable_dr_definition": "grade >= 2",
    "referable_sensitivity": float(sensitivity),
    "referable_specificity": float(specificity),
    "referable_tp": int(tp),
    "referable_tn": int(tn),
    "referable_fp": int(fp),
    "referable_fn": int(fn),
    "test_set_evaluated": False
}

pred.to_csv(PREDICTIONS_PATH, index=False)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("STEP 3.4B - VALIDATION CHECKPOINT AUDIT")
print("Checkpoint epoch:", checkpoint["epoch"])
print("Validation images:", len(pred))
print("Accuracy:", round(metrics["accuracy"], 6))
print("Macro-F1:", round(macro_f1, 6))
print("QWK:", round(qwk, 6))
print("Checkpoint metric reproducibility: PASS")

print("\nPER-CLASS PERFORMANCE")
class_report = pd.DataFrame({
    "DR grade": [0, 1, 2, 3, 4],
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "support": support
})
print(class_report.round(5).to_string(index=False))

print("\nCONFUSION MATRIX")
print("Rows: true grades; columns: predicted grades")
print(pd.DataFrame(
    cm,
    index=[f"True DR{i}" for i in range(5)],
    columns=[f"Pred DR{i}" for i in range(5)]
).to_string())

print("\nREFERABLE DR PERFORMANCE")
print("Definition: DR grade >= 2")
print("Sensitivity:", round(sensitivity, 6))
print("Specificity:", round(specificity, 6))
print("TP:", int(tp), "TN:", int(tn))
print("FP:", int(fp), "FN:", int(fn))

print("\nRISK-GROUP CONFLICT STRATA")
print(subgroups.round(5).to_string(index=False))

print("\nOUTPUT FILES")
print(PREDICTIONS_PATH)
print(METRICS_PATH)

print("\nSTEP 3.4B COMPLETED")
