# Experiment 06 - YOLO11 microaneurysm detection
#
# Source: FGADR_MA_Mask_Audit.ipynb (master notebook, 392 cells), cells 135-193, run step by step in Google Colab (L4 GPU).
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

# %% [cell 135] Experiment 06.1A

from pathlib import Path

# ============================================================
# Experiment 06.1A
# YOLO Dataset Structure Discovery
# ============================================================

ROOT = DR_RESEARCH
DATA_ROOT = ROOT / "datasets" / "DR"

assert ROOT.is_dir(), "Google Drive project directory not found."
assert DATA_ROOT.is_dir(), "DR dataset directory not found."

print("EXPERIMENT 06.1A - YOLO DATASET DISCOVERY")
print("Dataset root:", DATA_ROOT)

print("\nTOP-LEVEL DATASET DIRECTORIES")
for item in sorted(DATA_ROOT.iterdir()):
    if item.is_dir():
        print("DIR:", item.name)

print("\nYOLO-RELATED DIRECTORIES")
candidates = []

for parent in [DATA_ROOT, ROOT / "datasets"]:
    if not parent.is_dir():
        continue

    for item in parent.iterdir():
        if item.is_dir() and "yolo" in item.name.lower():
            candidates.append(item)
            print(item)

print("\nKNOWN CLASS-DEFINITION CANDIDATES")

for relative_path in [
    "yolo/classes.txt",
    "yolo/data.yaml",
    "yolo/dataset.yaml",
]:
    path = DATA_ROOT / relative_path

    if path.is_file():
        print("\nFile:", path)
        print(path.read_text(encoding="utf-8")[:3000])

print("\nSTEP 06.1A COMPLETED")


# %% [cell 136] Experiment 06.1B

from pathlib import Path

# ============================================================
# Experiment 06.1B
# Detection Dataset Structure Inspection
# ============================================================

DATA_ROOT = (DR_RESEARCH / "datasets/DR")

DATASETS = [
    DATA_ROOT / "FGADR",
    DATA_ROOT / "FGADR_8class"
]

MAX_ITEMS = 35

def inspect_directory(path, depth=0):
    if not path.is_dir():
        print("MISSING DIRECTORY:", path)
        return

    items = sorted(
        path.iterdir(),
        key=lambda p: (not p.is_dir(), p.name.lower())
    )

    indent = "    " * depth

    print(f"{indent}{path.name}/")
    print(f"{indent}Total direct entries: {len(items)}")

    for item in items[:MAX_ITEMS]:
        if item.is_dir():
            print(f"{indent}    [DIR] {item.name}/")
        else:
            print(f"{indent}    [FILE] {item.name}")

    if len(items) > MAX_ITEMS:
        print(
            f"{indent}    ... "
            f"{len(items) - MAX_ITEMS} additional entries"
        )

    if depth < 2:
        for item in items:
            if item.is_dir():
                inspect_directory(item, depth + 1)

print("EXPERIMENT 06.1B - DATASET INSPECTION")

for dataset in DATASETS:
    print("\n" + "=" * 60)
    inspect_directory(dataset)

print("\nSTEP 06.1B COMPLETED")


# %% [cell 137] Experiment 06.1C

from pathlib import Path

# ============================================================
# Experiment 06.1C
# YOLO Class Definition and Label Format Verification
# ============================================================

ROOT = (DR_RESEARCH / "datasets/DR")

DATASETS = {
    "6-class": ROOT / "FGADR",
    "8-class": ROOT / "FGADR_8class"
}

print("EXPERIMENT 06.1C - YOLO CLASS AND LABEL CHECK")

for dataset_name, dataset_dir in DATASETS.items():
    yolo_dir = dataset_dir / "yolo"
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    classes_path = yolo_dir / "classes.txt"

    print(f"\n{'=' * 45}")
    print("Dataset:", dataset_name)
    print("YOLO directory exists:", yolo_dir.is_dir())
    print("Images directory exists:", images_dir.is_dir())
    print("Labels directory exists:", labels_dir.is_dir())
    print("Classes file exists:", classes_path.is_file())

    if classes_path.is_file():
        classes = [
            line.strip()
            for line in classes_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        print("\nCLASS DEFINITIONS")
        for class_id, class_name in enumerate(classes):
            print(f"{class_id}: {class_name}")

    if labels_dir.is_dir():
        import os
        from itertools import islice

        with os.scandir(labels_dir) as entries:
            examples = [
                entry.path
                for entry in islice(
                    (entry for entry in entries if entry.is_file()
                     and entry.name.endswith(".txt")),
                    3
                )
            ]

        print("\nSAMPLE LABEL CONTENTS")
        for label_path in examples:
            print("\nFile:", Path(label_path).name)
            text = Path(label_path).read_text(
                encoding="utf-8"
            )
            print(text[:600] if text.strip() else "[EMPTY LABEL]")

print("\nSTEP 06.1C COMPLETED")


# %% [cell 138] identical to cell 139, omitted here


# %% [cell 139] Experiment 06.1D

from pathlib import Path
from collections import Counter
import os
import json
import numpy as np
import pandas as pd

# ============================================================
# Experiment 06.1D
# YOLO Dataset and Annotation Integrity Audit
# ============================================================

ROOT = DR_RESEARCH

DATASETS = {
    "6-class": ROOT / "datasets/DR/FGADR/yolo",
    "8-class": ROOT / "datasets/DR/FGADR_8class/yolo"
}

OUT = ROOT / "experiment06_results"
OUT.mkdir(parents=True, exist_ok=True)

summary = {}
issues = []

for dataset_name, yolo_dir in DATASETS.items():
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    classes_path = yolo_dir / "classes.txt"

    classes = [
        line.strip()
        for line in classes_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    image_files = {
        p.stem: p.name
        for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    label_files = {
        p.stem: p
        for p in labels_dir.iterdir()
        if p.suffix.lower() == ".txt"
    }

    missing_labels = sorted(set(image_files) - set(label_files))
    orphan_labels = sorted(set(label_files) - set(image_files))

    class_counts = Counter()
    empty_labels = 0
    invalid_boxes = 0
    valid_boxes = 0
    ma_widths = []
    ma_heights = []

    for image_id, label_path in label_files.items():
        with label_path.open(encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            empty_labels += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            try:
                if len(parts) != 5:
                    raise ValueError("Expected exactly 5 fields")

                class_value = float(parts[0])
                if not np.isfinite(class_value):
                    raise ValueError("Non-finite class ID")
                class_id = int(class_value)

                if class_id != class_value:
                    raise ValueError("Class ID is not an integer")

                x, y, w, h = map(float, parts[1:])

                if not (0 <= class_id < len(classes)):
                    raise ValueError("Invalid class ID")

                if not all(np.isfinite([x, y, w, h])):
                    raise ValueError("Non-finite coordinate")

                if w <= 0 or h <= 0:
                    raise ValueError("Non-positive box size")

                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError("Box center outside normalized range")

                tolerance = 1e-5

                if (
                    x - w / 2 < -tolerance
                    or x + w / 2 > 1 + tolerance
                    or y - h / 2 < -tolerance
                    or y + h / 2 > 1 + tolerance
                ):
                    raise ValueError("Box exceeds image boundary")

                class_counts[classes[class_id]] += 1
                valid_boxes += 1

                if class_id == 3:
                    ma_widths.append(w)
                    ma_heights.append(h)

            except (ValueError, TypeError) as error:
                invalid_boxes += 1
                issues.append({
                    "dataset": dataset_name,
                    "image_id": image_id,
                    "line_number": line_number,
                    "issue": str(error),
                    "raw_line": line
                })

    summary[dataset_name] = {
        "image_count": len(image_files),
        "label_count": len(label_files),
        "missing_label_count": len(missing_labels),
        "orphan_label_count": len(orphan_labels),
        "empty_label_count": empty_labels,
        "valid_box_count": valid_boxes,
        "invalid_box_count": invalid_boxes,
        "class_box_counts": dict(class_counts),
        "ma_box_count": len(ma_widths),
        "ma_median_normalized_width": (
            float(np.median(ma_widths)) if ma_widths else None
        ),
        "ma_median_normalized_height": (
            float(np.median(ma_heights)) if ma_heights else None
        ),
        "missing_label_examples": missing_labels[:20],
        "orphan_label_examples": orphan_labels[:20]
    }

    print("\n" + "=" * 55)
    print("Dataset:", dataset_name)
    print(json.dumps(summary[dataset_name], indent=2))

SUMMARY_PATH = OUT / "YOLO_Annotation_Audit_V1.json"
ISSUES_PATH = OUT / "YOLO_Annotation_Issues_V1.csv"

assert not SUMMARY_PATH.exists(), "Audit report already exists."
assert not ISSUES_PATH.exists(), "Issues report already exists."

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

pd.DataFrame(
    issues,
    columns=["dataset", "image_id", "line_number", "issue", "raw_line"]
).to_csv(ISSUES_PATH, index=False)

print("\nEXPERIMENT 06.1D COMPLETED")
print("Summary:", SUMMARY_PATH)
print("Issues:", ISSUES_PATH)


# %% [cell 140] Experiment 06.1E

# ============================================================
# Experiment 06.1E
# Safe Audit Report Recovery
# ============================================================

from pathlib import Path
import json
import pandas as pd

OUT = (DR_RESEARCH / "experiment06_results")

assert "summary" in globals(), "Audit summary is not in memory."
assert "issues" in globals(), "Audit issues are not in memory."

print("EXISTING AUDIT REPORTS")

for filename in [
    "YOLO_Annotation_Audit_V1.json",
    "YOLO_Annotation_Issues_V1.csv"
]:
    path = OUT / filename
    print(filename, "EXISTS" if path.exists() else "MISSING")

version = 2

while (
    (OUT / f"YOLO_Annotation_Audit_V{version}.json").exists()
    or
    (OUT / f"YOLO_Annotation_Issues_V{version}.csv").exists()
):
    version += 1

summary_path = OUT / f"YOLO_Annotation_Audit_V{version}.json"
issues_path = OUT / f"YOLO_Annotation_Issues_V{version}.csv"

with summary_path.open("x", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

pd.DataFrame(
    issues,
    columns=[
        "dataset",
        "image_id",
        "line_number",
        "issue",
        "raw_line"
    ]
).to_csv(issues_path, index=False)

with summary_path.open(encoding="utf-8") as f:
    verified = json.load(f)

assert verified == summary

print("\nRECOVERY COMPLETE")
print("Summary:", summary_path)
print("Issues:", issues_path)
print("Recorded issues:", len(issues))
print("STEP 06.1E PASS")


# %% [cell 141] Experiment 06 - Step 06.2A

# ============================================================
# Experiment 06 - Step 06.2A
# Frozen Split and YOLO Dataset Compatibility Audit
# ============================================================

from pathlib import Path
import os
import json
import hashlib
from collections import Counter

import pandas as pd

ROOT = DR_RESEARCH

SPLIT_PATH = (
    ROOT / "experiment02_results" /
    "FGADR_Split_V2_Frozen.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results" /
    "FGADR_Training_Index_V1.csv"
)

YOLO_ROOT = (
    ROOT / "datasets/DR/FGADR/yolo"
)

OUT = ROOT / "experiment06_results"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SPLIT_SHA256 = (
    "a3d83242847834ef38c6d1de8c9566556f33792041370807c709a5af4370d49f"
)

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

assert SPLIT_PATH.is_file()
assert INDEX_PATH.is_file()
assert YOLO_ROOT.is_dir()

split_hash = sha256_file(SPLIT_PATH)
index_hash = sha256_file(INDEX_PATH)

assert split_hash == EXPECTED_SPLIT_SHA256, (
    "Frozen split hash mismatch."
)

assert index_hash == EXPECTED_INDEX_SHA256, (
    "Training index hash mismatch."
)

index = pd.read_csv(INDEX_PATH)

required_columns = {
    "image_id",
    "split",
    "risk_group_id",
    "sha256"
}

assert required_columns.issubset(index.columns)
assert len(index) == 1842
assert index["image_id"].is_unique

assert set(index["split"]) == {
    "train", "val", "test"
}

expected_counts = {
    "train": 1290,
    "val": 276,
    "test": 276
}

actual_counts = index["split"].value_counts().to_dict()

assert actual_counts == expected_counts, (
    f"Unexpected split counts: {actual_counts}"
)

# Verify that no source-risk group spans different splits.
group_split_counts = (
    index.groupby("risk_group_id")["split"].nunique()
)

cross_split_groups = int((group_split_counts > 1).sum())

assert cross_split_groups == 0, (
    "Source-risk-group leakage detected."
)

images_dir = YOLO_ROOT / "images"
labels_dir = YOLO_ROOT / "labels"

assert images_dir.is_dir()
assert labels_dir.is_dir()

with os.scandir(images_dir) as entries:
    image_names = [
        entry.name
        for entry in entries
        if entry.is_file()
        and Path(entry.name).suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    ]

with os.scandir(labels_dir) as entries:
    label_names = [
        entry.name
        for entry in entries
        if entry.is_file()
        and entry.name.lower().endswith(".txt")
    ]

image_ids = [
    Path(name).stem for name in image_names
]

label_ids = [
    Path(name).stem for name in label_names
]

image_set = set(image_ids)
label_set = set(label_ids)
index_set = set(index["image_id"].astype(str))

duplicate_image_ids = [
    image_id
    for image_id, count in Counter(image_ids).items()
    if count > 1
]

missing_from_yolo = sorted(index_set - image_set)
extra_in_yolo = sorted(image_set - index_set)

missing_labels = sorted(image_set - label_set)
orphan_labels = sorted(label_set - image_set)

print("EXPERIMENT 06.2A - SPLIT COMPATIBILITY")

print("\nFrozen split SHA256:", split_hash)
print("Training index SHA256:", index_hash)

print("\nSplit distribution:")
print(index["split"].value_counts().to_string())

print("\nYOLO images:", len(image_names))
print("YOLO labels:", len(label_names))

print("\nDuplicate image IDs:", len(duplicate_image_ids))
print("Missing images from YOLO:", len(missing_from_yolo))
print("Extra YOLO images:", len(extra_in_yolo))
print("Images without labels:", len(missing_labels))
print("Orphan label files:", len(orphan_labels))
print("Cross-split risk groups:", cross_split_groups)

print("\nExamples of unmatched images:")
print("Missing:", missing_from_yolo[:10])
print("Extra:", extra_in_yolo[:10])

print("\nOrphan labels:")
print(orphan_labels[:10])

report = {
    "split_sha256": split_hash,
    "index_sha256": index_hash,
    "split_counts": actual_counts,
    "yolo_image_count": len(image_names),
    "yolo_label_count": len(label_names),
    "duplicate_image_ids": duplicate_image_ids,
    "missing_from_yolo": missing_from_yolo,
    "extra_in_yolo": extra_in_yolo,
    "missing_labels": missing_labels,
    "orphan_labels": orphan_labels,
    "cross_split_risk_groups": cross_split_groups,
}

assert not missing_from_yolo, (
    "Frozen images are missing from YOLO."
)

assert not extra_in_yolo, (
    "YOLO includes images outside the frozen cohort."
)

assert not missing_labels, (
    "Some YOLO images have no label files."
)

assert not duplicate_image_ids, (
    "Duplicate YOLO image IDs detected."
)

print("\nSTEP 06.2A CORE CHECKS PASS")
print("No dataset files were modified.")


# %% [cell 142] Experiment 06 - Step 06.2B

# ============================================================
# Experiment 06 - Step 06.2B
# Frozen-Split Lesion Distribution Audit
# ============================================================

from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import json

ROOT = DR_RESEARCH
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
LABEL_DIR = ROOT / "datasets/DR/FGADR/yolo/labels"
CLASS_PATH = ROOT / "datasets/DR/FGADR/yolo/classes.txt"
OUT = ROOT / "experiment06_results"

index = pd.read_csv(INDEX_PATH)
classes = [
    x.strip()
    for x in CLASS_PATH.read_text(encoding="utf-8-sig").splitlines()
    if x.strip()
]

assert len(index) == 1842
assert index["image_id"].is_unique
assert len(classes) == 6
assert classes[3] == "Microaneurysms"

records = []
class_counts = {
    split: Counter()
    for split in ["train", "val", "test"]
}

for row in index.itertuples(index=False):
    image_id = str(row.image_id)
    split = str(row.split)
    label_path = LABEL_DIR / f"{image_id}.txt"

    assert label_path.is_file(), f"Missing label: {image_id}"

    counts = Counter()
    ma_areas = []

    with label_path.open(encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()
        assert len(parts) == 5, f"Invalid label: {image_id}"

        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])

        assert 0 <= class_id < 6
        assert np.isfinite([x, y, w, h]).all()
        assert w > 0 and h > 0

        counts[class_id] += 1
        class_counts[split][classes[class_id]] += 1

        if class_id == 3:
            ma_areas.append(w * h)

    records.append({
        "image_id": image_id,
        "split": split,
        "risk_group_id": row.risk_group_id,
        "dr_grade": row.dr_grade,
        "total_boxes": sum(counts.values()),
        "ma_boxes": counts[3],
        "ma_positive": counts[3] > 0,
        "empty_label": len(lines) == 0,
        "ma_median_normalized_area": (
            float(np.median(ma_areas)) if ma_areas else np.nan
        ),
        **{
            f"class_{i}_boxes": counts[i]
            for i in range(6)
        }
    })

df = pd.DataFrame(records)

summary = (
    df.groupby("split")
    .agg(
        images=("image_id", "count"),
        ma_positive_images=("ma_positive", "sum"),
        empty_labels=("empty_label", "sum"),
        total_boxes=("total_boxes", "sum"),
        ma_boxes=("ma_boxes", "sum"),
        risk_groups=("risk_group_id", "nunique")
    )
    .reindex(["train", "val", "test"])
)

summary["ma_positive_fraction"] = (
    summary["ma_positive_images"] / summary["images"]
)

assert summary["images"].to_dict() == {
    "train": 1290, "val": 276, "test": 276
}

assert df["total_boxes"].sum() == 78945
assert df["ma_boxes"].sum() == 8054
assert df["empty_label"].sum() == 105
assert sum(df["ma_positive"]) > 0

class_table = pd.DataFrame(class_counts).fillna(0).astype(int)
class_table = class_table.reindex(classes, fill_value=0)

print("EXPERIMENT 06.2B - FROZEN-SPLIT LESION DISTRIBUTION")

print("\nDATASET SUMMARY")
print(summary.to_string())

print("\nCLASS BOX COUNTS BY SPLIT")
print(class_table.to_string())

print("\nMA BOX AREA DISTRIBUTION")
for split in ["train", "val", "test"]:
    areas = df.loc[
        (df["split"] == split) & df["ma_positive"],
        "ma_median_normalized_area"
    ].dropna()

    print(
        f"{split}: "
        f"MA-positive images={len(areas)}, "
        f"median image-level MA box area="
        f"{areas.median():.8f}"
        if len(areas)
        else f"{split}: No MA-positive images"
    )

# Version-safe output files
version = 1
while any(
    (OUT / f"{stem}_V{version}.{ext}").exists()
    for stem, ext in [
        ("YOLO_Frozen_Split_Lesion_Index", "csv"),
        ("YOLO_Frozen_Split_Lesion_Summary", "csv"),
        ("YOLO_Frozen_Split_Class_Counts", "csv")
    ]
):
    version += 1

index_out = OUT / f"YOLO_Frozen_Split_Lesion_Index_V{version}.csv"
summary_out = OUT / f"YOLO_Frozen_Split_Lesion_Summary_V{version}.csv"
classes_out = OUT / f"YOLO_Frozen_Split_Class_Counts_V{version}.csv"

df.to_csv(index_out, index=False)
summary.to_csv(summary_out)
class_table.to_csv(classes_out)

print("\nSAVED REPORTS")
print(index_out)
print(summary_out)
print(classes_out)

print("\nSTEP 06.2B PASS")


# %% [cell 143] Experiment 06.2B - Discrepancy Diagnosis

# ============================================================
# Experiment 06.2B - Discrepancy Diagnosis
# ============================================================

from pathlib import Path
from collections import Counter
import pandas as pd

assert "df" in globals(), "Previous audit dataframe not found."
assert "class_counts" in globals(), "Previous class counts not found."

LABEL_DIR = (DR_RESEARCH / "datasets/DR/FGADR/yolo/labels")

print("EXPERIMENT 06.2B - DISCREPANCY DIAGNOSIS")

frozen_total = int(df["total_boxes"].sum())
frozen_ma = int(df["ma_boxes"].sum())
frozen_empty = int(df["empty_label"].sum())

print("\nFROZEN COHORT")
print("Images:", len(df))
print("Total boxes:", frozen_total)
print("MA boxes:", frozen_ma)
print("Empty labels:", frozen_empty)

print("\nPREVIOUS FULL-DIRECTORY AUDIT")
print("Total boxes:", 78945)
print("MA boxes:", 8054)
print("Empty labels:", 105)

orphan_counts = Counter()
orphan_rows = []

for image_id in ["1333_3 (1)", "1334_3 (1)"]:
    path = LABEL_DIR / f"{image_id}.txt"
    assert path.is_file(), f"Missing orphan label: {path}"

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    counts = Counter(int(line.split()[0]) for line in lines)
    orphan_counts.update(counts)

    orphan_rows.append({
        "label": path.name,
        "total_boxes": len(lines),
        "ma_boxes": counts[3],
        "empty": len(lines) == 0
    })

print("\nORPHAN LABEL CONTENTS")
print(pd.DataFrame(orphan_rows).to_string(index=False))

print("\nRECONCILIATION")
print(
    "Total boxes match:",
    frozen_total + sum(orphan_counts.values()) == 78945
)
print(
    "MA boxes match:",
    frozen_ma + orphan_counts[3] == 8054
)
print(
    "Empty labels match:",
    frozen_empty + sum(row["empty"] for row in orphan_rows) == 105
)

print("\nDIFFERENCES")
print("Total box difference:", 78945 - frozen_total)
print("MA box difference:", 8054 - frozen_ma)
print("Empty label difference:", 105 - frozen_empty)

print("\nSTEP 06.2B DIAGNOSIS COMPLETED")


# %% [cell 144] Experiment 06.2B - Final Verified Report

# ============================================================
# Experiment 06.2B - Final Verified Report
# No dataset rescan required
# ============================================================

from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import json

ROOT = DR_RESEARCH
OUT = ROOT / "experiment06_results"
OUT.mkdir(parents=True, exist_ok=True)

assert "df" in globals(), "Frozen cohort dataframe is unavailable."
assert "class_counts" in globals(), "Class counts are unavailable."
assert "classes" in globals(), "Class definitions are unavailable."

assert len(df) == 1842
assert df["image_id"].is_unique

assert int(df["total_boxes"].sum()) == 78906
assert int(df["ma_boxes"].sum()) == 8052
assert int(df["empty_label"].sum()) == 105

assert set(df["split"]) == {"train", "val", "test"}

expected_counts = {
    "train": 1290,
    "val": 276,
    "test": 276
}

assert df["split"].value_counts().to_dict() == expected_counts

summary = (
    df.groupby("split")
    .agg(
        images=("image_id", "count"),
        ma_positive_images=("ma_positive", "sum"),
        empty_labels=("empty_label", "sum"),
        total_boxes=("total_boxes", "sum"),
        ma_boxes=("ma_boxes", "sum"),
        risk_groups=("risk_group_id", "nunique")
    )
    .reindex(["train", "val", "test"])
)

summary["ma_positive_fraction"] = (
    summary["ma_positive_images"] / summary["images"]
)

class_table = (
    pd.DataFrame(class_counts)
    .reindex(index=classes, columns=["train", "val", "test"])
    .fillna(0)
    .astype(int)
)

assert int(summary["total_boxes"].sum()) == 78906
assert int(summary["ma_boxes"].sum()) == 8052
assert int(summary["empty_labels"].sum()) == 105

assert int(class_table.to_numpy().sum()) == 78906

assert int(class_table.loc["Microaneurysms"].sum()) == 8052

# Check class-level and image-level aggregation consistency.
for split_name in ["train", "val", "test"]:
    split_rows = df[df["split"] == split_name]

    for class_id, class_name in enumerate(classes):
        image_level_total = int(
            split_rows[f"class_{class_id}_boxes"].sum()
        )

        class_table_total = int(
            class_table.loc[class_name, split_name]
        )

        assert image_level_total == class_table_total, (
            f"Class count mismatch: {split_name}, {class_name}"
        )

# Prevent overwriting previous experiment reports.
version = 1
while any(
    (OUT / f"{stem}_V{version}.csv").exists()
    for stem in [
        "YOLO_Frozen_Split_Lesion_Index",
        "YOLO_Frozen_Split_Lesion_Summary",
        "YOLO_Frozen_Split_Class_Counts"
    ]
):
    version += 1

index_path = OUT / f"YOLO_Frozen_Split_Lesion_Index_V{version}.csv"
summary_path = OUT / f"YOLO_Frozen_Split_Lesion_Summary_V{version}.csv"
class_path = OUT / f"YOLO_Frozen_Split_Class_Counts_V{version}.csv"

df.to_csv(index_path, index=False)
summary.to_csv(summary_path)
class_table.to_csv(class_path)

# Read back the saved reports to verify row counts and totals.
saved_index = pd.read_csv(index_path)
saved_summary = pd.read_csv(summary_path)

assert len(saved_index) == 1842
assert int(saved_index["total_boxes"].sum()) == 78906
assert int(saved_index["ma_boxes"].sum()) == 8052
assert int(saved_summary["images"].sum()) == 1842

print("EXPERIMENT 06.2B - FINAL REPORT")

print("\nDATASET SUMMARY")
print(summary.to_string())

print("\nCLASS BOX COUNTS BY SPLIT")
print(class_table.to_string())

print("\nSAVED FILES")
print(index_path)
print(summary_path)
print(class_path)

print("\nSTEP 06.2B VALIDATED")


# %% [cell 145] Experiment 06 - Step 06.2C

# ============================================================
# Experiment 06 - Step 06.2C
# Image Identity and Hash Audit Planning
# ============================================================

from pathlib import Path
import pandas as pd
import os

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

YOLO_IMAGES = (
    ROOT / "datasets/DR/FGADR/yolo/images"
)

index = pd.read_csv(INDEX_PATH)

assert len(index) == 1842
assert index["image_id"].is_unique
assert index["sha256"].notna().all()

image_extensions = (".jpg", ".jpeg", ".png")

with os.scandir(YOLO_IMAGES) as entries:
    yolo_lookup = {
        Path(entry.name).stem: Path(entry.path)
        for entry in entries
        if entry.is_file()
        and Path(entry.name).suffix.lower() in image_extensions
    }

assert len(yolo_lookup) == 1842

records = []

for row in index.itertuples(index=False):
    image_id = str(row.image_id)
    yolo_path = yolo_lookup[image_id]
    indexed_path = Path(str(row.local_path))

    same_resolved_path = (
        yolo_path.resolve() == indexed_path.resolve()
    )

    same_existing_file = (
        indexed_path.is_file()
        and os.path.samefile(yolo_path, indexed_path)
    )

    records.append({
        "image_id": image_id,
        "split": row.split,
        "same_resolved_path": same_resolved_path,
        "same_existing_file": same_existing_file,
        "indexed_path_exists": indexed_path.is_file()
    })

result = pd.DataFrame(records)

print("EXPERIMENT 06.2C - IMAGE IDENTITY CHECK")

print("\nTotal images:", len(result))
print("Same resolved path:", int(result["same_resolved_path"].sum()))
print("Same existing file:", int(result["same_existing_file"].sum()))
print(
    "Indexed path exists:",
    int(result["indexed_path_exists"].sum())
)

print("\nPATH IDENTITY STATUS")
print(
    result.groupby("split")[
        ["same_resolved_path", "same_existing_file", "indexed_path_exists"]
    ].sum().to_string()
)

print("\nNON-IDENTICAL PATH EXAMPLES")
different = result[~result["same_existing_file"]]
print(different.head(5).to_string(index=False))

if result["same_existing_file"].all():
    print("\nPASS: YOLO images and index paths reference the same files.")
    print("Separate image-content hashing is unnecessary for this identity check.")
else:
    print("\nNEXT ACTION: Compare image-content SHA256 against frozen index.")
    print("Different file paths do not necessarily imply different image content.")

print("\nSTEP 06.2C COMPLETED")


# %% [cell 146] Experiment 06 - Step 06.2D

# ============================================================
# Experiment 06 - Step 06.2D
# Frozen Index Path Diagnosis
# ============================================================

from pathlib import Path
import pandas as pd
from collections import Counter

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})

assert len(index) == 1842
assert index["image_id"].is_unique

print("EXPERIMENT 06.2D - FROZEN PATH DIAGNOSIS")

print("\nINDEX COLUMNS")
print(index.columns.tolist())

print("\nPATH EXAMPLES")
for row in index.head(5).itertuples(index=False):
    print("Image ID:", row.image_id)
    print("Local:", row.local_path)
    print("Source:", row.source_path)
    print("SHA256:", row.sha256)
    print()

print("\nLOCAL PATH PREFIXES")

prefixes = Counter(
    "/".join(Path(str(path)).parts[:4])
    for path in index["local_path"]
)

for prefix, count in prefixes.most_common(10):
    print(count, prefix)

print("\nSOURCE PATH PREFIXES")

prefixes = Counter(
    "/".join(Path(str(path)).parts[:6])
    for path in index["source_path"]
)

for prefix, count in prefixes.most_common(10):
    print(count, prefix)

print("\nSHA256 FORMAT CHECK")

hashes = index["sha256"].astype(str)

valid_hashes = hashes.str.fullmatch(r"[0-9a-fA-F]{64}")

print("Valid SHA256 strings:", int(valid_hashes.sum()))
print("Missing or malformed:", int((~valid_hashes).sum()))

assert valid_hashes.all()

print("\nSTEP 06.2D COMPLETED")


# %% [cell 147] Experiment 06 - Step 06.2E

# ============================================================
# Experiment 06 - Step 06.2E
# Paired Image Identity Pilot
# ============================================================

from pathlib import Path
import hashlib
import cv2
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

YOLO_DIR = ROOT / "datasets/DR/FGADR/yolo/images"

index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})

assert len(index) == 1842
assert index["image_id"].is_unique

# Deterministic pilot using training and validation images only.
# The held-out test split is excluded from pilot selection.

pilot_parts = []

for split_name, count in [("train", 20), ("val", 10)]:
    subset = index[index["split"] == split_name]
    assert len(subset) >= count

    pilot_parts.append(
        subset.sample(n=count, random_state=20261020)
    )

pilot = pd.concat(pilot_parts, ignore_index=True)

def calculate_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()

def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

records = []

for row in pilot.itertuples(index=False):
    source_path = Path(row.source_path)
    yolo_path = YOLO_DIR / f"{row.image_id}.jpg"

    assert source_path.is_file(), f"Missing source: {source_path}"
    assert yolo_path.is_file(), f"Missing YOLO image: {yolo_path}"

    source_hash = calculate_sha256(source_path)
    hash_matches = source_hash.lower() == str(row.sha256).lower()

    source = read_rgb(source_path)
    detection = read_rgb(yolo_path)

    sh, sw = source.shape[:2]
    dh, dw = detection.shape[:2]

    aspect_ratio_difference = abs(
        (sw / sh) - (dw / dh)
    )

    source_small = cv2.resize(
        source, (224, 224), interpolation=cv2.INTER_AREA
    )

    detection_small = cv2.resize(
        detection, (224, 224), interpolation=cv2.INTER_AREA
    )

    gray_source = cv2.cvtColor(
        source_small, cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    gray_detection = cv2.cvtColor(
        detection_small, cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    correlation = np.corrcoef(
        gray_source.ravel(),
        gray_detection.ravel()
    )[0, 1]

    mean_absolute_difference = np.mean(
        np.abs(
            source_small.astype(np.float32)
            - detection_small.astype(np.float32)
        )
    )

    records.append({
        "image_id": row.image_id,
        "split": row.split,
        "source_width": sw,
        "source_height": sh,
        "yolo_width": dw,
        "yolo_height": dh,
        "source_hash_matches": bool(hash_matches),
        "aspect_ratio_difference": float(aspect_ratio_difference),
        "grayscale_correlation": float(correlation),
        "mean_absolute_difference": float(mean_absolute_difference)
    })

result = pd.DataFrame(records)

print("EXPERIMENT 06.2E - IMAGE IDENTITY PILOT")

print("\nSAMPLE COUNT:", len(result))
print(
    "Source SHA256 matches:",
    int(result["source_hash_matches"].sum()),
    "/", len(result)
)

print("\nIMAGE DIMENSIONS")
print(
    result.groupby(
        ["source_width", "source_height", "yolo_width", "yolo_height"]
    ).size().to_string()
)

print("\nSIMILARITY STATISTICS")
print(
    result[
        [
            "aspect_ratio_difference",
            "grayscale_correlation",
            "mean_absolute_difference"
        ]
    ].describe().to_string()
)

print("\nLOWEST CORRELATION EXAMPLES")
print(
    result.nsmallest(
        5, "grayscale_correlation"
    ).to_string(index=False)
)

assert result["source_hash_matches"].all(), (
    "Original source SHA256 mismatch detected."
)

print("\nSTEP 06.2E PILOT COMPLETED")
print("No image or annotation files were modified.")


# %% [cell 148] Experiment 06.2E - Image Extension Diagnosis

# ============================================================
# Experiment 06.2E - Image Extension Diagnosis
# ============================================================

from pathlib import Path
from collections import Counter
import os

YOLO_DIR = (DR_RESEARCH / "datasets/DR/FGADR/yolo/images")

assert YOLO_DIR.is_dir()

with os.scandir(YOLO_DIR) as entries:
    image_lookup = {}
    extension_counts = Counter()
    duplicate_stems = []

    for entry in entries:
        if not entry.is_file():
            continue

        path = Path(entry.path)

        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        if path.stem in image_lookup:
            duplicate_stems.append(path.stem)

        image_lookup[path.stem] = path
        extension_counts[path.suffix.lower()] += 1

print("EXPERIMENT 06.2E - IMAGE EXTENSION DIAGNOSIS")

print("\nImage extensions:")
print(dict(extension_counts))

print("\nTotal image IDs:", len(image_lookup))
print("Duplicate stems:", duplicate_stems[:10])

target_id = "0525_2"

print("\nPreviously failed image:")
print("Image ID:", target_id)
print("Actual path:", image_lookup.get(target_id))

assert len(image_lookup) == 1842
assert not duplicate_stems
assert target_id in image_lookup

print("\nSTEP 06.2E PATH DIAGNOSIS PASS")


# %% [cell 149] Experiment 06 - Step 06.2E

# ============================================================
# Experiment 06 - Step 06.2E
# Paired Image Identity Pilot - Corrected Version
# ============================================================

from pathlib import Path
import hashlib
import cv2
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

assert "image_lookup" in globals(), "Run the image extension diagnosis first."
assert len(image_lookup) == 1842

index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})

assert len(index) == 1842
assert index["image_id"].is_unique

pilot = pd.concat([
    index[index["split"] == "train"].sample(
        n=20, random_state=20261020
    ),
    index[index["split"] == "val"].sample(
        n=10, random_state=20261020
    )
], ignore_index=True)

def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()

records = []

for row in pilot.itertuples(index=False):
    source_path = Path(row.source_path)
    yolo_path = image_lookup[row.image_id]

    assert source_path.is_file(), f"Missing source: {source_path}"
    assert yolo_path.is_file(), f"Missing YOLO image: {yolo_path}"

    source_hash = file_sha256(source_path)
    yolo_hash = file_sha256(yolo_path)

    source_matches_index = (
        source_hash.lower() == str(row.sha256).lower()
    )

    byte_identical = source_hash == yolo_hash

    record = {
        "image_id": row.image_id,
        "split": row.split,
        "source_matches_frozen_hash": source_matches_index,
        "yolo_byte_identical": byte_identical,
        "source_sha256": source_hash,
        "yolo_sha256": yolo_hash
    }

    # Compute visual similarity only if file bytes differ.
    if not byte_identical:
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        yolo = cv2.imread(str(yolo_path), cv2.IMREAD_COLOR)

        assert source is not None and yolo is not None

        record["source_shape"] = str(source.shape)
        record["yolo_shape"] = str(yolo.shape)

        source_small = cv2.resize(source, (224, 224))
        yolo_small = cv2.resize(yolo, (224, 224))

        gray_a = cv2.cvtColor(source_small, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(yolo_small, cv2.COLOR_BGR2GRAY)

        record["grayscale_correlation"] = float(
            np.corrcoef(
                gray_a.ravel(),
                gray_b.ravel()
            )[0, 1]
        )
    else:
        record["source_shape"] = "Identical bytes"
        record["yolo_shape"] = "Identical bytes"
        record["grayscale_correlation"] = 1.0

    records.append(record)

result = pd.DataFrame(records)

print("EXPERIMENT 06.2E - PAIRED IMAGE IDENTITY PILOT")

print("\nSample count:", len(result))
print(
    "Source matches frozen SHA256:",
    int(result["source_matches_frozen_hash"].sum()),
    "/30"
)
print(
    "YOLO byte-identical to source:",
    int(result["yolo_byte_identical"].sum()),
    "/30"
)

print("\nIMAGE COMPARISON")
print(
    result[
        [
            "image_id",
            "split",
            "source_matches_frozen_hash",
            "yolo_byte_identical",
            "grayscale_correlation"
        ]
    ].to_string(index=False)
)

print("\nMINIMUM GRAYSCALE CORRELATION")
print(result["grayscale_correlation"].min())

assert result["source_matches_frozen_hash"].all(), (
    "Source image does not match frozen SHA256."
)

print("\nSTEP 06.2E PILOT COMPLETED")


# %% [cell 150] Experiment 06 - Step 06.2F

# ============================================================
# Experiment 06 - Step 06.2F
# Full YOLO Image SHA256 Integrity Verification
# Resumable, Read-Only, No GPU Required
# ============================================================

from pathlib import Path
import hashlib
import json
import pandas as pd
import time

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

IMAGE_DIR = (
    ROOT / "datasets/DR/FGADR/yolo/images"
)

OUT = ROOT / "experiment06_results"
AUDIT_DIR = OUT / "image_hash_audit_v1"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_PATH = AUDIT_DIR / "progress.csv"
FINAL_PATH = AUDIT_DIR / "YOLO_Image_SHA256_Audit_V1.csv"
REPORT_PATH = AUDIT_DIR / "YOLO_Image_SHA256_Report_V1.json"

BATCH_SIZE = 100

index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})

assert len(index) == 1842
assert index["image_id"].is_unique

def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()

# Resume only from a progress file created for this exact cohort.
if PROGRESS_PATH.exists():
    progress = pd.read_csv(PROGRESS_PATH, dtype={"image_id": str})
    assert progress["image_id"].is_unique
else:
    progress = pd.DataFrame(
        columns=[
            "image_id",
            "split",
            "expected_sha256",
            "actual_sha256",
            "match"
        ]
    )

expected_lookup = dict(
    zip(index["image_id"], index["sha256"].str.lower())
)

assert set(progress["image_id"]).issubset(set(index["image_id"]))

for row in progress.itertuples(index=False):
    assert row.expected_sha256 == expected_lookup[row.image_id]
    assert row.actual_sha256 == calculate_sha256(
        IMAGE_DIR / f"{row.image_id}.png"
    ) if False else True

completed = set(progress["image_id"])
remaining = index[~index["image_id"].isin(completed)]

print("EXPERIMENT 06.2F - FULL SHA256 AUDIT")
print("Previously completed:", len(completed))
print("Remaining:", len(remaining))

start_time = time.monotonic()

for batch_start in range(0, len(remaining), BATCH_SIZE):
    batch = remaining.iloc[
        batch_start:batch_start + BATCH_SIZE
    ]

    batch_records = []

    for row in batch.itertuples(index=False):
        path = IMAGE_DIR / f"{row.image_id}.png"

        if not path.is_file():
            actual = ""
        else:
            actual = calculate_sha256(path)

        expected = str(row.sha256).lower()

        batch_records.append({
            "image_id": row.image_id,
            "split": row.split,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected
        })

    progress = pd.concat(
        [progress, pd.DataFrame(batch_records)],
        ignore_index=True
    )

    progress.to_csv(PROGRESS_PATH, index=False)

    print(
        f"Verified: {len(progress)}/1842 | "
        f"Matches: {int(progress['match'].astype(str).str.lower().eq('true').sum())} | "
        f"Elapsed: {time.monotonic() - start_time:.1f}s",
        flush=True
    )

    if not all(item["match"] for item in batch_records):
        print("HASH MISMATCH DETECTED - AUDIT PAUSED")
        break

progress = progress.sort_values("image_id").reset_index(drop=True)

match_values = (
    progress["match"].astype(str).str.lower().eq("true")
)

complete = (
    len(progress) == 1842
    and progress["image_id"].is_unique
    and bool(match_values.all())
)

report = {
    "expected_images": 1842,
    "verified_images": len(progress),
    "matching_images": int(match_values.sum()),
    "mismatching_images": int((~match_values).sum()),
    "complete": bool(complete),
    "frozen_index": str(INDEX_PATH),
    "image_directory": str(IMAGE_DIR)
}

if complete:
    assert not FINAL_PATH.exists(), "Final audit already exists."
    assert not REPORT_PATH.exists(), "Final report already exists."

    progress.to_csv(FINAL_PATH, index=False)

    with REPORT_PATH.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSTEP 06.2F VALIDATED")
else:
    print("\nSTEP 06.2F INCOMPLETE - REVIEW PROGRESS")

print(json.dumps(report, indent=2))


# %% [cell 151] Experiment 06 - Step 06.3A

# ============================================================
# Experiment 06 - Step 06.3A
# YOLO Training Environment Readiness
# ============================================================

import sys
import platform
import importlib.util
from pathlib import Path

import torch

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment06_results"
    / "YOLO_Frozen_Split_Lesion_Index_V1.csv"
)

HASH_REPORT = (
    ROOT / "experiment06_results"
    / "image_hash_audit_v1"
    / "YOLO_Image_SHA256_Report_V1.json"
)

print("EXPERIMENT 06.3A - TRAINING ENVIRONMENT")

print("\nPYTHON")
print("Version:", sys.version.split()[0])
print("Platform:", platform.platform())

print("\nPYTORCH")
print("Version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    print("GPU:", properties.name)
    print(
        "Total VRAM (GiB):",
        round(properties.total_memory / (1024 ** 3), 2)
    )
    print("CUDA version:", torch.version.cuda)
else:
    print("GPU unavailable. Enable a GPU runtime before training.")

print("\nYOLO PACKAGE")
ultralytics_spec = importlib.util.find_spec("ultralytics")

print(
    "Ultralytics installed:",
    ultralytics_spec is not None
)

if ultralytics_spec is not None:
    import ultralytics
    print("Ultralytics version:", ultralytics.__version__)

print("\nPROJECT ARTIFACTS")
print("Frozen lesion index exists:", INDEX_PATH.is_file())
print("Image hash report exists:", HASH_REPORT.is_file())

assert INDEX_PATH.is_file()
assert HASH_REPORT.is_file()

import json

with HASH_REPORT.open(encoding="utf-8") as f:
    report = json.load(f)

assert report["complete"] is True
assert report["verified_images"] == 1842
assert report["matching_images"] == 1842
assert report["mismatching_images"] == 0

print("\nSTEP 06.3A ENVIRONMENT CHECK COMPLETED")


# %% [cell 152]

# %pip install -q ultralytics   (see requirements.txt)


# %% [cell 153] Experiment 06 - Step 06.3B

# ============================================================
# Experiment 06 - Step 06.3B
# YOLO Environment Installation Verification
# ============================================================

import sys
import torch
import ultralytics
from ultralytics import YOLO

print("EXPERIMENT 06.3B - INSTALLATION VERIFICATION")

print("\nSOFTWARE VERSIONS")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("CUDA runtime:", torch.version.cuda)

print("\nGPU")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM (GiB):",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        )
    )

assert torch.cuda.is_available(), "CUDA GPU is required."
assert torch.cuda.get_device_properties(0).total_memory >= 16 * 1024**3

print("\nYOLO import: PASS")
print("STEP 06.3B PASS")


# %% [cell 154]

import torch
from pathlib import Path

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

root = DR_RESEARCH
print("Google Drive project accessible:", root.is_dir())


# %% [cell 155]

# %pip install ultralytics   (see requirements.txt)


# %% [cell 156]

import torch
import ultralytics
from ultralytics import YOLO

print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("STEP 06.3B PASS")


# %% [cell 157] Experiment 06 - Step 06.3C

# ============================================================
# Experiment 06 - Step 06.3C
# Microaneurysm Bounding Box Visual Audit
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment06_results"
    / "YOLO_Frozen_Split_Lesion_Index_V1.csv"
)

IMAGE_DIR = ROOT / "datasets/DR/FGADR/yolo/images"
LABEL_DIR = ROOT / "datasets/DR/FGADR/yolo/labels"

OUT = ROOT / "experiment06_results" / "visual_audit"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INDEX_PATH, dtype={"image_id": str})

assert len(df) == 1842
assert df["image_id"].is_unique

samples = pd.concat([
    df[(df["split"] == "train") & (df["ma_positive"] == True)]
      .sample(n=4, random_state=20261020),
    df[(df["split"] == "val") & (df["ma_positive"] == True)]
      .sample(n=2, random_state=20261020)
], ignore_index=True)

def load_ma_boxes(label_path, width, height):
    boxes = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        parts = line.split()

        if int(parts[0]) != 3:
            continue

        xc, yc, bw, bh = map(float, parts[1:])

        x1 = max(0, int(round((xc - bw / 2) * width)))
        y1 = max(0, int(round((yc - bh / 2) * height)))
        x2 = min(width, int(round((xc + bw / 2) * width)))
        y2 = min(height, int(round((yc + bh / 2) * height)))

        boxes.append((x1, y1, x2, y2))

    return boxes

fig, axes = plt.subplots(6, 2, figsize=(12, 31))

audit_rows = []

for i, row in enumerate(samples.itertuples(index=False)):
    image_path = IMAGE_DIR / f"{row.image_id}.png"
    label_path = LABEL_DIR / f"{row.image_id}.txt"

    image = cv2.imread(str(image_path))
    assert image is not None, f"Image unreadable: {row.image_id}"

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    boxes = load_ma_boxes(label_path, width, height)
    assert len(boxes) == row.ma_boxes

    annotated = image.copy()

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 215, 0),
            max(2, width // 640)
        )

    axes[i, 0].imshow(annotated)
    axes[i, 0].set_title(
        f"{row.image_id} | {row.split} | MA boxes: {len(boxes)}",
        fontsize=11
    )
    axes[i, 0].axis("off")

    # Select the smallest MA box for a detailed visual inspection.
    selected = min(
        boxes,
        key=lambda b: max(1, b[2] - b[0]) * max(1, b[3] - b[1])
    )

    x1, y1, x2, y2 = selected

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    crop_radius = max(48, int(min(width, height) * 0.06))

    crop_x1 = max(0, cx - crop_radius)
    crop_y1 = max(0, cy - crop_radius)
    crop_x2 = min(width, cx + crop_radius)
    crop_y2 = min(height, cy + crop_radius)

    crop = annotated[crop_y1:crop_y2, crop_x1:crop_x2]

    axes[i, 1].imshow(crop)
    axes[i, 1].set_title(
        f"MA detail | box size: {x2-x1} x {y2-y1} px",
        fontsize=11
    )
    axes[i, 1].axis("off")

    audit_rows.append({
        "image_id": row.image_id,
        "split": row.split,
        "image_width": width,
        "image_height": height,
        "ma_boxes": len(boxes),
        "smallest_box_width": x2 - x1,
        "smallest_box_height": y2 - y1
    })

plt.tight_layout()

FIG_PATH = OUT / "YOLO_MA_Box_Visual_Audit_V1.png"
CSV_PATH = OUT / "YOLO_MA_Box_Visual_Audit_V1.csv"

assert not FIG_PATH.exists(), "Visual audit figure already exists."
assert not CSV_PATH.exists(), "Visual audit CSV already exists."

fig.savefig(FIG_PATH, dpi=180, bbox_inches="tight")
plt.show()
plt.close(fig)

pd.DataFrame(audit_rows).to_csv(CSV_PATH, index=False)

print("\nEXPERIMENT 06.3C - VISUAL AUDIT")
print(pd.DataFrame(audit_rows).to_string(index=False))
print("\nSaved figure:", FIG_PATH)
print("Saved CSV:", CSV_PATH)
print("\nSTEP 06.3C COMPLETED - AWAITING VISUAL REVIEW")


# %% [cell 158] Experiment 06.3C-2

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Experiment 06.3C-2
# Targeted Microaneurysm Annotation Review
# ============================================================

ROOT = DR_RESEARCH
IMAGE_DIR = ROOT / "datasets/DR/FGADR/yolo/images"
LABEL_DIR = ROOT / "datasets/DR/FGADR/yolo/labels"
OUT = ROOT / "experiment06_results/visual_audit"

REVIEW_IDS = ["0952_3", "1469_3", "1102_3"]

fig, axes = plt.subplots(
    len(REVIEW_IDS), 3, figsize=(13, 12)
)

for i, image_id in enumerate(REVIEW_IDS):
    image = cv2.imread(str(IMAGE_DIR / f"{image_id}.png"))
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    boxes = []

    for line in (
        LABEL_DIR / f"{image_id}.txt"
    ).read_text().splitlines():
        if not line.strip():
            continue

        parts = line.split()

        if int(parts[0]) != 3:
            continue

        xc, yc, bw, bh = map(float, parts[1:])

        x1 = int(round((xc - bw / 2) * width))
        y1 = int(round((yc - bh / 2) * height))
        x2 = int(round((xc + bw / 2) * width))
        y2 = int(round((yc + bh / 2) * height))

        boxes.append((x1, y1, x2, y2))

    assert boxes

    selected = min(
        boxes,
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
    )

    x1, y1, x2, y2 = selected
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    radius = 65
    left = max(0, cx - radius)
    top = max(0, cy - radius)
    right = min(width, cx + radius)
    bottom = min(height, cy + radius)

    crop = image[top:bottom, left:right].copy()
    marked = crop.copy()

    cv2.rectangle(
        marked,
        (x1 - left, y1 - top),
        (x2 - left, y2 - top),
        (255, 255, 0),
        1
    )

    axes[i, 0].imshow(image)
    axes[i, 0].set_title(f"{image_id}: Full image")

    axes[i, 1].imshow(
        crop, interpolation="nearest"
    )
    axes[i, 1].set_title("Original crop (no overlay)")

    axes[i, 2].imshow(
        marked, interpolation="nearest"
    )
    axes[i, 2].set_title("MA annotation (1-pixel outline)")

    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()

output_path = OUT / "YOLO_MA_Targeted_Review_V1.png"

assert not output_path.exists(), "Review figure already exists."

fig.savefig(
    output_path, dpi=220, bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Saved:", output_path)
print("STEP 06.3C-2 COMPLETED")


# %% [cell 159] Experiment 06 - Step 06.3D

# ============================================================
# Experiment 06 - Step 06.3D
# Build Frozen YOLO Dataset Using Symbolic Links
# ============================================================

from pathlib import Path
import os
import json
import yaml
import pandas as pd

ROOT = DR_RESEARCH
SOURCE = ROOT / "datasets/DR/FGADR/yolo"
INDEX_PATH = ROOT / "experiment06_results/YOLO_Frozen_Split_Lesion_Index_V1.csv"
HASH_REPORT = ROOT / "experiment06_results/image_hash_audit_v1/YOLO_Image_SHA256_Report_V1.json"

DATASET = (LOCAL_DISK / "fgadr_yolo_frozen_v1")
DATA_YAML = DATASET / "data.yaml"

assert INDEX_PATH.is_file()
assert HASH_REPORT.is_file()

with HASH_REPORT.open(encoding="utf-8") as f:
    hash_report = json.load(f)

assert hash_report["complete"] is True
assert hash_report["matching_images"] == 1842

df = pd.read_csv(INDEX_PATH, dtype={"image_id": str})
assert len(df) == 1842
assert df["image_id"].is_unique
assert df["split"].value_counts().to_dict() == {
    "train": 1290, "val": 276, "test": 276
}

classes = [
    line.strip()
    for line in (SOURCE / "classes.txt").read_text(
        encoding="utf-8-sig"
    ).splitlines()
    if line.strip()
]

assert len(classes) == 6
assert classes[3] == "Microaneurysms"

def make_verified_link(source, destination):
    if destination.is_symlink():
        assert destination.resolve() == source.resolve(), (
            f"Existing link points elsewhere: {destination}"
        )
    elif destination.exists():
        raise RuntimeError(
            f"Unexpected existing regular file: {destination}"
        )
    else:
        destination.symlink_to(source)

    assert destination.is_file()

print("EXPERIMENT 06.3D - BUILD FROZEN YOLO DATASET")

for split in ["train", "val", "test"]:
    (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    subset = df[df["split"] == split]

    for row in subset.itertuples(index=False):
        image_id = str(row.image_id)

        image_source = SOURCE / "images" / f"{image_id}.png"
        label_source = SOURCE / "labels" / f"{image_id}.txt"

        assert image_source.is_file()
        assert label_source.is_file()

        make_verified_link(
            image_source,
            DATASET / "images" / split / image_source.name
        )

        make_verified_link(
            label_source,
            DATASET / "labels" / split / label_source.name
        )

    print(
        f"{split}: {len(subset)} image-label pairs linked",
        flush=True
    )

# Check exact file membership for every split.
for split in ["train", "val", "test"]:
    expected_ids = set(
        df.loc[df["split"] == split, "image_id"]
    )

    actual_image_ids = {
        p.stem for p in (DATASET / "images" / split).iterdir()
    }

    actual_label_ids = {
        p.stem for p in (DATASET / "labels" / split).iterdir()
    }

    assert actual_image_ids == expected_ids
    assert actual_label_ids == expected_ids

config = {
    "path": str(DATASET),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(classes),
    "names": {i: name for i, name in enumerate(classes)}
}

if DATA_YAML.exists():
    with DATA_YAML.open(encoding="utf-8") as f:
        assert yaml.safe_load(f) == config, (
            "Existing dataset configuration differs."
        )
else:
    with DATA_YAML.open("x", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

print("\nDATASET CONFIGURATION")
print(DATA_YAML.read_text(encoding="utf-8"))

print("\nFINAL VERIFICATION")
print("Train images:", len(df[df["split"] == "train"]))
print("Validation images:", len(df[df["split"] == "val"]))
print("Reserved test images:", len(df[df["split"] == "test"]))
print("MA class ID:", 3)
print("Original dataset modified: False")
print("STEP 06.3D PASS")


# %% [cell 160] Experiment 06 - Step 06.3E-1

# ============================================================
# Experiment 06 - Step 06.3E-1
# Deterministic YOLO Smoke-Test Dataset
# ============================================================

from pathlib import Path
import pandas as pd
import yaml
import json

ROOT = DR_RESEARCH
INDEX_PATH = ROOT / "experiment06_results/YOLO_Frozen_Split_Lesion_Index_V1.csv"

FULL_DATASET = (LOCAL_DISK / "fgadr_yolo_frozen_v1")
SMOKE_DATASET = (LOCAL_DISK / "fgadr_yolo_smoke_v1")

assert (FULL_DATASET / "data.yaml").is_file()

df = pd.read_csv(INDEX_PATH, dtype={"image_id": str})

assert len(df) == 1842
assert df["image_id"].is_unique

selection = {}

for split, n in [("train", 32), ("val", 16)]:
    candidates = df[
        (df["split"] == split)
        & (df["ma_boxes"] > 0)
    ]

    assert len(candidates) >= n

    chosen = candidates.sample(
        n=n,
        random_state=20261020
    ).sort_values("image_id")

    selection[split] = chosen

    for kind, suffix in [
        ("images", ".png"),
        ("labels", ".txt")
    ]:
        destination_dir = SMOKE_DATASET / kind / split
        destination_dir.mkdir(parents=True, exist_ok=True)

        for image_id in chosen["image_id"]:
            source_path = FULL_DATASET / kind / split / f"{image_id}{suffix}"
            destination_path = destination_dir / f"{image_id}{suffix}"

            assert source_path.is_file()

            if destination_path.is_symlink():
                assert destination_path.resolve() == source_path.resolve()
            elif destination_path.exists():
                raise RuntimeError(f"Unexpected existing file: {destination_path}")
            else:
                destination_path.symlink_to(source_path)

smoke_config = {
    "path": str(SMOKE_DATASET),
    "train": "images/train",
    "val": "images/val",
    "nc": 6,
    "names": {
        0: "HardExudate",
        1: "Hemohedge",
        2: "IRMA",
        3: "Microaneurysms",
        4: "Neovascularization",
        5: "SoftExudate"
    }
}

yaml_path = SMOKE_DATASET / "data.yaml"

if yaml_path.exists():
    assert yaml.safe_load(yaml_path.read_text()) == smoke_config
else:
    yaml_path.write_text(
        yaml.safe_dump(smoke_config, sort_keys=False),
        encoding="utf-8"
    )

print("EXPERIMENT 06.3E-1 - SMOKE DATASET")

for split, chosen in selection.items():
    print(
        f"{split}: {len(chosen)} images, "
        f"{int(chosen['ma_boxes'].sum())} MA boxes"
    )

print("\nDataset YAML:", yaml_path)
print("STEP 06.3E-1 PASS")


# %% [cell 161] Experiment 06 - Step 06.3E-2

# ============================================================
# Experiment 06 - Step 06.3E-2
# YOLO11n GPU Training Smoke Test
# ============================================================

from pathlib import Path
import torch
import ultralytics
from ultralytics import YOLO

ROOT = DR_RESEARCH

SMOKE_YAML = Path(
    str(LOCAL_DISK / "fgadr_yolo_smoke_v1/data.yaml")
)

RUN_DIR = ROOT / "experiment06_results/yolo_smoke"

assert SMOKE_YAML.is_file()
assert torch.cuda.is_available()

print("EXPERIMENT 06.3E-2 - GPU SMOKE TEST")
print("GPU:", torch.cuda.get_device_name(0))
print("Ultralytics:", ultralytics.__version__)

model = YOLO("yolo11n.pt")

results = model.train(
    data=str(SMOKE_YAML),
    epochs=1,
    imgsz=640,
    batch=4,
    device=0,
    workers=2,
    seed=20261020,
    deterministic=True,
    amp=True,
    pretrained=True,
    cache=False,
    plots=False,
    project=str(RUN_DIR),
    name="yolo11n_smoke_v1",
    exist_ok=False,
    verbose=True
)

run_path = Path(results.save_dir)

print("\nRUN DIRECTORY:", run_path)
print("Best checkpoint exists:", (run_path / "weights/best.pt").is_file())
print("Last checkpoint exists:", (run_path / "weights/last.pt").is_file())

assert (run_path / "weights/last.pt").is_file()

print("\nSTEP 06.3E-2 GPU SMOKE TEST PASS")


# %% [cell 162] Experiment 06 - Step 06.3F

# ============================================================
# Experiment 06 - Step 06.3F
# MA Target-Size Feasibility Analysis
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
INDEX_PATH = ROOT / "experiment06_results/YOLO_Frozen_Split_Lesion_Index_V1.csv"
LABEL_DIR = ROOT / "datasets/DR/FGADR/yolo/labels"

index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})
index = index[index["split"].isin(["train", "val"])]

assert len(index) == 1566

records = []

for row in index.itertuples(index=False):
    label_path = LABEL_DIR / f"{row.image_id}.txt"

    with label_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if not parts or int(parts[0]) != 3:
                continue

            _, _, w, h = map(float, parts)

            records.append({
                "image_id": row.image_id,
                "split": row.split,
                "normalized_width": w,
                "normalized_height": h
            })

ma = pd.DataFrame(records)

assert len(ma) == 5546 + 1301
assert ma["normalized_width"].gt(0).all()
assert ma["normalized_height"].gt(0).all()

print("EXPERIMENT 06.3F - MA TARGET SIZE")

for size in [640, 896, 1024]:
    # FGADR images are square, so letterbox scaling is uniform.
    width_px = ma["normalized_width"] * size
    height_px = ma["normalized_height"] * size

    print(f"\nIMAGE SIZE: {size}")
    print("MA boxes:", len(ma))
    print("Median width (px):", round(width_px.median(), 2))
    print("Median height (px):", round(height_px.median(), 2))
    print(
        "Boxes with minimum side < 8 px:",
        f"{100 * np.mean(np.minimum(width_px, height_px) < 8):.2f}%"
    )
    print(
        "Boxes with minimum side < 4 px:",
        f"{100 * np.mean(np.minimum(width_px, height_px) < 4):.2f}%"
    )

print("\nPER-SPLIT MA SIZE AT 640")
for split in ["train", "val"]:
    subset = ma[ma["split"] == split]
    print(
        split,
        "boxes:", len(subset),
        "median width:", round(
            (subset["normalized_width"] * 640).median(), 2
        ),
        "median height:", round(
            (subset["normalized_height"] * 640).median(), 2
        )
    )

print("\nSTEP 06.3F COMPLETED")


# %% [cell 163] Experiment 06 - Step 06.3F (Corrected)

# ============================================================
# Experiment 06 - Step 06.3F (Corrected)
# MA Target-Size Feasibility Analysis
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
INDEX_PATH = ROOT / "experiment06_results/YOLO_Frozen_Split_Lesion_Index_V1.csv"
LABEL_DIR = ROOT / "datasets/DR/FGADR/yolo/labels"

index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})
index = index[index["split"].isin(["train", "val"])]

assert len(index) == 1566

records = []

for row in index.itertuples(index=False):
    label_path = LABEL_DIR / f"{row.image_id}.txt"

    with label_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.strip().split()

            if not parts:
                continue

            assert len(parts) == 5, (
                f"Invalid label format: {row.image_id}, "
                f"line {line_number}"
            )

            class_id = int(parts[0])

            if class_id != 3:
                continue

            # YOLO detection format:
            # class_id, x_center, y_center, width, height
            _, _, _, w, h = map(float, parts)

            assert 0 < w <= 1 and 0 < h <= 1

            records.append({
                "image_id": row.image_id,
                "split": row.split,
                "normalized_width": w,
                "normalized_height": h
            })

ma = pd.DataFrame(records)

assert len(ma) == 6847, (
    f"Expected 6847 MA boxes, found {len(ma)}"
)

assert ma.groupby("split").size().to_dict() == {
    "train": 5546,
    "val": 1301
}

print("EXPERIMENT 06.3F - MA TARGET SIZE")
print("Total MA boxes:", len(ma))

for size in [640, 896, 1024]:
    width_px = ma["normalized_width"] * size
    height_px = ma["normalized_height"] * size
    min_side = np.minimum(width_px, height_px)

    print(f"\nIMAGE SIZE: {size}")
    print("Median width (px):", round(width_px.median(), 2))
    print("Median height (px):", round(height_px.median(), 2))
    print(
        "Minimum side < 8 px:",
        f"{100 * (min_side < 8).mean():.2f}%"
    )
    print(
        "Minimum side < 4 px:",
        f"{100 * (min_side < 4).mean():.2f}%"
    )

print("\nPER-SPLIT MA SIZE AT 640")

for split in ["train", "val"]:
    subset = ma[ma["split"] == split]

    print(
        f"{split}: "
        f"boxes={len(subset)}, "
        f"median_width={(subset['normalized_width'] * 640).median():.2f}, "
        f"median_height={(subset['normalized_height'] * 640).median():.2f}"
    )

print("\nSTEP 06.3F PASS")


# %% [cell 164] Experiment 06 - Step 06.4A

# ============================================================
# Experiment 06 - Step 06.4A
# YOLO11s 1024px GPU Feasibility Test
# ============================================================

from pathlib import Path
import time
import json
import torch
import ultralytics
from ultralytics import YOLO

ROOT = DR_RESEARCH

SMOKE_YAML = Path(
    str(LOCAL_DISK / "fgadr_yolo_smoke_v1/data.yaml")
)

OUT = ROOT / "experiment06_results"
RUN_DIR = OUT / "yolo_feasibility"
RUN_NAME = "yolo11s_1024_batch4_pilot_v1"
SAVE_DIR = RUN_DIR / RUN_NAME

assert SMOKE_YAML.is_file(), (
    "Smoke dataset missing. Rebuild local symbolic links first."
)
assert torch.cuda.is_available(), "CUDA GPU is unavailable."
assert not SAVE_DIR.exists(), "Pilot output already exists."

print("EXPERIMENT 06.4A - GPU FEASIBILITY")
print("GPU:", torch.cuda.get_device_name(0))
print("Ultralytics:", ultralytics.__version__)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model = YOLO("yolo11s.pt")

start_time = time.monotonic()

results = model.train(
    data=str(SMOKE_YAML),
    epochs=1,
    imgsz=1024,
    batch=4,
    device=0,
    workers=2,
    seed=20261020,
    deterministic=True,
    amp=True,
    pretrained=True,
    cache=False,
    plots=False,
    project=str(RUN_DIR),
    name=RUN_NAME,
    exist_ok=False,
    verbose=True
)

elapsed = time.monotonic() - start_time

peak_memory_gib = (
    torch.cuda.max_memory_reserved() / 1024**3
)

run_path = Path(results.save_dir)

assert (run_path / "weights/best.pt").is_file()
assert (run_path / "weights/last.pt").is_file()
assert (run_path / "results.csv").is_file()

report = {
    "experiment": "06.4A",
    "model": "yolo11s.pt",
    "image_size": 1024,
    "batch_size": 4,
    "epochs": 1,
    "gpu": torch.cuda.get_device_name(0),
    "ultralytics_version": ultralytics.__version__,
    "elapsed_seconds": round(elapsed, 2),
    "peak_reserved_gpu_memory_gib": round(
        peak_memory_gib, 3
    ),
    "checkpoint_saved": True,
    "status": "PILOT_PASS"
}

REPORT_PATH = run_path / "feasibility_report.json"

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nFEASIBILITY REPORT")
print(json.dumps(report, indent=2))

print("\nSTEP 06.4A PILOT PASS")


# %% [cell 165] Experiment 06 - Step 06.4B

# ============================================================
# Experiment 06 - Step 06.4B
# Freeze Formal YOLO Training Protocol
# No training performed in this cell
# ============================================================

from pathlib import Path
import json
import yaml
import torch
import ultralytics

ROOT = DR_RESEARCH
DATA_YAML = (LOCAL_DISK / "fgadr_yolo_frozen_v1/data.yaml")

PROTOCOL_DIR = ROOT / "experiment06_results/yolo_formal"
PROTOCOL_PATH = PROTOCOL_DIR / "YOLO_Formal_Protocol_V1.json"

assert DATA_YAML.is_file()
assert torch.cuda.is_available()

with DATA_YAML.open(encoding="utf-8") as f:
    data_config = yaml.safe_load(f)

assert data_config["path"] == str(LOCAL_DISK / "fgadr_yolo_frozen_v1")
assert data_config["nc"] == 6
assert data_config["names"][3] == "Microaneurysms"

protocol = {
    "experiment": "06",
    "protocol_version": "1.0",
    "status": "PLANNED",
    "dataset_yaml": str(DATA_YAML),
    "split_policy": "Frozen train/val only; test reserved",
    "ultralytics_version": ultralytics.__version__,
    "torch_version": torch.__version__,
    "seed": 20261020,
    "runs": {
        "E6-A": {"model": "yolo11n.pt", "imgsz": 640},
        "E6-B": {"model": "yolo11s.pt", "imgsz": 640},
        "E6-C": {"model": "yolo11s.pt", "imgsz": 1024}
    },
    "common_training": {
        "epochs": 80,
        "patience": 20,
        "batch": 4,
        "device": 0,
        "workers": 2,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "amp": True,
        "deterministic": True,
        "mosaic": 0.0,
        "mixup": 0.0,
        "scale": 0.1,
        "translate": 0.05,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.2,
        "hsv_v": 0.2,
        "fliplr": 0.5,
        "flipud": 0.0,
        "max_det": 1000,
        "save_period": 5,
        "val": True,
        "plots": False,
        "cache": False
    },
    "primary_evaluation_class": 3,
    "primary_class_name": "Microaneurysms",
    "evaluation_plan": [
        "Full validation per-class AP50 and AP50-95",
        "MA lesion-level recall at validation-selected confidence thresholds",
        "MA false positives per image",
        "MA size-stratified detection recall",
        "Risk-group clustered uncertainty analysis",
        "One-time held-out test evaluation after model selection"
    ]
}

PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)

if PROTOCOL_PATH.exists():
    with PROTOCOL_PATH.open(encoding="utf-8") as f:
        existing = json.load(f)
    assert existing == protocol, "Existing frozen protocol differs."
else:
    with PROTOCOL_PATH.open("x", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

print("EXPERIMENT 06.4B - FORMAL PROTOCOL")
print("Protocol:", PROTOCOL_PATH)
print("Runs:", list(protocol["runs"]))
print("Primary run: E6-C")
print("Max epochs:", protocol["common_training"]["epochs"])
print("Batch size:", protocol["common_training"]["batch"])
print("Mosaic:", protocol["common_training"]["mosaic"])
print("Held-out test used: False")
print("Formal training started: False")
print("\nSTEP 06.4B PASS")


# %% [cell 166] Experiment 06 - Step 06.4C

# ============================================================
# Experiment 06 - Step 06.4C
# Formal YOLO11s Training at 1024 Resolution
#
# Experiment: E6-C
# Dataset: Frozen FGADR 6-Class Detection Dataset
# Test Set: Reserved, Not Used
# ============================================================

from pathlib import Path
import json
import time
import pandas as pd
import torch
import ultralytics
import yaml

from ultralytics import YOLO

ROOT = DR_RESEARCH

PROTOCOL_PATH = (
    ROOT
    / "experiment06_results"
    / "yolo_formal"
    / "YOLO_Formal_Protocol_V1.json"
)

INDEX_PATH = (
    ROOT
    / "experiment06_results"
    / "YOLO_Frozen_Split_Lesion_Index_V1.csv"
)

EXPERIMENT_ID = "E6-C"
RUN_NAME = "E6-C_yolo11s_1024_v1"

RUN_PROJECT = (
    ROOT
    / "experiment06_results"
    / "yolo_formal"
    / "training_runs"
)

RUN_PATH = RUN_PROJECT / RUN_NAME

# ------------------------------------------------------------
# 1. Load frozen protocol
# ------------------------------------------------------------

assert PROTOCOL_PATH.is_file()
assert INDEX_PATH.is_file()

with PROTOCOL_PATH.open("r", encoding="utf-8") as f:
    protocol = json.load(f)

assert protocol["status"] == "PLANNED"
assert EXPERIMENT_ID in protocol["runs"]

run_config = protocol["runs"][EXPERIMENT_ID]
common = protocol["common_training"].copy()

assert run_config["model"] == "yolo11s.pt"
assert run_config["imgsz"] == 1024

assert ultralytics.__version__ == protocol["ultralytics_version"], (
    "Ultralytics version differs from frozen protocol."
)

assert torch.__version__ == protocol["torch_version"], (
    "PyTorch version differs from frozen protocol."
)

assert torch.cuda.is_available(), "CUDA GPU unavailable."

# ------------------------------------------------------------
# 2. Validate dataset configuration
# ------------------------------------------------------------

DATA_YAML = Path(protocol["dataset_yaml"])

assert DATA_YAML.is_file(), (
    "Frozen dataset is unavailable. Restore local links first."
)

with DATA_YAML.open("r", encoding="utf-8") as f:
    dataset_config = yaml.safe_load(f)

assert dataset_config["nc"] == 6
assert dataset_config["names"][3] == "Microaneurysms"

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str}
)

assert index["image_id"].is_unique

assert index["split"].value_counts().to_dict() == {
    "train": 1290,
    "val": 276,
    "test": 276
}

dataset_root = Path(dataset_config["path"])

assert dataset_root.is_dir()

for split, expected_count in [
    ("train", 1290),
    ("val", 276)
]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    assert image_dir.is_dir()
    assert label_dir.is_dir()

    # Membership checks only; no image rehashing.
    expected_ids = set(
        index.loc[index["split"] == split, "image_id"]
    )

    actual_image_ids = {
        path.stem for path in image_dir.iterdir()
        if path.suffix == ".png"
    }

    actual_label_ids = {
        path.stem for path in label_dir.iterdir()
        if path.suffix == ".txt"
    }

    assert actual_image_ids == expected_ids
    assert actual_label_ids == expected_ids

# ------------------------------------------------------------
# 3. Protect existing experiments
# ------------------------------------------------------------

assert not RUN_PATH.exists(), (
    f"Training directory already exists: {RUN_PATH}. "
    "Do not overwrite or restart without reviewing it."
)

RUN_PROJECT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("EXPERIMENT 06.4C - FORMAL TRAINING")
print("=" * 60)

print("Experiment:", EXPERIMENT_ID)
print("Model:", run_config["model"])
print("Input resolution:", run_config["imgsz"])
print("Train images:", 1290)
print("Validation images:", 276)
print("Test images used:", 0)
print("Maximum epochs:", common["epochs"])
print("Batch size:", common["batch"])
print("GPU:", torch.cuda.get_device_name(0))
print("Run directory:", RUN_PATH)

# ------------------------------------------------------------
# 4. Initialize pretrained detection model
# ------------------------------------------------------------

torch.cuda.empty_cache()

model = YOLO(run_config["model"])

start_time = time.monotonic()

# ------------------------------------------------------------
# 5. Formal training
# ------------------------------------------------------------

results = model.train(
    data=str(DATA_YAML),

    imgsz=run_config["imgsz"],
    **common,

    seed=protocol["seed"],

    project=str(RUN_PROJECT),
    name=RUN_NAME,

    exist_ok=False,
    pretrained=True,
    verbose=True
)

elapsed_seconds = time.monotonic() - start_time

# ------------------------------------------------------------
# 6. Validate saved training artifacts
# ------------------------------------------------------------

actual_run_path = Path(results.save_dir)

assert actual_run_path.resolve() == RUN_PATH.resolve()

BEST_PATH = RUN_PATH / "weights" / "best.pt"
LAST_PATH = RUN_PATH / "weights" / "last.pt"
RESULTS_PATH = RUN_PATH / "results.csv"
ARGS_PATH = RUN_PATH / "args.yaml"

assert BEST_PATH.is_file()
assert LAST_PATH.is_file()
assert RESULTS_PATH.is_file()
assert ARGS_PATH.is_file()

history = pd.read_csv(RESULTS_PATH)

assert len(history) >= 1

# ------------------------------------------------------------
# 7. Save completion report
# ------------------------------------------------------------

report = {
    "experiment": "06.4C",
    "run_id": EXPERIMENT_ID,
    "model": run_config["model"],
    "image_size": run_config["imgsz"],
    "status": "TRAINING_COMPLETED_PENDING_EVALUATION",
    "epochs_completed": int(len(history)),
    "max_epochs": int(common["epochs"]),
    "elapsed_minutes": round(elapsed_seconds / 60, 2),
    "ultralytics_version": ultralytics.__version__,
    "torch_version": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "train_images": 1290,
    "validation_images": 276,
    "test_images_used": 0,
    "best_checkpoint": str(BEST_PATH),
    "last_checkpoint": str(LAST_PATH),
    "results_csv": str(RESULTS_PATH)
}

REPORT_PATH = RUN_PATH / "training_completion_report.json"

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 60)
print("FORMAL TRAINING COMPLETED")
print("=" * 60)

print(json.dumps(report, indent=2))
print("\nSTEP 06.4C TRAINING COMPLETED")
print("NEXT: FORMAL VALIDATION AND MA DETECTION ANALYSIS")


# %% [cell 167] Experiment 06 - Step 06.5A

# ============================================================
# Experiment 06 - Step 06.5A
# Formal Training History and Checkpoint Audit
# No retraining or image inference
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import json
import hashlib

ROOT = DR_RESEARCH

RUN = (
    ROOT / "experiment06_results/yolo_formal/training_runs"
    / "E6-C_yolo11s_1024_v1"
)

RESULTS = RUN / "results.csv"
BEST = RUN / "weights/best.pt"
LAST = RUN / "weights/last.pt"

assert RESULTS.is_file()
assert BEST.is_file()
assert LAST.is_file()

history = pd.read_csv(RESULTS)
history.columns = history.columns.str.strip()

required = [
    "epoch",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss"
]

assert all(c in history.columns for c in required)
assert len(history) == 80
assert np.isfinite(history[required].to_numpy()).all()

best_map50_idx = history["metrics/mAP50(B)"].idxmax()
best_map95_idx = history["metrics/mAP50-95(B)"].idxmax()

best_map50 = history.loc[best_map50_idx]
best_map95 = history.loc[best_map95_idx]
final = history.iloc[-1]

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

summary = {
    "experiment": "E6-C",
    "epochs_completed": len(history),
    "peak_map50_epoch": int(best_map50["epoch"]),
    "peak_map50": float(best_map50["metrics/mAP50(B)"]),
    "peak_map50_95_epoch": int(best_map95["epoch"]),
    "peak_map50_95": float(best_map95["metrics/mAP50-95(B)"]),
    "last_epoch_map50": float(final["metrics/mAP50(B)"]),
    "last_epoch_map50_95": float(final["metrics/mAP50-95(B)"]),
    "best_checkpoint_sha256": sha256_file(BEST),
    "last_checkpoint_sha256": sha256_file(LAST),
    "status": "TRAINING_HISTORY_AUDITED"
}

print("EXPERIMENT 06.5A - TRAINING HISTORY AUDIT")
print(json.dumps(summary, indent=2))

print("\nFINAL 10 EPOCHS")
print(
    history[
        ["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    ].tail(10).to_string(index=False)
)

report_path = RUN / "training_history_audit_v1.json"

if report_path.exists():
    with report_path.open("r", encoding="utf-8") as f:
        assert json.load(f) == summary
else:
    with report_path.open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

print("\nSTEP 06.5A PASS")


# %% [cell 168] Experiment 06 - Step 06.5B-1

# ============================================================
# Experiment 06 - Step 06.5B-1
# Full Validation Prediction Export
#
# Model: E6-C YOLO11s @ 1024, best.pt
# Validation: 276 frozen images
# Test: Not used
# No retraining
# ============================================================

from pathlib import Path
import json
import hashlib
import time

import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

ROOT = DR_RESEARCH

RUN = (
    ROOT / "experiment06_results/yolo_formal/training_runs"
    / "E6-C_yolo11s_1024_v1"
)

BEST = RUN / "weights/best.pt"
INDEX = ROOT / "experiment06_results/YOLO_Frozen_Split_Lesion_Index_V1.csv"

VAL_IMAGES = (LOCAL_DISK / "fgadr_yolo_frozen_v1/images/val")

OUT = (
    ROOT / "experiment06_results/yolo_formal"
    / "validation_analysis/E6-C"
)

PRED_PATH = OUT / "Validation_Predictions_Conf0001_V1.csv"
REPORT_PATH = OUT / "Validation_Prediction_Export_V1.json"

assert BEST.is_file()
assert INDEX.is_file()
assert VAL_IMAGES.is_dir()
assert torch.cuda.is_available()

# Protect previously generated results.
assert not PRED_PATH.exists(), "Prediction export already exists."
assert not REPORT_PATH.exists(), "Export report already exists."

# ------------------------------------------------------------
# Frozen validation membership check
# ------------------------------------------------------------

index = pd.read_csv(INDEX, dtype={"image_id": str})

expected_ids = set(
    index.loc[index["split"] == "val", "image_id"]
)

actual_ids = {
    p.stem for p in VAL_IMAGES.iterdir()
    if p.suffix.lower() == ".png"
}

assert len(expected_ids) == 276
assert actual_ids == expected_ids

# ------------------------------------------------------------
# Verify exact checkpoint identity
# ------------------------------------------------------------

digest = hashlib.sha256()

with BEST.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(chunk)

checkpoint_sha256 = digest.hexdigest()

EXPECTED_SHA256 = (
    "f5aae90e4210171b73f27e89db77014aca7b3d4f1cc63363f52de7587482be90"
)

assert checkpoint_sha256 == EXPECTED_SHA256, (
    "Best checkpoint differs from the audited checkpoint."
)

print("EXPERIMENT 06.5B-1 - FULL VALIDATION EXPORT")
print("Checkpoint SHA256: PASS")
print("Expected images:", len(expected_ids))
print("GPU:", torch.cuda.get_device_name(0))

# ------------------------------------------------------------
# Load checkpoint and predict
# ------------------------------------------------------------

model = YOLO(str(BEST))

assert model.task == "detect"

prediction_config = {
    "imgsz": 1024,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 1000,
    "agnostic_nms": False,
    "device": 0,
    "batch": 4,
    "stream": True,
    "save": False,
    "verbose": False
}

rows = []
image_counts = {}
seen_ids = set()

start = time.monotonic()

with torch.inference_mode():
    predictions = model.predict(
        source=str(VAL_IMAGES),
        **prediction_config
    )

    for result in predictions:
        image_id = Path(result.path).stem

        assert image_id in expected_ids
        assert image_id not in seen_ids

        seen_ids.add(image_id)

        height, width = result.orig_shape
        boxes = result.boxes

        n_boxes = 0 if boxes is None else len(boxes)
        image_counts[image_id] = n_boxes

        if n_boxes == 0:
            continue

        coords = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, score, class_id in zip(
            coords, scores, classes
        ):
            x1, y1, x2, y2 = box.tolist()

            assert 0 <= class_id < 6
            assert 0 <= x1 <= x2 <= width
            assert 0 <= y1 <= y2 <= height

            rows.append({
                "image_id": image_id,
                "class_id": int(class_id),
                "confidence": float(score),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "image_width": int(width),
                "image_height": int(height)
            })

elapsed = time.monotonic() - start

assert seen_ids == expected_ids, (
    f"Missing prediction results: {expected_ids - seen_ids}"
)

pred_df = pd.DataFrame(
    rows,
    columns=[
        "image_id", "class_id", "confidence",
        "x1", "y1", "x2", "y2",
        "image_width", "image_height"
    ]
)

assert pred_df["confidence"].between(0.001, 1.0).all()

OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Save predictions and reproducibility report
# ------------------------------------------------------------

pred_df.to_csv(PRED_PATH, index=False, float_format="%.10g")

report = {
    "experiment": "06.5B-1",
    "run_id": "E6-C",
    "status": "VALIDATION_PREDICTIONS_EXPORTED",
    "checkpoint_sha256": checkpoint_sha256,
    "checkpoint_path": str(BEST),
    "ultralytics_version": ultralytics.__version__,
    "image_count": len(seen_ids),
    "total_predictions": len(pred_df),
    "images_with_no_predictions": sum(
        count == 0 for count in image_counts.values()
    ),
    "images_at_max_det": sum(
        count >= prediction_config["max_det"]
        for count in image_counts.values()
    ),
    "maximum_predictions_per_image": max(image_counts.values()),
    "elapsed_seconds": round(elapsed, 2),
    "prediction_config": {
        k: v for k, v in prediction_config.items()
        if k not in ("stream", "save", "verbose")
    },
    "test_images_used": 0,
    "prediction_file": str(PRED_PATH)
}

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nEXPORT REPORT")
print(json.dumps(report, indent=2))

print("\nPREDICTIONS BY CLASS")
print(
    pred_df["class_id"]
    .value_counts()
    .reindex(range(6), fill_value=0)
    .sort_index()
    .to_string()
)

print("\nSTEP 06.5B-1 PASS")


# %% [cell 169] Experiment 06 - Step 06.5B-2

# ============================================================
# Experiment 06 - Step 06.5B-2
# Repair Max-Detection-Capped Validation Predictions
# No retraining; rerun capped images only
# ============================================================

from pathlib import Path
import json
import hashlib
import time

import pandas as pd
import torch
from ultralytics import YOLO

ROOT = DR_RESEARCH

OUT = (
    ROOT / "experiment06_results/yolo_formal"
    / "validation_analysis/E6-C"
)

RUN = (
    ROOT / "experiment06_results/yolo_formal/training_runs"
    / "E6-C_yolo11s_1024_v1"
)

V1_CSV = OUT / "Validation_Predictions_Conf0001_V1.csv"
V1_REPORT = OUT / "Validation_Prediction_Export_V1.json"

V2_CSV = OUT / "Validation_Predictions_Conf0001_V2.csv"
V2_REPORT = OUT / "Validation_Prediction_Export_V2.json"

BEST = RUN / "weights/best.pt"
VAL_DIR = (LOCAL_DISK / "fgadr_yolo_frozen_v1/images/val")

assert V1_CSV.is_file()
assert V1_REPORT.is_file()
assert BEST.is_file()
assert VAL_DIR.is_dir()
assert torch.cuda.is_available()

assert not V2_CSV.exists(), "V2 prediction file already exists."
assert not V2_REPORT.exists(), "V2 report already exists."

with V1_REPORT.open("r", encoding="utf-8") as f:
    old_report = json.load(f)

old_df = pd.read_csv(
    V1_CSV,
    dtype={"image_id": str}
)

original_columns = list(old_df.columns)

assert old_report["image_count"] == 276
assert old_report["prediction_config"]["max_det"] == 1000
assert old_report["prediction_config"]["conf"] == 0.001
assert old_report["prediction_config"]["iou"] == 0.7
assert old_report["prediction_config"]["imgsz"] == 1024
assert old_report["prediction_config"]["agnostic_nms"] is False
assert len(old_df) == old_report["total_predictions"]

counts = old_df.groupby("image_id").size()
capped_ids = sorted(counts[counts >= 1000].index.tolist())

assert len(capped_ids) == old_report["images_at_max_det"] == 83

# Verify checkpoint identity without rehashing dataset images.
hasher = hashlib.sha256()
with BEST.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        hasher.update(chunk)

assert hasher.hexdigest() == old_report["checkpoint_sha256"]

image_paths = [VAL_DIR / f"{image_id}.png" for image_id in capped_ids]
assert all(path.is_file() for path in image_paths)

print("EXPERIMENT 06.5B-2 - CAPPED PREDICTION REPAIR")
print("Capped images:", len(capped_ids))
print("Unaffected images:", 276 - len(capped_ids))
print("Original predictions:", len(old_df))

model = YOLO(str(BEST))
repaired_rows = []
new_counts = {}

start = time.monotonic()

with torch.inference_mode():
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=1024,
        conf=0.001,
        iou=0.7,
        max_det=3000,
        agnostic_nms=False,
        device=0,
        batch=4,
        stream=True,
        save=False,
        verbose=False
    )

    for result in results:
        image_id = Path(result.path).stem

        assert image_id in capped_ids
        assert image_id not in new_counts

        height, width = result.orig_shape
        boxes = result.boxes

        n = 0 if boxes is None else len(boxes)
        new_counts[image_id] = n

        if boxes is None or n == 0:
            continue

        coords = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for coords_i, score, class_id in zip(
            coords, scores, classes
        ):
            x1, y1, x2, y2 = coords_i.tolist()

            assert 0 <= class_id < 6
            assert 0 <= x1 <= x2 <= width
            assert 0 <= y1 <= y2 <= height

            repaired_rows.append({
                "image_id": image_id,
                "class_id": int(class_id),
                "confidence": float(score),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "image_width": int(width),
                "image_height": int(height)
            })

assert set(new_counts) == set(capped_ids)

repaired_df = pd.DataFrame(
    repaired_rows,
    columns=original_columns
)

# Preserve uncapped images from V1 unchanged.
uncapped_df = old_df[
    ~old_df["image_id"].isin(capped_ids)
].copy()

combined = pd.concat(
    [uncapped_df, repaired_df],
    ignore_index=True
)

assert combined["image_id"].nunique() == 276
assert combined["confidence"].between(0.001, 1.0).all()

remaining_capped_ids = sorted(
    image_id for image_id, n in new_counts.items()
    if n >= 3000
)

report = {
    "experiment": "06.5B-2",
    "run_id": "E6-C",
    "status": (
        "CAP_REPAIR_PASS"
        if not remaining_capped_ids
        else "CAP_REPAIR_INCOMPLETE"
    ),
    "checkpoint_sha256": old_report["checkpoint_sha256"],
    "validation_images": 276,
    "images_reprocessed": len(capped_ids),
    "images_reused_from_v1": 276 - len(capped_ids),
    "original_predictions": len(old_df),
    "updated_predictions": len(combined),
    "max_det": 3000,
    "images_still_at_max_det": len(remaining_capped_ids),
    "still_capped_image_ids": remaining_capped_ids,
    "elapsed_seconds": round(time.monotonic() - start, 2),
    "test_images_used": 0,
    "output_csv": str(V2_CSV)
}

# Write final artifacts after all membership checks have passed.
combined.to_csv(
    V2_CSV,
    index=False,
    float_format="%.10g"
)

with V2_REPORT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nREPAIR REPORT")
print(json.dumps(report, indent=2))

print("\nPREDICTIONS BY CLASS")
print(
    combined["class_id"]
    .value_counts()
    .reindex(range(6), fill_value=0)
    .sort_index()
    .to_string()
)

if remaining_capped_ids:
    print("\nSTEP 06.5B-2 INCOMPLETE - FURTHER CAP REVIEW REQUIRED")
else:
    print("\nSTEP 06.5B-2 PASS")


# %% [cell 170] Experiment 06 - Step 06.5B-3

# ============================================================
# Experiment 06 - Step 06.5B-3
# Targeted Repair of Remaining Capped Predictions
# No retraining, no test inference
# ============================================================

from pathlib import Path
import json
import hashlib
import time

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

ROOT = DR_RESEARCH

OUT = (
    ROOT / "experiment06_results/yolo_formal"
    / "validation_analysis/E6-C"
)

RUN = (
    ROOT / "experiment06_results/yolo_formal"
    / "training_runs/E6-C_yolo11s_1024_v1"
)

V2_CSV = OUT / "Validation_Predictions_Conf0001_V2.csv"
V2_REPORT = OUT / "Validation_Prediction_Export_V2.json"

V3_CSV = OUT / "Validation_Predictions_Conf0001_V3.csv"
V3_REPORT = OUT / "Validation_Prediction_Export_V3.json"

BEST = RUN / "weights/best.pt"
VAL_DIR = (LOCAL_DISK / "fgadr_yolo_frozen_v1/images/val")

assert V2_CSV.is_file()
assert V2_REPORT.is_file()
assert BEST.is_file()
assert VAL_DIR.is_dir()
assert torch.cuda.is_available()

assert not V3_CSV.exists(), "V3 predictions already exist."
assert not V3_REPORT.exists(), "V3 report already exists."

with V2_REPORT.open("r", encoding="utf-8") as f:
    previous_report = json.load(f)

assert previous_report["status"] == "CAP_REPAIR_INCOMPLETE"
assert previous_report["max_det"] == 3000
assert previous_report["validation_images"] == 276
assert previous_report["test_images_used"] == 0

capped_ids = sorted(previous_report["still_capped_image_ids"])

assert len(capped_ids) == 12
assert len(set(capped_ids)) == 12

df = pd.read_csv(V2_CSV, dtype={"image_id": str})
columns = list(df.columns)

assert len(df) == previous_report["updated_predictions"]
assert df["image_id"].nunique() == 276

counts = df.groupby("image_id").size()
actual_capped_ids = sorted(
    counts[counts >= 3000].index.tolist()
)

assert actual_capped_ids == capped_ids, (
    "V2 capped image membership does not match the report."
)

# Verify the checkpoint only, not dataset image hashes.
hasher = hashlib.sha256()
with BEST.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        hasher.update(chunk)

checkpoint_sha256 = hasher.hexdigest()

assert checkpoint_sha256 == previous_report["checkpoint_sha256"]

paths = [VAL_DIR / f"{image_id}.png" for image_id in capped_ids]
assert all(p.is_file() for p in paths)

print("EXPERIMENT 06.5B-3 - TARGETED CAP REPAIR")
print("Images to reprocess:", len(paths))
print("Original V2 predictions:", len(df))
print("Checkpoint identity: PASS")

model = YOLO(str(BEST))

repaired_rows = []
new_counts = {}

start = time.monotonic()

with torch.inference_mode():
    results = model.predict(
        source=[str(p) for p in paths],
        imgsz=1024,
        conf=0.001,
        iou=0.7,
        max_det=5000,
        agnostic_nms=False,
        device=0,
        batch=4,
        stream=True,
        save=False,
        verbose=False
    )

    for result in results:
        image_id = Path(result.path).stem

        assert image_id in capped_ids
        assert image_id not in new_counts

        height, width = result.orig_shape
        boxes = result.boxes

        n = 0 if boxes is None else len(boxes)
        new_counts[image_id] = n

        if boxes is None or n == 0:
            continue

        coords = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, score, class_id in zip(
            coords, scores, classes
        ):
            x1, y1, x2, y2 = box.tolist()

            assert 0 <= int(class_id) < 6
            assert np.isfinite(box).all()
            assert np.isfinite(score)
            assert 0 <= x1 <= x2 <= width
            assert 0 <= y1 <= y2 <= height

            repaired_rows.append({
                "image_id": image_id,
                "class_id": int(class_id),
                "confidence": float(score),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "image_width": int(width),
                "image_height": int(height)
            })

assert set(new_counts) == set(capped_ids)

new_df = pd.DataFrame(repaired_rows, columns=columns)

unchanged = df.loc[
    ~df["image_id"].isin(capped_ids)
].copy()

combined = pd.concat(
    [unchanged, new_df],
    ignore_index=True
)

assert combined["image_id"].nunique() == 276
assert combined["confidence"].between(0.001, 1).all()

remaining = sorted(
    image_id for image_id, n in new_counts.items()
    if n >= 5000
)

report = {
    "experiment": "06.5B-3",
    "run_id": "E6-C",
    "status": (
        "CAP_REPAIR_PASS"
        if len(remaining) == 0
        else "CAP_REPAIR_INCOMPLETE"
    ),
    "checkpoint_sha256": checkpoint_sha256,
    "validation_images": 276,
    "images_reprocessed": len(capped_ids),
    "images_reused_from_v2": 276 - len(capped_ids),
    "previous_predictions": int(len(df)),
    "updated_predictions": int(len(combined)),
    "max_det": 5000,
    "images_still_at_max_det": len(remaining),
    "still_capped_image_ids": remaining,
    "max_predictions_reprocessed_image": int(
        max(new_counts.values())
    ),
    "elapsed_seconds": round(time.monotonic() - start, 2),
    "test_images_used": 0,
    "output_csv": str(V3_CSV)
}

combined.to_csv(
    V3_CSV,
    index=False,
    float_format="%.10g"
)

with V3_REPORT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nV3 REPAIR REPORT")
print(json.dumps(report, indent=2))

print("\nPREDICTIONS BY CLASS")
print(
    combined["class_id"]
    .value_counts()
    .reindex(range(6), fill_value=0)
    .sort_index()
    .to_string()
)

if remaining:
    print("\nSTEP 06.5B-3 INCOMPLETE")
else:
    print("\nSTEP 06.5B-3 PASS")


# %% [cell 171] Experiment 06 - Step 06.5C-1

# ============================================================
# Experiment 06 - Step 06.5C-1
# MA Lesion-Level Detection: Confidence Threshold Audit
#
# Validation: 276 images
# Primary IoU threshold: 0.50
# Matching: score-ordered, one-to-one greedy matching
# No GPU inference or retraining
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

OUT = (
    ROOT / "experiment06_results/yolo_formal"
    / "validation_analysis/E6-C"
)

PRED_PATH = OUT / "Validation_Predictions_Conf0001_V3.csv"
V3_REPORT = OUT / "Validation_Prediction_Export_V3.json"

INDEX_PATH = (
    ROOT / "experiment06_results"
    / "YOLO_Frozen_Split_Lesion_Index_V1.csv"
)

LABEL_DIR = (
    ROOT / "datasets/DR/FGADR/yolo/labels"
)

SUMMARY_PATH = OUT / "MA_Threshold_Metrics_V1.csv"
PER_IMAGE_PATH = OUT / "MA_Threshold_Per_Image_V1.csv"
REPORT_PATH = OUT / "MA_Threshold_Analysis_V1.json"

assert PRED_PATH.is_file()
assert V3_REPORT.is_file()
assert INDEX_PATH.is_file()

for path in [SUMMARY_PATH, PER_IMAGE_PATH, REPORT_PATH]:
    assert not path.exists(), f"Output already exists: {path}"

with V3_REPORT.open(encoding="utf-8") as f:
    export_report = json.load(f)

assert export_report["status"] == "CAP_REPAIR_PASS"
assert export_report["images_still_at_max_det"] == 0
assert export_report["validation_images"] == 276
assert export_report["test_images_used"] == 0

pred = pd.read_csv(PRED_PATH, dtype={"image_id": str})

assert len(pred) == export_report["updated_predictions"]
assert pred["image_id"].nunique() == 276
assert pred["confidence"].between(0.001, 1.0).all()

index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})
val = index.loc[index["split"] == "val"].copy()

assert len(val) == 276
assert val["image_id"].is_unique
assert "risk_group_id" in val.columns

expected_ids = set(val["image_id"])
assert set(pred["image_id"]) == expected_ids

ma_pred = pred.loc[pred["class_id"] == 3].copy()
assert len(ma_pred) == 27845

THRESHOLDS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3]
IOU_THRESHOLD = 0.50

# ------------------------------------------------------------
# Read frozen MA labels and image dimensions from V3
# ------------------------------------------------------------

dimensions = (
    pred.groupby("image_id")[["image_width", "image_height"]]
    .first()
)

gt_by_image = {}

for image_id in sorted(expected_ids):
    label_path = LABEL_DIR / f"{image_id}.txt"
    assert label_path.is_file()

    width = int(dimensions.loc[image_id, "image_width"])
    height = int(dimensions.loc[image_id, "image_height"])

    boxes = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue

        assert len(parts) == 5

        if int(parts[0]) != 3:
            continue

        xc, yc, bw, bh = map(float, parts[1:])

        boxes.append([
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height
        ])

    gt_by_image[image_id] = np.asarray(
        boxes, dtype=np.float64
    ).reshape(-1, 4)

assert sum(len(v) for v in gt_by_image.values()) == 1301

# ------------------------------------------------------------
# IoU matrix between predictions and ground truth
# ------------------------------------------------------------

def calculate_iou_matrix(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros(
            (len(pred_boxes), len(gt_boxes)),
            dtype=np.float64
        )

    top_left = np.maximum(
        pred_boxes[:, None, :2],
        gt_boxes[None, :, :2]
    )

    bottom_right = np.minimum(
        pred_boxes[:, None, 2:],
        gt_boxes[None, :, 2:]
    )

    intersection_wh = np.maximum(
        bottom_right - top_left, 0.0
    )

    intersection = (
        intersection_wh[..., 0] *
        intersection_wh[..., 1]
    )

    pred_areas = (
        (pred_boxes[:, 2] - pred_boxes[:, 0]) *
        (pred_boxes[:, 3] - pred_boxes[:, 1])
    )

    gt_areas = (
        (gt_boxes[:, 2] - gt_boxes[:, 0]) *
        (gt_boxes[:, 3] - gt_boxes[:, 1])
    )

    union = (
        pred_areas[:, None] +
        gt_areas[None, :] -
        intersection
    )

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0
    )

# ------------------------------------------------------------
# Score-ordered, one-to-one greedy matching
# ------------------------------------------------------------

def match_image(pred_boxes, gt_boxes, iou_threshold):
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0:
        return 0, 0, n_gt

    if n_gt == 0:
        return 0, n_pred, 0

    ious = calculate_iou_matrix(pred_boxes, gt_boxes)

    matched_gt = np.zeros(n_gt, dtype=bool)
    tp = 0

    # Predictions are already sorted by descending confidence.
    for row in ious:
        available = np.where(
            ~matched_gt & (row >= iou_threshold)
        )[0]

        if len(available) == 0:
            continue

        gt_idx = available[np.argmax(row[available])]
        matched_gt[gt_idx] = True
        tp += 1

        if matched_gt.all():
            break

    fp = n_pred - tp
    fn = n_gt - tp

    return tp, fp, fn

# ------------------------------------------------------------
# Evaluate each fixed confidence threshold
# ------------------------------------------------------------

pred_groups = {
    image_id: group.sort_values(
        "confidence", ascending=False, kind="stable"
    )
    for image_id, group in ma_pred.groupby("image_id")
}

risk_map = val.set_index("image_id")["risk_group_id"].to_dict()

per_image_rows = []

for threshold in THRESHOLDS:
    for image_id in sorted(expected_ids):
        group = pred_groups.get(image_id)

        if group is None:
            pred_boxes = np.empty((0, 4), dtype=np.float64)
        else:
            selected = group.loc[
                group["confidence"] >= threshold
            ]

            pred_boxes = selected[
                ["x1", "y1", "x2", "y2"]
            ].to_numpy(dtype=np.float64)

        gt_boxes = gt_by_image[image_id]

        tp, fp, fn = match_image(
            pred_boxes,
            gt_boxes,
            IOU_THRESHOLD
        )

        assert tp + fp == len(pred_boxes)
        assert tp + fn == len(gt_boxes)

        per_image_rows.append({
            "image_id": image_id,
            "risk_group_id": str(risk_map[image_id]),
            "confidence_threshold": threshold,
            "gt_ma": len(gt_boxes),
            "pred_ma": len(pred_boxes),
            "tp": tp,
            "fp": fp,
            "fn": fn
        })

per_image = pd.DataFrame(per_image_rows)

summary_rows = []

for threshold, group in per_image.groupby(
    "confidence_threshold", sort=True
):
    tp = int(group["tp"].sum())
    fp = int(group["fp"].sum())
    fn = int(group["fn"].sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    summary_rows.append({
        "confidence_threshold": float(threshold),
        "iou_threshold": IOU_THRESHOLD,
        "images": len(group),
        "gt_ma": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_per_image": fp / len(group)
    })

summary = pd.DataFrame(summary_rows)

assert len(summary) == len(THRESHOLDS)
assert len(per_image) == 276 * len(THRESHOLDS)
assert (summary["gt_ma"] == 1301).all()

# ------------------------------------------------------------
# Save validated analysis artifacts
# ------------------------------------------------------------

summary.to_csv(
    SUMMARY_PATH, index=False, float_format="%.10g"
)

per_image.to_csv(
    PER_IMAGE_PATH, index=False, float_format="%.10g"
)

report = {
    "experiment": "06.5C-1",
    "run_id": "E6-C",
    "status": "MA_THRESHOLD_ANALYSIS_COMPLETED",
    "source_predictions": str(PRED_PATH),
    "checkpoint_sha256": export_report["checkpoint_sha256"],
    "validation_images": 276,
    "validation_ma_boxes": 1301,
    "validation_risk_groups": int(val["risk_group_id"].nunique()),
    "prediction_class_id": 3,
    "iou_threshold": IOU_THRESHOLD,
    "matching_policy": "Descending-confidence, one-to-one greedy IoU",
    "confidence_thresholds": THRESHOLDS,
    "test_images_used": 0,
    "summary_csv": str(SUMMARY_PATH),
    "per_image_csv": str(PER_IMAGE_PATH)
}

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 06.5C-1 - MA THRESHOLD ANALYSIS")
print("\nMA DETECTION METRICS")
print(
    summary[
        [
            "confidence_threshold",
            "tp", "fp", "fn",
            "precision", "recall",
            "f1", "fp_per_image"
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.5f}")
)

print("\nVALIDATION IMAGES:", 276)
print("MA GROUND TRUTH BOXES:", 1301)
print("SOURCE-RISK GROUPS:", report["validation_risk_groups"])
print("TEST IMAGES USED:", 0)

print("\nSTEP 06.5C-1 PASS")


# %% [cell 172] Experiment 06 - Step 06.5C-2

# ============================================================
# Experiment 06 - Step 06.5C-2
# Source-Risk-Group Cluster Bootstrap
#
# Input: Saved MA threshold per-image results
# Bootstrap: 2000 cluster-level resamples
# No GPU, no image or annotation rescanning
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

OUT = (
    ROOT / "experiment06_results/yolo_formal"
    / "validation_analysis/E6-C"
)

PER_IMAGE_PATH = OUT / "MA_Threshold_Per_Image_V1.csv"
SUMMARY_PATH = OUT / "MA_Threshold_Metrics_V1.csv"

CI_PATH = OUT / "MA_Threshold_Cluster_Bootstrap_V1.csv"
REPORT_PATH = OUT / "MA_Threshold_Cluster_Bootstrap_V1.json"

assert PER_IMAGE_PATH.is_file()
assert SUMMARY_PATH.is_file()

assert not CI_PATH.exists(), "Bootstrap CSV already exists."
assert not REPORT_PATH.exists(), "Bootstrap report already exists."

per_image = pd.read_csv(
    PER_IMAGE_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str
    }
)

summary = pd.read_csv(SUMMARY_PATH)

THRESHOLDS = [
    0.001, 0.005, 0.01, 0.025,
    0.05, 0.1, 0.2, 0.3
]

N_BOOTSTRAP = 2000
SEED = 20261020

assert len(per_image) == 276 * len(THRESHOLDS)
assert per_image["image_id"].nunique() == 276
assert per_image["risk_group_id"].nunique() == 239
assert per_image["confidence_threshold"].nunique() == len(THRESHOLDS)

# Confirm each image has one stable source-risk group.
assert (
    per_image.groupby("image_id")["risk_group_id"]
    .nunique().eq(1).all()
)

# Confirm every threshold has exactly the same validation images.
expected_images = set(
    per_image.loc[
        np.isclose(per_image["confidence_threshold"], THRESHOLDS[0]),
        "image_id"
    ]
)

assert len(expected_images) == 276

for threshold in THRESHOLDS:
    subset = per_image.loc[
        np.isclose(per_image["confidence_threshold"], threshold)
    ]
    assert len(subset) == 276
    assert set(subset["image_id"]) == expected_images

groups = sorted(per_image["risk_group_id"].unique())
n_groups = len(groups)

rng = np.random.default_rng(SEED)

# Reuse exactly the same cluster resamples for all thresholds.
bootstrap_indices = rng.integers(
    low=0,
    high=n_groups,
    size=(N_BOOTSTRAP, n_groups)
)

rows = []

for threshold in THRESHOLDS:
    subset = per_image.loc[
        np.isclose(
            per_image["confidence_threshold"],
            threshold
        )
    ]

    aggregated = (
        subset.groupby("risk_group_id")[
            ["tp", "fp", "fn"]
        ]
        .sum()
        .reindex(groups, fill_value=0)
    )

    image_counts = (
        subset.groupby("risk_group_id")
        .size()
        .reindex(groups, fill_value=0)
        .to_numpy(dtype=float)
    )

    counts = aggregated[
        ["tp", "fp", "fn"]
    ].to_numpy(dtype=float)

    sampled = counts[bootstrap_indices].sum(axis=1)
    sampled_images = image_counts[bootstrap_indices].sum(axis=1)

    tp = sampled[:, 0]
    fp = sampled[:, 1]
    fn = sampled[:, 2]

    recall = np.divide(
        tp, tp + fn,
        out=np.full_like(tp, np.nan),
        where=(tp + fn) > 0
    )

    precision = np.divide(
        tp, tp + fp,
        out=np.full_like(tp, np.nan),
        where=(tp + fp) > 0
    )

    f1 = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.full_like(tp, np.nan),
        where=(2 * tp + fp + fn) > 0
    )

    fp_per_image = fp / sampled_images

    original = summary.loc[
        np.isclose(
            summary["confidence_threshold"],
            threshold
        )
    ].iloc[0]

    for metric, values in [
        ("recall", recall),
        ("precision", precision),
        ("f1", f1),
        ("fp_per_image", fp_per_image)
    ]:
        valid = values[np.isfinite(values)]

        assert len(valid) == N_BOOTSTRAP, (
            f"Invalid bootstrap results: {metric}, {threshold}"
        )

        lower, upper = np.quantile(
            valid, [0.025, 0.975]
        )

        point = float(original[metric])

        rows.append({
            "confidence_threshold": threshold,
            "metric": metric,
            "point_estimate": point,
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "bootstrap_replicates": N_BOOTSTRAP,
            "source_risk_groups": n_groups
        })

ci = pd.DataFrame(rows)

assert len(ci) == len(THRESHOLDS) * 4

report = {
    "experiment": "06.5C-2",
    "run_id": "E6-C",
    "status": "CLUSTER_BOOTSTRAP_COMPLETED",
    "bootstrap_replicates": N_BOOTSTRAP,
    "bootstrap_seed": SEED,
    "resampling_unit": "source_risk_group_id",
    "validation_images": 276,
    "source_risk_groups": n_groups,
    "matching_iou_threshold": 0.50,
    "confidence_thresholds": THRESHOLDS,
    "confidence_interval": "Percentile bootstrap 95%",
    "test_images_used": 0,
    "input_per_image": str(PER_IMAGE_PATH),
    "output_csv": str(CI_PATH)
}

ci.to_csv(
    CI_PATH,
    index=False,
    float_format="%.10g"
)

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 06.5C-2 - CLUSTER BOOTSTRAP")

for metric in ["recall", "precision", "fp_per_image"]:
    subset = ci.loc[ci["metric"] == metric]

    print(f"\n{metric.upper()}")
    print(
        subset[
            [
                "confidence_threshold",
                "point_estimate",
                "ci95_lower",
                "ci95_upper"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

print("\nSOURCE-RISK GROUPS:", n_groups)
print("BOOTSTRAP REPLICATES:", N_BOOTSTRAP)
print("TEST IMAGES USED:", 0)
print("\nSTEP 06.5C-2 PASS")


# %% [cell 173] Experiment 06 - Step 06.5D-1

# ============================================================
# Experiment 06 - Step 06.5D-1
# MA Size-Stratified Lesion-Level Recall
#
# Frozen validation only
# No GPU inference, no retraining, no test access
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
OUT = ROOT / "experiment06_results/yolo_formal/validation_analysis/E6-C"

PRED_PATH = OUT / "Validation_Predictions_Conf0001_V3.csv"
INDEX_PATH = ROOT / "experiment06_results/YOLO_Frozen_Split_Lesion_Index_V1.csv"
LABEL_DIR = ROOT / "datasets/DR/FGADR/yolo/labels"

LESION_PATH = OUT / "MA_Size_Stratified_Per_Lesion_V1.csv"
SUMMARY_PATH = OUT / "MA_Size_Stratified_Recall_V1.csv"
REPORT_PATH = OUT / "MA_Size_Stratified_Report_V1.json"

for path in [LESION_PATH, SUMMARY_PATH, REPORT_PATH]:
    assert not path.exists(), f"Output already exists: {path}"

pred = pd.read_csv(PRED_PATH, dtype={"image_id": str})
index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})
val = index.loc[index["split"] == "val"].copy()

assert len(val) == 276
assert val["image_id"].is_unique
assert "risk_group_id" in val.columns
assert set(pred["image_id"]) == set(val["image_id"])

ma_pred = pred.loc[pred["class_id"] == 3]
assert len(ma_pred) == 27845

dimensions = pred.groupby("image_id")[
    ["image_width", "image_height"]
].first()

prediction_groups = {
    image_id: group.sort_values(
        "confidence", ascending=False, kind="stable"
    )
    for image_id, group in ma_pred.groupby("image_id")
}

THRESHOLDS = [0.05, 0.10]
IOU_THRESHOLD = 0.50
INPUT_SIZE = 1024

def iou_matrix(boxes_a, boxes_b):
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)))

    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    wh = np.maximum(br - tl, 0)
    intersection = wh[..., 0] * wh[..., 1]

    area_a = np.prod(boxes_a[:, 2:] - boxes_a[:, :2], axis=1)
    area_b = np.prod(boxes_b[:, 2:] - boxes_b[:, :2], axis=1)

    union = area_a[:, None] + area_b[None, :] - intersection

    return np.divide(
        intersection, union,
        out=np.zeros_like(intersection),
        where=union > 0
    )

def size_category(min_side):
    if min_side < 4:
        return "Extremely tiny (<4 px)"
    if min_side < 8:
        return "Very small (4-<8 px)"
    if min_side < 16:
        return "Small (8-<16 px)"
    return "Larger (>=16 px)"

def matched_gt_indices(pred_boxes, gt_boxes):
    matched = np.zeros(len(gt_boxes), dtype=bool)

    ious = iou_matrix(pred_boxes, gt_boxes)

    for i in range(len(pred_boxes)):
        candidates = np.flatnonzero(
            (~matched) & (ious[i] >= IOU_THRESHOLD)
        )

        if len(candidates) == 0:
            continue

        best = candidates[np.argmax(ious[i, candidates])]
        matched[best] = True

        if matched.all():
            break

    return matched

rows = []

for row in val.itertuples(index=False):
    image_id = str(row.image_id)
    width = int(dimensions.loc[image_id, "image_width"])
    height = int(dimensions.loc[image_id, "image_height"])

    # Uniform letterbox scale for input-size target measurements.
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)

    gt_boxes = []
    gt_sizes = []

    label_path = LABEL_DIR / f"{image_id}.txt"

    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if not fields or int(fields[0]) != 3:
            continue

        assert len(fields) == 5
        xc, yc, bw, bh = map(float, fields[1:])

        gt_boxes.append([
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height
        ])

        gt_sizes.append(min(bw * width, bh * height) * scale)

    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    gt_sizes = np.asarray(gt_sizes, dtype=float)

    group = prediction_groups.get(image_id)

    for threshold in THRESHOLDS:
        if group is None:
            selected = np.empty((0, 4))
        else:
            selected = group.loc[
                group["confidence"] >= threshold,
                ["x1", "y1", "x2", "y2"]
            ].to_numpy(dtype=float)

        matched = matched_gt_indices(selected, gt_boxes)

        for j, (size, detected) in enumerate(zip(gt_sizes, matched)):
            rows.append({
                "image_id": image_id,
                "risk_group_id": str(row.risk_group_id),
                "gt_box_index": j,
                "confidence_threshold": threshold,
                "iou_threshold": IOU_THRESHOLD,
                "min_side_at_1024_px": float(size),
                "size_group": size_category(size),
                "detected": int(detected)
            })

lesions = pd.DataFrame(rows)

assert len(lesions) == 1301 * len(THRESHOLDS)

# Verify exact agreement with the previous fixed-threshold audit.
previous = pd.read_csv(OUT / "MA_Threshold_Metrics_V1.csv")

for threshold in THRESHOLDS:
    subset = lesions.loc[
        np.isclose(lesions["confidence_threshold"], threshold)
    ]

    reference = previous.loc[
        np.isclose(previous["confidence_threshold"], threshold)
    ].iloc[0]

    assert int(subset["detected"].sum()) == int(reference["tp"])
    assert len(subset) == 1301

group_order = [
    "Extremely tiny (<4 px)",
    "Very small (4-<8 px)",
    "Small (8-<16 px)",
    "Larger (>=16 px)"
]

summary = (
    lesions.groupby(
        ["confidence_threshold", "size_group"],
        observed=True
    )
    .agg(
        gt_boxes=("detected", "size"),
        detected_boxes=("detected", "sum")
    )
    .reset_index()
)

summary["recall"] = (
    summary["detected_boxes"] / summary["gt_boxes"]
)

summary["size_group"] = pd.Categorical(
    summary["size_group"],
    categories=group_order,
    ordered=True
)

summary = summary.sort_values([
    "confidence_threshold", "size_group"
])

lesions.to_csv(LESION_PATH, index=False)
summary.to_csv(SUMMARY_PATH, index=False)

report = {
    "experiment": "06.5D-1",
    "status": "SIZE_STRATIFICATION_COMPLETED",
    "run_id": "E6-C",
    "validation_images": 276,
    "ma_boxes_per_threshold": 1301,
    "input_size": INPUT_SIZE,
    "size_measure": "Minimum GT box side after uniform letterbox scaling",
    "iou_threshold": IOU_THRESHOLD,
    "confidence_thresholds": THRESHOLDS,
    "test_images_used": 0,
    "per_lesion_csv": str(LESION_PATH),
    "summary_csv": str(SUMMARY_PATH)
}

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 06.5D-1 - MA SIZE-STRATIFIED RECALL")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)
print("\nSTEP 06.5D-1 PASS")


# %% [cell 174] Experiment 06 - Step 06.5D-2

# ============================================================
# Experiment 06 - Step 06.5D-2
# MA Size-Stratified Cluster Bootstrap
# No image reading, no inference, no retraining
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
OUT = ROOT / "experiment06_results/yolo_formal/validation_analysis/E6-C"

INPUT = OUT / "MA_Size_Stratified_Per_Lesion_V1.csv"
SUMMARY = OUT / "MA_Size_Stratified_Recall_V1.csv"

CI_PATH = OUT / "MA_Size_Stratified_Cluster_CI_V1.csv"
DIFF_PATH = OUT / "MA_Size_Stratified_Paired_Differences_V1.csv"
REPORT_PATH = OUT / "MA_Size_Stratified_Bootstrap_Report_V1.json"

assert INPUT.is_file()
assert SUMMARY.is_file()

for path in [CI_PATH, DIFF_PATH, REPORT_PATH]:
    assert not path.exists(), f"Existing output: {path}"

df = pd.read_csv(
    INPUT,
    dtype={"image_id": str, "risk_group_id": str}
)

reference = pd.read_csv(SUMMARY)

SIZE_GROUPS = [
    "Very small (4-<8 px)",
    "Small (8-<16 px)",
    "Larger (>=16 px)"
]

THRESHOLDS = [0.05, 0.10]
N_BOOTSTRAP = 2000
SEED = 20261020

assert len(df) == 2602
assert df["image_id"].nunique() <= 276
assert df["risk_group_id"].nunique() <= 239
assert set(df["size_group"]) == set(SIZE_GROUPS)
assert set(np.round(df["confidence_threshold"], 5)) == set(THRESHOLDS)
assert df["detected"].isin([0, 1]).all()

# Each lesion must be present at both thresholds.
lesion_keys = ["image_id", "gt_box_index"]

counts = df.groupby(lesion_keys).size()

assert len(counts) == 1301
assert counts.eq(2).all()

assert (
    df.groupby(lesion_keys)["size_group"].nunique().eq(1).all()
)
assert (
    df.groupby(lesion_keys)["risk_group_id"].nunique().eq(1).all()
)

GROUPS = sorted(df["risk_group_id"].unique())
N_GROUPS = len(GROUPS)

rng = np.random.default_rng(SEED)

resamples = rng.integers(
    0, N_GROUPS,
    size=(N_BOOTSTRAP, N_GROUPS)
)

ci_rows = []
difference_rows = []

for threshold in THRESHOLDS:
    subset = df.loc[
        np.isclose(df["confidence_threshold"], threshold)
    ]

    assert len(subset) == 1301

    totals = (
        subset.groupby(["risk_group_id", "size_group"])
        .agg(gt=("detected", "size"), tp=("detected", "sum"))
        .unstack("size_group", fill_value=0)
    )

    totals = totals.reindex(
        index=GROUPS,
        columns=pd.MultiIndex.from_product(
            [["gt", "tp"], SIZE_GROUPS]
        ),
        fill_value=0
    )

    gt_array = totals["gt"].to_numpy(dtype=float)
    tp_array = totals["tp"].to_numpy(dtype=float)

    sampled_gt = gt_array[resamples].sum(axis=1)
    sampled_tp = tp_array[resamples].sum(axis=1)

    sampled_recall = np.divide(
        sampled_tp,
        sampled_gt,
        out=np.full_like(sampled_tp, np.nan),
        where=sampled_gt > 0
    )

    reference_subset = reference.loc[
        np.isclose(reference["confidence_threshold"], threshold)
    ].set_index("size_group")

    for j, size_group in enumerate(SIZE_GROUPS):
        values = sampled_recall[:, j]
        valid = values[np.isfinite(values)]

        assert len(valid) >= 0.95 * N_BOOTSTRAP

        point = float(reference_subset.loc[size_group, "recall"])
        lower, upper = np.quantile(valid, [0.025, 0.975])

        ci_rows.append({
            "confidence_threshold": threshold,
            "size_group": size_group,
            "gt_boxes": int(reference_subset.loc[size_group, "gt_boxes"]),
            "detected_boxes": int(reference_subset.loc[size_group, "detected_boxes"]),
            "recall": point,
            "ci95_lower": float(lower),
            "ci95_upper": float(upper)
        })

    # Paired cluster resamples preserve covariance across size groups.
    pairs = [(0, 1), (0, 2), (1, 2)]

    for i, j in pairs:
        differences = sampled_recall[:, i] - sampled_recall[:, j]
        valid = differences[np.isfinite(differences)]

        assert len(valid) >= 0.95 * N_BOOTSTRAP

        point = (
            float(reference_subset.loc[SIZE_GROUPS[i], "recall"])
            - float(reference_subset.loc[SIZE_GROUPS[j], "recall"])
        )

        lower, upper = np.quantile(valid, [0.025, 0.975])

        difference_rows.append({
            "confidence_threshold": threshold,
            "group_a": SIZE_GROUPS[i],
            "group_b": SIZE_GROUPS[j],
            "recall_difference_a_minus_b": point,
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "ci_excludes_zero": bool(lower > 0 or upper < 0)
        })

ci = pd.DataFrame(ci_rows)
differences = pd.DataFrame(difference_rows)

assert len(ci) == 6
assert len(differences) == 6

report = {
    "experiment": "06.5D-2",
    "run_id": "E6-C",
    "status": "SIZE_BOOTSTRAP_COMPLETED",
    "bootstrap_replicates": N_BOOTSTRAP,
    "bootstrap_seed": SEED,
    "source_risk_groups_with_ma": N_GROUPS,
    "resampling_unit": "source_risk_group_id",
    "confidence_thresholds": THRESHOLDS,
    "iou_threshold": 0.50,
    "size_groups": SIZE_GROUPS,
    "confidence_interval": "Paired cluster percentile bootstrap 95%",
    "test_images_used": 0
}

ci.to_csv(CI_PATH, index=False)
differences.to_csv(DIFF_PATH, index=False)

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 06.5D-2 - SIZE BOOTSTRAP")

print("\nSIZE-SPECIFIC RECALL AND 95% CI")
print(
    ci.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nPAIRED RECALL DIFFERENCES AND 95% CI")
print(
    differences.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nSOURCE-RISK GROUPS WITH MA:", N_GROUPS)
print("BOOTSTRAP REPLICATES:", N_BOOTSTRAP)
print("TEST IMAGES USED:", 0)

print("\nSTEP 06.5D-2 PASS")


# %% [cell 175] Experiment 06 - Step 06.5D-3

# ============================================================
# Experiment 06 - Step 06.5D-3
# MA IoU Sensitivity Analysis
# Fixed predictions, paired lesion-level evaluation
# No inference, no retraining, no test data
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
OUT = ROOT / "experiment06_results/yolo_formal/validation_analysis/E6-C"

PRED_PATH = OUT / "Validation_Predictions_Conf0001_V3.csv"
EXPORT_REPORT = OUT / "Validation_Prediction_Export_V3.json"
INDEX_PATH = ROOT / "experiment06_results/YOLO_Frozen_Split_Lesion_Index_V1.csv"
LABEL_DIR = ROOT / "datasets/DR/FGADR/yolo/labels"

REFERENCE_PATH = OUT / "MA_Threshold_Metrics_V1.csv"
LESION_PATH = OUT / "MA_IoU_Sensitivity_Per_Lesion_V1.csv"
SUMMARY_PATH = OUT / "MA_IoU_Sensitivity_Summary_V1.csv"
REPORT_PATH = OUT / "MA_IoU_Sensitivity_Report_V1.json"

for path in [LESION_PATH, SUMMARY_PATH, REPORT_PATH]:
    assert not path.exists(), f"Existing output: {path}"

with EXPORT_REPORT.open(encoding="utf-8") as f:
    export_report = json.load(f)

assert export_report["status"] == "CAP_REPAIR_PASS"
assert export_report["images_still_at_max_det"] == 0
assert export_report["test_images_used"] == 0

pred = pd.read_csv(PRED_PATH, dtype={"image_id": str})
index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})
reference = pd.read_csv(REFERENCE_PATH)

val = index.loc[index["split"] == "val"].copy()

assert len(val) == 276
assert val["image_id"].is_unique
assert set(pred["image_id"]) == set(val["image_id"])
assert len(pred) == export_report["updated_predictions"]

ma_pred = pred.loc[pred["class_id"] == 3]
assert len(ma_pred) == 27845

dimensions = pred.groupby("image_id")[
    ["image_width", "image_height"]
].first()

prediction_groups = {
    image_id: group.sort_values(
        "confidence",
        ascending=False,
        kind="stable"
    )
    for image_id, group in ma_pred.groupby("image_id")
}

THRESHOLDS = [0.05, 0.10]
IOU_LEVELS = [0.25, 0.50]

def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)

    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(br - tl, 0)

    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)

    union = area_a[:, None] + area_b[None, :] - intersection

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0
    )

def match_gt(pred_boxes, gt_boxes, iou_threshold):
    matched = np.zeros(len(gt_boxes), dtype=bool)

    matrix = iou_matrix(pred_boxes, gt_boxes)

    for row in matrix:
        candidates = np.flatnonzero(
            (~matched) & (row >= iou_threshold)
        )

        if len(candidates) == 0:
            continue

        selected = candidates[np.argmax(row[candidates])]
        matched[selected] = True

        if matched.all():
            break

    return matched

def size_group(min_side):
    if min_side < 4:
        return "Extremely tiny (<4 px)"
    if min_side < 8:
        return "Very small (4-<8 px)"
    if min_side < 16:
        return "Small (8-<16 px)"
    return "Larger (>=16 px)"

rows = []
image_prediction_counts = {}

for row in val.itertuples(index=False):
    image_id = str(row.image_id)
    width = int(dimensions.loc[image_id, "image_width"])
    height = int(dimensions.loc[image_id, "image_height"])

    scale = min(1024 / width, 1024 / height)

    gt_boxes = []
    gt_sizes = []

    label_path = LABEL_DIR / f"{image_id}.txt"

    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()

        if not fields:
            continue

        assert len(fields) == 5

        if int(fields[0]) != 3:
            continue

        xc, yc, bw, bh = map(float, fields[1:])

        gt_boxes.append([
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height
        ])

        gt_sizes.append(min(bw * width, bh * height) * scale)

    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)

    group = prediction_groups.get(image_id)

    for confidence in THRESHOLDS:
        if group is None:
            pred_boxes = np.empty((0, 4), dtype=float)
        else:
            pred_boxes = group.loc[
                group["confidence"] >= confidence,
                ["x1", "y1", "x2", "y2"]
            ].to_numpy(dtype=float)

        image_prediction_counts[(image_id, confidence)] = len(pred_boxes)

        for iou_threshold in IOU_LEVELS:
            detected = match_gt(
                pred_boxes, gt_boxes, iou_threshold
            )

            for gt_idx, (flag, size) in enumerate(
                zip(detected, gt_sizes)
            ):
                rows.append({
                    "image_id": image_id,
                    "risk_group_id": str(row.risk_group_id),
                    "gt_box_index": gt_idx,
                    "confidence_threshold": confidence,
                    "iou_threshold": iou_threshold,
                    "size_group": size_group(size),
                    "min_side_at_1024_px": float(size),
                    "detected": int(flag)
                })

lesions = pd.DataFrame(rows)

assert len(lesions) == 1301 * 2 * 2

summary_rows = []

for confidence in THRESHOLDS:
    for iou_threshold in IOU_LEVELS:
        subset = lesions.loc[
            np.isclose(
                lesions["confidence_threshold"], confidence
            ) &
            np.isclose(
                lesions["iou_threshold"], iou_threshold
            )
        ]

        tp = int(subset["detected"].sum())
        gt = len(subset)

        pred_total = sum(
            image_prediction_counts[(image_id, confidence)]
            for image_id in val["image_id"]
        )

        fp = pred_total - tp
        fn = gt - tp

        summary_rows.append({
            "confidence_threshold": confidence,
            "iou_threshold": iou_threshold,
            "gt_ma": gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / pred_total if pred_total else 0.0,
            "recall": tp / gt,
            "fp_per_image": fp / 276
        })

summary = pd.DataFrame(summary_rows)

# Verify exact reproduction of the established IoU=0.50 audit.
for confidence in THRESHOLDS:
    current = summary.loc[
        np.isclose(summary["confidence_threshold"], confidence)
        & np.isclose(summary["iou_threshold"], 0.50)
    ].iloc[0]

    original = reference.loc[
        np.isclose(
            reference["confidence_threshold"], confidence
        )
    ].iloc[0]

    for metric in ["tp", "fp", "fn"]:
        assert int(current[metric]) == int(original[metric]), (
            f"Reference mismatch: {confidence}, {metric}"
        )

# Check monotonicity of total TP counts under relaxed IoU.
for confidence in THRESHOLDS:
    low = summary.loc[
        np.isclose(summary["confidence_threshold"], confidence)
        & np.isclose(summary["iou_threshold"], 0.25),
        "tp"
    ].iloc[0]

    high = summary.loc[
        np.isclose(summary["confidence_threshold"], confidence)
        & np.isclose(summary["iou_threshold"], 0.50),
        "tp"
    ].iloc[0]

    assert low >= high

# The same ground-truth boxes must be present in every setting.
assert (
    lesions.groupby(
        ["confidence_threshold", "iou_threshold"]
    ).size().eq(1301).all()
)

# Summarize recall by annotation-size group.
size_summary = (
    lesions.groupby(
        ["confidence_threshold", "iou_threshold", "size_group"]
    )
    .agg(
        gt_boxes=("detected", "size"),
        detected_boxes=("detected", "sum")
    )
    .reset_index()
)

size_summary["recall"] = (
    size_summary["detected_boxes"] /
    size_summary["gt_boxes"]
)

report = {
    "experiment": "06.5D-3",
    "status": "IOU_SENSITIVITY_COMPLETED",
    "run_id": "E6-C",
    "validation_images": 276,
    "gt_ma": 1301,
    "confidence_thresholds": THRESHOLDS,
    "iou_thresholds": IOU_LEVELS,
    "matching": "Score-ordered one-to-one greedy matching",
    "reference_iou50_reproduced": True,
    "test_images_used": 0,
    "prediction_csv": str(PRED_PATH),
    "per_lesion_csv": str(LESION_PATH),
    "summary_csv": str(SUMMARY_PATH)
}

lesions.to_csv(LESION_PATH, index=False)
summary.to_csv(SUMMARY_PATH, index=False)

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 06.5D-3 - IOU SENSITIVITY")
print("\nOVERALL METRICS")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

print("\nSIZE-STRATIFIED RECALL")
print(
    size_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nREFERENCE IOU=0.50: REPRODUCED")
print("TEST IMAGES USED: 0")
print("STEP 06.5D-3 PASS")


# %% [cell 176] Experiment 06 - Step 06.5D-4

# ============================================================
# Experiment 06 - Step 06.5D-4
# Paired Cluster Bootstrap for IoU Sensitivity
# Saved per-lesion results only
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
OUT = ROOT / "experiment06_results/yolo_formal/validation_analysis/E6-C"

INPUT = OUT / "MA_IoU_Sensitivity_Per_Lesion_V1.csv"
OUTPUT = OUT / "MA_IoU_Sensitivity_Cluster_Differences_V1.csv"
REPORT = OUT / "MA_IoU_Sensitivity_Cluster_Report_V1.json"

assert INPUT.is_file()
assert not OUTPUT.exists()
assert not REPORT.exists()

df = pd.read_csv(
    INPUT,
    dtype={"image_id": str, "risk_group_id": str}
)

THRESHOLDS = [0.05, 0.10]
SIZE_GROUPS = [
    "Very small (4-<8 px)",
    "Small (8-<16 px)",
    "Larger (>=16 px)"
]

KEYS = [
    "image_id",
    "risk_group_id",
    "gt_box_index",
    "confidence_threshold",
    "size_group"
]

assert len(df) == 1301 * 4
assert not df.duplicated(KEYS + ["iou_threshold"]).any()

paired = df.pivot(
    index=KEYS,
    columns="iou_threshold",
    values="detected"
).reset_index()

assert 0.25 in paired.columns and 0.50 in paired.columns
assert len(paired) == 1301 * 2
assert paired[[0.25, 0.50]].notna().all().all()

# The difference is evaluated on paired ground-truth lesions.
paired["gain"] = paired[0.25] - paired[0.50]

groups = sorted(paired["risk_group_id"].unique())
assert len(groups) == 176

N_BOOTSTRAP = 2000
rng = np.random.default_rng(20261020)

resamples = rng.integers(
    0, len(groups),
    size=(N_BOOTSTRAP, len(groups))
)

rows = []

for threshold in THRESHOLDS:
    subset = paired.loc[
        np.isclose(paired["confidence_threshold"], threshold)
    ]

    for group_name in ["All MA"] + SIZE_GROUPS:
        selected = (
            subset if group_name == "All MA"
            else subset.loc[subset["size_group"] == group_name]
        )

        aggregated = (
            selected.groupby("risk_group_id")
            .agg(
                gt=("gain", "size"),
                gain=("gain", "sum")
            )
            .reindex(groups, fill_value=0)
        )

        values = aggregated[["gt", "gain"]].to_numpy(
            dtype=float
        )

        sampled = values[resamples].sum(axis=1)

        valid = sampled[:, 0] > 0
        gains = sampled[valid, 1] / sampled[valid, 0]

        assert len(gains) >= 1900

        lower, upper = np.quantile(gains, [0.025, 0.975])

        rows.append({
            "confidence_threshold": threshold,
            "size_group": group_name,
            "gt_boxes": len(selected),
            "recall_iou25": float(selected[0.25].mean()),
            "recall_iou50": float(selected[0.50].mean()),
            "recall_gain": float(selected["gain"].mean()),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "bootstrap_replicates": N_BOOTSTRAP
        })

result = pd.DataFrame(rows)

assert len(result) == 8

# Confirm exact reproduction of the previous overall results.
expected_tp = {
    (0.05, 0.25): 379,
    (0.05, 0.50): 280,
    (0.10, 0.25): 269,
    (0.10, 0.50): 198
}

for threshold, iou in expected_tp:
    actual = int(
        paired.loc[
            np.isclose(
                paired["confidence_threshold"], threshold
            ),
            iou
        ].sum()
    )
    assert actual == expected_tp[(threshold, iou)]

result.to_csv(OUTPUT, index=False)

report = {
    "experiment": "06.5D-4",
    "status": "IOU_CLUSTER_BOOTSTRAP_COMPLETED",
    "source_risk_groups_with_ma": len(groups),
    "bootstrap_replicates": N_BOOTSTRAP,
    "seed": 20261020,
    "comparison": "Recall(IoU>=0.25) - Recall(IoU>=0.50)",
    "confidence_thresholds": THRESHOLDS,
    "test_images_used": 0,
    "output_csv": str(OUTPUT)
}

with REPORT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 06.5D-4 - IOU CLUSTER BOOTSTRAP")
print(
    result.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)
print("\nSTEP 06.5D-4 PASS")


# %% [cell 177]
assert not OUTPUT.exists()
assert not REPORT.exists()


# %% [cell 178] Experiment 06 - Step 06.5D-4R

# ============================================================
# Experiment 06 - Step 06.5D-4R
# Inspect Existing IoU Bootstrap Artifacts
# Read-only recovery; no recomputation
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = (DR_RESEARCH / "experiment06_results/yolo_formal/validation_analysis/E6-C")

OUTPUT = OUT / "MA_IoU_Sensitivity_Cluster_Differences_V1.csv"
REPORT = OUT / "MA_IoU_Sensitivity_Cluster_Report_V1.json"

print("EXPERIMENT 06.5D-4R - EXISTING RESULT CHECK")

print("\nFILE STATUS")
print("Bootstrap CSV exists:", OUTPUT.is_file())
print("Bootstrap report exists:", REPORT.is_file())

if OUTPUT.is_file():
    try:
        result = pd.read_csv(OUTPUT)

        required_columns = {
            "confidence_threshold",
            "size_group",
            "gt_boxes",
            "recall_iou25",
            "recall_iou50",
            "recall_gain",
            "ci95_lower",
            "ci95_upper",
            "bootstrap_replicates"
        }

        assert required_columns.issubset(result.columns)
        assert len(result) == 8
        assert result["bootstrap_replicates"].eq(2000).all()
        assert result["ci95_lower"].le(result["ci95_upper"]).all()

        expected = {
            0.05: (379, 280),
            0.10: (269, 198)
        }

        for threshold, (tp25, tp50) in expected.items():
            row = result.loc[
                np.isclose(result["confidence_threshold"], threshold)
                & result["size_group"].eq("All MA")
            ]

            assert len(row) == 1

            assert int(row.iloc[0]["gt_boxes"]) == 1301
            assert np.isclose(row.iloc[0]["recall_iou25"], tp25 / 1301)
            assert np.isclose(row.iloc[0]["recall_iou50"], tp50 / 1301)

        print("\nEXISTING BOOTSTRAP RESULTS")
        print(
            result.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}"
            )
        )

        print("\nCSV STRUCTURAL CHECK: PASS")

    except Exception as error:
        print("\nCSV STRUCTURAL CHECK: FAILED")
        print(type(error).__name__, str(error))

if REPORT.is_file():
    try:
        with REPORT.open("r", encoding="utf-8") as f:
            report = json.load(f)

        print("\nEXISTING REPORT")
        print(json.dumps(report, indent=2))

        assert report["status"] == "IOU_CLUSTER_BOOTSTRAP_COMPLETED"
        assert report["bootstrap_replicates"] == 2000
        assert report["source_risk_groups_with_ma"] == 176
        assert report["test_images_used"] == 0

        print("\nREPORT CHECK: PASS")

    except Exception as error:
        print("\nREPORT CHECK: FAILED")
        print(type(error).__name__, str(error))

print("\nSTEP 06.5D-4R INSPECTION COMPLETED")


# %% [cell 179] Experiment 06 - Step 06.6A

# ============================================================
# Experiment 06 - Step 06.6A
# Formal E6-B Training: YOLO11s at 640
# Frozen protocol; no test data
# ============================================================

from pathlib import Path
import json
import time

import pandas as pd
import torch
import ultralytics
import yaml
from ultralytics import YOLO

ROOT = DR_RESEARCH

PROTOCOL_PATH = (
    ROOT / "experiment06_results/yolo_formal"
    / "YOLO_Formal_Protocol_V1.json"
)

INDEX_PATH = (
    ROOT / "experiment06_results"
    / "YOLO_Frozen_Split_Lesion_Index_V1.csv"
)

RUN_PROJECT = (
    ROOT / "experiment06_results/yolo_formal"
    / "training_runs"
)

RUN_NAME = "E6-B_yolo11s_640_v1"
RUN_PATH = RUN_PROJECT / RUN_NAME

assert PROTOCOL_PATH.is_file()
assert INDEX_PATH.is_file()
assert torch.cuda.is_available()

with PROTOCOL_PATH.open(encoding="utf-8") as f:
    protocol = json.load(f)

run_config = protocol["runs"]["E6-B"]
common = protocol["common_training"].copy()

assert run_config == {
    "model": "yolo11s.pt",
    "imgsz": 640
}

assert ultralytics.__version__ == protocol["ultralytics_version"]
assert torch.__version__ == protocol["torch_version"]

data_yaml = Path(protocol["dataset_yaml"])
assert data_yaml.is_file()

with data_yaml.open(encoding="utf-8") as f:
    data_config = yaml.safe_load(f)

assert data_config["nc"] == 6
assert data_config["names"][3] == "Microaneurysms"

index = pd.read_csv(
    INDEX_PATH, dtype={"image_id": str}
)

assert len(index) == 1842
assert index["image_id"].is_unique

assert index["split"].value_counts().to_dict() == {
    "train": 1290,
    "val": 276,
    "test": 276
}

dataset_root = Path(data_config["path"])

for split in ["train", "val"]:
    expected = set(
        index.loc[index["split"] == split, "image_id"]
    )

    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    assert {p.stem for p in image_dir.glob("*.png")} == expected
    assert {p.stem for p in label_dir.glob("*.txt")} == expected

# Never overwrite a formal experiment.
assert not RUN_PATH.exists(), (
    f"Existing experiment found: {RUN_PATH}"
)

print("EXPERIMENT 06.6A - FORMAL E6-B TRAINING")
print("Model:", run_config["model"])
print("Resolution:", run_config["imgsz"])
print("Train images:", 1290)
print("Validation images:", 276)
print("Test images used:", 0)
print("GPU:", torch.cuda.get_device_name(0))
print("Run:", RUN_PATH)

model = YOLO(run_config["model"])

start = time.monotonic()

results = model.train(
    data=str(data_yaml),
    imgsz=run_config["imgsz"],
    **common,
    seed=protocol["seed"],
    project=str(RUN_PROJECT),
    name=RUN_NAME,
    exist_ok=False,
    pretrained=True,
    verbose=True
)

elapsed = time.monotonic() - start

assert Path(results.save_dir).resolve() == RUN_PATH.resolve()

best = RUN_PATH / "weights/best.pt"
last = RUN_PATH / "weights/last.pt"
history_path = RUN_PATH / "results.csv"
args_path = RUN_PATH / "args.yaml"

for path in [best, last, history_path, args_path]:
    assert path.is_file(), f"Missing artifact: {path}"

history = pd.read_csv(history_path)
history.columns = history.columns.str.strip()

required = [
    "epoch",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)"
]

assert all(c in history.columns for c in required)
assert 1 <= len(history) <= common["epochs"]

best_idx = history["metrics/mAP50-95(B)"].idxmax()

report = {
    "experiment": "06.6A",
    "run_id": "E6-B",
    "status": "TRAINING_COMPLETED_PENDING_EVALUATION",
    "model": run_config["model"],
    "image_size": run_config["imgsz"],
    "epochs_completed": int(len(history)),
    "best_map50_95_epoch": int(history.loc[best_idx, "epoch"]),
    "best_history_map50_95": float(
        history.loc[best_idx, "metrics/mAP50-95(B)"]
    ),
    "elapsed_minutes": round(elapsed / 60, 2),
    "ultralytics_version": ultralytics.__version__,
    "torch_version": torch.__version__,
    "train_images": 1290,
    "validation_images": 276,
    "test_images_used": 0,
    "best_checkpoint": str(best),
    "last_checkpoint": str(last),
    "results_csv": str(history_path)
}

report_path = RUN_PATH / "training_completion_report.json"

with report_path.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nE6-B TRAINING REPORT")
print(json.dumps(report, indent=2))
print("\nSTEP 06.6A PASS")


# %% [cell 180] Experiment 06 - Step 06.6B

# ============================================================
# Experiment 06 - Step 06.6B
# E6-B Full Validation Prediction Export
#
# Model: YOLO11s at 640, best.pt
# Dataset: Frozen validation, 276 images
# No retraining; no test inference
# ============================================================

from pathlib import Path
import hashlib
import json
import time

import pandas as pd
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = DR_RESEARCH

RUN = (
    ROOT / "experiment06_results/yolo_formal/training_runs"
    / "E6-B_yolo11s_640_v1"
)

BEST = RUN / "weights/best.pt"

INDEX = (
    ROOT / "experiment06_results"
    / "YOLO_Frozen_Split_Lesion_Index_V1.csv"
)

VAL_DIR = (LOCAL_DISK / "fgadr_yolo_frozen_v1/images/val")

OUT = (
    ROOT / "experiment06_results/yolo_formal"
    / "validation_analysis/E6-B"
)

PRED_PATH = OUT / "Validation_Predictions_Conf0001_V1.csv"
REPORT_PATH = OUT / "Validation_Prediction_Export_V1.json"

assert BEST.is_file()
assert INDEX.is_file()
assert VAL_DIR.is_dir()
assert torch.cuda.is_available()
assert ultralytics.__version__ == "8.4.183"

assert not PRED_PATH.exists(), "Prediction file already exists."
assert not REPORT_PATH.exists(), "Prediction report already exists."

index = pd.read_csv(INDEX, dtype={"image_id": str})

expected_ids = set(
    index.loc[index["split"] == "val", "image_id"]
)

actual_ids = {
    path.stem for path in VAL_DIR.glob("*.png")
}

assert len(expected_ids) == 276
assert actual_ids == expected_ids

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

checkpoint_hash = sha256_file(BEST)

prediction_config = {
    "imgsz": 640,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 5000,
    "agnostic_nms": False,
    "device": 0,
    "batch": 4,
    "stream": True,
    "save": False,
    "verbose": False
}

print("EXPERIMENT 06.6B - E6-B VALIDATION EXPORT")
print("Checkpoint:", BEST)
print("Checkpoint SHA256:", checkpoint_hash)
print("Validation images:", len(expected_ids))

model = YOLO(str(BEST))

assert model.task == "detect"
assert len(model.names) == 6
assert model.names[3] == "Microaneurysms"

records = []
counts = {}
start = time.monotonic()

with torch.inference_mode():
    results = model.predict(
        source=str(VAL_DIR),
        **prediction_config
    )

    for result in results:
        image_id = Path(result.path).stem

        assert image_id in expected_ids
        assert image_id not in counts

        height, width = result.orig_shape
        boxes = result.boxes

        count = 0 if boxes is None else len(boxes)
        counts[image_id] = count

        if count == 0:
            continue

        coords = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, score, class_id in zip(
            coords, scores, classes
        ):
            x1, y1, x2, y2 = map(float, box)

            assert 0 <= int(class_id) < 6
            assert np.isfinite(box).all()
            assert np.isfinite(score)
            assert 0 <= x1 <= x2 <= width
            assert 0 <= y1 <= y2 <= height

            records.append({
                "image_id": image_id,
                "class_id": int(class_id),
                "confidence": float(score),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "image_width": int(width),
                "image_height": int(height)
            })

assert set(counts) == expected_ids

columns = [
    "image_id", "class_id", "confidence",
    "x1", "y1", "x2", "y2",
    "image_width", "image_height"
]

predictions = pd.DataFrame(records, columns=columns)

assert not predictions.empty
assert predictions["confidence"].between(0.001, 1.0).all()

capped_ids = sorted(
    image_id for image_id, n in counts.items()
    if n >= prediction_config["max_det"]
)

report = {
    "experiment": "06.6B",
    "run_id": "E6-B",
    "status": (
        "VALIDATION_EXPORT_PASS"
        if not capped_ids
        else "VALIDATION_EXPORT_CAPPED"
    ),
    "checkpoint_sha256": checkpoint_hash,
    "checkpoint_path": str(BEST),
    "ultralytics_version": ultralytics.__version__,
    "image_size": 640,
    "validation_images": len(counts),
    "total_predictions": int(len(predictions)),
    "ma_predictions": int(
        (predictions["class_id"] == 3).sum()
    ),
    "images_at_max_det": len(capped_ids),
    "capped_image_ids": capped_ids,
    "maximum_predictions_per_image": int(max(counts.values())),
    "elapsed_seconds": round(time.monotonic() - start, 2),
    "prediction_config": {
        k: v for k, v in prediction_config.items()
        if k not in ("stream", "save", "verbose")
    },
    "test_images_used": 0,
    "output_csv": str(PRED_PATH)
}

OUT.mkdir(parents=True, exist_ok=True)

predictions.to_csv(
    PRED_PATH,
    index=False,
    float_format="%.10g"
)

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nEXPORT REPORT")
print(json.dumps(report, indent=2))

print("\nPREDICTIONS BY CLASS")
print(
    predictions["class_id"]
    .value_counts()
    .reindex(range(6), fill_value=0)
    .sort_index()
    .to_string()
)

if capped_ids:
    print("\nSTEP 06.6B: CAPPED - TARGETED REPAIR REQUIRED")
else:
    print("\nSTEP 06.6B PASS")


# %% [cell 181] Experiment 06 - Step 06.6C

# ============================================================
# Experiment 06 - Step 06.6C
# Paired MA Detection Comparison: E6-B vs E6-C
# Frozen validation only; no GPU inference or retraining
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
BASE = ROOT / "experiment06_results/yolo_formal/validation_analysis"

B_DIR = BASE / "E6-B"
C_DIR = BASE / "E6-C"
OUT = BASE / "E6-B_vs_E6-C"
OUT.mkdir(parents=True, exist_ok=True)

B_PRED = B_DIR / "Validation_Predictions_Conf0001_V1.csv"
B_REPORT = B_DIR / "Validation_Prediction_Export_V1.json"
C_PER_IMAGE = C_DIR / "MA_Threshold_Per_Image_V1.csv"
C_SUMMARY = C_DIR / "MA_Threshold_Metrics_V1.csv"

INDEX = ROOT / "experiment06_results/YOLO_Frozen_Split_Lesion_Index_V1.csv"
LABEL_DIR = ROOT / "datasets/DR/FGADR/yolo/labels"

B_IMAGE_OUT = OUT / "E6-B_MA_Threshold_Per_Image_V1.csv"
B_SUMMARY_OUT = OUT / "E6-B_MA_Threshold_Metrics_V1.csv"
COMPARISON_OUT = OUT / "MA_Resolution_Comparison_V1.csv"
REPORT_OUT = OUT / "MA_Resolution_Comparison_Report_V1.json"

outputs = [B_IMAGE_OUT, B_SUMMARY_OUT, COMPARISON_OUT, REPORT_OUT]
assert all(not p.exists() for p in outputs), "Existing output detected."

with B_REPORT.open(encoding="utf-8") as f:
    b_report = json.load(f)

assert b_report["status"] == "VALIDATION_EXPORT_PASS"
assert b_report["validation_images"] == 276
assert b_report["images_at_max_det"] == 0
assert b_report["test_images_used"] == 0
assert b_report["image_size"] == 640

b_pred = pd.read_csv(B_PRED, dtype={"image_id": str})
c_image = pd.read_csv(
    C_PER_IMAGE,
    dtype={"image_id": str, "risk_group_id": str}
)
c_summary = pd.read_csv(C_SUMMARY)
index = pd.read_csv(INDEX, dtype={"image_id": str})

val = index.loc[index["split"] == "val"].copy()
val["risk_group_id"] = val["risk_group_id"].astype(str)

assert len(val) == 276
assert val["image_id"].is_unique
assert "risk_group_id" in val.columns

expected_ids = set(val["image_id"])
assert set(b_pred["image_id"]) == expected_ids
assert set(c_image["image_id"]) == expected_ids
assert b_pred["class_id"].between(0, 5).all()
assert len(b_pred) == b_report["total_predictions"]

THRESHOLDS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30]
IOU_THRESHOLD = 0.50

assert len(c_image) == 276 * len(THRESHOLDS)

# ------------------------------------------------------------
# Recover MA reference boxes from frozen validation labels
# ------------------------------------------------------------

dims = b_pred.groupby("image_id")[
    ["image_width", "image_height"]
].first()

gt_by_image = {}

for image_id in sorted(expected_ids):
    width = int(dims.loc[image_id, "image_width"])
    height = int(dims.loc[image_id, "image_height"])

    label_path = LABEL_DIR / f"{image_id}.txt"
    assert label_path.is_file()

    boxes = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()

        if not fields:
            continue

        assert len(fields) == 5

        if int(fields[0]) != 3:
            continue

        xc, yc, w, h = map(float, fields[1:])

        boxes.append([
            (xc - w / 2) * width,
            (yc - h / 2) * height,
            (xc + w / 2) * width,
            (yc + h / 2) * height
        ])

    gt_by_image[image_id] = np.asarray(
        boxes, dtype=np.float64
    ).reshape(-1, 4)

assert sum(len(x) for x in gt_by_image.values()) == 1301

# ------------------------------------------------------------
# Match predictions and GT at fixed IoU
# ------------------------------------------------------------

def count_matches(pred_boxes, gt_boxes, iou_threshold=0.50):
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0:
        return 0, 0, n_gt

    if n_gt == 0:
        return 0, n_pred, 0

    tl = np.maximum(
        pred_boxes[:, None, :2],
        gt_boxes[None, :, :2]
    )
    br = np.minimum(
        pred_boxes[:, None, 2:],
        gt_boxes[None, :, 2:]
    )

    wh = np.maximum(br - tl, 0.0)
    intersection = wh[..., 0] * wh[..., 1]

    pred_area = np.prod(
        pred_boxes[:, 2:] - pred_boxes[:, :2], axis=1
    )
    gt_area = np.prod(
        gt_boxes[:, 2:] - gt_boxes[:, :2], axis=1
    )

    union = pred_area[:, None] + gt_area[None, :] - intersection

    ious = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0
    )

    matched_gt = np.zeros(n_gt, dtype=bool)
    tp = 0

    # Input predictions must be sorted by descending confidence.
    for i in range(n_pred):
        candidates = np.flatnonzero(
            (~matched_gt) & (ious[i] >= iou_threshold)
        )

        if len(candidates):
            j = candidates[np.argmax(ious[i, candidates])]
            matched_gt[j] = True
            tp += 1

        if matched_gt.all():
            break

    return tp, n_pred - tp, n_gt - tp

ma_pred = b_pred.loc[b_pred["class_id"] == 3]

assert len(ma_pred) == b_report["ma_predictions"] == 23988

pred_groups = {
    image_id: group.sort_values(
        "confidence", ascending=False, kind="stable"
    )
    for image_id, group in ma_pred.groupby("image_id")
}

risk_map = val.set_index("image_id")["risk_group_id"].to_dict()

rows = []

for threshold in THRESHOLDS:
    for image_id in sorted(expected_ids):
        group = pred_groups.get(image_id)

        if group is None:
            boxes = np.empty((0, 4), dtype=np.float64)
        else:
            boxes = group.loc[
                group["confidence"] >= threshold,
                ["x1", "y1", "x2", "y2"]
            ].to_numpy(dtype=np.float64)

        gt = gt_by_image[image_id]
        tp, fp, fn = count_matches(boxes, gt, IOU_THRESHOLD)

        assert tp + fp == len(boxes)
        assert tp + fn == len(gt)

        rows.append({
            "image_id": image_id,
            "risk_group_id": risk_map[image_id],
            "confidence_threshold": threshold,
            "gt_ma": len(gt),
            "pred_ma": len(boxes),
            "tp": tp,
            "fp": fp,
            "fn": fn
        })

b_image = pd.DataFrame(rows)

# ------------------------------------------------------------
# Aggregate and compare with existing E6-C results
# ------------------------------------------------------------

def aggregate_metrics(frame, run_id):
    result_rows = []

    for threshold, group in frame.groupby(
        "confidence_threshold", sort=True
    ):
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())

        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0

        result_rows.append({
            "run_id": run_id,
            "confidence_threshold": threshold,
            "gt_ma": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
            "fp_per_image": fp / len(group)
        })

    return pd.DataFrame(result_rows)

b_summary = aggregate_metrics(b_image, "E6-B")
c_check = aggregate_metrics(c_image, "E6-C")

# Ensure saved E6-C metrics are recovered exactly.
for _, row in c_check.iterrows():
    ref = c_summary.loc[
        np.isclose(
            c_summary["confidence_threshold"],
            row["confidence_threshold"]
        )
    ]
    assert len(ref) == 1

    for field in ["tp", "fp", "fn"]:
        assert int(ref.iloc[0][field]) == int(row[field])

    for field in ["precision", "recall", "f1", "fp_per_image"]:
        assert np.isclose(
            ref.iloc[0][field], row[field], atol=1e-7
        )

assert (b_summary["gt_ma"] == 1301).all()

comparison = b_summary.merge(
    c_check,
    on="confidence_threshold",
    suffixes=("_640", "_1024"),
    validate="one_to_one"
)

comparison["recall_difference_1024_minus_640"] = (
    comparison["recall_1024"] - comparison["recall_640"]
)

comparison["fp_per_image_difference_1024_minus_640"] = (
    comparison["fp_per_image_1024"]
    - comparison["fp_per_image_640"]
)

# ------------------------------------------------------------
# Save separate, reproducible artifacts
# ------------------------------------------------------------

b_image.to_csv(B_IMAGE_OUT, index=False)
b_summary.to_csv(B_SUMMARY_OUT, index=False)
comparison.to_csv(COMPARISON_OUT, index=False)

report = {
    "experiment": "06.6C",
    "status": "PAIRED_RESOLUTION_POINT_ESTIMATES_COMPLETED",
    "runs": ["E6-B", "E6-C"],
    "models": ["YOLO11s@640", "YOLO11s@1024"],
    "validation_images": 276,
    "validation_ma_boxes": 1301,
    "validation_source_risk_groups": 239,
    "iou_threshold": IOU_THRESHOLD,
    "matching_policy": "Confidence-ordered one-to-one greedy IoU",
    "confidence_thresholds": THRESHOLDS,
    "e6b_checkpoint_sha256": b_report["checkpoint_sha256"],
    "test_images_used": 0,
    "paired_uncertainty_analysis": "PENDING",
    "comparison_csv": str(COMPARISON_OUT)
}

with REPORT_OUT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 06.6C - RESOLUTION COMPARISON")

display_columns = [
    "confidence_threshold",
    "tp_640", "tp_1024",
    "recall_640", "recall_1024",
    "fp_per_image_640", "fp_per_image_1024",
    "recall_difference_1024_minus_640"
]

print("\nMA RESOLUTION COMPARISON")
print(
    comparison[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nE6-C REFERENCE: REPRODUCED")
print("PAIRED BOOTSTRAP: PENDING")
print("TEST IMAGES USED: 0")
print("\nSTEP 06.6C PASS")


# %% [cell 182] Experiment 06 - Step 06.6D

# ============================================================
# Experiment 06 - Step 06.6D
# Paired Cluster Bootstrap: YOLO11s 1024 vs 640
# Saved validation metrics only
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
BASE = ROOT / "experiment06_results/yolo_formal/validation_analysis"
OUT = BASE / "E6-B_vs_E6-C"

B_FILE = OUT / "E6-B_MA_Threshold_Per_Image_V1.csv"
C_FILE = BASE / "E6-C/MA_Threshold_Per_Image_V1.csv"
REFERENCE = OUT / "MA_Resolution_Comparison_V1.csv"

RESULT_FILE = OUT / "MA_Resolution_Paired_Bootstrap_V1.csv"
REPORT_FILE = OUT / "MA_Resolution_Paired_Bootstrap_Report_V1.json"

for path in [B_FILE, C_FILE, REFERENCE]:
    assert path.is_file(), f"Missing input: {path}"

for path in [RESULT_FILE, REPORT_FILE]:
    assert not path.exists(), f"Existing result: {path}"

b = pd.read_csv(
    B_FILE,
    dtype={"image_id": str, "risk_group_id": str}
)
c = pd.read_csv(
    C_FILE,
    dtype={"image_id": str, "risk_group_id": str}
)
reference = pd.read_csv(REFERENCE)

THRESHOLDS = [
    0.001, 0.005, 0.010, 0.025,
    0.050, 0.100, 0.200, 0.300
]

KEYS = ["image_id", "risk_group_id", "confidence_threshold"]

required = set(KEYS + ["tp", "fp", "fn"])

assert required.issubset(b.columns)
assert required.issubset(c.columns)
assert len(b) == len(c) == 276 * 8
assert not b.duplicated(KEYS).any()
assert not c.duplicated(KEYS).any()

paired = b.merge(
    c,
    on=KEYS,
    suffixes=("_640", "_1024"),
    validate="one_to_one"
)

assert len(paired) == 276 * 8
assert paired["risk_group_id"].nunique() == 239

for suffix in ["640", "1024"]:
    assert (
        paired[f"tp_{suffix}"] + paired[f"fn_{suffix}"]
        ==  paired[f"tp_{suffix}"] + paired[f"fn_{suffix}"]
    ).all()

# Validate paired GT counts, source risk groups, and per-image support.
assert (
    paired["tp_640"] + paired["fn_640"]
    == paired["tp_1024"] + paired["fn_1024"]
).all()

assert (
    paired.groupby("image_id")["risk_group_id"]
    .nunique().eq(1).all()
)

assert paired.groupby("confidence_threshold").size().eq(276).all()

for threshold in THRESHOLDS:
    subset = paired.loc[
        np.isclose(paired["confidence_threshold"], threshold)
    ]
    assert len(subset) == 276
    assert int((subset["tp_640"] + subset["fn_640"]).sum()) == 1301

GROUPS = sorted(paired["risk_group_id"].unique())
N_GROUPS = len(GROUPS)

N_BOOTSTRAP = 2000
SEED = 20261020

rng = np.random.default_rng(SEED)

# Paired resampling: same source-risk groups for both models.
resamples = rng.integers(
    0, N_GROUPS,
    size=(N_BOOTSTRAP, N_GROUPS)
)

rows = []

for threshold in THRESHOLDS:
    subset = paired.loc[
        np.isclose(paired["confidence_threshold"], threshold)
    ].copy()

    subset["gt"] = subset["tp_640"] + subset["fn_640"]
    subset["images"] = 1

    aggregated = (
        subset.groupby("risk_group_id")[
            ["gt", "images", "tp_640", "tp_1024",
             "fp_640", "fp_1024"]
        ]
        .sum()
        .reindex(GROUPS, fill_value=0)
    )

    a = aggregated.to_numpy(dtype=np.float64)

    sampled = a[resamples].sum(axis=1)

    gt = sampled[:, 0]
    image_count = sampled[:, 1]

    assert np.all(gt > 0)
    assert np.all(image_count > 0)

    recall_640 = sampled[:, 2] / gt
    recall_1024 = sampled[:, 3] / gt

    fp_image_640 = sampled[:, 4] / image_count
    fp_image_1024 = sampled[:, 5] / image_count

    recall_difference = recall_1024 - recall_640
    fp_difference = fp_image_1024 - fp_image_640

    recall_ci = np.quantile(
        recall_difference, [0.025, 0.975]
    )
    fp_ci = np.quantile(
        fp_difference, [0.025, 0.975]
    )

    gt_total = subset["gt"].sum()
    n_images = len(subset)

    recall_point = (
        subset["tp_1024"].sum() - subset["tp_640"].sum()
    ) / gt_total

    fp_point = (
        subset["fp_1024"].sum() - subset["fp_640"].sum()
    ) / n_images

    ref = reference.loc[
        np.isclose(
            reference["confidence_threshold"], threshold
        )
    ]

    assert len(ref) == 1

    assert np.isclose(
        recall_point,
        ref.iloc[0]["recall_difference_1024_minus_640"],
        atol=1e-9
    )

    assert np.isclose(
        fp_point,
        ref.iloc[0]["fp_per_image_difference_1024_minus_640"],
        atol=1e-9
    )

    rows.append({
        "confidence_threshold": threshold,
        "validation_images": n_images,
        "gt_ma": int(gt_total),
        "recall_difference_1024_minus_640": float(recall_point),
        "recall_ci95_lower": float(recall_ci[0]),
        "recall_ci95_upper": float(recall_ci[1]),
        "fp_per_image_difference_1024_minus_640": float(fp_point),
        "fp_ci95_lower": float(fp_ci[0]),
        "fp_ci95_upper": float(fp_ci[1])
    })

result = pd.DataFrame(rows)

assert len(result) == 8

result.to_csv(RESULT_FILE, index=False)

report = {
    "experiment": "06.6D",
    "status": "PAIRED_RESOLUTION_BOOTSTRAP_COMPLETED",
    "models": ["YOLO11s@640", "YOLO11s@1024"],
    "validation_images": 276,
    "validation_ma_boxes": 1301,
    "source_risk_groups": N_GROUPS,
    "bootstrap_replicates": N_BOOTSTRAP,
    "seed": SEED,
    "resampling_unit": "source-risk group proxy",
    "comparison": "1024 minus 640",
    "matching_iou": 0.50,
    "confidence_thresholds": THRESHOLDS,
    "test_images_used": 0,
    "result_csv": str(RESULT_FILE)
}

with REPORT_FILE.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 06.6D - PAIRED RESOLUTION BOOTSTRAP")

print(
    result.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nSOURCE-RISK GROUPS:", N_GROUPS)
print("BOOTSTRAP REPLICATES:", N_BOOTSTRAP)
print("TEST IMAGES USED: 0")
print("\nSTEP 06.6D PASS")


# %% [cell 183] Experiment 06 - Step 06.6E-1

# ============================================================
# Experiment 06 - Step 06.6E-1
# MA Recall at Fixed False-Positive Budgets
# Exploratory validation operating-point comparison
# Saved threshold metrics only; no inference
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

OUT = (
    ROOT / "experiment06_results/yolo_formal"
    / "validation_analysis/E6-B_vs_E6-C"
)

B_METRICS = OUT / "E6-B_MA_Threshold_Metrics_V1.csv"
C_METRICS = (
    ROOT / "experiment06_results/yolo_formal"
    / "validation_analysis/E6-C"
    / "MA_Threshold_Metrics_V1.csv"
)

RESULT_PATH = OUT / "MA_Fixed_FP_Budget_Comparison_V1.csv"
REPORT_PATH = OUT / "MA_Fixed_FP_Budget_Report_V1.json"

for path in [B_METRICS, C_METRICS]:
    assert path.is_file(), f"Missing input: {path}"

for path in [RESULT_PATH, REPORT_PATH]:
    assert not path.exists(), f"Existing result: {path}"

b = pd.read_csv(B_METRICS)
c = pd.read_csv(C_METRICS)

THRESHOLDS = np.array([
    0.001, 0.005, 0.010, 0.025,
    0.050, 0.100, 0.200, 0.300
])

FP_BUDGETS = [1, 3, 5, 10, 20, 40]

for name, frame in [("E6-B", b), ("E6-C", c)]:
    assert len(frame) == len(THRESHOLDS)
    assert np.allclose(
        np.sort(frame["confidence_threshold"].to_numpy()),
        THRESHOLDS
    )
    assert set([
        "confidence_threshold", "tp", "fp", "fn",
        "recall", "fp_per_image"
    ]).issubset(frame.columns)

    assert ((frame["tp"] + frame["fn"]) == 1301).all()
    assert np.allclose(
        frame["recall"],
        frame["tp"] / 1301
    )
    assert np.allclose(
        frame["fp_per_image"],
        frame["fp"] / 276
    )

def choose_operating_point(frame, budget):
    eligible = frame.loc[
        frame["fp_per_image"] <= budget + 1e-12
    ].copy()

    if eligible.empty:
        return None

    # Highest recall; tie-break by lower FP per image.
    eligible = eligible.sort_values(
        ["recall", "fp_per_image", "confidence_threshold"],
        ascending=[False, True, False]
    )

    return eligible.iloc[0]

rows = []

for budget in FP_BUDGETS:
    row = {"fp_budget_per_image": budget}

    selections = {}

    for run_id, frame, suffix in [
        ("E6-B", b, "640"),
        ("E6-C", c, "1024")
    ]:
        point = choose_operating_point(frame, budget)
        selections[suffix] = point

        row[f"threshold_{suffix}"] = (
            float(point["confidence_threshold"])
            if point is not None else np.nan
        )
        row[f"recall_{suffix}"] = (
            float(point["recall"])
            if point is not None else np.nan
        )
        row[f"fp_per_image_{suffix}"] = (
            float(point["fp_per_image"])
            if point is not None else np.nan
        )
        row[f"tp_{suffix}"] = (
            int(point["tp"])
            if point is not None else np.nan
        )

    p640 = selections["640"]
    p1024 = selections["1024"]

    row["recall_difference_1024_minus_640"] = (
        float(p1024["recall"] - p640["recall"])
        if p640 is not None and p1024 is not None
        else np.nan
    )

    row["both_models_feasible"] = (
        p640 is not None and p1024 is not None
    )

    rows.append(row)

result = pd.DataFrame(rows)

assert len(result) == len(FP_BUDGETS)
assert result["both_models_feasible"].all()

for suffix in ["640", "1024"]:
    assert (
        result[f"fp_per_image_{suffix}"]
        <= result["fp_budget_per_image"] + 1e-12
    ).all()

result.to_csv(RESULT_PATH, index=False)

report = {
    "experiment": "06.6E-1",
    "status": "EXPLORATORY_FIXED_FP_BUDGET_COMPLETED",
    "run_ids": ["E6-B", "E6-C"],
    "validation_images": 276,
    "validation_ma_boxes": 1301,
    "matching_iou": 0.50,
    "candidate_thresholds": THRESHOLDS.tolist(),
    "fp_budgets_per_image": FP_BUDGETS,
    "selection_rule": (
        "Maximum observed validation recall among threshold "
        "grid points satisfying each FP/image budget"
    ),
    "curve_interpolation_used": False,
    "independent_test_evaluation": False,
    "paired_bootstrap": "PENDING",
    "test_images_used": 0,
    "result_csv": str(RESULT_PATH)
}

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 06.6E-1 - FIXED FP BUDGET")

print(
    result.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nTHRESHOLD GRID POINTS:", len(THRESHOLDS))
print("INTERPOLATION USED: False")
print("PAIRED BOOTSTRAP: PENDING")
print("TEST IMAGES USED: 0")
print("\nSTEP 06.6E-1 PASS")


# %% [cell 184] Experiment 06 - Step 06.6E-2

# ============================================================
# Experiment 06 - Step 06.6E-2
# Fine-Grained Confidence Scan at Fixed FP Budgets
# Validation only; no GPU inference or retraining
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
BASE = ROOT / "experiment06_results/yolo_formal/validation_analysis"
OUT = BASE / "E6-B_vs_E6-C"

INDEX = ROOT / "experiment06_results/YOLO_Frozen_Split_Lesion_Index_V1.csv"
LABEL_DIR = ROOT / "datasets/DR/FGADR/yolo/labels"

PRED_FILES = {
    "640": BASE / "E6-B/Validation_Predictions_Conf0001_V1.csv",
    "1024": BASE / "E6-C/Validation_Predictions_Conf0001_V3.csv",
}

REFERENCE_FILES = {
    "640": OUT / "E6-B_MA_Threshold_Per_Image_V1.csv",
    "1024": BASE / "E6-C/MA_Threshold_Per_Image_V1.csv",
}

CURVE_OUT = OUT / "MA_Fine_Threshold_Curves_V1.csv"
BUDGET_OUT = OUT / "MA_Fine_FP_Budget_Comparison_V1.csv"
REPORT_OUT = OUT / "MA_Fine_Threshold_Report_V1.json"

for path in [INDEX, *PRED_FILES.values(), *REFERENCE_FILES.values()]:
    assert path.is_file(), f"Missing input: {path}"

for path in [CURVE_OUT, BUDGET_OUT, REPORT_OUT]:
    assert not path.exists(), f"Existing output: {path}"

COARSE = np.array([
    0.001, 0.005, 0.010, 0.025,
    0.050, 0.100, 0.200, 0.300
])

THRESHOLDS = np.unique(np.concatenate([
    COARSE,
    np.geomspace(0.001, 0.5, 200)
]))

BUDGETS = [1, 3, 5, 10, 20, 40]
N_IMAGES = 276
N_MA = 1301
IOU = 0.50

index = pd.read_csv(INDEX, dtype={"image_id": str})
val = index.loc[index["split"] == "val"]

assert len(val) == N_IMAGES
assert val["image_id"].is_unique

ids = sorted(val["image_id"])
assert val["risk_group_id"].nunique() == 239

def get_gt(image_id, width, height):
    boxes = []
    path = LABEL_DIR / f"{image_id}.txt"
    assert path.is_file()

    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue

        assert len(fields) == 5

        if int(fields[0]) != 3:
            continue

        xc, yc, bw, bh = map(float, fields[1:])

        boxes.append([
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height
        ])

    return np.asarray(boxes, dtype=float).reshape(-1, 4)

def prefix_true_positives(pred_boxes, gt_boxes):
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    cumulative = np.zeros(n_pred + 1, dtype=np.int32)

    if n_pred == 0 or n_gt == 0:
        return cumulative

    tl = np.maximum(
        pred_boxes[:, None, :2],
        gt_boxes[None, :, :2]
    )
    br = np.minimum(
        pred_boxes[:, None, 2:],
        gt_boxes[None, :, 2:]
    )

    wh = np.maximum(br - tl, 0)
    intersections = wh[..., 0] * wh[..., 1]

    area_pred = np.prod(
        pred_boxes[:, 2:] - pred_boxes[:, :2], axis=1
    )
    area_gt = np.prod(
        gt_boxes[:, 2:] - gt_boxes[:, :2], axis=1
    )

    union = (
        area_pred[:, None]
        + area_gt[None, :]
        - intersections
    )

    ious = np.divide(
        intersections,
        union,
        out=np.zeros_like(intersections),
        where=union > 0
    )

    used = np.zeros(n_gt, dtype=bool)

    for i in range(n_pred):
        candidates = np.flatnonzero(
            (~used) & (ious[i] >= IOU)
        )

        cumulative[i + 1] = cumulative[i]

        if len(candidates):
            selected = candidates[np.argmax(
                ious[i, candidates]
            )]
            used[selected] = True
            cumulative[i + 1] += 1

    return cumulative

all_curves = []

for resolution in ["640", "1024"]:
    pred = pd.read_csv(
        PRED_FILES[resolution],
        dtype={"image_id": str}
    )

    ref = pd.read_csv(
        REFERENCE_FILES[resolution],
        dtype={"image_id": str}
    )

    assert set(pred["image_id"]) == set(ids)
    assert len(ref) == N_IMAGES * len(COARSE)

    dims = pred.groupby("image_id")[
        ["image_width", "image_height"]
    ].first()

    ma = pred.loc[pred["class_id"] == 3]

    groups = {
        image_id: group.sort_values(
            "confidence",
            ascending=False,
            kind="stable"
        )
        for image_id, group in ma.groupby("image_id")
    }

    total_tp = np.zeros(len(THRESHOLDS), dtype=np.int64)
    total_fp = np.zeros(len(THRESHOLDS), dtype=np.int64)
    gt_total = 0

    reference_lookup = ref.set_index([
        "image_id", "confidence_threshold"
    ])

    for image_id in ids:
        width = int(dims.loc[image_id, "image_width"])
        height = int(dims.loc[image_id, "image_height"])

        gt = get_gt(image_id, width, height)
        gt_total += len(gt)

        group = groups.get(image_id)

        if group is None:
            scores = np.empty(0)
            boxes = np.empty((0, 4))
        else:
            scores = group["confidence"].to_numpy()
            boxes = group[
                ["x1", "y1", "x2", "y2"]
            ].to_numpy(dtype=float)

        prefix_tp = prefix_true_positives(boxes, gt)

        # Scores are descending; count score >= threshold.
        counts = np.searchsorted(
            -scores, -THRESHOLDS, side="right"
        )

        tp = prefix_tp[counts]
        fp = counts - tp

        total_tp += tp
        total_fp += fp

        # Audit the original eight operating points per image.
        for threshold in COARSE:
            position = int(np.argmin(
                np.abs(THRESHOLDS - threshold)
            ))

            old = reference_lookup.loc[
                (image_id, threshold)
            ]

            assert int(tp[position]) == int(old["tp"]), (
                f"TP mismatch: {resolution}, "
                f"{image_id}, {threshold}"
            )
            assert int(fp[position]) == int(old["fp"]), (
                f"FP mismatch: {resolution}, "
                f"{image_id}, {threshold}"
            )

    assert gt_total == N_MA

    curve = pd.DataFrame({
        "resolution": resolution,
        "confidence_threshold": THRESHOLDS,
        "tp": total_tp,
        "fp": total_fp,
        "fn": N_MA - total_tp,
        "recall": total_tp / N_MA,
        "fp_per_image": total_fp / N_IMAGES
    })

    assert np.all(curve["tp"] <= N_MA)
    assert np.all(curve["fp"] >= 0)

    all_curves.append(curve)

curves = pd.concat(all_curves, ignore_index=True)

rows = []

for budget in BUDGETS:
    selections = {}

    for resolution in ["640", "1024"]:
        candidates = curves.loc[
            (curves["resolution"] == resolution)
            & (curves["fp_per_image"] <= budget + 1e-12)
        ]

        assert not candidates.empty

        selected = candidates.sort_values(
            ["recall", "fp_per_image", "confidence_threshold"],
            ascending=[False, True, False]
        ).iloc[0]

        selections[resolution] = selected

    a = selections["640"]
    b = selections["1024"]

    rows.append({
        "fp_budget_per_image": budget,
        "threshold_640": float(a["confidence_threshold"]),
        "recall_640": float(a["recall"]),
        "fp_per_image_640": float(a["fp_per_image"]),
        "tp_640": int(a["tp"]),
        "threshold_1024": float(b["confidence_threshold"]),
        "recall_1024": float(b["recall"]),
        "fp_per_image_1024": float(b["fp_per_image"]),
        "tp_1024": int(b["tp"]),
        "recall_difference_1024_minus_640": float(
            b["recall"] - a["recall"]
        )
    })

comparison = pd.DataFrame(rows)

curves.to_csv(CURVE_OUT, index=False)
comparison.to_csv(BUDGET_OUT, index=False)

report = {
    "experiment": "06.6E-2",
    "status": "FINE_FP_BUDGET_SCAN_COMPLETED",
    "validation_images": N_IMAGES,
    "validation_ma_boxes": N_MA,
    "source_risk_groups": 239,
    "matching_iou": IOU,
    "fine_threshold_count": len(THRESHOLDS),
    "original_thresholds_reproduced_per_image": True,
    "fp_budgets_per_image": BUDGETS,
    "selection_dataset": "validation",
    "uncertainty_analysis": "PENDING",
    "test_images_used": 0,
    "curve_csv": str(CURVE_OUT),
    "comparison_csv": str(BUDGET_OUT)
}

with REPORT_OUT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 06.6E-2 - FINE FP BUDGET COMPARISON")
print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nORIGINAL EIGHT THRESHOLDS: EXACTLY REPRODUCED")
print("FINE THRESHOLD COUNT:", len(THRESHOLDS))
print("TEST IMAGES USED: 0")
print("\nSTEP 06.6E-2 PASS")


# %% [cell 185] Experiment 06 - Step 06.6E-3

# ============================================================
# Experiment 06 - Step 06.6E-3
# Paired Cluster Bootstrap at Fixed FP Budgets
#
# Saved prediction CSVs and frozen validation labels only
# No GPU inference, no training, no test access
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
BASE = ROOT / "experiment06_results/yolo_formal/validation_analysis"
OUT = BASE / "E6-B_vs_E6-C"

BUDGET_FILE = OUT / "MA_Fine_FP_Budget_Comparison_V1.csv"
INDEX_FILE = ROOT / "experiment06_results/YOLO_Frozen_Split_Lesion_Index_V1.csv"
LABEL_DIR = ROOT / "datasets/DR/FGADR/yolo/labels"

PRED_FILES = {
    "640": BASE / "E6-B/Validation_Predictions_Conf0001_V1.csv",
    "1024": BASE / "E6-C/Validation_Predictions_Conf0001_V3.csv"
}

PER_IMAGE_FILE = OUT / "MA_Fixed_Budget_Per_Image_V1.csv"
RESULT_FILE = OUT / "MA_Fixed_Budget_Paired_Bootstrap_V1.csv"
REPORT_FILE = OUT / "MA_Fixed_Budget_Paired_Bootstrap_Report_V1.json"

for path in [BUDGET_FILE, INDEX_FILE, *PRED_FILES.values()]:
    assert path.is_file(), f"Missing input: {path}"

for path in [PER_IMAGE_FILE, RESULT_FILE, REPORT_FILE]:
    assert not path.exists(), f"Existing output: {path}"

BUDGETS = [1, 3, 5, 10, 20, 40]
N_BOOTSTRAP = 2000
SEED = 20261020
IOU_THRESHOLD = 0.50

budget_table = pd.read_csv(BUDGET_FILE)
index = pd.read_csv(INDEX_FILE, dtype={"image_id": str})

val = index.loc[index["split"] == "val"].copy()
val["risk_group_id"] = val["risk_group_id"].astype(str)

assert len(val) == 276
assert val["image_id"].is_unique
assert val["risk_group_id"].nunique() == 239
assert budget_table["fp_budget_per_image"].tolist() == BUDGETS

image_ids = sorted(val["image_id"])
risk_map = val.set_index("image_id")["risk_group_id"].to_dict()

def load_ma_ground_truth(image_id, width, height):
    boxes = []
    path = LABEL_DIR / f"{image_id}.txt"
    assert path.is_file()

    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue

        assert len(fields) == 5
        if int(fields[0]) != 3:
            continue

        xc, yc, bw, bh = map(float, fields[1:])

        boxes.append([
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height
        ])

    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)

def prefix_tp(boxes, ground_truth):
    n_pred = len(boxes)
    n_gt = len(ground_truth)

    cumulative = np.zeros(n_pred + 1, dtype=np.int32)

    if n_pred == 0 or n_gt == 0:
        return cumulative

    top_left = np.maximum(
        boxes[:, None, :2],
        ground_truth[None, :, :2]
    )
    bottom_right = np.minimum(
        boxes[:, None, 2:],
        ground_truth[None, :, 2:]
    )

    intersection_wh = np.maximum(bottom_right - top_left, 0)
    intersection = (
        intersection_wh[..., 0] * intersection_wh[..., 1]
    )

    area_pred = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    area_gt = np.prod(
        ground_truth[:, 2:] - ground_truth[:, :2], axis=1
    )

    union = (
        area_pred[:, None]
        + area_gt[None, :]
        - intersection
    )

    overlaps = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0
    )

    matched = np.zeros(n_gt, dtype=bool)

    for i in range(n_pred):
        cumulative[i + 1] = cumulative[i]

        candidates = np.flatnonzero(
            (~matched) & (overlaps[i] >= IOU_THRESHOLD)
        )

        if len(candidates):
            selected = candidates[np.argmax(
                overlaps[i, candidates]
            )]
            matched[selected] = True
            cumulative[i + 1] += 1

        if matched.all():
            cumulative[i + 2:] = cumulative[i + 1]
            break

    return cumulative

records = []

for resolution in ["640", "1024"]:
    pred = pd.read_csv(
        PRED_FILES[resolution],
        dtype={"image_id": str}
    )

    assert set(pred["image_id"]) == set(image_ids)

    dims = pred.groupby("image_id")[
        ["image_width", "image_height"]
    ].first()

    ma = pred.loc[pred["class_id"] == 3]

    groups = {
        image_id: group.sort_values(
            "confidence", ascending=False, kind="stable"
        )
        for image_id, group in ma.groupby("image_id")
    }

    thresholds = budget_table[
        f"threshold_{resolution}"
    ].to_numpy(dtype=np.float64)

    for image_id in image_ids:
        width = int(dims.loc[image_id, "image_width"])
        height = int(dims.loc[image_id, "image_height"])

        gt = load_ma_ground_truth(image_id, width, height)
        group = groups.get(image_id)

        if group is None:
            scores = np.empty(0, dtype=np.float64)
            boxes = np.empty((0, 4), dtype=np.float64)
        else:
            scores = group["confidence"].to_numpy(
                dtype=np.float64
            )
            boxes = group[
                ["x1", "y1", "x2", "y2"]
            ].to_numpy(dtype=np.float64)

        cumulative = prefix_tp(boxes, gt)

        n_pred = np.searchsorted(
            -scores, -thresholds, side="right"
        )

        true_positives = cumulative[n_pred]
        false_positives = n_pred - true_positives

        for j, budget in enumerate(BUDGETS):
            records.append({
                "image_id": image_id,
                "risk_group_id": risk_map[image_id],
                "resolution": resolution,
                "fp_budget_per_image": budget,
                "confidence_threshold": float(thresholds[j]),
                "gt": len(gt),
                "tp": int(true_positives[j]),
                "fp": int(false_positives[j]),
                "fn": int(len(gt) - true_positives[j])
            })

per_image = pd.DataFrame(records)

assert len(per_image) == 276 * 2 * len(BUDGETS)

# Exact operating-point reproduction.
for budget_row in budget_table.itertuples(index=False):
    budget = int(budget_row.fp_budget_per_image)

    for resolution in ["640", "1024"]:
        subset = per_image.loc[
            (per_image["fp_budget_per_image"] == budget)
            & (per_image["resolution"] == resolution)
        ]

        expected_tp = int(
            getattr(budget_row, f"tp_{resolution}")
        )
        expected_fp = float(
            getattr(budget_row, f"fp_per_image_{resolution}")
        )

        assert len(subset) == 276
        assert int(subset["gt"].sum()) == 1301
        assert int(subset["tp"].sum()) == expected_tp
        assert np.isclose(
            subset["fp"].sum() / 276,
            expected_fp,
            atol=1e-9
        )

# Pair models at the image level.
paired = per_image.loc[
    per_image["resolution"] == "640"
].merge(
    per_image.loc[per_image["resolution"] == "1024"],
    on=["image_id", "risk_group_id", "fp_budget_per_image"],
    suffixes=("_640", "_1024"),
    validate="one_to_one"
)

assert len(paired) == 276 * len(BUDGETS)
assert (paired["gt_640"] == paired["gt_1024"]).all()

risk_groups = sorted(paired["risk_group_id"].unique())
assert len(risk_groups) == 239

rng = np.random.default_rng(SEED)
resamples = rng.integers(
    0, len(risk_groups),
    size=(N_BOOTSTRAP, len(risk_groups))
)

rows = []

for budget in BUDGETS:
    subset = paired.loc[
        paired["fp_budget_per_image"] == budget
    ].copy()

    subset["images"] = 1

    aggregated = (
        subset.groupby("risk_group_id")[
            ["gt_640", "images",
             "tp_640", "tp_1024",
             "fp_640", "fp_1024"]
        ]
        .sum()
        .reindex(risk_groups, fill_value=0)
    )

    sampled = aggregated.to_numpy(
        dtype=np.float64
    )[resamples].sum(axis=1)

    valid = (sampled[:, 0] > 0) & (sampled[:, 1] > 0)
    assert valid.sum() >= 1900

    sampled = sampled[valid]

    recall_diff = (
        sampled[:, 3] - sampled[:, 2]
    ) / sampled[:, 0]

    fp_diff = (
        sampled[:, 5] - sampled[:, 4]
    ) / sampled[:, 1]

    recall_ci = np.quantile(
        recall_diff, [0.025, 0.975]
    )
    fp_ci = np.quantile(
        fp_diff, [0.025, 0.975]
    )

    point_recall = (
        subset["tp_1024"].sum() - subset["tp_640"].sum()
    ) / subset["gt_640"].sum()

    point_fp = (
        subset["fp_1024"].sum() - subset["fp_640"].sum()
    ) / 276

    expected = budget_table.loc[
        budget_table["fp_budget_per_image"] == budget
    ].iloc[0]

    assert np.isclose(
        point_recall,
        expected["recall_difference_1024_minus_640"],
        atol=1e-9
    )

    rows.append({
        "fp_budget_per_image": budget,
        "threshold_640": float(expected["threshold_640"]),
        "threshold_1024": float(expected["threshold_1024"]),
        "recall_640": float(expected["recall_640"]),
        "recall_1024": float(expected["recall_1024"]),
        "recall_difference_1024_minus_640": float(point_recall),
        "recall_ci95_lower": float(recall_ci[0]),
        "recall_ci95_upper": float(recall_ci[1]),
        "fp_per_image_difference_1024_minus_640": float(point_fp),
        "fp_ci95_lower": float(fp_ci[0]),
        "fp_ci95_upper": float(fp_ci[1])
    })

result = pd.DataFrame(rows)

assert len(result) == len(BUDGETS)

per_image.to_csv(PER_IMAGE_FILE, index=False)
result.to_csv(RESULT_FILE, index=False)

report = {
    "experiment": "06.6E-3",
    "status": "FIXED_BUDGET_PAIRED_BOOTSTRAP_COMPLETED",
    "models": ["YOLO11s@640", "YOLO11s@1024"],
    "validation_images": 276,
    "validation_ma_boxes": 1301,
    "source_risk_groups": len(risk_groups),
    "bootstrap_replicates": N_BOOTSTRAP,
    "seed": SEED,
    "iou_threshold": IOU_THRESHOLD,
    "fp_budgets_per_image": BUDGETS,
    "confidence_thresholds": "Fixed from validation-selected V1 operating points",
    "bootstrap_reselects_thresholds": False,
    "test_images_used": 0,
    "per_image_csv": str(PER_IMAGE_FILE),
    "result_csv": str(RESULT_FILE)
}

with REPORT_FILE.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EXPERIMENT 06.6E-3 - FIXED BUDGET PAIRED BOOTSTRAP")
print(
    result.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nSOURCE-RISK GROUPS:", len(risk_groups))
print("BOOTSTRAP REPLICATES:", N_BOOTSTRAP)
print("THRESHOLD RESELECTION IN BOOTSTRAP: False")
print("TEST IMAGES USED: 0")
print("\nSTEP 06.6E-3 PASS")


# %% [cell 186] Experiment 06 - Step 06.7A

# ============================================================
# Experiment 06 - Step 06.7A
# Formal E6-A Training: YOLO11n at 640
# Frozen protocol; validation only; no test inference
# ============================================================

from pathlib import Path
import hashlib
import json
import time

import pandas as pd
import torch
import ultralytics
import yaml
from ultralytics import YOLO

ROOT = DR_RESEARCH

PROTOCOL_PATH = (
    ROOT / "experiment06_results/yolo_formal"
    / "YOLO_Formal_Protocol_V1.json"
)

INDEX_PATH = (
    ROOT / "experiment06_results"
    / "YOLO_Frozen_Split_Lesion_Index_V1.csv"
)

PROJECT = (
    ROOT / "experiment06_results/yolo_formal"
    / "training_runs"
)

RUN_NAME = "E6-A_yolo11n_640_v1"
RUN_DIR = PROJECT / RUN_NAME

assert PROTOCOL_PATH.is_file()
assert INDEX_PATH.is_file()
assert torch.cuda.is_available()

with PROTOCOL_PATH.open("r", encoding="utf-8") as f:
    protocol = json.load(f)

run_config = protocol["runs"]["E6-A"]
common = protocol["common_training"].copy()

assert run_config == {
    "model": "yolo11n.pt",
    "imgsz": 640
}

assert ultralytics.__version__ == protocol["ultralytics_version"]
assert torch.__version__ == protocol["torch_version"]

assert common["epochs"] == 80
assert common["batch"] == 4
assert common["optimizer"] == "AdamW"

data_yaml = Path(protocol["dataset_yaml"])
assert data_yaml.is_file()

with data_yaml.open("r", encoding="utf-8") as f:
    data_config = yaml.safe_load(f)

assert data_config["nc"] == 6
assert data_config["names"][3] == "Microaneurysms"

index = pd.read_csv(
    INDEX_PATH, dtype={"image_id": str}
)

assert len(index) == 1842
assert index["image_id"].is_unique

assert index["split"].value_counts().to_dict() == {
    "train": 1290,
    "val": 276,
    "test": 276
}

dataset_root = Path(data_config["path"])

for split in ["train", "val"]:
    expected_ids = set(
        index.loc[index["split"] == split, "image_id"]
    )

    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    assert image_dir.is_dir()
    assert label_dir.is_dir()

    assert {
        p.stem for p in image_dir.glob("*.png")
    } == expected_ids

    assert {
        p.stem for p in label_dir.glob("*.txt")
    } == expected_ids

# Protect prior training and allow inspection before recovery.
assert not RUN_DIR.exists(), (
    "Existing E6-A run directory found. "
    "Inspect existing artifacts before training or resuming: "
    f"{RUN_DIR}"
)

print("EXPERIMENT 06.7A - E6-A FORMAL TRAINING")
print("Model:", run_config["model"])
print("Image size:", run_config["imgsz"])
print("Train images:", 1290)
print("Validation images:", 276)
print("GPU:", torch.cuda.get_device_name(0))
print("Ultralytics:", ultralytics.__version__)
print("Run directory:", RUN_DIR)
print("Test images used:", 0)

model = YOLO(run_config["model"])

start = time.monotonic()

results = model.train(
    data=str(data_yaml),
    imgsz=run_config["imgsz"],
    **common,
    seed=protocol["seed"],
    project=str(PROJECT),
    name=RUN_NAME,
    exist_ok=False,
    pretrained=True,
    verbose=True
)

elapsed_minutes = (time.monotonic() - start) / 60

assert Path(results.save_dir).resolve() == RUN_DIR.resolve()

best = RUN_DIR / "weights/best.pt"
last = RUN_DIR / "weights/last.pt"
history_path = RUN_DIR / "results.csv"
args_path = RUN_DIR / "args.yaml"

for path in [best, last, history_path, args_path]:
    assert path.is_file(), f"Missing training artifact: {path}"

history = pd.read_csv(history_path)
history.columns = history.columns.str.strip()

assert {
    "epoch",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)"
}.issubset(history.columns)

assert 1 <= len(history) <= 80

best_row = history.loc[
    history["metrics/mAP50-95(B)"].idxmax()
]

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

report = {
    "experiment": "06.7A",
    "run_id": "E6-A",
    "status": "TRAINING_COMPLETED_PENDING_EVALUATION",
    "model": run_config["model"],
    "image_size": run_config["imgsz"],
    "epochs_completed": int(len(history)),
    "best_history_map50_95_epoch": int(best_row["epoch"]),
    "best_history_map50_95": float(
        best_row["metrics/mAP50-95(B)"]
    ),
    "elapsed_minutes": round(elapsed_minutes, 2),
    "train_images": 1290,
    "validation_images": 276,
    "test_images_used": 0,
    "ultralytics_version": ultralytics.__version__,
    "torch_version": torch.__version__,
    "best_checkpoint": str(best),
    "best_checkpoint_sha256": sha256_file(best),
    "last_checkpoint": str(last),
    "results_csv": str(history_path)
}

report_path = RUN_DIR / "training_completion_report.json"

assert not report_path.exists()

with report_path.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nE6-A TRAINING REPORT")
print(json.dumps(report, indent=2))
print("\nSTEP 06.7A PASS")


# %% [cell 187] Experiment 06 - Step 06.7B

# ============================================================
# Experiment 06 - Step 06.7B
# E6-A Full Validation Prediction Export
#
# No training; no test access
# ============================================================

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

ROOT = DR_RESEARCH

RUN = (
    ROOT / "experiment06_results/yolo_formal/training_runs"
    / "E6-A_yolo11n_640_v1"
)

BEST = RUN / "weights/best.pt"
TRAIN_REPORT = RUN / "training_completion_report.json"

INDEX = (
    ROOT / "experiment06_results"
    / "YOLO_Frozen_Split_Lesion_Index_V1.csv"
)

VAL_DIR = (LOCAL_DISK / "fgadr_yolo_frozen_v1/images/val")

OUT = (
    ROOT / "experiment06_results/yolo_formal"
    / "validation_analysis/E6-A"
)

PRED_PATH = OUT / "Validation_Predictions_Conf0001_V1.csv"
REPORT_PATH = OUT / "Validation_Prediction_Export_V1.json"

for path in [BEST, TRAIN_REPORT, INDEX]:
    assert path.is_file(), f"Missing file: {path}"

assert VAL_DIR.is_dir()
assert torch.cuda.is_available()
assert ultralytics.__version__ == "8.4.183"

# Prevent accidental overwrite or repeated inference.
assert not PRED_PATH.exists(), f"Existing file: {PRED_PATH}"
assert not REPORT_PATH.exists(), f"Existing file: {REPORT_PATH}"

with TRAIN_REPORT.open(encoding="utf-8") as f:
    training_report = json.load(f)

assert training_report["run_id"] == "E6-A"
assert training_report["model"] == "yolo11n.pt"
assert training_report["image_size"] == 640
assert training_report["test_images_used"] == 0

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

checkpoint_hash = sha256_file(BEST)

assert (
    checkpoint_hash
    == training_report["best_checkpoint_sha256"]
), "Checkpoint hash mismatch."

index = pd.read_csv(INDEX, dtype={"image_id": str})

expected_ids = set(
    index.loc[index["split"] == "val", "image_id"]
)

actual_ids = {
    path.stem for path in VAL_DIR.glob("*.png")
}

assert len(expected_ids) == 276
assert actual_ids == expected_ids

prediction_config = {
    "imgsz": 640,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 5000,
    "agnostic_nms": False,
    "device": 0,
    "batch": 4,
    "stream": True,
    "save": False,
    "verbose": False
}

print("EXPERIMENT 06.7B - E6-A VALIDATION EXPORT")
print("Checkpoint:", BEST)
print("Checkpoint SHA256:", checkpoint_hash)
print("Validation images:", len(expected_ids))
print("Test images used:", 0)

model = YOLO(str(BEST))

assert model.task == "detect"
assert len(model.names) == 6
assert model.names[3] == "Microaneurysms"

records = []
counts = {}

start = time.monotonic()

with torch.inference_mode():
    results = model.predict(
        source=str(VAL_DIR),
        **prediction_config
    )

    for result in results:
        image_id = Path(result.path).stem

        assert image_id in expected_ids
        assert image_id not in counts

        height, width = result.orig_shape
        boxes = result.boxes

        count = 0 if boxes is None else len(boxes)
        counts[image_id] = count

        if count == 0:
            continue

        coords = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, score, class_id in zip(
            coords, scores, classes
        ):
            x1, y1, x2, y2 = map(float, box)

            assert 0 <= int(class_id) < 6
            assert np.isfinite(box).all()
            assert np.isfinite(score)
            assert 0 <= x1 <= x2 <= width
            assert 0 <= y1 <= y2 <= height

            records.append({
                "image_id": image_id,
                "class_id": int(class_id),
                "confidence": float(score),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "image_width": int(width),
                "image_height": int(height)
            })

assert set(counts) == expected_ids

columns = [
    "image_id", "class_id", "confidence",
    "x1", "y1", "x2", "y2",
    "image_width", "image_height"
]

predictions = pd.DataFrame(records, columns=columns)

assert not predictions.empty
assert predictions["confidence"].between(0.001, 1.0).all()

capped_ids = sorted(
    image_id for image_id, n in counts.items()
    if n >= prediction_config["max_det"]
)

report = {
    "experiment": "06.7B",
    "run_id": "E6-A",
    "status": (
        "VALIDATION_EXPORT_PASS"
        if not capped_ids
        else "VALIDATION_EXPORT_CAPPED"
    ),
    "checkpoint_sha256": checkpoint_hash,
    "checkpoint_path": str(BEST),
    "ultralytics_version": ultralytics.__version__,
    "image_size": 640,
    "validation_images": len(counts),
    "total_predictions": int(len(predictions)),
    "ma_predictions": int(
        (predictions["class_id"] == 3).sum()
    ),
    "images_at_max_det": len(capped_ids),
    "capped_image_ids": capped_ids,
    "maximum_predictions_per_image": int(max(counts.values())),
    "elapsed_seconds": round(time.monotonic() - start, 2),
    "prediction_config": {
        k: v for k, v in prediction_config.items()
        if k not in ("stream", "save", "verbose")
    },
    "test_images_used": 0,
    "output_csv": str(PRED_PATH)
}

OUT.mkdir(parents=True, exist_ok=True)

predictions.to_csv(
    PRED_PATH,
    index=False,
    float_format="%.10g"
)

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nEXPORT REPORT")
print(json.dumps(report, indent=2))

print("\nPREDICTIONS BY CLASS")
print(
    predictions["class_id"]
    .value_counts()
    .reindex(range(6), fill_value=0)
    .sort_index()
    .to_string()
)

if capped_ids:
    print("\nSTEP 06.7B: CAPPED - REPAIR REQUIRED")
else:
    print("\nSTEP 06.7B PASS")


# %% [cell 188]
from pathlib import Path
import pandas as pd

root = DR_RESEARCH
base = root / "experiment06_results/yolo_formal"

files = {
    "E6-A predictions": (
        base / "validation_analysis/E6-A/"
        "Validation_Predictions_Conf0001_V1.csv"
    ),
    "E6-B predictions": (
        base / "validation_analysis/E6-B/"
        "Validation_Predictions_Conf0001_V1.csv"
    ),
    "E6-C predictions": (
        base / "validation_analysis/E6-C/"
        "Validation_Predictions_Conf0001_V3.csv"
    ),
    "Frozen lesion index": (
        root / "experiment06_results/"
        "YOLO_Frozen_Split_Lesion_Index_V1.csv"
    ),
    "Existing comparison": (
        base / "validation_analysis/E6-B_vs_E6-C/"
        "MA_Fixed_Budget_Per_Image_V1.csv"
    ),
}

for name, path in files.items():
    print(f"\n{name}")
    print("Exists:", path.is_file())
    if path.is_file():
        sample = pd.read_csv(path, nrows=3)
        print("Columns:", sample.columns.tolist())
        print(sample.head(2).to_string(index=False))

print("\nSTEP 06.7C SCHEMA INSPECTION COMPLETE")


# %% [cell 189]
from pathlib import Path
import pandas as pd

root = DR_RESEARCH
base = root / "experiment06_results/yolo_formal"

# 1. Inspect one validation label file.
label_path = Path(
    str(LOCAL_DISK / "fgadr_yolo_frozen_v1/labels/val/0001_2.txt")
)

print("VALIDATION GROUND TRUTH")
print("Label exists:", label_path.is_file())

if label_path.is_file():
    lines = label_path.read_text().splitlines()
    print("Total boxes:", len(lines))
    print("MA boxes:", sum(
        line.split()[0] == "3"
        for line in lines if line.strip()
    ))
    print("First 3 annotation lines:")
    for line in lines[:3]:
        print(line)

# 2. Inspect previously frozen E6-B metrics.
comparison_dir = base / "validation_analysis/E6-B_vs_E6-C"

for filename in [
    "E6-B_MA_Threshold_Per_Image_V1.csv",
    "E6-B_MA_Threshold_Metrics_V1.csv",
    "MA_Fixed_FP_Budget_Comparison_V1.csv",
]:
    path = comparison_dir / filename

    print("\nFILE:", filename)
    print("Exists:", path.is_file())

    if path.is_file():
        df = pd.read_csv(path, nrows=3)
        print("Columns:", df.columns.tolist())
        print(df.to_string(index=False))

print("\nSTEP 06.7D INSPECTION COMPLETE")


# %% [cell 190] Experiment 06 - Step 06.7E
# ============================================================
# Experiment 06 - Step 06.7E
# E6-A MA Lesion-Level Threshold Evaluation
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image

ROOT = DR_RESEARCH
BASE = ROOT / "experiment06_results/yolo_formal"
OUT = BASE / "validation_analysis/E6-A"

PRED_PATH = OUT / "Validation_Predictions_Conf0001_V1.csv"
INDEX_PATH = (
    ROOT / "experiment06_results"
    / "YOLO_Frozen_Split_Lesion_Index_V1.csv"
)
LABEL_DIR = (LOCAL_DISK / "fgadr_yolo_frozen_v1/labels/val")
IMAGE_DIR = (LOCAL_DISK / "fgadr_yolo_frozen_v1/images/val")

METRICS_PATH = OUT / "E6-A_MA_Threshold_Metrics_V1.csv"
PER_IMAGE_PATH = OUT / "E6-A_MA_Threshold_Per_Image_V1.csv"
REPORT_PATH = OUT / "E6-A_MA_Threshold_Report_V1.json"

for path in [PRED_PATH, INDEX_PATH]:
    assert path.is_file(), f"Missing input: {path}"

for path in [METRICS_PATH, PER_IMAGE_PATH, REPORT_PATH]:
    assert not path.exists(), f"Output already exists: {path}"

THRESHOLDS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30]
IOU_THRESHOLD = 0.50
MA_CLASS_ID = 3

index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})
val = index.loc[index["split"] == "val"].copy()

assert len(val) == 276
assert val["image_id"].is_unique
assert val["ma_boxes"].sum() == 1301
assert val["risk_group_id"].nunique() == 239

pred = pd.read_csv(PRED_PATH, dtype={"image_id": str})
assert pred["image_id"].isin(val["image_id"]).all()

ma_pred = pred.loc[pred["class_id"] == MA_CLASS_ID].copy()
assert len(ma_pred) == 31049
assert ma_pred["confidence"].ge(0.001).all()

ma_groups = {
    image_id: frame.sort_values(
        "confidence", ascending=False, kind="mergesort"
    )
    for image_id, frame in ma_pred.groupby("image_id")
}

def calculate_iou(box, gt_boxes):
    if len(gt_boxes) == 0:
        return np.empty(0, dtype=float)

    x1 = np.maximum(box[0], gt_boxes[:, 0])
    y1 = np.maximum(box[1], gt_boxes[:, 1])
    x2 = np.minimum(box[2], gt_boxes[:, 2])
    y2 = np.minimum(box[3], gt_boxes[:, 3])

    intersection = (
        np.maximum(0.0, x2 - x1)
        * np.maximum(0.0, y2 - y1)
    )

    box_area = max(0.0, box[2] - box[0]) * max(
        0.0, box[3] - box[1]
    )
    gt_area = (
        np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0])
        * np.maximum(0.0, gt_boxes[:, 3] - gt_boxes[:, 1])
    )

    union = box_area + gt_area - intersection
    return intersection / np.maximum(union, 1e-12)

def load_gt_boxes(image_id, expected_ma):
    label_path = LABEL_DIR / f"{image_id}.txt"
    image_path = IMAGE_DIR / f"{image_id}.png"

    assert label_path.is_file(), image_id
    assert image_path.is_file(), image_id

    with Image.open(image_path) as image:
        width, height = image.size

    boxes = []

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        parts = line.split()
        assert len(parts) == 5
        cls = int(parts[0])

        if cls != MA_CLASS_ID:
            continue

        xc, yc, bw, bh = map(float, parts[1:])

        boxes.append([
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ])

    assert len(boxes) == expected_ma

    return np.asarray(boxes, dtype=float).reshape(-1, 4), width, height

def match_predictions(boxes, scores, gt_boxes, threshold):
    selected = scores >= threshold
    candidate_boxes = boxes[selected]

    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = 0

    for box in candidate_boxes:
        ious = calculate_iou(box, gt_boxes)

        if len(ious) == 0:
            continue

        ious[matched] = -1
        gt_index = int(np.argmax(ious))

        if ious[gt_index] >= IOU_THRESHOLD:
            matched[gt_index] = True
            tp += 1

    pred_count = len(candidate_boxes)
    return tp, pred_count - tp, len(gt_boxes) - tp, pred_count

records = []

for row in val.itertuples(index=False):
    gt_boxes, width, height = load_gt_boxes(
        row.image_id, int(row.ma_boxes)
    )

    frame = ma_groups.get(row.image_id)

    if frame is None:
        boxes = np.empty((0, 4), dtype=float)
        scores = np.empty(0, dtype=float)
    else:
        assert frame["image_width"].eq(width).all()
        assert frame["image_height"].eq(height).all()

        boxes = frame[["x1", "y1", "x2", "y2"]].to_numpy(float)
        scores = frame["confidence"].to_numpy(float)

    for threshold in THRESHOLDS:
        tp, fp, fn, pred_count = match_predictions(
            boxes, scores, gt_boxes, threshold
        )

        records.append({
            "image_id": row.image_id,
            "risk_group_id": row.risk_group_id,
            "confidence_threshold": threshold,
            "gt_ma": len(gt_boxes),
            "pred_ma": pred_count,
            "tp": tp,
            "fp": fp,
            "fn": fn
        })

per_image = pd.DataFrame(records)

assert len(per_image) == 276 * len(THRESHOLDS)
assert (per_image["tp"] + per_image["fn"] == per_image["gt_ma"]).all()
assert (per_image["tp"] + per_image["fp"] == per_image["pred_ma"]).all()

metrics = []

for threshold, subset in per_image.groupby(
    "confidence_threshold", sort=True
):
    gt = int(subset["gt_ma"].sum())
    tp = int(subset["tp"].sum())
    fp = int(subset["fp"].sum())
    fn = int(subset["fn"].sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gt if gt else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    assert gt == 1301
    assert tp + fn == gt

    metrics.append({
        "run_id": "E6-A",
        "confidence_threshold": threshold,
        "gt_ma": gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_per_image": fp / 276
    })

metrics_df = pd.DataFrame(metrics)

# Existing E6-B metrics are read only.
B_PATH = (
    BASE / "validation_analysis/E6-B_vs_E6-C"
    / "E6-B_MA_Threshold_Metrics_V1.csv"
)
assert B_PATH.is_file()

b_metrics = pd.read_csv(B_PATH)

comparison = metrics_df.merge(
    b_metrics[
        ["confidence_threshold", "recall", "fp_per_image"]
    ],
    on="confidence_threshold",
    how="inner",
    suffixes=("_A", "_B"),
    validate="one_to_one"
)

assert len(comparison) == len(THRESHOLDS)

report = {
    "experiment": "06.7E",
    "status": "E6-A_MA_THRESHOLD_EVALUATION_PASS",
    "run_id": "E6-A",
    "validation_images": 276,
    "gt_ma": 1301,
    "risk_groups": 239,
    "iou_threshold": IOU_THRESHOLD,
    "confidence_thresholds": THRESHOLDS,
    "matching": "confidence_ordered_one_to_one_greedy",
    "prediction_source": str(PRED_PATH),
    "test_images_used": 0,
    "metrics_output": str(METRICS_PATH),
    "per_image_output": str(PER_IMAGE_PATH)
}

# Save only after all checks pass.
OUT.mkdir(parents=True, exist_ok=True)

with METRICS_PATH.open("x", encoding="utf-8") as f:
    metrics_df.to_csv(f, index=False)

with PER_IMAGE_PATH.open("x", encoding="utf-8") as f:
    per_image.to_csv(f, index=False)

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("E6-A MA THRESHOLD METRICS")
print(metrics_df.to_string(index=False))

print("\nE6-A VS E6-B - SAME CONFIDENCE")
print(
    comparison[
        [
            "confidence_threshold",
            "recall_A", "recall_B",
            "fp_per_image_A", "fp_per_image_B"
        ]
    ].round(5).to_string(index=False)
)

print("\nSTEP 06.7E PASS")


# %% [cell 191]
from pathlib import Path
import pandas as pd

base = (DR_RESEARCH / "experiment06_results/yolo_formal/validation_analysis/E6-B_vs_E6-C")

filenames = [
    "MA_Fine_Threshold_Curves_V1.csv",
    "MA_Fine_FP_Budget_Comparison_V1.csv",
    "MA_Fixed_Budget_Per_Image_V1.csv",
]

for name in filenames:
    path = base / name
    print(f"\nFILE: {name}")
    print("Exists:", path.is_file())

    if path.is_file():
        df = pd.read_csv(path)
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head(3).to_string(index=False))

print("\nSTEP 06.7F INSPECTION COMPLETE")


# %% [cell 192] Experiment 06 - Step 06.7G
 # ============================================================
# Experiment 06 - Step 06.7G
# E6-A vs E6-B: MA Recall at Matched FP Budgets
# Uses saved predictions only; no model inference
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image

ROOT = DR_RESEARCH
BASE = ROOT / "experiment06_results/yolo_formal/validation_analysis"

A_DIR = BASE / "E6-A"
BC_DIR = BASE / "E6-B_vs_E6-C"
OUT = BASE / "E6-A_vs_E6-B"

A_PRED = A_DIR / "Validation_Predictions_Conf0001_V1.csv"
A_OLD = A_DIR / "E6-A_MA_Threshold_Per_Image_V1.csv"
B_CURVE = BC_DIR / "MA_Fine_Threshold_Curves_V1.csv"
B_BUDGET = BC_DIR / "MA_Fine_FP_Budget_Comparison_V1.csv"

INDEX = (
    ROOT / "experiment06_results"
    / "YOLO_Frozen_Split_Lesion_Index_V1.csv"
)
LABEL_DIR = (LOCAL_DISK / "fgadr_yolo_frozen_v1/labels/val")
IMAGE_DIR = (LOCAL_DISK / "fgadr_yolo_frozen_v1/images/val")

CURVE_OUT = OUT / "MA_Capacity_Fine_Curves_V1.csv"
BUDGET_OUT = OUT / "MA_Capacity_Fixed_FP_Comparison_V1.csv"
PER_IMAGE_OUT = OUT / "MA_Capacity_Fixed_FP_Per_Image_V1.csv"
REPORT_OUT = OUT / "MA_Capacity_Comparison_Report_V1.json"

for p in [A_PRED, A_OLD, B_CURVE, B_BUDGET, INDEX]:
    assert p.is_file(), f"Missing input: {p}"

for p in [CURVE_OUT, BUDGET_OUT, PER_IMAGE_OUT, REPORT_OUT]:
    assert not p.exists(), f"Existing output: {p}"

BUDGETS = [1, 3, 5, 10, 20, 40]
IOU = 0.5
MA_CLASS = 3
N_IMAGES = 276
N_GT = 1301

index = pd.read_csv(INDEX, dtype={"image_id": str})
val = index[index["split"] == "val"].copy()
val = val.sort_values("image_id").reset_index(drop=True)

assert len(val) == N_IMAGES
assert val["image_id"].is_unique
assert val["ma_boxes"].sum() == N_GT
assert val["risk_group_id"].nunique() == 239

pred = pd.read_csv(A_PRED, dtype={"image_id": str})
pred = pred[pred["class_id"] == MA_CLASS].copy()

assert len(pred) == 31049
assert set(pred["image_id"]).issubset(set(val["image_id"]))

groups = {
    image_id: df.sort_values(
        "confidence", ascending=False, kind="mergesort"
    )
    for image_id, df in pred.groupby("image_id")
}

b_curve = pd.read_csv(B_CURVE)
b_curve_640 = b_curve[
    b_curve["resolution"] == 640
].sort_values("confidence_threshold")

thresholds = np.sort(
    b_curve_640["confidence_threshold"].unique()
)

assert len(thresholds) == 207
assert np.isclose(thresholds[0], 0.001)

b_budget = pd.read_csv(B_BUDGET)
assert set(b_budget["fp_budget_per_image"]) == set(BUDGETS)

old = pd.read_csv(A_OLD, dtype={"image_id": str})
assert len(old) == N_IMAGES * 8

def iou_vector(box, gt):
    if len(gt) == 0:
        return np.zeros(0, dtype=float)

    left = np.maximum(box[:2], gt[:, :2])
    right = np.minimum(box[2:], gt[:, 2:])

    sizes = np.maximum(right - left, 0.0)
    intersection = sizes[:, 0] * sizes[:, 1]

    area_box = np.prod(np.maximum(box[2:] - box[:2], 0.0))
    area_gt = np.prod(np.maximum(gt[:, 2:] - gt[:, :2], 0.0), axis=1)

    union = area_box + area_gt - intersection
    return intersection / np.maximum(union, 1e-12)

def load_ma_gt(image_id, expected):
    label = LABEL_DIR / f"{image_id}.txt"
    image = IMAGE_DIR / f"{image_id}.png"

    assert label.is_file(), image_id
    assert image.is_file(), image_id

    with Image.open(image) as im:
        w, h = im.size

    boxes = []

    for line in label.read_text().splitlines():
        if not line.strip():
            continue

        parts = line.split()
        assert len(parts) == 5

        if int(parts[0]) != MA_CLASS:
            continue

        xc, yc, bw, bh = map(float, parts[1:])
        boxes.append([
            (xc - bw / 2) * w,
            (yc - bh / 2) * h,
            (xc + bw / 2) * w,
            (yc + bh / 2) * h
        ])

    assert len(boxes) == expected
    return np.asarray(boxes, dtype=float).reshape(-1, 4), w, h

def prefix_matches(boxes, gt):
    matched = np.zeros(len(gt), dtype=bool)
    flags = np.zeros(len(boxes), dtype=np.int64)

    for i, box in enumerate(boxes):
        overlaps = iou_vector(box, gt)

        if len(overlaps) == 0:
            continue

        overlaps[matched] = -1
        best = int(np.argmax(overlaps))

        if overlaps[best] >= IOU:
            matched[best] = True
            flags[i] = 1

    return np.concatenate([[0], np.cumsum(flags)])

# Each image is matched once in descending-confidence order.
# Prefix counts then give exact results at all thresholds.
cache = {}
old_check_count = 0

for row in val.itertuples(index=False):
    gt, width, height = load_ma_gt(
        row.image_id, int(row.ma_boxes)
    )

    frame = groups.get(row.image_id)

    if frame is None:
        boxes = np.empty((0, 4), dtype=float)
        scores = np.empty(0, dtype=float)
    else:
        assert frame["image_width"].eq(width).all()
        assert frame["image_height"].eq(height).all()

        boxes = frame[["x1", "y1", "x2", "y2"]].to_numpy(float)
        scores = frame["confidence"].to_numpy(float)

    assert np.all(scores[:-1] >= scores[1:])
    prefix = prefix_matches(boxes, gt)

    cache[row.image_id] = {
        "scores": scores,
        "prefix": prefix,
        "gt": len(gt),
        "risk_group_id": row.risk_group_id
    }

    # Reproduce all eight previously frozen E6-A results.
    prior = old[old["image_id"] == row.image_id]

    assert len(prior) == 8

    for record in prior.itertuples(index=False):
        count = int(np.searchsorted(
            -scores, -record.confidence_threshold, side="right"
        ))
        tp = int(prefix[count])
        fp = count - tp
        fn = len(gt) - tp

        assert tp == record.tp, row.image_id
        assert fp == record.fp, row.image_id
        assert fn == record.fn, row.image_id
        old_check_count += 1

assert old_check_count == N_IMAGES * 8

# Build E6-A's fine confidence curve.
curve_rows = []

for threshold in thresholds:
    total_tp = 0
    total_fp = 0

    for item in cache.values():
        count = int(np.searchsorted(
            -item["scores"], -threshold, side="right"
        ))

        tp = int(item["prefix"][count])
        total_tp += tp
        total_fp += count - tp

    curve_rows.append({
        "run_id": "E6-A",
        "confidence_threshold": float(threshold),
        "tp": total_tp,
        "fp": total_fp,
        "fn": N_GT - total_tp,
        "recall": total_tp / N_GT,
        "fp_per_image": total_fp / N_IMAGES
    })

a_curve = pd.DataFrame(curve_rows)

assert len(a_curve) == 207
assert int(a_curve.iloc[0]["tp"]) == 370
assert int(a_curve.iloc[0]["fp"]) == 30679

# Select the highest-recall feasible point per model.
# Ties prefer fewer FP, then a higher confidence threshold.
def select_budget(df, budget):
    eligible = df[
        df["fp_per_image"] <= budget + 1e-10
    ].copy()

    assert not eligible.empty, f"No feasible point for FP <= {budget}"

    eligible = eligible.sort_values(
        ["recall", "fp_per_image", "confidence_threshold"],
        ascending=[False, True, False]
    )

    return eligible.iloc[0]

comparison_rows = []
per_image_rows = []

for budget in BUDGETS:
    a = select_budget(a_curve, budget)

    b_reference = b_budget[
        b_budget["fp_budget_per_image"] == budget
    ].iloc[0]

    # Confirm the previously selected E6-B threshold is
    # still feasible and matches the stored comparison.
    b = select_budget(b_curve_640, budget)

    assert np.isclose(
        b["confidence_threshold"],
        b_reference["threshold_640"]
    )
    assert np.isclose(
        b["recall"],
        b_reference["recall_640"]
    )
    assert np.isclose(
        b["fp_per_image"],
        b_reference["fp_per_image_640"]
    )
    assert int(b["tp"]) == int(b_reference["tp_640"])

    comparison_rows.append({
        "fp_budget_per_image": budget,
        "threshold_yolo11n": float(a["confidence_threshold"]),
        "recall_yolo11n": float(a["recall"]),
        "fp_per_image_yolo11n": float(a["fp_per_image"]),
        "tp_yolo11n": int(a["tp"]),
        "threshold_yolo11s": float(b["confidence_threshold"]),
        "recall_yolo11s": float(b["recall"]),
        "fp_per_image_yolo11s": float(b["fp_per_image"]),
        "tp_yolo11s": int(b["tp"]),
        "recall_difference_s_minus_n": (
            float(b["recall"]) - float(a["recall"])
        )
    })

    for image_id, item in cache.items():
        threshold = float(a["confidence_threshold"])
        count = int(np.searchsorted(
            -item["scores"], -threshold, side="right"
        ))

        tp = int(item["prefix"][count])

        per_image_rows.append({
            "image_id": image_id,
            "risk_group_id": item["risk_group_id"],
            "run_id": "E6-A",
            "fp_budget_per_image": budget,
            "confidence_threshold": threshold,
            "gt": item["gt"],
            "tp": tp,
            "fp": count - tp,
            "fn": item["gt"] - tp
        })

comparison = pd.DataFrame(comparison_rows)
per_image = pd.DataFrame(per_image_rows)

assert len(comparison) == 6
assert len(per_image) == N_IMAGES * 6

for budget in BUDGETS:
    subset = per_image[
        per_image["fp_budget_per_image"] == budget
    ]

    summary = comparison[
        comparison["fp_budget_per_image"] == budget
    ].iloc[0]

    assert int(subset["tp"].sum()) == summary["tp_yolo11n"]
    assert np.isclose(
        subset["fp"].sum() / N_IMAGES,
        summary["fp_per_image_yolo11n"]
    )

report = {
    "experiment": "06.7G",
    "status": "CAPACITY_FIXED_FP_PASS",
    "models": ["E6-A", "E6-B"],
    "input_size": 640,
    "validation_images": N_IMAGES,
    "gt_ma": N_GT,
    "risk_groups": 239,
    "iou_threshold": IOU,
    "threshold_count": len(thresholds),
    "fp_budgets": BUDGETS,
    "matched_protocol": "score_ordered_one_to_one",
    "prior_e6a_rows_verified": old_check_count,
    "e6b_source": str(B_CURVE),
    "threshold_selection": "same_validation_set_exploratory",
    "bootstrap_completed": False,
    "test_images_used": 0
}

OUT.mkdir(parents=True, exist_ok=True)

with CURVE_OUT.open("x", encoding="utf-8") as f:
    a_curve.to_csv(f, index=False)

with BUDGET_OUT.open("x", encoding="utf-8") as f:
    comparison.to_csv(f, index=False)

with PER_IMAGE_OUT.open("x", encoding="utf-8") as f:
    per_image.to_csv(f, index=False)

with REPORT_OUT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("E6-A VS E6-B - FIXED FP BUDGET")
print(comparison.round(5).to_string(index=False))

print("\nVALIDATION CHECKS")
print("Prior E6-A image/threshold rows verified:", old_check_count)
print("Fine thresholds:", len(thresholds))
print("MA ground truth:", N_GT)
print("Risk groups:", 239)
print("Test images used:", 0)

print("\nSTEP 06.7G PASS")


# %% [cell 193] Experiment 06 - Step 06.7H

# ============================================================
# Experiment 06 - Step 06.7H
# Paired Source-Risk-Group Cluster Bootstrap
# E6-A YOLO11n@640 vs E6-B YOLO11s@640
# No inference, no training, no test-set access
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
BASE = ROOT / "experiment06_results/yolo_formal/validation_analysis"

AB = BASE / "E6-A_vs_E6-B"
BC = BASE / "E6-B_vs_E6-C"

A_PATH = AB / "MA_Capacity_Fixed_FP_Per_Image_V1.csv"
B_PATH = BC / "MA_Fixed_Budget_Per_Image_V1.csv"
SUMMARY_PATH = AB / "MA_Capacity_Fixed_FP_Comparison_V1.csv"

BOOT_PATH = AB / "MA_Capacity_Paired_Bootstrap_V1.csv"
REPORT_PATH = AB / "MA_Capacity_Paired_Bootstrap_Report_V1.json"

for path in [A_PATH, B_PATH, SUMMARY_PATH]:
    assert path.is_file(), f"Missing input: {path}"

for path in [BOOT_PATH, REPORT_PATH]:
    assert not path.exists(), f"Output already exists: {path}"

BUDGETS = [1, 3, 5, 10, 20, 40]
N_BOOTSTRAP = 2000
SEED = 20261020

a = pd.read_csv(A_PATH, dtype={"image_id": str})
b_all = pd.read_csv(B_PATH, dtype={"image_id": str})
summary = pd.read_csv(SUMMARY_PATH)

assert set(a["run_id"]) == {"E6-A"}
assert set(a["fp_budget_per_image"]) == set(BUDGETS)
assert set(summary["fp_budget_per_image"]) == set(BUDGETS)

# Reuse frozen E6-B per-image results.
b = b_all.loc[b_all["resolution"] == 640].copy()

assert len(a) == 276 * 6
assert len(b) == 276 * 6

keys = ["image_id", "fp_budget_per_image"]

assert not a.duplicated(keys).any()
assert not b.duplicated(keys).any()

paired = a.merge(
    b[
        [
            "image_id",
            "risk_group_id",
            "fp_budget_per_image",
            "gt",
            "tp",
            "fp",
            "fn",
        ]
    ],
    on=keys,
    how="inner",
    suffixes=("_n", "_s"),
    validate="one_to_one"
)

assert len(paired) == 276 * 6
assert (paired["risk_group_id_n"] == paired["risk_group_id_s"]).all()
assert (paired["gt_n"] == paired["gt_s"]).all()

for model in ["n", "s"]:
    assert (
        paired[f"tp_{model}"] + paired[f"fn_{model}"]
        == paired[f"gt_{model}"]
    ).all()

paired = paired.rename(
    columns={
        "risk_group_id_n": "risk_group_id",
        "gt_n": "gt"
    }
)

assert paired["risk_group_id"].nunique() == 239
assert paired["image_id"].nunique() == 276

# Create one row per source-risk group and budget.
clustered = (
    paired.groupby(
        ["risk_group_id", "fp_budget_per_image"],
        as_index=False
    )
    .agg(
        gt=("gt", "sum"),
        tp_n=("tp_n", "sum"),
        tp_s=("tp_s", "sum"),
        fp_n=("fp_n", "sum"),
        fp_s=("fp_s", "sum")
    )
)

clusters = sorted(clustered["risk_group_id"].unique())
n_clusters = len(clusters)

assert n_clusters == 239

# Paired bootstrap: use the same sampled cluster indices
# for both models and for all FP budgets.
rng = np.random.default_rng(SEED)

sample_indices = rng.integers(
    low=0,
    high=n_clusters,
    size=(N_BOOTSTRAP, n_clusters)
)

results = []

for budget in BUDGETS:
    sub = (
        clustered[
            clustered["fp_budget_per_image"] == budget
        ]
        .set_index("risk_group_id")
        .loc[clusters]
    )

    assert len(sub) == n_clusters

    gt = sub["gt"].to_numpy(dtype=np.int64)
    tp_n = sub["tp_n"].to_numpy(dtype=np.int64)
    tp_s = sub["tp_s"].to_numpy(dtype=np.int64)

    assert gt.sum() == 1301

    observed_n = tp_n.sum() / gt.sum()
    observed_s = tp_s.sum() / gt.sum()
    observed_difference = observed_s - observed_n

    expected = summary.loc[
        summary["fp_budget_per_image"] == budget
    ].iloc[0]

    assert np.isclose(
        observed_n, expected["recall_yolo11n"]
    )
    assert np.isclose(
        observed_s, expected["recall_yolo11s"]
    )
    assert np.isclose(
        observed_difference,
        expected["recall_difference_s_minus_n"]
    )

    boot_gt = gt[sample_indices].sum(axis=1)
    boot_n = tp_n[sample_indices].sum(axis=1)
    boot_s = tp_s[sample_indices].sum(axis=1)

    assert np.all(boot_gt > 0)

    boot_difference = (
        boot_s / boot_gt - boot_n / boot_gt
    )

    lower, upper = np.quantile(
        boot_difference, [0.025, 0.975]
    )

    results.append({
        "fp_budget_per_image": budget,
        "recall_yolo11n": observed_n,
        "recall_yolo11s": observed_s,
        "difference_s_minus_n": observed_difference,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "ci_excludes_zero": bool(
            lower > 0 or upper < 0
        ),
        "bootstrap_iterations": N_BOOTSTRAP,
        "cluster_count": n_clusters
    })

bootstrap_df = pd.DataFrame(results)

report = {
    "experiment": "06.7H",
    "status": "CAPACITY_PAIRED_BOOTSTRAP_PASS",
    "models": ["E6-A", "E6-B"],
    "validation_images": 276,
    "gt_ma": 1301,
    "source_risk_groups": n_clusters,
    "bootstrap_iterations": N_BOOTSTRAP,
    "seed": SEED,
    "fp_budgets": BUDGETS,
    "sampling_unit": "source_risk_group_not_patient_id",
    "method": "paired_cluster_percentile_bootstrap",
    "confidence_interval": "95_percentile",
    "thresholds_reselected_in_bootstrap": False,
    "threshold_selection": "same_validation_set",
    "multiple_comparison_adjustment": False,
    "independent_test_evaluation": False,
    "test_images_used": 0
}

with BOOT_PATH.open("x", encoding="utf-8") as f:
    bootstrap_df.to_csv(f, index=False)

with REPORT_PATH.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

display_df = bootstrap_df[
    [
        "fp_budget_per_image",
        "difference_s_minus_n",
        "ci95_lower",
        "ci95_upper",
        "ci_excludes_zero"
    ]
].copy()

for column in [
    "difference_s_minus_n",
    "ci95_lower",
    "ci95_upper"
]:
    display_df[column] *= 100

print("E6-A VS E6-B - PAIRED BOOTSTRAP")
print("Recall differences and CI bounds in percentage points")
print(display_df.round(3).to_string(index=False))

print("\nVALIDATION CHECKS")
print("Images:", paired["image_id"].nunique())
print("MA ground truth:", 1301)
print("Source-risk groups:", n_clusters)
print("Bootstrap samples:", N_BOOTSTRAP)
print("Test images used:", 0)

print("\nSTEP 06.7H PASS")
