# Experiment 05 - Grad-CAM vs microaneurysm masks with a retina-constrained spatial null
#
# Source: FGADR_MA_Mask_Audit.ipynb (master notebook, 392 cells), cells 105-134, run step by step in Google Colab (L4 GPU).
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

# %% [cell 105] Experiment 05 - Step 5.1A

from pathlib import Path
import pandas as pd

# ============================================================
# Experiment 05 - Step 5.1A
# Existing Lesion Mask and Audit File Inventory
# ============================================================

ROOT = DR_RESEARCH

TARGET_DIRS = {
    "Dataset directory": ROOT / "datasets" / "DR",
    "Phase 1 results": ROOT / "phase1_results",
    "Phase 2 results": ROOT / "phase2_results",
    "Phase 3 results": ROOT / "phase3_results",
    "Experiment 02": ROOT / "experiment02_results",
    "Experiment 04": ROOT / "experiment04_results",
}

KEYWORDS = (
    "mask", "lesion", "seg", "micro",
    "ma_", "annotation", "audit", "manifest"
)

print("STEP 5.1A - EXISTING MASK INVENTORY")

for name, folder in TARGET_DIRS.items():
    print(f"\n{name}: {folder}")

    if not folder.is_dir():
        print("Status: NOT FOUND")
        continue

    entries = sorted(
        folder.iterdir(),
        key=lambda p: p.name.lower()
    )

    print("Direct entries:", len(entries))

    matches = [
        entry for entry in entries
        if entry.is_dir()
        or any(
            keyword in entry.name.lower()
            for keyword in KEYWORDS
        )
    ]

    for entry in matches[:40]:
        entry_type = "DIR" if entry.is_dir() else "FILE"
        print(f"  [{entry_type}] {entry.name}")

    if len(matches) > 40:
        print("  Additional entries omitted:", len(matches) - 40)

print("\nKNOWN DATASET SEGMENTATION DIRECTORY")

seg_dir = (
    ROOT / "datasets" / "DR"
    / "FGADR" / "images4seg"
)

print("Expected directory:", seg_dir)
print("Exists:", seg_dir.is_dir())

if seg_dir.is_dir():
    entries = sorted(seg_dir.iterdir())
    print("Direct entries:", len(entries))
    for entry in entries[:25]:
        print(
            "  [DIR]" if entry.is_dir() else "  [FILE]",
            entry.name
        )

print("\nSTEP 5.1A COMPLETED")


# %% [cell 106] Experiment 05 - Step 5.1B

from pathlib import Path
import pandas as pd

# ============================================================
# Experiment 05 - Step 5.1B
# Existing Metadata Schema Inspection
# ============================================================

ROOT = DR_RESEARCH

FILES = {
    "Master manifest": (
        ROOT / "experiment02_results"
        / "FGADR_Master_Manifest.csv"
    ),
    "MA quality categories": (
        ROOT / "phase3_results"
        / "MA_Mask_Quality_Categorization.csv"
    ),
    "MA threshold sensitivity": (
        ROOT / "phase3_results"
        / "MA_Broad_Region_Threshold_Sensitivity.csv"
    ),
    "Paired validation predictions": (
        ROOT / "experiment04_results"
        / "FGADR_Validation_Paired_Predictions_V1.csv"
    )
}

print("STEP 5.1B - METADATA SCHEMA INSPECTION")

for name, path in FILES.items():
    print(f"\n{'=' * 55}")
    print("Dataset:", name)
    print("File exists:", path.is_file())

    if not path.is_file():
        print("Status: FILE NOT FOUND")
        continue

    sample = pd.read_csv(path, nrows=2)

    print("Columns:")
    print(sample.columns.tolist())

    print("\nFirst two rows:")
    print(sample.to_string(index=False, max_colwidth=45))

print("\nMA MASK DIRECTORY")

MASK_DIR = (
    ROOT / "datasets" / "DR" / "FGADR"
    / "images4seg" / "Microaneurysms_Masks"
)

print("Exists:", MASK_DIR.is_dir())

if MASK_DIR.is_dir():
    filenames = sorted(
        path.name
        for path in MASK_DIR.iterdir()
        if path.is_file()
    )

    print("Mask files:", len(filenames))
    print("First 10 filenames:", filenames[:10])

print("\nSTEP 5.1B COMPLETED")


# %% [cell 107] Experiment 05 - Step 5.2A

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

# ============================================================
# Experiment 05 - Step 5.2A
# Frozen Validation MA Attribution Cohort
# ============================================================

ROOT = DR_RESEARCH
OUTPUT_DIR = ROOT / "experiment05_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_PATH = (
    ROOT / "phase3_results"
    / "MA_Mask_Quality_Categorization.csv"
)

MANIFEST_PATH = (
    ROOT / "experiment02_results"
    / "FGADR_Master_Manifest.csv"
)

PAIRED_PATH = (
    ROOT / "experiment04_results"
    / "FGADR_Validation_Paired_Predictions_V1.csv"
)

MASK_DIR = (
    ROOT / "datasets" / "DR" / "FGADR"
    / "images4seg" / "Microaneurysms_Masks"
)

INDEX_PATH = OUTPUT_DIR / "FGADR_Validation_MA_Cohort_V1.csv"
REPORT_PATH = OUTPUT_DIR / "FGADR_Validation_MA_Cohort_Report_V1.json"

for path in [QUALITY_PATH, MANIFEST_PATH, PAIRED_PATH]:
    assert path.is_file(), f"Missing file: {path}"

assert MASK_DIR.is_dir()
assert not INDEX_PATH.exists(), "Cohort index already exists."
assert not REPORT_PATH.exists(), "Cohort report already exists."

# ============================================================
# 1. Load existing audited metadata
# ============================================================

quality = pd.read_csv(QUALITY_PATH)
manifest = pd.read_csv(MANIFEST_PATH)
paired = pd.read_csv(PAIRED_PATH)

assert len(paired) == 276
assert paired["image_id"].is_unique
assert quality["image_id"].is_unique
assert manifest["image_id"].is_unique

quality_columns = [
    "image_id",
    "dr_grade",
    "review_category",
    "review_reason",
    "clinical_validation_status",
    "is_empty_mask",
    "dimensions_match",
    "candidate_count",
    "largest_component_area_px",
    "total_mask_area_px",
    "mask_coverage",
    "threshold_sensitive_flag",
    "grayscale_encoding_flag",
    "broad_region_flag",
    "reference_threshold",
    "fundus_width",
    "fundus_height",
]

quality = quality[quality_columns].rename(
    columns={"dr_grade": "quality_dr_grade"}
)

manifest = manifest[
    ["image_id", "dr_grade", "image_path", "mask_path"]
].rename(
    columns={"dr_grade": "manifest_dr_grade"}
)

# ============================================================
# 2. Join strictly by image ID
# ============================================================

cohort = paired.merge(
    quality,
    on="image_id",
    how="left",
    validate="one_to_one",
    indicator=True
)

assert cohort["_merge"].eq("both").all(), (
    "Some validation images have no quality audit."
)

cohort = cohort.drop(columns="_merge")

cohort = cohort.merge(
    manifest,
    on="image_id",
    how="left",
    validate="one_to_one",
    indicator=True
)

assert cohort["_merge"].eq("both").all(), (
    "Some validation images have no master manifest."
)

cohort = cohort.drop(columns="_merge")

assert len(cohort) == 276
assert cohort["image_id"].is_unique

assert (
    cohort["true_grade"].astype(int)
    == cohort["quality_dr_grade"].astype(int)
).all()

assert (
    cohort["true_grade"].astype(int)
    == cohort["manifest_dr_grade"].astype(int)
).all()

assert (
    cohort["true_grade"].value_counts().sort_index().tolist()
    == [15, 32, 89, 97, 43]
)

# ============================================================
# 3. Resolve known mask directory
# ============================================================

cohort["resolved_mask_path"] = cohort["image_id"].map(
    lambda image_id: str(MASK_DIR / f"{image_id}.png")
)

cohort["mask_file_exists"] = cohort[
    "resolved_mask_path"
].map(lambda path: Path(path).is_file())

assert cohort["mask_file_exists"].all(), (
    "At least one validation MA mask file is missing."
)

# The original path is preserved for audit purposes.
# The resolved path is the verified path used later.

# ============================================================
# 4. Define analysis groups before inspecting attribution
# ============================================================

boolean_fields = [
    "is_empty_mask",
    "dimensions_match",
    "threshold_sensitive_flag",
    "grayscale_encoding_flag",
    "broad_region_flag",
]

for column in boolean_fields:
    assert cohort[column].notna().all(), (
        f"Missing quality flag: {column}"
    )
    assert cohort[column].isin([True, False]).all(), (
        f"Unexpected quality flag values: {column}"
    )

cohort["ma_nonempty"] = ~cohort["is_empty_mask"]

cohort["primary_ma_candidate"] = (
    cohort["review_category"].eq(
        "strict_lesion_level_candidate"
    )
    & cohort["ma_nonempty"]
    & cohort["dimensions_match"]
    & ~cohort["threshold_sensitive_flag"]
    & ~cohort["grayscale_encoding_flag"]
    & ~cohort["broad_region_flag"]
    & cohort["total_mask_area_px"].gt(0)
    & cohort["mask_file_exists"]
)

cohort["analysis_group"] = np.select(
    [
        cohort["primary_ma_candidate"],
        cohort["is_empty_mask"],
    ],
    [
        "primary_ma_candidate",
        "empty_ma_mask",
    ],
    default="nonprimary_quality_review"
)

# These are historical mask-audit measurements,
# not clinically adjudicated lesion measurements.

cohort = cohort.sort_values(
    "image_id"
).reset_index(drop=True)

# ============================================================
# 5. Summarize inclusion and exclusion
# ============================================================

print("STEP 5.2A - VALIDATION MA COHORT")

print("\nDATA LINKAGE")
print("Paired validation images:", len(cohort))
print("Matched quality records:", len(cohort))
print("Verified MA mask files:", int(cohort["mask_file_exists"].sum()))
print("Ground-truth grade consistency: PASS")

print("\nANALYSIS GROUP COUNTS")
print(cohort["analysis_group"].value_counts().to_string())

print("\nQUALITY REVIEW CATEGORIES")
print(cohort["review_category"].value_counts().to_string())

print("\nPRIMARY COHORT BY DR GRADE")
print(
    pd.crosstab(
        cohort["true_grade"],
        cohort["analysis_group"]
    ).to_string()
)

print("\nMASK QUALITY FLAGS")
for column in [
    "is_empty_mask",
    "threshold_sensitive_flag",
    "grayscale_encoding_flag",
    "broad_region_flag",
]:
    print(column, ":", int(cohort[column].sum()))

# ============================================================
# 6. Save locked index and report
# ============================================================

cohort.to_csv(INDEX_PATH, index=False)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(block)
    return digest.hexdigest()

report = {
    "experiment": "Experiment05_Step5.2A",
    "split": "val",
    "total_validation_images": int(len(cohort)),
    "verified_mask_files": int(cohort["mask_file_exists"].sum()),
    "analysis_group_counts": {
        key: int(value)
        for key, value in cohort[
            "analysis_group"
        ].value_counts().items()
    },
    "primary_cohort_size": int(
        cohort["primary_ma_candidate"].sum()
    ),
    "cohort_csv_sha256": sha256_file(INDEX_PATH),
    "selection_rule": (
        "strict_lesion_level_candidate AND nonempty AND "
        "matching_dimensions AND no threshold_sensitive, "
        "grayscale_encoding, or broad_region flag AND "
        "positive audited mask area"
    ),
    "clinical_ground_truth_verified": False,
    "mask_pixel_values_recomputed": False,
    "test_split_evaluated": False,
    "created_utc": datetime.now(timezone.utc).isoformat(),
}

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nCOHORT FILES")
print("Index:", INDEX_PATH)
print("Index SHA256:", report["cohort_csv_sha256"])
print("Report:", REPORT_PATH)

print("\nSTEP 5.2A COMPLETED")


# %% [cell 108] Experiment 05 - Step 5.3A

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

# ============================================================
# Experiment 05 - Step 5.3A
# MA Mask Size Distribution and Resolution Audit
# ============================================================

ROOT = DR_RESEARCH
OUT = ROOT / "experiment05_results"

INDEX_PATH = OUT / "FGADR_Validation_MA_Cohort_V1.csv"
REPORT_PATH = OUT / "FGADR_MA_Size_Distribution_V1.json"

EXPECTED_SHA256 = (
    "542c4f21780ba4060945e893a22c4e133b0f44a006fecb94ea5da4b76b93b286"
)

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

assert file_sha256(INDEX_PATH) == EXPECTED_SHA256
assert not REPORT_PATH.exists()

df = pd.read_csv(INDEX_PATH)
primary = df.loc[df["primary_ma_candidate"] == True].copy()

assert len(df) == 276
assert len(primary) == 156
assert primary["true_grade"].between(1, 4).all()

required = [
    "candidate_count",
    "largest_component_area_px",
    "total_mask_area_px",
    "mask_coverage",
    "fundus_width",
    "fundus_height",
]

assert primary[required].notna().all().all()
assert (primary["total_mask_area_px"] > 0).all()
assert (primary["candidate_count"] > 0).all()
assert (primary["largest_component_area_px"] > 0).all()

INPUT_SIZE = 224

primary["area_scale_factor"] = (
    INPUT_SIZE / primary["fundus_width"]
) * (
    INPUT_SIZE / primary["fundus_height"]
)

primary["scaled_total_area_estimate"] = (
    primary["total_mask_area_px"]
    * primary["area_scale_factor"]
)

primary["scaled_largest_component_estimate"] = (
    primary["largest_component_area_px"]
    * primary["area_scale_factor"]
)

# These estimates are geometric approximations,
# not measurements from resized binary masks.

def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }

metrics = {
    "candidate_count": summarize(primary["candidate_count"]),
    "largest_component_area_px": summarize(
        primary["largest_component_area_px"]
    ),
    "total_mask_area_px": summarize(
        primary["total_mask_area_px"]
    ),
    "mask_coverage": summarize(primary["mask_coverage"]),
    "scaled_largest_component_estimate": summarize(
        primary["scaled_largest_component_estimate"]
    ),
    "scaled_total_area_estimate": summarize(
        primary["scaled_total_area_estimate"]
    ),
}

print("STEP 5.3A - MA MASK SIZE AUDIT")
print("Validation images:", len(df))
print("Primary MA candidates:", len(primary))

print("\nSIZE DISTRIBUTIONS")
for name, summary in metrics.items():
    print(f"\n{name}")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}")

print("\nRESOLUTION ESTIMATES")
for threshold in [1, 4, 16]:
    count = int((
        primary["scaled_largest_component_estimate"]
        < threshold
    ).sum())

    print(
        f"Largest component estimated below {threshold} px:",
        count,
        f"({100 * count / len(primary):.1f}%)"
    )

print("\nBY DR GRADE")
grade_summary = primary.groupby("true_grade").agg(
    images=("image_id", "count"),
    median_candidate_count=("candidate_count", "median"),
    median_largest_area_px=(
        "largest_component_area_px", "median"
    ),
    median_total_area_px=("total_mask_area_px", "median"),
    median_mask_coverage=("mask_coverage", "median")
)

print(grade_summary.to_string())

report = {
    "experiment": "Experiment05_Step5.3A",
    "cohort_sha256": EXPECTED_SHA256,
    "primary_image_count": len(primary),
    "original_resolution": "1280x1280",
    "model_input_resolution": "224x224",
    "distribution_metrics": metrics,
    "size_measurement_level": "image-level audited mask summary",
    "scaled_area_is_estimate": True,
    "clinical_lesion_validation": False,
    "test_split_evaluated": False
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nREPORT:", REPORT_PATH)
print("STEP 5.3A COMPLETED")


# %% [cell 109] Experiment 05 - Step 5.3B

from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
import cv2
from PIL import Image

# ============================================================
# Experiment 05 - Step 5.3B
# MA Mask Resizing Pilot
# ============================================================

ROOT = DR_RESEARCH
RESULT_DIR = ROOT / "experiment05_results"

COHORT_PATH = (
    RESULT_DIR / "FGADR_Validation_MA_Cohort_V1.csv"
)

PILOT_PATH = RESULT_DIR / "MA_Mask_Resize_Pilot_V1.csv"

EXPECTED_SHA256 = (
    "542c4f21780ba4060945e893a22c4e133b0f44a006fecb94ea5da4b76b93b286"
)

assert not PILOT_PATH.exists()

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

assert sha256_file(COHORT_PATH) == EXPECTED_SHA256

cohort = pd.read_csv(COHORT_PATH)

primary = cohort.loc[
    cohort["primary_ma_candidate"] == True
].copy()

assert len(primary) == 156

# Select three images per DR grade using a fixed rule.
# This selection does not depend on model predictions.
pilot = (
    primary.sort_values("image_id")
    .groupby("true_grade", group_keys=False)
    .head(3)
    .copy()
)

assert len(pilot) == 12

rows = []

for _, row in pilot.iterrows():
    mask_path = Path(row["resolved_mask_path"])

    with Image.open(mask_path) as img:
        original = np.asarray(img)

    if original.ndim == 3:
        assert np.array_equal(
            original[..., 0],
            original[..., 1]
        )
        assert np.array_equal(
            original[..., 0],
            original[..., 2]
        )
        original = original[..., 0]

    assert original.shape == (1280, 1280)

    # Match the audited reference threshold.
    original_binary = (
        original > int(row["reference_threshold"])
    ).astype(np.uint8)

    nearest = cv2.resize(
        original_binary,
        (224, 224),
        interpolation=cv2.INTER_NEAREST
    )

    area_weighted = cv2.resize(
        original_binary.astype(np.float32),
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    assert np.isfinite(area_weighted).all()
    assert area_weighted.min() >= 0
    assert area_weighted.max() <= 1.00001

    original_area = int(original_binary.sum())
    nearest_area = int(nearest.sum())
    area_weight_sum = float(area_weighted.sum())

    rows.append({
        "image_id": row["image_id"],
        "dr_grade": int(row["true_grade"]),
        "original_mask_pixels": original_area,
        "nearest_mask_pixels": nearest_area,
        "area_weighted_pixel_sum": area_weight_sum,
        "nearest_mask_empty": nearest_area == 0,
        "area_weighted_nonzero_pixels": int(
            (area_weighted > 0).sum()
        ),
        "area_weighted_pixels_above_0_01": int(
            (area_weighted >= 0.01).sum()
        )
    })

results = pd.DataFrame(rows)
results.to_csv(PILOT_PATH, index=False)

print("STEP 5.3B - MASK RESIZING PILOT")
print("Pilot images:", len(results))

print("\nPER-IMAGE RESULTS")
print(results.to_string(index=False))

print("\nSUMMARY")
print(
    "Nearest-neighbor empty masks:",
    int(results["nearest_mask_empty"].sum())
)
print(
    "Median nearest-neighbor mask pixels:",
    results["nearest_mask_pixels"].median()
)
print(
    "Median area-weighted pixel sum:",
    round(results["area_weighted_pixel_sum"].median(), 4)
)

print("\nOUTPUT FILE:", PILOT_PATH)
print("STEP 5.3B COMPLETED")


# %% [cell 110] Experiment 05 - Step 5.3C

from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ============================================================
# Experiment 05 - Step 5.3C
# Final MA Mask Resampling Validation
# ============================================================

ROOT = DR_RESEARCH
RESULT_DIR = ROOT / "experiment05_results"

COHORT_PATH = RESULT_DIR / "FGADR_Validation_MA_Cohort_V1.csv"
OUTPUT_PATH = RESULT_DIR / "MA_Mask_Resampling_Validation_V1.csv"
REPORT_PATH = RESULT_DIR / "MA_Mask_Resampling_Protocol_V1.json"

EXPECTED_COHORT_SHA256 = (
    "542c4f21780ba4060945e893a22c4e133b0f44a006fecb94ea5da4b76b93b286"
)

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()

assert file_sha256(COHORT_PATH) == EXPECTED_COHORT_SHA256
assert not OUTPUT_PATH.exists()
assert not REPORT_PATH.exists()

cohort = pd.read_csv(COHORT_PATH)
primary = cohort.loc[
    cohort["primary_ma_candidate"].astype(bool)
].sort_values("image_id")

assert len(primary) == 156

rows = []

for row in primary.itertuples(index=False):

    with Image.open(row.resolved_mask_path) as img:
        mask = np.asarray(img)

    if mask.ndim == 3:
        assert np.array_equal(mask[..., 0], mask[..., 1])
        assert np.array_equal(mask[..., 0], mask[..., 2])
        mask = mask[..., 0]

    assert mask.shape == (1280, 1280)

    original = (
        mask > int(row.reference_threshold)
    ).astype(np.uint8)

    nearest = cv2.resize(
        original,
        (224, 224),
        interpolation=cv2.INTER_NEAREST
    )

    weighted = cv2.resize(
        original.astype(np.float32),
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    original_area = int(original.sum())
    expected_area = original_area * (224 / 1280) ** 2
    weighted_area = float(weighted.sum())
    nearest_area = int(nearest.sum())

    assert original_area > 0
    assert np.isfinite(weighted).all()
    assert weighted.min() >= 0
    assert weighted.max() <= 1.00001

    rows.append({
        "image_id": row.image_id,
        "dr_grade": int(row.true_grade),
        "original_area_px": original_area,
        "expected_scaled_area": expected_area,
        "weighted_area_sum": weighted_area,
        "nearest_area_px": nearest_area,
        "weighted_nonzero_pixels": int((weighted > 0).sum()),
        "nearest_empty": nearest_area == 0,
        "weighted_empty": weighted_area <= 0,
        "weighted_relative_area_error": (
            abs(weighted_area - expected_area) / expected_area
        )
    })

result = pd.DataFrame(rows)

print("STEP 5.3C - FINAL MA RESAMPLING VALIDATION")
print("Images:", len(result))

print("\nRESAMPLING RESULTS")
print("Nearest empty masks:", int(result["nearest_empty"].sum()))
print("Weighted empty masks:", int(result["weighted_empty"].sum()))
print(
    "Median relative area error:",
    round(result["weighted_relative_area_error"].median(), 6)
)
print(
    "Maximum relative area error:",
    round(result["weighted_relative_area_error"].max(), 6)
)

print("\nRELATIVE AREA ERROR QUANTILES")
print(
    result["weighted_relative_area_error"]
    .quantile([0, 0.25, 0.5, 0.75, 0.9, 1.0])
    .to_string()
)

print("\nBY DR GRADE")
print(
    result.groupby("dr_grade").agg(
        images=("image_id", "count"),
        nearest_empty=("nearest_empty", "sum"),
        weighted_empty=("weighted_empty", "sum"),
        median_weighted_area=("weighted_area_sum", "median"),
        median_area_error=("weighted_relative_area_error", "median")
    ).to_string()
)

result.to_csv(OUTPUT_PATH, index=False)

report = {
    "experiment": "Experiment05_Step5.3C",
    "cohort_sha256": EXPECTED_COHORT_SHA256,
    "image_count": len(result),
    "source_mask_threshold_rule": "pixel_value > reference_threshold",
    "primary_resampling": "cv2.INTER_AREA",
    "sensitivity_resampling": "cv2.INTER_NEAREST",
    "target_resolution": [224, 224],
    "nearest_empty_count": int(result["nearest_empty"].sum()),
    "weighted_empty_count": int(result["weighted_empty"].sum()),
    "median_weighted_area_error": float(
        result["weighted_relative_area_error"].median()
    ),
    "maximum_weighted_area_error": float(
        result["weighted_relative_area_error"].max()
    ),
    "result_sha256": file_sha256(OUTPUT_PATH),
    "clinical_lesion_validation": False,
    "test_split_evaluated": False
}

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nOUTPUT FILES")
print(OUTPUT_PATH)
print(REPORT_PATH)
print("STEP 5.3C COMPLETED")


# %% [cell 111] Experiment 05 - Step 5.4A

import gc
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import timm
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ============================================================
# Experiment 05 - Step 5.4A
# Paired Grad-CAM Smoke Test
# ============================================================

ROOT = DR_RESEARCH
OUTPUT_DIR = ROOT / "experiment05_results" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

EFF_CHECKPOINT = (
    ROOT / "experiment03_results"
    / "efficientnet_b0_baseline_v1"
    / "best_model.pt"
)

RET_CHECKPOINT = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_model.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert DEVICE.type == "cuda", "A CUDA GPU is required for this test."

for path in [COHORT_PATH, INDEX_PATH, EFF_CHECKPOINT, RET_CHECKPOINT]:
    assert path.is_file(), f"Missing file: {path}"

cohort = pd.read_csv(COHORT_PATH)
training_index = pd.read_csv(INDEX_PATH)

# Choose one fixed primary-cohort image without using model predictions.
sample = (
    cohort.loc[cohort["primary_ma_candidate"] == True]
    .sort_values("image_id")
    .iloc[0]
)

image_id = str(sample["image_id"])
true_grade = int(sample["true_grade"])

index_row = training_index.loc[
    training_index["image_id"] == image_id
]

assert len(index_row) == 1
assert index_row.iloc[0]["split"] == "val"
assert int(index_row.iloc[0]["dr_grade"]) == true_grade

image_path = Path(str(index_row.iloc[0]["local_path"]))

if not image_path.is_file():
    image_path = Path(str(index_row.iloc[0]["source_path"]))

assert image_path.is_file(), f"Image not found: {image_id}"

with Image.open(image_path) as image:
    original_rgb = image.convert("RGB")
    original_rgb.load()

with Image.open(sample["resolved_mask_path"]) as image:
    raw_mask = np.asarray(image)

if raw_mask.ndim == 3:
    assert np.array_equal(raw_mask[..., 0], raw_mask[..., 1])
    assert np.array_equal(raw_mask[..., 0], raw_mask[..., 2])
    raw_mask = raw_mask[..., 0]

binary_mask = (
    raw_mask > int(sample["reference_threshold"])
).astype(np.float32)

mask_224 = cv2.resize(
    binary_mask,
    (224, 224),
    interpolation=cv2.INTER_AREA
)

assert mask_224.sum() > 0

validation_transform = transforms.Compose([
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

input_tensor = validation_transform(
    original_rgb
).unsqueeze(0).to(DEVICE)

display_rgb = np.asarray(
    original_rgb.resize((224, 224), Image.Resampling.BICUBIC)
)

# ============================================================
# Checkpoint loading
# ============================================================

def load_checkpoint_model(model_name, checkpoint_path, global_pool=None):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    if "model_state_dict" in checkpoint:
        weights = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        weights = checkpoint["model"]
    else:
        weights = checkpoint

    assert isinstance(weights, dict)

    kwargs = {
        "pretrained": False,
        "num_classes": 5
    }

    if global_pool is not None:
        kwargs["global_pool"] = global_pool

    model = timm.create_model(model_name, **kwargs)

    model.load_state_dict(weights, strict=True)
    model.eval()

    if callable(getattr(model, "set_grad_checkpointing", None)):
        model.set_grad_checkpointing(enable=False)

    model = model.to(DEVICE)

    del checkpoint, weights
    gc.collect()

    return model

# ============================================================
# Grad-CAM implementation
# ============================================================

def calculate_gradcam(model, target_layer, input_tensor, target_class, is_vit):
    captured = {}

    def forward_hook(module, inputs, output):
        assert isinstance(output, torch.Tensor)
        captured["activation"] = output

        if not output.requires_grad:
            raise RuntimeError("Target activation does not require gradients.")

        output.register_hook(
            lambda grad: captured.__setitem__("gradient", grad)
        )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(input_tensor)
            assert logits.shape == (1, 5)
            assert torch.isfinite(logits).all()

            target_logit = logits[0, target_class]
            target_logit.backward()

        activation = captured["activation"].detach().float()
        gradient = captured["gradient"].detach().float()

        if is_vit:
            # ViT patch tokens form a 14x14 spatial grid.
            # The first token is the CLS token.
            assert activation.shape == (1, 197, 1024)
            assert gradient.shape == activation.shape

            activation = (
                activation[:, 1:, :]
                .transpose(1, 2)
                .reshape(1, 1024, 14, 14)
            )

            gradient = (
                gradient[:, 1:, :]
                .transpose(1, 2)
                .reshape(1, 1024, 14, 14)
            )
        else:
            assert activation.ndim == 4
            assert gradient.shape == activation.shape

        weights = gradient.mean(dim=(2, 3), keepdim=True)

        cam = torch.relu(
            (weights * activation).sum(dim=1)
        )[0]

        cam = cam.cpu().numpy()

        assert np.isfinite(cam).all()

        cam = cv2.resize(
            cam,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        )

        max_value = float(cam.max())

        if max_value > 1e-12:
            cam = cam / max_value
        else:
            cam = np.zeros_like(cam)

        probabilities = torch.softmax(
            logits.detach().float(), dim=1
        )[0].cpu().numpy()

        return {
            "cam": cam,
            "probabilities": probabilities,
            "predicted_grade": int(probabilities.argmax()),
            "raw_cam_max": max_value,
            "target_logit": float(target_logit.detach().item())
        }

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

# ============================================================
# EfficientNet Grad-CAM
# ============================================================

print("STEP 5.4A - PAIRED GRAD-CAM TEST")
print("Image ID:", image_id)
print("True DR grade:", true_grade)
print("Device:", DEVICE)

eff_model = load_checkpoint_model(
    "efficientnet_b0",
    EFF_CHECKPOINT
)

eff_result = calculate_gradcam(
    eff_model,
    eff_model.conv_head,
    input_tensor,
    true_grade,
    is_vit=False
)

del eff_model
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# RETFound Grad-CAM
# ============================================================

ret_model = load_checkpoint_model(
    "vit_large_patch16_224",
    RET_CHECKPOINT,
    global_pool="avg"
)

ret_result = calculate_gradcam(
    ret_model,
    ret_model.blocks[-1].norm1,
    input_tensor,
    true_grade,
    is_vit=True
)

del ret_model
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(display_rgb)
axes[0].set_title("Fundus image")

axes[1].imshow(display_rgb)
axes[1].imshow(
    mask_224,
    cmap="Reds",
    alpha=np.clip(mask_224 * 0.85, 0, 0.85),
    vmin=0,
    vmax=1
)
axes[1].set_title("MA mask overlay")

axes[2].imshow(display_rgb)
axes[2].imshow(
    eff_result["cam"],
    cmap="jet",
    alpha=0.45,
    vmin=0,
    vmax=1
)
axes[2].set_title("EfficientNet Grad-CAM")

axes[3].imshow(display_rgb)
axes[3].imshow(
    ret_result["cam"],
    cmap="jet",
    alpha=0.45,
    vmin=0,
    vmax=1
)
axes[3].set_title("RETFound Grad-CAM")

for ax in axes:
    ax.axis("off")

fig.suptitle(
    f"Image {image_id} | Ground-truth DR{true_grade}",
    fontsize=12
)
fig.tight_layout()

FIGURE_PATH = OUTPUT_DIR / f"GradCAM_Pilot_{image_id}.png"
REPORT_PATH = OUTPUT_DIR / f"GradCAM_Pilot_{image_id}.json"

assert not FIGURE_PATH.exists()
assert not REPORT_PATH.exists()

fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
plt.show()
plt.close(fig)

report = {
    "experiment": "Experiment05_Step5.4A",
    "image_id": image_id,
    "true_grade": true_grade,
    "mask_area_weight_sum": float(mask_224.sum()),
    "attribution_target": "ground_truth_grade_logit",
    "efficientnet": {
        "target_layer": "conv_head",
        "predicted_grade": eff_result["predicted_grade"],
        "raw_cam_max": eff_result["raw_cam_max"],
        "probabilities": eff_result["probabilities"].tolist()
    },
    "retfound": {
        "target_layer": "blocks.23.norm1",
        "predicted_grade": ret_result["predicted_grade"],
        "raw_cam_max": ret_result["raw_cam_max"],
        "probabilities": ret_result["probabilities"].tolist()
    },
    "test_split_evaluated": False
}

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nGRAD-CAM SMOKE TEST RESULTS")

for name, result in [
    ("EfficientNet", eff_result),
    ("RETFound", ret_result)
]:
    print(f"\n{name}")
    print("Predicted grade:", result["predicted_grade"])
    print("CAM shape:", result["cam"].shape)
    print("CAM finite:", bool(np.isfinite(result["cam"]).all()))
    print("Raw CAM maximum:", result["raw_cam_max"])
    print("CAM range:", float(result["cam"].min()), float(result["cam"].max()))

print("\nOUTPUT FILES")
print(FIGURE_PATH)
print(REPORT_PATH)
print("\nSTEP 5.4A COMPLETED")


# %% [cell 112] Experiment 05 - Step 5.4B

import gc
import numpy as np
import matplotlib.pyplot as plt
import torch

# ============================================================
# Experiment 05 - Step 5.4B
# RETFound Attribution Layer Validation
# ============================================================

assert "load_checkpoint_model" in globals()
assert "calculate_gradcam" in globals()
assert "input_tensor" in globals()
assert "mask_224" in globals()
assert "display_rgb" in globals()
assert "RET_CHECKPOINT" in globals()

ret_model = load_checkpoint_model(
    "vit_large_patch16_224",
    RET_CHECKPOINT,
    global_pool="avg"
)

ret_model.eval()
ret_model.set_grad_checkpointing(enable=False)

# Compare two target layers using the same DR target.
norm_result = calculate_gradcam(
    ret_model,
    ret_model.blocks[-1].norm1,
    input_tensor,
    true_grade,
    is_vit=True
)

block_result = calculate_gradcam(
    ret_model,
    ret_model.blocks[-1],
    input_tensor,
    true_grade,
    is_vit=True
)

# Verify that the candidate last-block attribution is valid.
assert block_result["raw_cam_max"] > 1e-12
assert np.isfinite(block_result["cam"]).all()
assert block_result["cam"].std() > 1e-8
assert np.isfinite(norm_result["cam"]).all()

# Build an explicit RGBA overlay for the MA annotation.
rgba = np.zeros((224, 224, 4), dtype=np.float32)
rgba[..., 0] = 1.0
rgba[..., 3] = np.clip(mask_224 * 0.95, 0, 0.95)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(display_rgb)
axes[0].set_title("Fundus image")

axes[1].imshow(display_rgb)
axes[1].imshow(rgba)
axes[1].set_title("MA mask overlay")

axes[2].imshow(display_rgb)
axes[2].imshow(
    norm_result["cam"],
    cmap="jet",
    alpha=0.45,
    vmin=0,
    vmax=1
)
axes[2].set_title("RETFound: Last Block Norm1")

axes[3].imshow(display_rgb)
axes[3].imshow(
    block_result["cam"],
    cmap="jet",
    alpha=0.45,
    vmin=0,
    vmax=1
)
axes[3].set_title("RETFound: Last Block Output")

for ax in axes:
    ax.axis("off")

fig.tight_layout()
plt.show()
plt.close(fig)

print("STEP 5.4B - ATTRIBUTION LAYER VALIDATION")

for name, result in [
    ("Last Block Norm1", norm_result),
    ("Last Block Output", block_result)
]:
    print(f"\n{name}")
    print("Raw CAM maximum:", result["raw_cam_max"])
    print("CAM standard deviation:", float(result["cam"].std()))
    print("CAM finite:", bool(np.isfinite(result["cam"]).all()))
    print("Predicted DR grade:", result["predicted_grade"])

print("\nMA overlay fixed: True")
print("Test split evaluated: False")
print("STEP 5.4B COMPLETED")

del ret_model
gc.collect()
torch.cuda.empty_cache()


# %% [cell 113] Experiment 05 - Step 5.4C

# ============================================================
# Experiment 05 - Step 5.4C
# Batch Grad-CAM: 156 Primary MA Images
# ============================================================

from pathlib import Path
import gc
import json
import hashlib
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

ROOT = DR_RESEARCH
OUT = ROOT / "experiment05_results" / "gradcam"
OUT.mkdir(parents=True, exist_ok=True)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)
INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)
PAIRED_PATH = (
    ROOT / "experiment04_results"
    / "FGADR_Validation_Paired_Predictions_V1.csv"
)

CSV_PATH = OUT / "MA_Attribution_Descriptive_V1.csv"
MAP_PATH = OUT / "MA_Attribution_Maps_V1.npz"

assert not CSV_PATH.exists()
assert not MAP_PATH.exists()
assert callable(load_checkpoint_model)
assert callable(calculate_gradcam)
assert DEVICE.type == "cuda"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

assert sha256_file(COHORT_PATH) == (
    "542c4f21780ba4060945e893a22c4e133b0f44a006fecb94ea5da4b76b93b286"
)

cohort = pd.read_csv(COHORT_PATH)
index = pd.read_csv(INDEX_PATH)
paired = pd.read_csv(PAIRED_PATH)

primary = (
    cohort.loc[cohort["primary_ma_candidate"] == True]
    .merge(
        index[["image_id", "split", "local_path", "source_path"]],
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        paired[["image_id", "efficientnet_pred", "retfound_pred"]],
        on="image_id",
        validate="one_to_one",
        suffixes=("", "_audit")
    )
    .sort_values("image_id")
    .reset_index(drop=True)
)

assert len(primary) == 156
assert primary["image_id"].is_unique
assert primary["split"].eq("val").all()

# Disallow use of another split.
assert set(primary["split"]) == {"val"}

model_configs = [
    ("efficientnet", "efficientnet_b0", EFF_CHECKPOINT, None),
    ("retfound", "vit_large_patch16_224", RET_CHECKPOINT, "avg"),
]

records = {}
cam_maps = {}

print("STEP 5.4C - BATCH GRAD-CAM")
print("Primary MA images:", len(primary))
print("Device:", DEVICE)

for model_key, architecture, checkpoint_path, pooling in model_configs:

    print(f"\nPROCESSING {model_key.upper()}")

    current_model = load_checkpoint_model(
        architecture,
        checkpoint_path,
        global_pool=pooling
    )

    target_layer = (
        current_model.conv_head
        if model_key == "efficientnet"
        else current_model.blocks[-1]
    )

    layer_name = (
        "conv_head"
        if model_key == "efficientnet"
        else "blocks.23"
    )

    model_records = []
    model_maps = []

    for row in tqdm(
        primary.itertuples(index=False),
        total=len(primary),
        desc=model_key
    ):

        image_path = Path(str(row.local_path))
        if not image_path.is_file():
            image_path = Path(str(row.source_path))

        assert image_path.is_file(), (
            f"Image missing: {row.image_id}"
        )

        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            input_image = validation_transform(rgb)

        batch = input_image.unsqueeze(0).to(DEVICE)

        result = calculate_gradcam(
            current_model,
            target_layer,
            batch,
            int(row.true_grade),
            is_vit=(model_key == "retfound")
        )

        assert result["cam"].shape == (224, 224)
        assert np.isfinite(result["cam"]).all()

        expected_prediction = int(
            getattr(row, f"{model_key}_pred")
        )

        assert result["predicted_grade"] == expected_prediction, (
            f"Prediction mismatch: {row.image_id}, {model_key}"
        )

        with Image.open(row.resolved_mask_path) as img:
            mask = np.asarray(img)

        if mask.ndim == 3:
            assert np.array_equal(mask[..., 0], mask[..., 1])
            assert np.array_equal(mask[..., 0], mask[..., 2])
            mask = mask[..., 0]

        binary_mask = (
            mask > int(row.reference_threshold)
        ).astype(np.float32)

        weighted_mask = cv2.resize(
            binary_mask,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )

        assert weighted_mask.sum() > 0

        cam = result["cam"].astype(np.float32)

        mean_full = float(cam.mean())
        mean_ma = float(
            np.sum(cam * weighted_mask)
            / np.sum(weighted_mask)
        )

        enrichment = (
            mean_ma / mean_full
            if mean_full > 1e-12
            else np.nan
        )

        model_records.append({
            "image_id": row.image_id,
            "true_grade": int(row.true_grade),
            "model": model_key,
            "target_layer": layer_name,
            "predicted_grade": result["predicted_grade"],
            "correct": (
                result["predicted_grade"] == int(row.true_grade)
            ),
            "mask_weight_sum": float(weighted_mask.sum()),
            "mean_cam_full": mean_full,
            "mean_cam_ma": mean_ma,
            "ma_enrichment": enrichment,
            "raw_cam_max": result["raw_cam_max"],
            "cam_std": float(cam.std())
        })

        # Keep full-resolution CAM maps for spatial-null analysis.
        model_maps.append(cam)

        del batch

    records[model_key] = pd.DataFrame(model_records)
    cam_maps[model_key] = np.stack(
        model_maps, axis=0
    ).astype(np.float32)

    del current_model
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# Save descriptive metrics and spatial CAM maps
# ============================================================

results = pd.concat(
    [records["efficientnet"], records["retfound"]],
    ignore_index=True
)

assert len(results) == 312
assert results.groupby("model").size().to_dict() == {
    "efficientnet": 156,
    "retfound": 156
}

assert np.isfinite(results["ma_enrichment"]).all(), (
    "Non-finite enrichment values detected."
)

# Use local temporary files and copy to final paths.
local_dir = (LOCAL_DISK / "retfound_ma_attribution")
local_dir.mkdir(parents=True, exist_ok=True)

local_csv = local_dir / "descriptive.csv"
local_maps = local_dir / "maps.npz"

results.to_csv(local_csv, index=False)

np.savez_compressed(
    local_maps,
    image_ids=primary["image_id"].to_numpy(dtype=str),
    efficientnet_cam=cam_maps["efficientnet"],
    retfound_cam=cam_maps["retfound"]
)

import shutil

shutil.copyfile(local_csv, CSV_PATH)
shutil.copyfile(local_maps, MAP_PATH)

assert sha256_file(local_csv) == sha256_file(CSV_PATH)
assert sha256_file(local_maps) == sha256_file(MAP_PATH)

print("\nBATCH ATTRIBUTION RESULTS")
print("Processed images:", len(primary))
print("Total model-image records:", len(results))

print("\nDESCRIPTIVE ENRICHMENT")
print(
    results.groupby("model")["ma_enrichment"]
    .agg(["count", "mean", "median", "std"])
    .to_string()
)

print("\nCAM QUALITY")
print(
    results.groupby("model")["cam_std"]
    .agg(["min", "median", "max"])
    .to_string()
)

print("\nOUTPUT FILES")
print(CSV_PATH)
print(MAP_PATH)
print("File transfer verification: PASS")
print("Test evaluated: False")
print("\nSTEP 5.4C COMPLETED")


# %% [cell 114] Experiment 05 - Step 5.4C Diagnostic

import gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image

# ============================================================
# Experiment 05 - Step 5.4C Diagnostic
# RETFound FP32 vs BF16 Prediction Consistency
# ============================================================

ROOT = DR_RESEARCH
IMAGE_ID = "0032_3"

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

PRED_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_validation_predictions.csv"
)

CHECKPOINT_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_model.pt"
)

index = pd.read_csv(INDEX_PATH)
predictions = pd.read_csv(PRED_PATH)

image_row = index.loc[index["image_id"] == IMAGE_ID]
prediction_row = predictions.loc[
    predictions["image_id"] == IMAGE_ID
]

assert len(image_row) == 1
assert len(prediction_row) == 1
assert image_row.iloc[0]["split"] == "val"

image_path = Path(str(image_row.iloc[0]["local_path"]))

if not image_path.is_file():
    image_path = Path(str(image_row.iloc[0]["source_path"]))

assert image_path.is_file()

with Image.open(image_path) as img:
    image_tensor = validation_transform(
        img.convert("RGB")
    ).unsqueeze(0).to("cuda")

# Remove the failed batch model if it is still in memory.
if "current_model" in globals():
    del current_model

gc.collect()
torch.cuda.empty_cache()

diagnostic_model = load_checkpoint_model(
    "vit_large_patch16_224",
    CHECKPOINT_PATH,
    global_pool="avg"
)

diagnostic_model.eval()
diagnostic_model.set_grad_checkpointing(enable=False)

with torch.inference_mode():
    fp32_logits = diagnostic_model(image_tensor)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):
        bf16_logits = diagnostic_model(image_tensor)

fp32_probs = torch.softmax(
    fp32_logits.float(), dim=1
)[0].cpu().numpy()

bf16_probs = torch.softmax(
    bf16_logits.float(), dim=1
)[0].cpu().numpy()

saved_probs = prediction_row[
    [f"prob_dr{i}" for i in range(5)]
].iloc[0].to_numpy(dtype=float)

saved_prediction = int(
    prediction_row.iloc[0]["predicted_grade"]
)

fp32_prediction = int(fp32_probs.argmax())
bf16_prediction = int(bf16_probs.argmax())

print("STEP 5.4C - PREDICTION CONSISTENCY DIAGNOSTIC")
print("Image ID:", IMAGE_ID)
print("True grade:", int(prediction_row.iloc[0]["true_grade"]))

print("\nSAVED VALIDATION RESULT")
print("Prediction:", saved_prediction)
print("Probabilities:", np.round(saved_probs, 8).tolist())

print("\nCURRENT FP32 RESULT")
print("Prediction:", fp32_prediction)
print("Probabilities:", np.round(fp32_probs, 8).tolist())

print("\nCURRENT BF16 RESULT")
print("Prediction:", bf16_prediction)
print("Probabilities:", np.round(bf16_probs, 8).tolist())

print("\nCONSISTENCY CHECK")
print(
    "FP32 matches saved prediction:",
    fp32_prediction == saved_prediction
)
print(
    "BF16 matches saved prediction:",
    bf16_prediction == saved_prediction
)
print(
    "FP32 maximum probability difference:",
    float(np.max(np.abs(fp32_probs - saved_probs)))
)
print(
    "BF16 maximum probability difference:",
    float(np.max(np.abs(bf16_probs - saved_probs)))
)

print("\nDIAGNOSTIC COMPLETED")

del diagnostic_model
gc.collect()
torch.cuda.empty_cache()


# %% [cell 115] Experiment 05 - Step 5.4D

import gc
import numpy as np
import pandas as pd
import torch

# ============================================================
# Experiment 05 - Step 5.4D
# RETFound BF16 Batch Context Reproducibility
# ============================================================

IMAGE_ID = "0032_3"

assert "val_loader" in globals()
assert "load_checkpoint_model" in globals()
assert "RET_CHECKPOINT" in globals()

saved_df = pd.read_csv(
    str(DR_RESEARCH / "experiment04_results/retfound_vit_large_v1/best_validation_predictions.csv")
)

saved_row = saved_df.loc[saved_df.image_id == IMAGE_ID]
assert len(saved_row) == 1

saved_probs = saved_row[
    [f"prob_dr{i}" for i in range(5)]
].iloc[0].to_numpy(dtype=np.float32)

matching_batch = None
matching_index = None

for images, labels, image_ids in val_loader:
    ids = list(image_ids)
    if IMAGE_ID in ids:
        matching_batch = images
        matching_index = ids.index(IMAGE_ID)
        break

assert matching_batch is not None
assert int(labels[matching_index]) == 4

if "diagnostic_model" in globals():
    del diagnostic_model

gc.collect()
torch.cuda.empty_cache()

model = load_checkpoint_model(
    "vit_large_patch16_224",
    RET_CHECKPOINT,
    global_pool="avg"
)
model.eval()
model.set_grad_checkpointing(enable=False)

def predict_bf16(inputs):
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(inputs.to("cuda"))

    return torch.softmax(
        logits.float(), dim=1
    ).cpu().numpy()

single_probs = predict_bf16(
    matching_batch[matching_index:matching_index + 1]
)[0]

full_probs = predict_bf16(
    matching_batch
)[matching_index]

print("STEP 5.4D - BATCH CONTEXT DIAGNOSTIC")
print("Image ID:", IMAGE_ID)
print("Validation batch size:", len(matching_batch))
print("Position within batch:", matching_index)

for name, probs in [
    ("Saved validation", saved_probs),
    ("BF16 single image", single_probs),
    ("BF16 original batch", full_probs)
]:
    print(f"\n{name}")
    print("Prediction:", int(probs.argmax()))
    print("Probabilities:", np.round(probs, 7).tolist())
    print(
        "Max difference from saved:",
        float(np.max(np.abs(probs - saved_probs)))
    )

print("\nCONSISTENCY")
print(
    "Batch prediction matches saved:",
    int(full_probs.argmax()) == int(saved_probs.argmax())
)
print(
    "Full batch probability close (atol=1e-4):",
    bool(np.allclose(full_probs, saved_probs, atol=1e-4, rtol=0))
)
print(
    "Batch context changes probabilities:",
    bool(not np.allclose(full_probs, single_probs, atol=1e-6, rtol=0))
)

print("\nSTEP 5.4D COMPLETED")

del model
gc.collect()
torch.cuda.empty_cache()


# %% [cell 116] Experiment 05 - Step 5.4D Fix

import gc
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ============================================================
# Experiment 05 - Step 5.4D Fix
# BF16 Batch Context Without Existing DataLoader
# ============================================================

ROOT = DR_RESEARCH
IMAGE_ID = "0032_3"
BATCH_SIZE = 16

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

PRED_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_validation_predictions.csv"
)

RET_CHECKPOINT = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_model.pt"
)

assert INDEX_PATH.is_file()
assert PRED_PATH.is_file()
assert RET_CHECKPOINT.is_file()
assert "load_checkpoint_model" in globals()

index = pd.read_csv(INDEX_PATH)
saved = pd.read_csv(PRED_PATH)

validation = index.loc[index["split"] == "val"].copy()

assert len(validation) == 276
assert validation["image_id"].is_unique
assert saved["image_id"].is_unique
assert len(saved) == 276

# Use the saved validation prediction ordering as the batch
# reconstruction order. This requires confirmation against
# the original validation dataset implementation.
validation = saved[["image_id"]].merge(
    validation,
    on="image_id",
    validate="one_to_one",
    how="left"
)

assert validation["local_path"].notna().all()

position = validation.index[
    validation["image_id"] == IMAGE_ID
].tolist()

assert len(position) == 1
position = position[0]

batch_start = (position // BATCH_SIZE) * BATCH_SIZE
batch_end = min(batch_start + BATCH_SIZE, len(validation))

batch_frame = validation.iloc[batch_start:batch_end]
batch_position = position - batch_start

validation_transform = transforms.Compose([
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

images = []

for row in batch_frame.itertuples(index=False):
    image_path = Path(str(row.local_path))

    if not image_path.is_file():
        image_path = Path(str(row.source_path))

    assert image_path.is_file()

    with Image.open(image_path) as image:
        tensor = validation_transform(image.convert("RGB"))

    images.append(tensor)

batch = torch.stack(images).to("cuda")

saved_row = saved.loc[saved["image_id"] == IMAGE_ID]
assert len(saved_row) == 1

saved_probs = saved_row[
    [f"prob_dr{i}" for i in range(5)]
].iloc[0].to_numpy(dtype=np.float32)

gc.collect()
torch.cuda.empty_cache()

model = load_checkpoint_model(
    "vit_large_patch16_224",
    RET_CHECKPOINT,
    global_pool="avg"
)

model.eval()
model.set_grad_checkpointing(enable=False)

def predict_bf16(inputs):
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits = model(inputs)

    return torch.softmax(
        logits.float(), dim=1
    ).cpu().numpy()

single_probs = predict_bf16(
    batch[batch_position:batch_position + 1]
)[0]

batch_probs = predict_bf16(batch)[batch_position]

print("STEP 5.4D - BF16 BATCH CONTEXT")
print("Image ID:", IMAGE_ID)
print("Reconstructed batch size:", len(batch_frame))
print("Image position:", batch_position)

for name, probs in [
    ("Saved validation", saved_probs),
    ("BF16 single image", single_probs),
    ("BF16 reconstructed batch", batch_probs)
]:
    print(f"\n{name}")
    print("Prediction:", int(probs.argmax()))
    print("Probabilities:", np.round(probs, 7).tolist())
    print(
        "Maximum difference from saved:",
        float(np.max(np.abs(probs - saved_probs)))
    )

print("\nDIAGNOSTIC RESULTS")
print(
    "Single-image prediction matches:",
    int(single_probs.argmax()) == int(saved_probs.argmax())
)
print(
    "Batch prediction matches:",
    int(batch_probs.argmax()) == int(saved_probs.argmax())
)
print(
    "Batch probability close:",
    bool(np.allclose(
        batch_probs,
        saved_probs,
        atol=1e-4,
        rtol=0
    ))
)

print("\nSTEP 5.4D COMPLETED")

del model, batch
gc.collect()
torch.cuda.empty_cache()


# %% [cell 117] Experiment 05 - Step 5.4E

# ============================================================
# Experiment 05 - Step 5.4E
# Recover Completed Attribution Results
# ============================================================

import numpy as np

print("STEP 5.4E - ATTRIBUTION RECOVERY CHECK")

has_records = (
    "records" in globals()
    and isinstance(records, dict)
    and "efficientnet" in records
)

has_maps = (
    "cam_maps" in globals()
    and isinstance(cam_maps, dict)
    and "efficientnet" in cam_maps
)

print("EfficientNet records available:", has_records)
print("EfficientNet CAM maps available:", has_maps)

if has_records:
    print("EfficientNet record count:", len(records["efficientnet"]))

if has_maps:
    print("EfficientNet CAM shape:", cam_maps["efficientnet"].shape)
    print(
        "EfficientNet CAM finite:",
        bool(np.isfinite(cam_maps["efficientnet"]).all())
    )

recovery_pass = (
    has_records
    and has_maps
    and len(records["efficientnet"]) == 156
    and cam_maps["efficientnet"].shape == (156, 224, 224)
    and np.isfinite(cam_maps["efficientnet"]).all()
)

print("\nRecovery status:", "PASS" if recovery_pass else "NOT AVAILABLE")
print("STEP 5.4E COMPLETED")


# %% [cell 118] Experiment 05 - Step 5.4F

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

# ============================================================
# Experiment 05 - Step 5.4F
# Persist Recovered EfficientNet Attribution Results
# ============================================================

ROOT = DR_RESEARCH
OUT = ROOT / "experiment05_results" / "gradcam"
OUT.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT / "EfficientNet_MA_Attribution_V1.csv"
MAP_PATH = OUT / "EfficientNet_MA_CAM_Maps_V1.npz"
REPORT_PATH = OUT / "EfficientNet_MA_CAM_Recovery_V1.json"

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

EXPECTED_COHORT_SHA256 = (
    "542c4f21780ba4060945e893a22c4e133b0f44a006fecb94ea5da4b76b93b286"
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

assert sha256_file(COHORT_PATH) == EXPECTED_COHORT_SHA256

assert "records" in globals()
assert "cam_maps" in globals()
assert "efficientnet" in records
assert "efficientnet" in cam_maps

primary = (
    pd.read_csv(COHORT_PATH)
    .query("primary_ma_candidate == True")
    .sort_values("image_id")
    .reset_index(drop=True)
)

eff_records = records["efficientnet"].copy()
eff_maps = np.asarray(cam_maps["efficientnet"], dtype=np.float32)

assert len(primary) == 156
assert len(eff_records) == 156
assert eff_maps.shape == (156, 224, 224)

assert (
    eff_records["image_id"].astype(str).tolist()
    == primary["image_id"].astype(str).tolist()
), "CAM order and cohort order do not match."

assert np.isfinite(eff_maps).all()
assert np.isfinite(eff_records["ma_enrichment"]).all()

assert not CSV_PATH.exists(), "Existing CSV must not be overwritten."
assert not MAP_PATH.exists(), "Existing maps must not be overwritten."
assert not REPORT_PATH.exists(), "Existing report must not be overwritten."

local_dir = (LOCAL_DISK / "ma_attribution_recovery")
local_dir.mkdir(parents=True, exist_ok=True)

local_csv = local_dir / "efficientnet.csv"
local_maps = local_dir / "efficientnet_maps.npz"

eff_records.to_csv(local_csv, index=False)

np.savez_compressed(
    local_maps,
    image_ids=primary["image_id"].astype(str).to_numpy(),
    efficientnet_cam=eff_maps
)

shutil.copyfile(local_csv, CSV_PATH)
shutil.copyfile(local_maps, MAP_PATH)

assert sha256_file(local_csv) == sha256_file(CSV_PATH)
assert sha256_file(local_maps) == sha256_file(MAP_PATH)

report = {
    "experiment": "Experiment05_Step5.4F",
    "model": "efficientnet_b0",
    "sample_count": 156,
    "cam_shape": [156, 224, 224],
    "attribution_target": "ground_truth_grade_logit",
    "target_layer": "conv_head",
    "cohort_sha256": EXPECTED_COHORT_SHA256,
    "csv_sha256": sha256_file(CSV_PATH),
    "maps_sha256": sha256_file(MAP_PATH),
    "test_split_evaluated": False,
}

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("STEP 5.4F - EFFICIENTNET RECOVERY")
print("Recovered images:", len(eff_records))
print("CAM array shape:", eff_maps.shape)
print("All CAM values finite: PASS")
print("Image ID alignment: PASS")
print("CSV saved:", CSV_PATH.is_file())
print("CAM maps saved:", MAP_PATH.is_file())
print("Transfer integrity: PASS")
print("Test split evaluated: False")
print("\nSTEP 5.4F COMPLETED")


# %% [cell 119] Experiment 05 - Step 5.4G

import gc
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ============================================================
# Experiment 05 - Step 5.4G
# RETFound Batch-Context Grad-CAM Pilot
# ============================================================

ROOT = DR_RESEARCH
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
PRED_PATH = ROOT / "experiment04_results/retfound_vit_large_v1/best_validation_predictions.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
MODEL_PATH = ROOT / "experiment04_results/retfound_vit_large_v1/best_model.pt"

for path in [INDEX_PATH, PRED_PATH, COHORT_PATH, MODEL_PATH]:
    assert path.is_file(), f"Missing: {path}"

index = pd.read_csv(INDEX_PATH)
saved = pd.read_csv(PRED_PATH)
cohort = pd.read_csv(COHORT_PATH)

assert len(saved) == 276
assert saved["image_id"].is_unique
assert len(cohort) == 276

validation = saved.merge(
    index[["image_id", "split", "local_path", "source_path"]],
    on="image_id",
    how="left",
    validate="one_to_one"
)

assert validation["split"].eq("val").all()

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_validation_batch(frame):
    images = []
    for row in frame.itertuples(index=False):
        path = Path(str(row.local_path))
        if not path.is_file():
            path = Path(str(row.source_path))
        assert path.is_file(), f"Image missing: {row.image_id}"
        with Image.open(path) as img:
            images.append(transform(img.convert("RGB")))
    return torch.stack(images).to("cuda")

# Remove possible leftover models from earlier diagnostics.
for name in ["current_model", "diagnostic_model", "ret_model"]:
    if name in globals():
        del globals()[name]

gc.collect()
torch.cuda.empty_cache()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
    mmap=True
)

retfound_cam_model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=5,
    global_pool="avg"
)

retfound_cam_model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

del checkpoint
retfound_cam_model.eval()
retfound_cam_model.set_grad_checkpointing(enable=False)

# Freeze parameters. The target activation itself will
# receive gradients through a detached leaf tensor.
retfound_cam_model.requires_grad_(False)
retfound_cam_model = retfound_cam_model.cuda()

captured = {}

def activation_hook(module, inputs, output):
    activation = output.detach().requires_grad_(True)
    captured["activation"] = activation
    return activation

hook_handle = retfound_cam_model.blocks[-1].register_forward_hook(
    activation_hook
)

try:
    first_batch = validation.iloc[:16].copy()
    images = load_validation_batch(first_batch)

    torch.cuda.reset_peak_memory_stats()

    with torch.enable_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = retfound_cam_model(images)

    probabilities = torch.softmax(
        logits.float(), dim=1
    ).detach().cpu().numpy()

    expected = first_batch[
        [f"prob_dr{i}" for i in range(5)]
    ].to_numpy(dtype=np.float32)

    max_difference = float(
        np.max(np.abs(probabilities - expected))
    )

    prediction_matches = bool(
        np.array_equal(
            probabilities.argmax(axis=1),
            first_batch["predicted_grade"].to_numpy(dtype=int)
        )
    )

    print("STEP 5.4G - BATCH CONTEXT PILOT")
    print("Batch images:", len(first_batch))
    print("Maximum probability difference:", max_difference)
    print("All predicted grades match:", prediction_matches)

    assert prediction_matches
    assert max_difference <= 1e-4, (
        "Batch probabilities do not reproduce validation."
    )

    target_position = 0
    target_class = int(
        first_batch.iloc[target_position]["true_grade"]
    )

    gradient = torch.autograd.grad(
        outputs=logits[target_position, target_class],
        inputs=captured["activation"],
        retain_graph=False
    )[0]

    activation = captured["activation"][target_position].float()
    grad = gradient[target_position].float()

    assert activation.shape == (197, 1024)
    assert grad.shape == activation.shape

    activation = activation[1:].transpose(0, 1).reshape(1024, 14, 14)
    grad = grad[1:].transpose(0, 1).reshape(1024, 14, 14)

    channel_weights = grad.mean(dim=(1, 2), keepdim=True)
    raw_cam = torch.relu(
        (channel_weights * activation).sum(dim=0)
    ).detach().cpu().numpy()

    assert np.isfinite(raw_cam).all()
    assert float(raw_cam.max()) > 1e-12

    cam_224 = cv2.resize(
        raw_cam,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )
    cam_224 = cam_224 / float(cam_224.max())

    print("\nGRAD-CAM RESULTS")
    print("Image ID:", first_batch.iloc[0]["image_id"])
    print("Attribution target grade:", target_class)
    print("CAM shape:", cam_224.shape)
    print("CAM standard deviation:", float(cam_224.std()))
    print("CAM finite:", bool(np.isfinite(cam_224).all()))
    print(
        "Peak GPU allocated GiB:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    )

    print("\nSTEP 5.4G COMPLETED")

finally:
    hook_handle.remove()
    captured.clear()
    if "images" in locals():
        del images
    if "logits" in locals():
        del logits
    gc.collect()
    torch.cuda.empty_cache()


# %% [cell 120] Experiment 05 - Step 5.4H

# ============================================================
# Experiment 05 - Step 5.4H
# RETFound Resumable Batch-Context Grad-CAM
# ============================================================

import os
import gc
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

# ============================================================
# 1. Paths and locked settings
# ============================================================

ROOT = DR_RESEARCH
RESULT_DIR = ROOT / "experiment05_results"
OUT_DIR = RESULT_DIR / "gradcam"
BATCH_DIR = OUT_DIR / "retfound_batch_cache_v1"

OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

PRED_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_validation_predictions.csv"
)

COHORT_PATH = (
    RESULT_DIR / "FGADR_Validation_MA_Cohort_V1.csv"
)

CHECKPOINT_PATH = (
    ROOT / "experiment04_results"
    / "retfound_vit_large_v1"
    / "best_model.pt"
)

FINAL_CSV = OUT_DIR / "RETFound_MA_Attribution_V1.csv"
FINAL_NPZ = OUT_DIR / "RETFound_MA_CAM_Maps_V1.npz"
FINAL_REPORT = OUT_DIR / "RETFound_MA_Attribution_Report_V1.json"

EXPECTED_COHORT_SHA256 = (
    "542c4f21780ba4060945e893a22c4e133b0f44a006fecb94ea5da4b76b93b286"
)

BATCH_SIZE = 16
PROBABILITY_TOLERANCE = 1e-4
DEVICE = torch.device("cuda")

assert torch.cuda.is_available()

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

for path in [
    INDEX_PATH, PRED_PATH, COHORT_PATH, CHECKPOINT_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

assert sha256_file(COHORT_PATH) == EXPECTED_COHORT_SHA256

if any(path.exists() for path in [
    FINAL_CSV, FINAL_NPZ, FINAL_REPORT
]):
    raise FileExistsError(
        "Final outputs already exist. Inspect them before rerunning."
    )

# ============================================================
# 2. Read fixed validation metadata
# ============================================================

index = pd.read_csv(INDEX_PATH)
saved = pd.read_csv(PRED_PATH)
cohort = pd.read_csv(COHORT_PATH)

assert len(saved) == 276
assert len(cohort) == 276
assert saved["image_id"].is_unique
assert cohort["image_id"].is_unique

validation = saved.merge(
    index[["image_id", "split", "local_path", "source_path"]],
    on="image_id",
    how="left",
    validate="one_to_one"
)

assert validation["split"].eq("val").all()
assert len(validation) == 276

primary = cohort.loc[
    cohort["primary_ma_candidate"] == True
].copy()

assert len(primary) == 156
assert primary["image_id"].is_unique

primary_lookup = primary.set_index("image_id")
primary_ids = set(primary_lookup.index)

assert primary_ids.issubset(set(validation["image_id"]))

# ============================================================
# 3. Reproduce formal validation preprocessing
# ============================================================

validation_transform = transforms.Compose([
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

def load_image_batch(frame):
    tensors = []

    for row in frame.itertuples(index=False):
        path = Path(str(row.local_path))

        if not path.is_file():
            path = Path(str(row.source_path))

        assert path.is_file(), (
            f"Image not found: {row.image_id}"
        )

        with Image.open(path) as image:
            tensors.append(
                validation_transform(image.convert("RGB"))
            )

    return torch.stack(tensors).to(DEVICE)

def load_weighted_mask(row):
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
        binary,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    assert float(weighted.sum()) > 0

    return weighted

# ============================================================
# 4. Identify unfinished batches before loading model
# ============================================================

batch_frames = [
    validation.iloc[start:start + BATCH_SIZE].copy()
    for start in range(0, len(validation), BATCH_SIZE)
]

assert len(batch_frames) == 18

def batch_paths(batch_idx):
    name = f"batch_{batch_idx:03d}"
    return (
        BATCH_DIR / f"{name}.csv",
        BATCH_DIR / f"{name}.npz"
    )

def batch_is_complete(batch_idx, frame):
    csv_path, npz_path = batch_paths(batch_idx)

    if not csv_path.exists() or not npz_path.exists():
        return False

    expected_ids = [
        image_id for image_id in frame["image_id"]
        if image_id in primary_ids
    ]

    batch_csv = pd.read_csv(csv_path)

    with np.load(npz_path, allow_pickle=False) as data:
        stored_ids = data["image_ids"].astype(str).tolist()
        stored_cams = data["cam"]

        assert stored_ids == expected_ids
        assert stored_cams.shape == (
            len(expected_ids), 224, 224
        )
        assert np.isfinite(stored_cams).all()

    assert batch_csv["image_id"].astype(str).tolist() == expected_ids

    return True

pending = [
    i for i, frame in enumerate(batch_frames)
    if not batch_is_complete(i, frame)
]

print("STEP 5.4H - RETFOUND BATCH ATTRIBUTION")
print("Total validation batches:", len(batch_frames))
print("Primary MA images:", len(primary))
print("Completed batches:", len(batch_frames) - len(pending))
print("Pending batches:", len(pending))
print("Device:", torch.cuda.get_device_name(0))

# ============================================================
# 5. Load model once
# ============================================================

if pending:

    # Release previous pilot models if still present.
    for name in [
        "retfound_cam_model",
        "model",
        "diagnostic_model",
        "current_model"
    ]:
        if name in globals():
            del globals()[name]

    gc.collect()
    torch.cuda.empty_cache()

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
    model = model.to(DEVICE)

    captured = {}

    def activation_hook(module, inputs, output):
        # Preserve the exact forward activation values while
        # making this layer an independent gradient target.
        activation = output.detach().requires_grad_(True)
        captured["activation"] = activation
        return activation

    hook_handle = model.blocks[-1].register_forward_hook(
        activation_hook
    )

    try:

        for batch_idx in tqdm(
            pending,
            desc="RETFound validation batches"
        ):

            frame = batch_frames[batch_idx]

            target_positions = [
                position
                for position, image_id in enumerate(
                    frame["image_id"].tolist()
                )
                if image_id in primary_ids
            ]

            assert len(target_positions) > 0

            images = load_image_batch(frame)

            captured.clear()

            with torch.enable_grad():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16
                ):
                    logits = model(images)

            probabilities = torch.softmax(
                logits.float(),
                dim=1
            ).detach().cpu().numpy()

            expected_probabilities = frame[
                [f"prob_dr{i}" for i in range(5)]
            ].to_numpy(dtype=np.float32)

            max_difference = float(
                np.max(
                    np.abs(
                        probabilities - expected_probabilities
                    )
                )
            )

            predictions_match = np.array_equal(
                probabilities.argmax(axis=1),
                frame["predicted_grade"].to_numpy(dtype=int)
            )

            assert predictions_match, (
                f"Prediction mismatch in batch {batch_idx}"
            )

            assert max_difference <= PROBABILITY_TOLERANCE, (
                f"Probability mismatch in batch {batch_idx}: "
                f"{max_difference}"
            )

            assert "activation" in captured

            activation = captured["activation"]

            assert activation.shape == (
                len(frame), 197, 1024
            )

            batch_records = []
            batch_cam_maps = []
            batch_image_ids = []

            for target_number, position in enumerate(
                target_positions
            ):

                row = frame.iloc[position]
                image_id = str(row["image_id"])
                lesion_row = primary_lookup.loc[image_id]

                target_class = int(row["true_grade"])

                # Retain the graph until the final target
                # in the current validation batch.
                keep_graph = (
                    target_number < len(target_positions) - 1
                )

                gradient = torch.autograd.grad(
                    outputs=logits[position, target_class],
                    inputs=activation,
                    retain_graph=keep_graph,
                    create_graph=False
                )[0]

                spatial_activation = (
                    activation[position, 1:, :]
                    .detach()
                    .float()
                    .transpose(0, 1)
                    .reshape(1024, 14, 14)
                )

                spatial_gradient = (
                    gradient[position, 1:, :]
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
                    (
                        channel_weights
                        * spatial_activation
                    ).sum(dim=0)
                ).cpu().numpy()

                assert np.isfinite(raw_cam).all(), (
                    f"Non-finite CAM: {image_id}"
                )

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

                weighted_mask = load_weighted_mask(
                    lesion_row
                )

                full_mean = float(cam.mean())
                ma_mean = float(
                    (cam * weighted_mask).sum()
                    / weighted_mask.sum()
                )

                enrichment = (
                    ma_mean / full_mean
                    if full_mean > 1e-12
                    else np.nan
                )

                assert np.isfinite(enrichment), (
                    f"Invalid enrichment: {image_id}"
                )

                batch_records.append({
                    "image_id": image_id,
                    "true_grade": target_class,
                    "model": "retfound",
                    "target_layer": "blocks.23",
                    "predicted_grade": int(
                        probabilities[position].argmax()
                    ),
                    "correct": (
                        int(probabilities[position].argmax())
                        == target_class
                    ),
                    "mask_weight_sum": float(
                        weighted_mask.sum()
                    ),
                    "mean_cam_full": full_mean,
                    "mean_cam_ma": ma_mean,
                    "ma_enrichment": enrichment,
                    "raw_cam_max": raw_max,
                    "cam_std": float(cam.std()),
                    "batch_index": batch_idx,
                    "batch_position": position,
                    "probability_max_difference": float(
                        np.max(np.abs(
                            probabilities[position]
                            - expected_probabilities[position]
                        ))
                    )
                })

                batch_cam_maps.append(cam)
                batch_image_ids.append(image_id)

                del gradient

            # =================================================
            # Save one completed batch atomically
            # =================================================

            csv_path, npz_path = batch_paths(batch_idx)

            assert not csv_path.exists()
            assert not npz_path.exists()

            csv_temp = Path(str(csv_path) + ".partial")
            npz_temp = Path(str(npz_path) + ".partial")

            batch_df = pd.DataFrame(batch_records)

            with open(csv_temp, "w", encoding="utf-8") as handle:
                batch_df.to_csv(handle, index=False)

            with open(npz_temp, "wb") as handle:
                np.savez_compressed(
                    handle,
                    image_ids=np.asarray(
                        batch_image_ids, dtype=str
                    ),
                    cam=np.stack(
                        batch_cam_maps
                    ).astype(np.float32)
                )

            os.replace(npz_temp, npz_path)
            os.replace(csv_temp, csv_path)

            print(
                f"Batch {batch_idx:02d}: "
                f"{len(batch_records)} CAMs saved, "
                f"max probability difference "
                f"{max_difference:.8f}"
            )

            del images, logits, probabilities, activation
            captured.clear()
            gc.collect()

    finally:
        hook_handle.remove()
        captured.clear()
        del model
        gc.collect()
        torch.cuda.empty_cache()

# ============================================================
# 6. Merge completed batch outputs
# ============================================================

all_frames = []
all_ids = []
all_cams = []

for batch_idx, frame in enumerate(batch_frames):

    assert batch_is_complete(batch_idx, frame), (
        f"Incomplete batch: {batch_idx}"
    )

    csv_path, npz_path = batch_paths(batch_idx)
    all_frames.append(pd.read_csv(csv_path))

    with np.load(npz_path, allow_pickle=False) as data:
        all_ids.extend(data["image_ids"].astype(str).tolist())
        all_cams.append(data["cam"].copy())

final_df = pd.concat(all_frames, ignore_index=True)
final_cams = np.concatenate(all_cams, axis=0)

assert len(final_df) == 156
assert final_cams.shape == (156, 224, 224)
assert final_df["image_id"].is_unique
assert final_df["image_id"].astype(str).tolist() == all_ids

assert set(all_ids) == primary_ids
assert np.isfinite(final_cams).all()
assert np.isfinite(final_df["ma_enrichment"]).all()

# Reorder according to the frozen primary cohort.
order = {
    image_id: idx
    for idx, image_id in enumerate(all_ids)
}

sorted_ids = sorted(primary_ids)
sorted_positions = [
    order[image_id] for image_id in sorted_ids
]

final_df = (
    final_df.set_index("image_id")
    .loc[sorted_ids]
    .reset_index()
)

final_cams = final_cams[sorted_positions]

# ============================================================
# 7. Persist final results
# ============================================================

csv_temp = Path(str(FINAL_CSV) + ".partial")
npz_temp = Path(str(FINAL_NPZ) + ".partial")

final_df.to_csv(csv_temp, index=False)

with open(npz_temp, "wb") as handle:
    np.savez_compressed(
        handle,
        image_ids=np.asarray(sorted_ids, dtype=str),
        retfound_cam=final_cams
    )

os.replace(npz_temp, FINAL_NPZ)
os.replace(csv_temp, FINAL_CSV)

report = {
    "experiment": "Experiment05_Step5.4H",
    "model": "RETFound_ViT_Large",
    "checkpoint": str(CHECKPOINT_PATH),
    "target_layer": "blocks.23",
    "attribution_target": "ground_truth_grade_logit",
    "precision": "bfloat16",
    "validation_batch_size": BATCH_SIZE,
    "cohort_sha256": EXPECTED_COHORT_SHA256,
    "n_primary_images": 156,
    "n_validation_batches": 18,
    "probability_tolerance": PROBABILITY_TOLERANCE,
    "max_observed_probability_difference": float(
        final_df["probability_max_difference"].max()
    ),
    "csv_sha256": sha256_file(FINAL_CSV),
    "maps_sha256": sha256_file(FINAL_NPZ),
    "test_split_evaluated": False
}

with open(FINAL_REPORT, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nSTEP 5.4H - FINAL RESULTS")
print("Processed primary images:", len(final_df))
print("CAM tensor shape:", final_cams.shape)
print(
    "Maximum probability difference:",
    report["max_observed_probability_difference"]
)
print(
    "All attribution values finite:",
    bool(np.isfinite(final_cams).all())
)

print("\nMA ENRICHMENT SUMMARY")
print(
    final_df["ma_enrichment"]
    .agg(["count", "mean", "median", "std"])
    .to_string()
)

print("\nOUTPUT FILES")
print(FINAL_CSV)
print(FINAL_NPZ)
print(FINAL_REPORT)
print("Test evaluated: False")
print("STEP 5.4H COMPLETED")


# %% [cell 121] Experiment 05 - Step 5.5A

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

# ============================================================
# Experiment 05 - Step 5.5A
# Paired MA Attribution Enrichment Audit
# ============================================================

ROOT = DR_RESEARCH
OUT = ROOT / "experiment05_results"
CAM_DIR = OUT / "gradcam"

EFF_CSV = CAM_DIR / "EfficientNet_MA_Attribution_V1.csv"
RET_CSV = CAM_DIR / "RETFound_MA_Attribution_V1.csv"

EFF_NPZ = CAM_DIR / "EfficientNet_MA_CAM_Maps_V1.npz"
RET_NPZ = CAM_DIR / "RETFound_MA_CAM_Maps_V1.npz"

COHORT_CSV = OUT / "FGADR_Validation_MA_Cohort_V1.csv"

PAIRED_CSV = OUT / "MA_Attribution_Paired_V1.csv"
REPORT_JSON = OUT / "MA_Attribution_Paired_Report_V1.json"

for path in [
    EFF_CSV, RET_CSV, EFF_NPZ, RET_NPZ, COHORT_CSV
]:
    assert path.is_file(), f"Missing file: {path}"

assert not PAIRED_CSV.exists()
assert not REPORT_JSON.exists()

eff = pd.read_csv(EFF_CSV)
ret = pd.read_csv(RET_CSV)

cohort = pd.read_csv(COHORT_CSV)
primary = cohort.loc[
    cohort["primary_ma_candidate"] == True
].copy()

assert len(primary) == 156

# ============================================================
# 1. Strict image-level pairing
# ============================================================

columns = [
    "image_id",
    "true_grade",
    "predicted_grade",
    "ma_enrichment",
    "mean_cam_full",
    "mean_cam_ma",
    "cam_std",
    "mask_weight_sum"
]

eff = eff[columns].rename(
    columns={
        col: f"efficientnet_{col}"
        for col in columns if col != "image_id"
    }
)

ret = ret[columns].rename(
    columns={
        col: f"retfound_{col}"
        for col in columns if col != "image_id"
    }
)

paired = eff.merge(
    ret,
    on="image_id",
    how="outer",
    validate="one_to_one",
    indicator=True
)

assert len(paired) == 156
assert paired["_merge"].eq("both").all()

paired = paired.drop(columns="_merge")

paired = paired.merge(
    primary[[
        "image_id",
        "true_grade",
        "mask_coverage",
        "candidate_count",
        "largest_component_area_px",
        "resolved_mask_path",
        "reference_threshold"
    ]],
    on="image_id",
    validate="one_to_one"
)

assert (
    paired["efficientnet_true_grade"]
    == paired["true_grade"]
).all()

assert (
    paired["retfound_true_grade"]
    == paired["true_grade"]
).all()

assert np.allclose(
    paired["efficientnet_mask_weight_sum"],
    paired["retfound_mask_weight_sum"],
    atol=1e-5,
    rtol=1e-5
)

# ============================================================
# 2. Verify saved CAM array identity and quality
# ============================================================

with np.load(EFF_NPZ, allow_pickle=False) as data:
    eff_ids = data["image_ids"].astype(str)
    eff_maps = data["efficientnet_cam"].copy()

with np.load(RET_NPZ, allow_pickle=False) as data:
    ret_ids = data["image_ids"].astype(str)
    ret_maps = data["retfound_cam"].copy()

assert eff_maps.shape == (156, 224, 224)
assert ret_maps.shape == (156, 224, 224)

assert np.array_equal(eff_ids, ret_ids)
assert set(eff_ids) == set(paired["image_id"])

assert np.isfinite(eff_maps).all()
assert np.isfinite(ret_maps).all()

# Verify CSV whole-image mean against the saved CAM tensors.
eff_mean_lookup = dict(zip(
    eff_ids, eff_maps.mean(axis=(1, 2))
))
ret_mean_lookup = dict(zip(
    ret_ids, ret_maps.mean(axis=(1, 2))
))

assert np.allclose(
    paired["efficientnet_mean_cam_full"].to_numpy(),
    paired["image_id"].map(eff_mean_lookup).to_numpy(),
    atol=1e-5
)

assert np.allclose(
    paired["retfound_mean_cam_full"].to_numpy(),
    paired["image_id"].map(ret_mean_lookup).to_numpy(),
    atol=1e-5
)

# ============================================================
# 3. Paired descriptive metrics
# ============================================================

paired["delta_enrichment"] = (
    paired["retfound_ma_enrichment"]
    - paired["efficientnet_ma_enrichment"]
)

paired["efficientnet_correct"] = (
    paired["efficientnet_predicted_grade"]
    == paired["true_grade"]
)

paired["retfound_correct"] = (
    paired["retfound_predicted_grade"]
    == paired["true_grade"]
)

paired = paired.sort_values(
    "image_id"
).reset_index(drop=True)

print("STEP 5.5A - PAIRED MA ATTRIBUTION AUDIT")
print("Paired images:", len(paired))
print("CAM array alignment: PASS")
print("Prediction label consistency: PASS")
print("Mask area consistency: PASS")

print("\nENRICHMENT BY MODEL")

for name in ["efficientnet", "retfound"]:
    values = paired[f"{name}_ma_enrichment"]
    print(f"\n{name.upper()}")
    print("Mean:", round(values.mean(), 6))
    print("Median:", round(values.median(), 6))
    print("Standard deviation:", round(values.std(), 6))
    print("Images with enrichment > 1:", int((values > 1).sum()))

print("\nPAIRED DIFFERENCES")
print(
    "Mean RETFound - EfficientNet:",
    round(paired["delta_enrichment"].mean(), 6)
)
print(
    "Median RETFound - EfficientNet:",
    round(paired["delta_enrichment"].median(), 6)
)
print(
    "RETFound higher images:",
    int((paired["delta_enrichment"] > 0).sum())
)
print(
    "EfficientNet higher images:",
    int((paired["delta_enrichment"] < 0).sum())
)

print("\nBY DR GRADE")
print(
    paired.groupby("true_grade").agg(
        images=("image_id", "count"),
        efficientnet_median=(
            "efficientnet_ma_enrichment", "median"
        ),
        retfound_median=(
            "retfound_ma_enrichment", "median"
        ),
        median_difference=(
            "delta_enrichment", "median"
        )
    ).to_string()
)

# ============================================================
# 4. Save paired analysis
# ============================================================

paired.to_csv(PAIRED_CSV, index=False)

report = {
    "experiment": "Experiment05_Step5.5A",
    "split": "val",
    "sample_count": 156,
    "attribution_target": "ground_truth_grade_logit",
    "efficientnet_layer": "conv_head",
    "retfound_layer": "blocks.23",
    "retfound_forward_precision": "BF16",
    "efficientnet_forward_precision": "FP32",
    "efficientnet_mean_enrichment": float(
        paired["efficientnet_ma_enrichment"].mean()
    ),
    "retfound_mean_enrichment": float(
        paired["retfound_ma_enrichment"].mean()
    ),
    "mean_paired_difference": float(
        paired["delta_enrichment"].mean()
    ),
    "median_paired_difference": float(
        paired["delta_enrichment"].median()
    ),
    "spatial_null_control_completed": False,
    "test_split_evaluated": False
}

with open(REPORT_JSON, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nOUTPUT FILES")
print(PAIRED_CSV)
print(REPORT_JSON)
print("STEP 5.5A COMPLETED")


# %% [cell 122] Experiment 05 - Step 5.5A Fix

from pathlib import Path
import numpy as np
import pandas as pd
import os

# ============================================================
# Experiment 05 - Step 5.5A Fix
# Repair EfficientNet NPZ String Encoding
# ============================================================

ROOT = DR_RESEARCH
CAM_DIR = ROOT / "experiment05_results" / "gradcam"

CSV_PATH = CAM_DIR / "EfficientNet_MA_Attribution_V1.csv"
NPZ_PATH = CAM_DIR / "EfficientNet_MA_CAM_Maps_V1.npz"

assert CSV_PATH.is_file()
assert NPZ_PATH.is_file()

records = pd.read_csv(CSV_PATH)

assert len(records) == 156
assert records["image_id"].is_unique

# Use the saved CSV ordering as the canonical image order.
image_ids = records["image_id"].astype(str).tolist()

with np.load(NPZ_PATH, allow_pickle=False) as data:
    print("NPZ keys:", data.files)

    # Read the CAM tensor without deserializing the
    # object-encoded image ID array.
    cam_maps = data["efficientnet_cam"].copy()

assert cam_maps.shape == (156, 224, 224)
assert np.isfinite(cam_maps).all()

# Confirm the saved CSV is in the original cohort order.
cohort = pd.read_csv(
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

expected_ids = (
    cohort.loc[
        cohort["primary_ma_candidate"] == True,
        "image_id"
    ]
    .astype(str)
    .sort_values()
    .tolist()
)

assert image_ids == expected_ids, (
    "CSV order differs from the frozen cohort order."
)

# Write Unicode strings, not object arrays.
safe_ids = np.asarray(image_ids, dtype="U64")

temporary_path = Path(str(NPZ_PATH) + ".fixed.npz")

assert not temporary_path.exists()

np.savez_compressed(
    temporary_path,
    image_ids=safe_ids,
    efficientnet_cam=cam_maps
)

# Verify the repaired file before replacing the old version.
with np.load(temporary_path, allow_pickle=False) as data:
    verified_ids = data["image_ids"].astype(str)
    verified_maps = data["efficientnet_cam"]

    assert verified_ids.tolist() == image_ids
    assert np.array_equal(verified_maps, cam_maps)

# Preserve the original file as a backup.
backup_path = CAM_DIR / "EfficientNet_MA_CAM_Maps_V1_original.npz"

assert not backup_path.exists()
os.replace(NPZ_PATH, backup_path)
os.replace(temporary_path, NPZ_PATH)

print("\nSTEP 5.5A - NPZ FORMAT REPAIR")
print("Images:", len(image_ids))
print("CAM shape:", cam_maps.shape)
print("Image ID dtype:", safe_ids.dtype)
print("CAM values unchanged: PASS")
print("Safe NPZ loading: PASS")
print("Original file preserved:", backup_path)
print("Repaired file:", NPZ_PATH)
print("STEP 5.5A FIX COMPLETED")


# %% [cell 123] Experiment 05 - Step 5.5B

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ============================================================
# Experiment 05 - Step 5.5B
# Paired MA Attribution Analysis and NPZ Repair Audit
# ============================================================

ROOT = DR_RESEARCH
OUT = ROOT / "experiment05_results"
CAM_DIR = OUT / "gradcam"

EFF_CSV = CAM_DIR / "EfficientNet_MA_Attribution_V1.csv"
RET_CSV = CAM_DIR / "RETFound_MA_Attribution_V1.csv"

EFF_NPZ = CAM_DIR / "EfficientNet_MA_CAM_Maps_V1.npz"
EFF_BACKUP = CAM_DIR / "EfficientNet_MA_CAM_Maps_V1_original.npz"
RET_NPZ = CAM_DIR / "RETFound_MA_CAM_Maps_V1.npz"

COHORT_PATH = OUT / "FGADR_Validation_MA_Cohort_V1.csv"

PAIRED_PATH = OUT / "MA_Attribution_Paired_V1.csv"
REPORT_PATH = OUT / "MA_Attribution_Paired_Report_V1.json"
REPAIR_PATH = CAM_DIR / "EfficientNet_NPZ_Format_Repair_Audit_V1.json"

for path in [
    EFF_CSV, RET_CSV, EFF_NPZ,
    EFF_BACKUP, RET_NPZ, COHORT_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

assert not PAIRED_PATH.exists()
assert not REPORT_PATH.exists()
assert not REPAIR_PATH.exists()

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(block)
    return digest.hexdigest()

# ============================================================
# 1. Verify frozen cohort
# ============================================================

EXPECTED_COHORT_SHA256 = (
    "542c4f21780ba4060945e893a22c4e133b0f44a006fecb94ea5da4b76b93b286"
)

assert sha256_file(COHORT_PATH) == EXPECTED_COHORT_SHA256

cohort = pd.read_csv(COHORT_PATH)

primary = cohort.loc[
    cohort["primary_ma_candidate"] == True
].copy()

assert len(primary) == 156
assert primary["image_id"].is_unique

expected_ids = sorted(primary["image_id"].astype(str).tolist())

# ============================================================
# 2. Load attribution records
# ============================================================

eff = pd.read_csv(EFF_CSV)
ret = pd.read_csv(RET_CSV)

required_columns = [
    "image_id",
    "true_grade",
    "predicted_grade",
    "ma_enrichment",
    "mean_cam_full",
    "mean_cam_ma",
    "cam_std",
    "mask_weight_sum",
]

for name, frame in [
    ("EfficientNet", eff),
    ("RETFound", ret)
]:
    assert len(frame) == 156, name
    assert frame["image_id"].is_unique, name
    assert set(frame["image_id"]) == set(expected_ids), name
    assert frame[required_columns].notna().all().all(), name

eff = eff[required_columns].rename(
    columns={
        column: f"efficientnet_{column}"
        for column in required_columns
        if column != "image_id"
    }
)

ret = ret[required_columns].rename(
    columns={
        column: f"retfound_{column}"
        for column in required_columns
        if column != "image_id"
    }
)

paired = eff.merge(
    ret,
    on="image_id",
    validate="one_to_one",
    how="inner"
)

assert len(paired) == 156

paired = paired.merge(
    primary[
        [
            "image_id",
            "true_grade",
            "mask_coverage",
            "candidate_count",
            "largest_component_area_px"
        ]
    ],
    on="image_id",
    validate="one_to_one"
)

assert (
    paired["efficientnet_true_grade"]
    == paired["true_grade"]
).all()

assert (
    paired["retfound_true_grade"]
    == paired["true_grade"]
).all()

assert np.allclose(
    paired["efficientnet_mask_weight_sum"],
    paired["retfound_mask_weight_sum"],
    atol=1e-5,
    rtol=1e-5
)

paired = paired.sort_values("image_id").reset_index(drop=True)

# ============================================================
# 3. Load CAM arrays safely
# ============================================================

with np.load(EFF_NPZ, allow_pickle=False) as data:
    eff_ids = data["image_ids"].astype(str)
    eff_maps = data["efficientnet_cam"].copy()

with np.load(RET_NPZ, allow_pickle=False) as data:
    ret_ids = data["image_ids"].astype(str)
    ret_maps = data["retfound_cam"].copy()

assert eff_maps.shape == (156, 224, 224)
assert ret_maps.shape == (156, 224, 224)
assert np.isfinite(eff_maps).all()
assert np.isfinite(ret_maps).all()

assert eff_ids.tolist() == expected_ids
assert ret_ids.tolist() == expected_ids

# Compare saved full-image CAM means against actual arrays.
for name, maps in [
    ("efficientnet", eff_maps),
    ("retfound", ret_maps)
]:
    recorded_means = paired[
        f"{name}_mean_cam_full"
    ].to_numpy(dtype=float)

    actual_means = maps.mean(
        axis=(1, 2),
        dtype=np.float64
    )

    assert np.allclose(
        recorded_means,
        actual_means,
        rtol=1e-4,
        atol=1e-6
    ), f"{name}: saved CAM means do not match."

# ============================================================
# 4. Calculate paired enrichment
# ============================================================

paired["delta_enrichment"] = (
    paired["retfound_ma_enrichment"]
    - paired["efficientnet_ma_enrichment"]
)

paired["efficientnet_correct"] = (
    paired["efficientnet_predicted_grade"]
    == paired["true_grade"]
)

paired["retfound_correct"] = (
    paired["retfound_predicted_grade"]
    == paired["true_grade"]
)

assert np.isfinite(
    paired["delta_enrichment"].to_numpy()
).all()

print("STEP 5.5B - PAIRED MA ATTRIBUTION")
print("Paired images:", len(paired))
print("Image ordering: PASS")
print("Ground-truth consistency: PASS")
print("Mask area consistency: PASS")
print("CAM array consistency: PASS")

print("\nENRICHMENT BY MODEL")

for name in ["efficientnet", "retfound"]:
    values = paired[f"{name}_ma_enrichment"]

    print(f"\n{name.upper()}")
    print("Mean:", round(float(values.mean()), 6))
    print("Median:", round(float(values.median()), 6))
    print("Standard deviation:", round(float(values.std()), 6))
    print("Enrichment > 1:", int((values > 1).sum()))

print("\nPAIRED DIFFERENCES")
print(
    "Mean difference:",
    round(float(paired["delta_enrichment"].mean()), 6)
)
print(
    "Median difference:",
    round(float(paired["delta_enrichment"].median()), 6)
)
print(
    "RETFound higher:",
    int((paired["delta_enrichment"] > 0).sum())
)
print(
    "EfficientNet higher:",
    int((paired["delta_enrichment"] < 0).sum())
)

print("\nBY DR GRADE")
print(
    paired.groupby("true_grade").agg(
        images=("image_id", "count"),
        efficientnet_median=(
            "efficientnet_ma_enrichment", "median"
        ),
        retfound_median=(
            "retfound_ma_enrichment", "median"
        ),
        median_difference=(
            "delta_enrichment", "median"
        )
    ).to_string()
)

# ============================================================
# 5. Save analytical results
# ============================================================

paired.to_csv(PAIRED_PATH, index=False)

report = {
    "experiment": "Experiment05_Step5.5B",
    "split": "val",
    "primary_images": 156,
    "attribution_target": "ground_truth_grade_logit",
    "efficientnet_layer": "conv_head",
    "retfound_layer": "blocks.23",
    "efficientnet_precision": "FP32",
    "retfound_precision": "BF16",
    "efficientnet_mean_enrichment": float(
        paired["efficientnet_ma_enrichment"].mean()
    ),
    "retfound_mean_enrichment": float(
        paired["retfound_ma_enrichment"].mean()
    ),
    "mean_difference": float(
        paired["delta_enrichment"].mean()
    ),
    "median_difference": float(
        paired["delta_enrichment"].median()
    ),
    "spatial_null_control_completed": False,
    "test_split_evaluated": False,
}

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

# Preserve provenance of the repaired NPZ without
# changing historical audit records.
repair_report = {
    "experiment": "Experiment05_NPZ_Format_Repair",
    "reason": "Object-encoded image IDs replaced with NumPy Unicode.",
    "original_npz": str(EFF_BACKUP),
    "repaired_npz": str(EFF_NPZ),
    "original_sha256": sha256_file(EFF_BACKUP),
    "repaired_sha256": sha256_file(EFF_NPZ),
    "cam_values_changed": False,
    "image_count": 156,
    "cam_shape": [156, 224, 224],
    "original_order_validation": (
        "The original CSV and CAM tensor ordering were "
        "validated against the frozen primary cohort "
        "during Step 5.4F."
    ),
    "created_utc": datetime.now(timezone.utc).isoformat()
}

with open(REPAIR_PATH, "w", encoding="utf-8") as handle:
    json.dump(repair_report, handle, indent=2)

print("\nOUTPUT FILES")
print(PAIRED_PATH)
print(REPORT_PATH)
print(REPAIR_PATH)
print("STEP 5.5B COMPLETED")


# %% [cell 124] Experiment 05 - Step 5.6A

# ============================================================
# Experiment 05 - Step 5.6A
# Exhaustive Circular-Shift Spatial Null
# ============================================================

from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

ROOT = DR_RESEARCH
OUT = ROOT / "experiment05_results"
CAM_DIR = OUT / "gradcam"

PAIRED_PATH = OUT / "MA_Attribution_Paired_V1.csv"
COHORT_PATH = OUT / "FGADR_Validation_MA_Cohort_V1.csv"

EFF_NPZ = CAM_DIR / "EfficientNet_MA_CAM_Maps_V1.npz"
RET_NPZ = CAM_DIR / "RETFound_MA_CAM_Maps_V1.npz"

RESULT_PATH = OUT / "MA_Spatial_Null_Unrestricted_V1.csv"
REPORT_PATH = OUT / "MA_Spatial_Null_Unrestricted_Report_V1.json"

assert not RESULT_PATH.exists()
assert not REPORT_PATH.exists()

paired = pd.read_csv(PAIRED_PATH).sort_values(
    "image_id"
).reset_index(drop=True)

cohort = pd.read_csv(COHORT_PATH)
primary = cohort.loc[
    cohort["primary_ma_candidate"] == True
].set_index("image_id")

with np.load(EFF_NPZ, allow_pickle=False) as data:
    eff_ids = data["image_ids"].astype(str)
    eff_maps = data["efficientnet_cam"].copy()

with np.load(RET_NPZ, allow_pickle=False) as data:
    ret_ids = data["image_ids"].astype(str)
    ret_maps = data["retfound_cam"].copy()

assert len(paired) == 156
assert eff_maps.shape == ret_maps.shape == (156, 224, 224)
assert eff_ids.tolist() == ret_ids.tolist()
assert eff_ids.tolist() == paired["image_id"].astype(str).tolist()

def load_ma_mask(image_id):
    row = primary.loc[image_id]

    with Image.open(row["resolved_mask_path"]) as image:
        mask = np.asarray(image)

    if mask.ndim == 3:
        assert np.array_equal(mask[..., 0], mask[..., 1])
        assert np.array_equal(mask[..., 0], mask[..., 2])
        mask = mask[..., 0]

    binary = (
        mask > int(row["reference_threshold"])
    ).astype(np.float32)

    return cv2.resize(
        binary,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

def compute_spatial_null(cam, mask):
    """
    Calculate lesion enrichment for every circular translation
    of the weighted mask using FFT cross-correlation.
    """

    mean_cam = float(cam.mean())
    mask_area = float(mask.sum())

    assert mean_cam > 1e-12
    assert mask_area > 0

    correlation = np.fft.ifft2(
        np.fft.fft2(cam.astype(np.float64))
        * np.conj(np.fft.fft2(mask.astype(np.float64)))
    ).real

    enrichment_map = (
        correlation / mask_area / mean_cam
    )

    observed = float(enrichment_map[0, 0])

    # Exclude the zero shift from the null distribution.
    null_values = enrichment_map.ravel()[1:]

    assert np.isfinite(null_values).all()

    upper_tail = float(
        (1 + np.count_nonzero(null_values >= observed))
        / (len(null_values) + 1)
    )

    return {
        "observed": observed,
        "null_mean": float(null_values.mean()),
        "null_median": float(np.median(null_values)),
        "null_std": float(null_values.std()),
        "null_q95": float(np.quantile(null_values, 0.95)),
        "upper_tail_fraction": upper_tail,
        "percentile_rank": float(
            100.0 * np.mean(null_values < observed)
        ),
        "null_adjusted_enrichment": float(
            observed - null_values.mean()
        )
    }

records = []

for i, image_id in enumerate(
    tqdm(eff_ids.tolist(), desc="Spatial null evaluation")
):

    mask = load_ma_mask(image_id)

    assert mask.shape == (224, 224)
    assert float(mask.sum()) > 0

    eff_result = compute_spatial_null(
        eff_maps[i], mask
    )
    ret_result = compute_spatial_null(
        ret_maps[i], mask
    )

    # Check zero-translation FFT computation against
    # the directly computed descriptive enrichment.
    source_row = paired.iloc[i]

    assert np.isclose(
        eff_result["observed"],
        source_row["efficientnet_ma_enrichment"],
        rtol=1e-4,
        atol=1e-5
    )

    assert np.isclose(
        ret_result["observed"],
        source_row["retfound_ma_enrichment"],
        rtol=1e-4,
        atol=1e-5
    )

    record = {"image_id": image_id}

    for prefix, result in [
        ("efficientnet", eff_result),
        ("retfound", ret_result)
    ]:
        for key, value in result.items():
            record[f"{prefix}_{key}"] = value

    record["delta_null_adjusted"] = (
        ret_result["null_adjusted_enrichment"]
        - eff_result["null_adjusted_enrichment"]
    )

    record["delta_percentile_rank"] = (
        ret_result["percentile_rank"]
        - eff_result["percentile_rank"]
    )

    records.append(record)

results = pd.DataFrame(records)

assert len(results) == 156
assert results["image_id"].is_unique

print("STEP 5.6A - UNRESTRICTED SPATIAL NULL")
print("Evaluated images:", len(results))
print("Translations per image:", 224 * 224)
print("Zero-shift enrichment consistency: PASS")

for name in ["efficientnet", "retfound"]:

    print(f"\n{name.upper()}")

    print(
        "Mean observed enrichment:",
        round(results[f"{name}_observed"].mean(), 6)
    )

    print(
        "Mean null enrichment:",
        round(results[f"{name}_null_mean"].mean(), 6)
    )

    print(
        "Median spatial percentile:",
        round(results[f"{name}_percentile_rank"].median(), 3)
    )

    print(
        "Images above 95th null percentile:",
        int((
            results[f"{name}_upper_tail_fraction"] <= 0.05
        ).sum())
    )

print("\nPAIRED SPATIAL COMPARISON")
print(
    "Mean difference in null-adjusted enrichment:",
    round(results["delta_null_adjusted"].mean(), 6)
)
print(
    "Median difference in spatial percentile rank:",
    round(results["delta_percentile_rank"].median(), 6)
)
print(
    "RETFound higher spatial percentile:",
    int((results["delta_percentile_rank"] > 0).sum())
)
print(
    "EfficientNet higher spatial percentile:",
    int((results["delta_percentile_rank"] < 0).sum())
)

results.to_csv(RESULT_PATH, index=False)

report = {
    "experiment": "Experiment05_Step5.6A",
    "method": "exhaustive_circular_shift_spatial_null",
    "sample_count": 156,
    "image_shape": [224, 224],
    "total_positions": 50176,
    "null_positions": 50175,
    "zero_shift_excluded": True,
    "retinal_foreground_constrained": False,
    "preserves_mask_area_and_shape": True,
    "uses_same_shift_domain_for_both_models": True,
    "test_split_evaluated": False
}

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nOUTPUT FILES")
print(RESULT_PATH)
print(REPORT_PATH)
print("STEP 5.6A COMPLETED")


# %% [cell 125] Experiment 05 - Step 5.6B

# ============================================================
# Experiment 05 - Step 5.6B
# Retinal Foreground Spatial-Null Feasibility Pilot
# ============================================================

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = DR_RESEARCH

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)
COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

index = pd.read_csv(INDEX_PATH)
cohort = pd.read_csv(COHORT_PATH)

pilot = (
    cohort.loc[cohort["primary_ma_candidate"] == True]
    .merge(
        index[["image_id", "split", "local_path", "source_path"]],
        on="image_id",
        validate="one_to_one"
    )
    .sort_values("image_id")
    .iloc[:12]
    .reset_index(drop=True)
)

assert len(pilot) == 12
assert pilot["split"].eq("val").all()

SIZE = 224
MIN_FOREGROUND_COVERAGE = 0.95
MIN_NULL_POSITIONS = 100

def load_pilot_image(row):
    path = Path(str(row.local_path))

    if not path.is_file():
        path = Path(str(row.source_path))

    assert path.is_file()

    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (SIZE, SIZE),
                Image.Resampling.BICUBIC
            )
        )

    return rgb


def estimate_retinal_foreground(rgb):
    """
    Estimate the largest illuminated retinal field.
    This is an image-processing proxy, not a clinical ROI.
    """
    intensity = rgb.max(axis=2)

    candidate = (intensity > 20).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)
    )

    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    assert contours, "No retinal foreground detected."

    foreground = np.zeros(
        (SIZE, SIZE), dtype=np.uint8
    )

    largest = max(contours, key=cv2.contourArea)

    cv2.drawContours(
        foreground, [largest], -1, 1, thickness=-1
    )

    # Erode slightly to reduce uncertainty near the border.
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)
    )

    foreground = cv2.erode(
        foreground, erosion_kernel
    )

    return foreground.astype(np.float32)


def load_weighted_ma(row):
    with Image.open(row.resolved_mask_path) as image:
        raw = np.asarray(image)

    if raw.ndim == 3:
        assert np.array_equal(raw[..., 0], raw[..., 1])
        assert np.array_equal(raw[..., 0], raw[..., 2])
        raw = raw[..., 0]

    binary = (
        raw > int(row.reference_threshold)
    ).astype(np.float32)

    mask = cv2.resize(
        binary,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    )

    assert mask.sum() > 0
    return mask


def measure_eligible_shifts(mask, foreground):
    """
    Require no boundary wrapping and >=95% weighted
    MA-mask overlap with the estimated retinal foreground.
    """

    ys, xs = np.nonzero(mask > 0)

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())

    # Integer shifts that keep the entire mask inside
    # the image without circular wraparound.
    dx = np.arange(-xmin, SIZE - xmax, dtype=int)
    dy = np.arange(-ymin, SIZE - ymax, dtype=int)

    possible_count = int(len(dx) * len(dy))

    # For each circular-shift index, evaluate the fraction
    # of the weighted mask lying inside the retinal field.
    overlap = np.fft.ifft2(
        np.fft.fft2(foreground.astype(np.float64))
        * np.conj(np.fft.fft2(mask.astype(np.float64)))
    ).real

    coverage = overlap / float(mask.sum())

    shifts_y, shifts_x = np.meshgrid(
        dy % SIZE, dx % SIZE, indexing="ij"
    )

    valid_coverage = coverage[
        shifts_y, shifts_x
    ]

    eligible_count = int(
        np.count_nonzero(
            valid_coverage >= MIN_FOREGROUND_COVERAGE
        )
    )

    observed_coverage = float(
        (foreground * mask).sum() / mask.sum()
    )

    # Exclude the observed zero shift from null candidates.
    if observed_coverage >= MIN_FOREGROUND_COVERAGE:
        eligible_null_count = eligible_count - 1
    else:
        eligible_null_count = eligible_count

    return {
        "retinal_fraction": float(foreground.mean()),
        "observed_ma_foreground_coverage": observed_coverage,
        "no_wrap_shifts": possible_count,
        "eligible_null_shifts": eligible_null_count,
        "sufficient_null_positions": (
            eligible_null_count >= MIN_NULL_POSITIONS
        )
    }


pilot_results = []

for row in pilot.itertuples(index=False):

    rgb = load_pilot_image(row)
    foreground = estimate_retinal_foreground(rgb)
    mask = load_weighted_ma(row)

    result = measure_eligible_shifts(
        mask, foreground
    )

    pilot_results.append({
        "image_id": row.image_id,
        **result
    })

pilot_df = pd.DataFrame(pilot_results)

print("STEP 5.6B - RETINAL NULL FEASIBILITY")
print("Pilot images:", len(pilot_df))
print("Minimum required null positions:", MIN_NULL_POSITIONS)
print("Minimum retinal overlap:", MIN_FOREGROUND_COVERAGE)

print("\nPER-IMAGE RESULTS")
print(
    pilot_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSUMMARY")
print(
    "Median retinal fraction:",
    round(pilot_df["retinal_fraction"].median(), 4)
)
print(
    "Median observed MA foreground coverage:",
    round(
        pilot_df["observed_ma_foreground_coverage"].median(),
        4
    )
)
print(
    "Median eligible null shifts:",
    pilot_df["eligible_null_shifts"].median()
)
print(
    "Images with >=100 eligible null shifts:",
    int(pilot_df["sufficient_null_positions"].sum()),
    "/", len(pilot_df)
)

print("\nSTEP 5.6B PILOT COMPLETED")


# %% [cell 126] Experiment 05 - Step 5.6C

# ============================================================
# Experiment 05 - Step 5.6C
# Retinal Foreground-Constrained Spatial Null
# ============================================================

from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

ROOT = DR_RESEARCH
OUT = ROOT / "experiment05_results"
CAM_DIR = OUT / "gradcam"

PAIRED_PATH = OUT / "MA_Attribution_Paired_V1.csv"
COHORT_PATH = OUT / "FGADR_Validation_MA_Cohort_V1.csv"
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"

EFF_PATH = CAM_DIR / "EfficientNet_MA_CAM_Maps_V1.npz"
RET_PATH = CAM_DIR / "RETFound_MA_CAM_Maps_V1.npz"

RESULT_PATH = OUT / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
REPORT_PATH = OUT / "MA_Spatial_Null_Retinal_Constrained_Report_V1.json"

assert not RESULT_PATH.exists()
assert not REPORT_PATH.exists()

SIZE = 224
MIN_COVERAGE = 0.95
MIN_NULL = 100

paired = pd.read_csv(PAIRED_PATH).sort_values("image_id")
cohort = pd.read_csv(COHORT_PATH)
index = pd.read_csv(INDEX_PATH)

primary = (
    cohort.loc[cohort["primary_ma_candidate"] == True]
    .merge(
        index[["image_id", "split", "local_path", "source_path"]],
        on="image_id",
        validate="one_to_one"
    )
    .set_index("image_id")
)

with np.load(EFF_PATH, allow_pickle=False) as data:
    eff_ids = data["image_ids"].astype(str).tolist()
    eff_maps = data["efficientnet_cam"].copy()

with np.load(RET_PATH, allow_pickle=False) as data:
    ret_ids = data["image_ids"].astype(str).tolist()
    ret_maps = data["retfound_cam"].copy()

assert len(primary) == len(paired) == 156
assert primary["split"].eq("val").all()
assert eff_ids == ret_ids == paired["image_id"].astype(str).tolist()
assert eff_maps.shape == ret_maps.shape == (156, SIZE, SIZE)

def load_retinal_image(row):
    path = Path(str(row["local_path"]))
    if not path.is_file():
        path = Path(str(row["source_path"]))
    assert path.is_file()

    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (SIZE, SIZE), Image.Resampling.BICUBIC
            )
        )
    return rgb

def retinal_foreground(rgb):
    intensity = rgb.max(axis=2)
    candidate = (intensity > 20).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)
    )
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    assert contours, "Retinal foreground not detected."

    foreground = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.drawContours(
        foreground,
        [max(contours, key=cv2.contourArea)],
        -1, 1, thickness=-1
    )
    foreground = cv2.erode(foreground, kernel)
    return foreground.astype(np.float64)

def weighted_ma_mask(row):
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

def fft_shift_values(reference, mask):
    """
    Return sums of reference values weighted by every
    circular translation of the MA mask.
    """
    return np.fft.ifft2(
        np.fft.fft2(reference.astype(np.float64))
        * np.conj(np.fft.fft2(mask))
    ).real

def eligible_shift_geometry(mask, foreground):
    ys, xs = np.nonzero(mask > 0)
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())

    dx = np.arange(-xmin, SIZE - xmax, dtype=int)
    dy = np.arange(-ymin, SIZE - ymax, dtype=int)

    shifts_y, shifts_x = np.meshgrid(
        dy % SIZE, dx % SIZE, indexing="ij"
    )

    coverage_map = (
        fft_shift_values(foreground, mask)
        / float(mask.sum())
    )

    coverage = coverage_map[shifts_y, shifts_x]

    permitted = coverage >= MIN_COVERAGE

    # Exclude the unshifted observed position.
    original = (shifts_y == 0) & (shifts_x == 0)
    permitted &= ~original

    observed_coverage = float(
        (foreground * mask).sum() / mask.sum()
    )

    coords_y = shifts_y[permitted]
    coords_x = shifts_x[permitted]

    return {
        "observed_coverage": observed_coverage,
        "foreground_fraction": float(foreground.mean()),
        "possible_shifts": int(len(dx) * len(dy)),
        "null_shifts": int(len(coords_y)),
        "coords_y": coords_y,
        "coords_x": coords_x,
        "eligible": (
            observed_coverage >= MIN_COVERAGE
            and len(coords_y) >= MIN_NULL
        )
    }

def model_spatial_metrics(cam, mask, geometry):
    if not geometry["eligible"]:
        return {
            "observed_enrichment": np.nan,
            "null_mean_enrichment": np.nan,
            "null_adjusted_enrichment": np.nan,
            "spatial_percentile": np.nan,
            "upper_tail_fraction": np.nan
        }

    weighted_scores = fft_shift_values(cam, mask)

    scale = float(mask.sum()) * float(cam.mean())
    assert scale > 1e-12

    enrichment = weighted_scores / scale
    observed = float(enrichment[0, 0])

    null = enrichment[
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

results = []

for i, image_id in enumerate(
    tqdm(eff_ids, desc="Retinal-constrained spatial null")
):
    row = primary.loc[image_id]
    rgb = load_retinal_image(row)
    mask = weighted_ma_mask(row)
    foreground = retinal_foreground(rgb)

    geometry = eligible_shift_geometry(mask, foreground)

    record = {
        "image_id": image_id,
        "true_grade": int(row["true_grade"]),
        "risk_group_id": str(
            row["risk_group_id"]
        ) if "risk_group_id" in row.index else "",
        "foreground_fraction": geometry["foreground_fraction"],
        "observed_coverage": geometry["observed_coverage"],
        "possible_shifts": geometry["possible_shifts"],
        "null_shifts": geometry["null_shifts"],
        "eligible": bool(geometry["eligible"]),
    }

    if geometry["observed_coverage"] < MIN_COVERAGE:
        record["exclusion_reason"] = "observed_coverage_below_95pct"
    elif geometry["null_shifts"] < MIN_NULL:
        record["exclusion_reason"] = "insufficient_null_shifts"
    else:
        record["exclusion_reason"] = ""

    for model_name, cam in [
        ("efficientnet", eff_maps[i]),
        ("retfound", ret_maps[i])
    ]:
        metrics = model_spatial_metrics(
            cam, mask, geometry
        )
        for key, value in metrics.items():
            record[f"{model_name}_{key}"] = value

    if geometry["eligible"]:
        assert np.isclose(
            record["efficientnet_observed_enrichment"],
            paired.iloc[i]["efficientnet_ma_enrichment"],
            rtol=1e-4, atol=1e-5
        )
        assert np.isclose(
            record["retfound_observed_enrichment"],
            paired.iloc[i]["retfound_ma_enrichment"],
            rtol=1e-4, atol=1e-5
        )

    results.append(record)

results = pd.DataFrame(results)
eligible = results.loc[results["eligible"]].copy()

assert len(results) == 156
assert results["image_id"].is_unique

eligible["delta_null_adjusted"] = (
    eligible["retfound_null_adjusted_enrichment"]
    - eligible["efficientnet_null_adjusted_enrichment"]
)
eligible["delta_spatial_percentile"] = (
    eligible["retfound_spatial_percentile"]
    - eligible["efficientnet_spatial_percentile"]
)

results = results.merge(
    eligible[
        ["image_id", "delta_null_adjusted",
         "delta_spatial_percentile"]
    ],
    on="image_id",
    how="left",
    validate="one_to_one"
)

print("\nSTEP 5.6C - RETINAL-CONSTRAINED SPATIAL NULL")
print("Total primary images:", len(results))
print("Eligible images:", len(eligible))
print("Excluded images:", len(results) - len(eligible))
print(
    "Median valid null shifts:",
    float(eligible["null_shifts"].median())
    if len(eligible) else None
)

print("\nEXCLUSION REASONS")
print(
    results.loc[
        ~results["eligible"], "exclusion_reason"
    ].value_counts().to_string()
)

if len(eligible):
    print("\nMODEL COMPARISON")
    for model_name in ["efficientnet", "retfound"]:
        print(f"\n{model_name.upper()}")
        print(
            "Median spatial percentile:",
            round(float(
                eligible[f"{model_name}_spatial_percentile"].median()
            ), 4)
        )
        print(
            "Mean null-adjusted enrichment:",
            round(float(
                eligible[
                    f"{model_name}_null_adjusted_enrichment"
                ].mean()
            ), 6)
        )
        print(
            "Images above 95th null percentile:",
            int((
                eligible[f"{model_name}_upper_tail_fraction"] <= 0.05
            ).sum())
        )

    print("\nPAIRED COMPARISON")
    print(
        "Mean null-adjusted difference:",
        round(float(
            eligible["delta_null_adjusted"].mean()
        ), 6)
    )
    print(
        "Median spatial percentile difference:",
        round(float(
            eligible["delta_spatial_percentile"].median()
        ), 4)
    )
    print(
        "RETFound higher percentile:",
        int((eligible["delta_spatial_percentile"] > 0).sum())
    )
    print(
        "EfficientNet higher percentile:",
        int((eligible["delta_spatial_percentile"] < 0).sum())
    )

results.to_csv(RESULT_PATH, index=False)

report = {
    "experiment": "Experiment05_Step5.6C",
    "method": "foreground_constrained_nonwrapping_rigid_translation",
    "total_primary_images": int(len(results)),
    "eligible_images": int(len(eligible)),
    "excluded_images": int(len(results) - len(eligible)),
    "minimum_retinal_coverage": MIN_COVERAGE,
    "minimum_null_shifts": MIN_NULL,
    "foreground_method": (
        "max_RGB_intensity_gt_20_largest_contour_"
        "morphological_closing_and_erosion_9x9"
    ),
    "spatial_null_scope": "per_image",
    "patient_independent_significance_test": False,
    "test_split_evaluated": False
}

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nOUTPUT FILES")
print(RESULT_PATH)
print(REPORT_PATH)
print("STEP 5.6C COMPLETED")


# %% [cell 127] Experiment 05 - Step 5.7

# ============================================================
# Experiment 05 - Step 5.7
# Source-Risk-Group Cluster Bootstrap
# and Cohort Selection Sensitivity Analysis
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
OUT = ROOT / "experiment05_results"

PAIRED_PATH = OUT / "MA_Attribution_Paired_V1.csv"
SPATIAL_PATH = OUT / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

RESULT_PATH = OUT / "MA_Cluster_Bootstrap_V1.json"
BOOTSTRAP_PATH = OUT / "MA_Cluster_Bootstrap_Draws_V1.csv"

assert not RESULT_PATH.exists()
assert not BOOTSTRAP_PATH.exists()

N_BOOTSTRAP = 2000
SEED = 20261020

paired = pd.read_csv(PAIRED_PATH)
spatial = pd.read_csv(SPATIAL_PATH)
index = pd.read_csv(INDEX_PATH)

assert len(paired) == 156
assert len(spatial) == 156
assert paired["image_id"].is_unique
assert spatial["image_id"].is_unique

risk_groups = index.loc[
    index["split"] == "val",
    ["image_id", "risk_group_id"]
].copy()

assert risk_groups["image_id"].is_unique
assert risk_groups["risk_group_id"].notna().all()

data = (
    paired.merge(
        spatial[
            [
                "image_id",
                "eligible",
                "exclusion_reason",
                "efficientnet_null_adjusted_enrichment",
                "retfound_null_adjusted_enrichment",
                "efficientnet_spatial_percentile",
                "retfound_spatial_percentile",
            ]
        ],
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        risk_groups,
        on="image_id",
        validate="one_to_one"
    )
)

assert len(data) == 156
assert data["risk_group_id"].notna().all()

data["delta_raw"] = (
    data["retfound_ma_enrichment"]
    - data["efficientnet_ma_enrichment"]
)

data["delta_adjusted"] = (
    data["retfound_null_adjusted_enrichment"]
    - data["efficientnet_null_adjusted_enrichment"]
)

data["delta_percentile"] = (
    data["retfound_spatial_percentile"]
    - data["efficientnet_spatial_percentile"]
)

eligible = data.loc[data["eligible"] == True].copy()

assert len(eligible) == 137
assert eligible["delta_adjusted"].notna().all()
assert eligible["delta_percentile"].notna().all()

# ============================================================
# 1. Cluster bootstrap with image-weighted means
# ============================================================

def cluster_bootstrap(frame, columns, n_boot=2000, seed=0):
    """
    Resample source-risk groups, keeping all images
    in every selected group.

    The estimator is the image-weighted sample mean,
    not the unweighted mean of cluster means.
    """

    grouped = frame.groupby(
        "risk_group_id",
        sort=True
    )

    group_sizes = grouped.size().to_numpy(dtype=np.float64)
    group_keys = list(grouped.groups.keys())

    group_sums = np.stack([
        grouped[col].sum().reindex(group_keys).to_numpy(
            dtype=np.float64
        )
        for col in columns
    ], axis=1)

    n_groups = len(group_keys)
    assert n_groups >= 2

    rng = np.random.default_rng(seed)

    draws = np.empty(
        (n_boot, len(columns)),
        dtype=np.float64
    )

    for b in range(n_boot):
        sampled = rng.integers(
            0, n_groups, size=n_groups
        )

        total_size = group_sizes[sampled].sum()
        total_values = group_sums[sampled].sum(axis=0)

        draws[b] = total_values / total_size

    summary = {}

    for j, col in enumerate(columns):
        observed = float(frame[col].mean())
        lower, upper = np.quantile(
            draws[:, j], [0.025, 0.975]
        )

        summary[col] = {
            "observed_mean": observed,
            "ci_95_lower": float(lower),
            "ci_95_upper": float(upper),
            "ci_contains_zero": bool(
                lower <= 0 <= upper
            ),
        }

    return summary, draws, n_groups

# ============================================================
# 2. Main 137-image spatial analysis
# ============================================================

main_columns = [
    "delta_adjusted",
    "delta_percentile",
    "delta_raw",
]

main_summary, main_draws, n_main_groups = cluster_bootstrap(
    eligible,
    main_columns,
    n_boot=N_BOOTSTRAP,
    seed=SEED
)

# ============================================================
# 3. Full 156-image raw-enrichment sensitivity
# ============================================================

full_summary, full_draws, n_full_groups = cluster_bootstrap(
    data,
    ["delta_raw"],
    n_boot=N_BOOTSTRAP,
    seed=SEED + 1
)

# ============================================================
# 4. Report same-cohort descriptive comparisons
# ============================================================

def summarize_raw(frame):
    return {
        "n_images": int(len(frame)),
        "efficientnet_mean": float(
            frame["efficientnet_ma_enrichment"].mean()
        ),
        "retfound_mean": float(
            frame["retfound_ma_enrichment"].mean()
        ),
        "paired_mean_difference": float(
            frame["delta_raw"].mean()
        ),
        "paired_median_difference": float(
            frame["delta_raw"].median()
        ),
    }

raw_all = summarize_raw(data)
raw_eligible = summarize_raw(eligible)

print("STEP 5.7 - CLUSTER BOOTSTRAP")
print("Primary cohort:", len(data))
print("Spatial-null eligible:", len(eligible))
print("Excluded:", len(data) - len(eligible))
print("Eligible risk groups:", n_main_groups)
print("Full-cohort risk groups:", n_full_groups)
print("Bootstrap replicates:", N_BOOTSTRAP)

print("\nRAW ENRICHMENT - SAME-COHORT CHECK")
print("All 156:", raw_all)
print("Eligible 137:", raw_eligible)

print("\nMAIN SPATIAL ANALYSIS - ELIGIBLE 137")

for column in main_columns:
    entry = main_summary[column]

    print(f"\n{column}")
    print("Mean delta:", round(entry["observed_mean"], 6))
    print(
        "95% cluster bootstrap CI:",
        (
            round(entry["ci_95_lower"], 6),
            round(entry["ci_95_upper"], 6)
        )
    )
    print("CI includes zero:", entry["ci_contains_zero"])

print("\nFULL COHORT - RAW ENRICHMENT SENSITIVITY")

entry = full_summary["delta_raw"]
print("Mean delta:", round(entry["observed_mean"], 6))
print(
    "95% cluster bootstrap CI:",
    (
        round(entry["ci_95_lower"], 6),
        round(entry["ci_95_upper"], 6)
    )
)
print("CI includes zero:", entry["ci_contains_zero"])

# ============================================================
# 5. Save reproducible bootstrap draws
# ============================================================

draws_frame = pd.DataFrame({
    "replicate": np.arange(N_BOOTSTRAP),
    "eligible_delta_adjusted": main_draws[:, 0],
    "eligible_delta_percentile": main_draws[:, 1],
    "eligible_delta_raw": main_draws[:, 2],
    "full_delta_raw": full_draws[:, 0],
})

draws_frame.to_csv(BOOTSTRAP_PATH, index=False)

report = {
    "experiment": "Experiment05_Step5.7",
    "split": "val",
    "bootstrap_method": (
        "source_risk_group_cluster_resampling_"
        "image_weighted_mean"
    ),
    "bootstrap_replicates": N_BOOTSTRAP,
    "random_seed": SEED,
    "primary_images": int(len(data)),
    "eligible_images": int(len(eligible)),
    "excluded_images": int(len(data) - len(eligible)),
    "eligible_risk_groups": n_main_groups,
    "full_risk_groups": n_full_groups,
    "main_analysis": main_summary,
    "full_raw_sensitivity": full_summary,
    "raw_all_images": raw_all,
    "raw_eligible_images": raw_eligible,
    "risk_groups_are_confirmed_patient_ids": False,
    "causal_pretraining_effect_estimated": False,
    "test_split_evaluated": False,
}

with open(REPORT_PATH := RESULT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nOUTPUT FILES")
print(RESULT_PATH)
print(BOOTSTRAP_PATH)
print("Test evaluated: False")
print("STEP 5.7 COMPLETED")


# %% [cell 128] Experiment 05 - Step 5.8

# ============================================================
# Experiment 05 - Step 5.8
# Common 7x7 CAM Resolution Sensitivity Analysis
# ============================================================

from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = DR_RESEARCH
OUT = ROOT / "experiment05_results"
CAM_DIR = OUT / "gradcam"

RESULT_PATH = OUT / "MA_Common_Resolution_Sensitivity_V1.csv"
REPORT_PATH = OUT / "MA_Common_Resolution_Sensitivity_Report_V1.json"

assert not RESULT_PATH.exists()
assert not REPORT_PATH.exists()

# Reuse the tested geometric protocol from Step 5.6C.
for name in [
    "load_retinal_image",
    "retinal_foreground",
    "weighted_ma_mask",
    "eligible_shift_geometry",
    "model_spatial_metrics",
]:
    assert name in globals() and callable(globals()[name]), (
        f"Required spatial-null function is unavailable: {name}"
    )

cohort = pd.read_csv(
    OUT / "FGADR_Validation_MA_Cohort_V1.csv"
)
index = pd.read_csv(
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)
reference = pd.read_csv(
    OUT / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

primary = (
    cohort.loc[cohort["primary_ma_candidate"] == True]
    .merge(
        index[
            ["image_id", "split", "risk_group_id",
             "local_path", "source_path"]
        ],
        on="image_id",
        validate="one_to_one"
    )
    .set_index("image_id")
)

with np.load(
    CAM_DIR / "EfficientNet_MA_CAM_Maps_V1.npz",
    allow_pickle=False
) as data:
    eff_ids = data["image_ids"].astype(str).tolist()
    eff_maps = data["efficientnet_cam"].copy()

with np.load(
    CAM_DIR / "RETFound_MA_CAM_Maps_V1.npz",
    allow_pickle=False
) as data:
    ret_ids = data["image_ids"].astype(str).tolist()
    ret_maps = data["retfound_cam"].copy()

assert eff_ids == ret_ids
assert len(eff_ids) == 156

reference = reference.set_index("image_id")

def common_resolution_cam(cam):
    # Project the saved 224x224 attribution to a shared
    # 7x7 representation using area-based downsampling.
    coarse = cv2.resize(
        cam.astype(np.float32),
        (7, 7),
        interpolation=cv2.INTER_AREA
    )

    reconstructed = cv2.resize(
        coarse,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    assert reconstructed.shape == (224, 224)
    assert np.isfinite(reconstructed).all()
    assert reconstructed.mean() > 1e-12

    return reconstructed.astype(np.float32)

records = []

for i, image_id in enumerate(
    tqdm(eff_ids, desc="Common-resolution spatial null")
):
    row = primary.loc[image_id]

    rgb = load_retinal_image(row)
    mask = weighted_ma_mask(row)
    foreground = retinal_foreground(rgb)

    geometry = eligible_shift_geometry(mask, foreground)

    # Ensure the eligibility cohort is identical to Step 5.6C.
    assert bool(geometry["eligible"]) == bool(
        reference.loc[image_id, "eligible"]
    ), f"Eligibility changed: {image_id}"

    result = {
        "image_id": image_id,
        "risk_group_id": str(row["risk_group_id"]),
        "eligible": bool(geometry["eligible"]),
        "null_shifts": int(geometry["null_shifts"]),
    }

    for model_name, maps in [
        ("efficientnet", eff_maps),
        ("retfound", ret_maps)
    ]:
        matched_cam = common_resolution_cam(maps[i])

        metrics = model_spatial_metrics(
            matched_cam,
            mask,
            geometry
        )

        for key, value in metrics.items():
            result[f"{model_name}_{key}"] = value

    records.append(result)

data = pd.DataFrame(records)
eligible = data.loc[data["eligible"]].copy()

assert len(data) == 156
assert len(eligible) == 137

eligible["delta_adjusted"] = (
    eligible["retfound_null_adjusted_enrichment"]
    - eligible["efficientnet_null_adjusted_enrichment"]
)

eligible["delta_percentile"] = (
    eligible["retfound_spatial_percentile"]
    - eligible["efficientnet_spatial_percentile"]
)

# ============================================================
# Source-risk-group cluster bootstrap
# ============================================================

grouped = eligible.groupby("risk_group_id", sort=True)

group_sizes = grouped.size().to_numpy(dtype=float)
group_sums = grouped[
    ["delta_adjusted", "delta_percentile"]
].sum().to_numpy(dtype=float)

n_groups = len(group_sizes)
assert n_groups >= 2

rng = np.random.default_rng(20261020)
n_boot = 2000
draws = np.empty((n_boot, 2), dtype=float)

for b in range(n_boot):
    selected = rng.integers(
        0, n_groups, size=n_groups
    )
    draws[b] = (
        group_sums[selected].sum(axis=0)
        / group_sizes[selected].sum()
    )

summary = {}

for j, metric in enumerate([
    "delta_adjusted",
    "delta_percentile"
]):
    ci = np.quantile(draws[:, j], [0.025, 0.975])

    summary[metric] = {
        "mean": float(eligible[metric].mean()),
        "ci_95": [float(ci[0]), float(ci[1])],
        "ci_contains_zero": bool(ci[0] <= 0 <= ci[1])
    }

data.to_csv(RESULT_PATH, index=False)

report = {
    "experiment": "Experiment05_Step5.8",
    "analysis": "common_7x7_cam_resolution",
    "source_maps": "saved_224x224_gradcam",
    "downsampling": "opencv_inter_area",
    "upsampling": "opencv_inter_linear",
    "images": 156,
    "eligible_images": 137,
    "risk_groups": n_groups,
    "bootstrap_replicates": n_boot,
    "bootstrap_seed": 20261020,
    "summary": summary,
    "native_layer_resolution_equivalence_claimed": False,
    "test_split_evaluated": False
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nSTEP 5.8 - COMMON RESOLUTION SENSITIVITY")
print("Common CAM grid: 7x7")
print("Eligible images:", len(eligible))
print("Source-risk groups:", n_groups)

for metric, values in summary.items():
    print(f"\n{metric}")
    print("Mean difference:", round(values["mean"], 6))
    print(
        "95% cluster bootstrap CI:",
        tuple(round(x, 6) for x in values["ci_95"])
    )
    print("CI includes zero:", values["ci_contains_zero"])

print("\nRETFOUND HIGHER SPATIAL PERCENTILE")
print(int((eligible["delta_percentile"] > 0).sum()))

print("\nOUTPUT FILES")
print(RESULT_PATH)
print(REPORT_PATH)
print("STEP 5.8 COMPLETED")


# %% [cell 129]

pass  # Google Drive mount removed


# %% [cell 130]
from pathlib import Path

ROOT = DR_RESEARCH

print("Google Drive connected:", ROOT.exists())
print("Project directory:", ROOT)

assert ROOT.exists(), "Google Drive is not mounted correctly."

print("DR Research project is ready.")


# %% [cell 131] Experiment 05 - Step 5.8

# ============================================================
# Experiment 05 - Step 5.8
# Standalone Common-Resolution Sensitivity Analysis
# ============================================================

from pathlib import Path
import json
import hashlib

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

ROOT = DR_RESEARCH
OUT = ROOT / "experiment05_results"
CAM_DIR = OUT / "gradcam"

COHORT_PATH = OUT / "FGADR_Validation_MA_Cohort_V1.csv"
INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
REFERENCE_PATH = OUT / "MA_Spatial_Null_Retinal_Constrained_V1.csv"

EFF_PATH = CAM_DIR / "EfficientNet_MA_CAM_Maps_V1.npz"
RET_PATH = CAM_DIR / "RETFound_MA_CAM_Maps_V1.npz"

RESULT_PATH = OUT / "MA_Common_Resolution_Sensitivity_V1.csv"
REPORT_PATH = OUT / "MA_Common_Resolution_Sensitivity_Report_V1.json"

EXPECTED_COHORT_SHA256 = (
    "542c4f21780ba4060945e893a22c4e133b0f44a006fecb94ea5da4b76b93b286"
)

SIZE = 224
MIN_COVERAGE = 0.95
MIN_NULL = 100
N_BOOTSTRAP = 2000
SEED = 20261020

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()

for path in [
    COHORT_PATH, INDEX_PATH, REFERENCE_PATH,
    EFF_PATH, RET_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

assert sha256_file(COHORT_PATH) == EXPECTED_COHORT_SHA256
assert not RESULT_PATH.exists(), "Result already exists."
assert not REPORT_PATH.exists(), "Report already exists."

# ============================================================
# 1. Restore frozen analysis cohort
# ============================================================

cohort = pd.read_csv(COHORT_PATH)
index = pd.read_csv(INDEX_PATH)
reference = pd.read_csv(REFERENCE_PATH).set_index("image_id")

primary = (
    cohort.loc[cohort["primary_ma_candidate"] == True]
    .merge(
        index[
            ["image_id", "split", "risk_group_id",
             "local_path", "source_path"]
        ],
        on="image_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_index")
    )
    .set_index("image_id")
)

# Prefer the authoritative frozen training index.
for col in [
    "split", "risk_group_id", "local_path", "source_path"
]:
    if col + "_index" in primary.columns:
        primary[col] = primary[col + "_index"]

assert len(primary) == 156
assert primary["split"].eq("val").all()
assert primary["risk_group_id"].notna().all()

with np.load(EFF_PATH, allow_pickle=False) as data:
    eff_ids = data["image_ids"].astype(str).tolist()
    eff_maps = data["efficientnet_cam"].copy()

with np.load(RET_PATH, allow_pickle=False) as data:
    ret_ids = data["image_ids"].astype(str).tolist()
    ret_maps = data["retfound_cam"].copy()

assert eff_ids == ret_ids
assert set(eff_ids) == set(primary.index)
assert eff_maps.shape == ret_maps.shape == (156, 224, 224)
assert np.isfinite(eff_maps).all()
assert np.isfinite(ret_maps).all()

# ============================================================
# 2. Reconstruct the Step 5.6C retinal geometry
# ============================================================

def load_retinal_image(row):
    path = Path(str(row["local_path"]))
    if not path.is_file():
        path = Path(str(row["source_path"]))
    assert path.is_file()

    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (SIZE, SIZE),
                Image.Resampling.BICUBIC
            )
        )
    return rgb

def retinal_foreground(rgb):
    intensity = rgb.max(axis=2)
    candidate = (intensity > 20).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)
    )

    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        candidate,
        cv2.RETR_EXTERNAL,
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

def weighted_ma_mask(row):
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
        binary,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    ).astype(np.float64)

    assert weighted.sum() > 0
    return weighted

def fft_shift_values(reference_map, mask):
    return np.fft.ifft2(
        np.fft.fft2(reference_map.astype(np.float64))
        * np.conj(np.fft.fft2(mask))
    ).real

def eligible_shift_geometry(mask, foreground):
    ys, xs = np.nonzero(mask > 0)

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())

    dx = np.arange(-xmin, SIZE - xmax, dtype=int)
    dy = np.arange(-ymin, SIZE - ymax, dtype=int)

    shifts_y, shifts_x = np.meshgrid(
        dy % SIZE, dx % SIZE, indexing="ij"
    )

    coverage_map = (
        fft_shift_values(foreground, mask)
        / float(mask.sum())
    )

    coverage = coverage_map[shifts_y, shifts_x]
    permitted = coverage >= MIN_COVERAGE

    # Exclude the original, unshifted mask.
    permitted &= ~(
        (shifts_y == 0) & (shifts_x == 0)
    )

    observed_coverage = float(
        (foreground * mask).sum() / mask.sum()
    )

    coords_y = shifts_y[permitted]
    coords_x = shifts_x[permitted]

    return {
        "eligible": bool(
            observed_coverage >= MIN_COVERAGE
            and len(coords_y) >= MIN_NULL
        ),
        "observed_coverage": observed_coverage,
        "null_shifts": len(coords_y),
        "coords_y": coords_y,
        "coords_x": coords_x
    }

# ============================================================
# 3. Common 7x7 CAM representation
# ============================================================

def match_resolution(cam):
    coarse = cv2.resize(
        cam.astype(np.float32),
        (7, 7),
        interpolation=cv2.INTER_AREA
    )

    restored = cv2.resize(
        coarse,
        (SIZE, SIZE),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float64)

    assert np.isfinite(restored).all()
    assert restored.mean() > 1e-12
    return restored

def spatial_metrics(cam, mask, geometry):
    if not geometry["eligible"]:
        return {
            "observed": np.nan,
            "null_mean": np.nan,
            "null_adjusted": np.nan,
            "percentile": np.nan
        }

    scores = fft_shift_values(cam, mask)
    scores /= float(mask.sum()) * float(cam.mean())

    observed = float(scores[0, 0])

    null = scores[
        geometry["coords_y"],
        geometry["coords_x"]
    ]

    assert len(null) >= MIN_NULL

    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_adjusted": float(observed - null.mean()),
        "percentile": float(
            100.0 * np.mean(null < observed)
        )
    }

# ============================================================
# 4. Run matched-resolution spatial analysis
# ============================================================

records = []

for i, image_id in enumerate(
    tqdm(eff_ids, desc="Common-resolution sensitivity")
):
    row = primary.loc[image_id]
    mask = weighted_ma_mask(row)
    rgb = load_retinal_image(row)
    foreground = retinal_foreground(rgb)

    geometry = eligible_shift_geometry(mask, foreground)

    # Eligibility must reproduce Step 5.6C exactly.
    assert geometry["eligible"] == bool(
        reference.loc[image_id, "eligible"]
    ), f"Eligibility mismatch: {image_id}"

    assert geometry["null_shifts"] == int(
        reference.loc[image_id, "null_shifts"]
    ), f"Null-position mismatch: {image_id}"

    record = {
        "image_id": image_id,
        "risk_group_id": str(row["risk_group_id"]),
        "eligible": geometry["eligible"],
        "null_shifts": geometry["null_shifts"]
    }

    for name, cam in [
        ("efficientnet", eff_maps[i]),
        ("retfound", ret_maps[i])
    ]:
        metrics = spatial_metrics(
            match_resolution(cam),
            mask,
            geometry
        )

        for key, value in metrics.items():
            record[f"{name}_{key}"] = value

    records.append(record)

data = pd.DataFrame(records)
eligible = data.loc[data["eligible"]].copy()

assert len(data) == 156
assert len(eligible) == 137

eligible["delta_adjusted"] = (
    eligible["retfound_null_adjusted"]
    - eligible["efficientnet_null_adjusted"]
)

eligible["delta_percentile"] = (
    eligible["retfound_percentile"]
    - eligible["efficientnet_percentile"]
)

# ============================================================
# 5. Source-risk-group cluster bootstrap
# ============================================================

grouped = eligible.groupby("risk_group_id", sort=True)

group_sizes = grouped.size().to_numpy(dtype=float)

group_sums = grouped[
    ["delta_adjusted", "delta_percentile"]
].sum().to_numpy(dtype=float)

n_groups = len(group_sizes)
assert n_groups >= 2

rng = np.random.default_rng(SEED)
draws = np.empty((N_BOOTSTRAP, 2), dtype=float)

for b in range(N_BOOTSTRAP):
    selected = rng.integers(
        0, n_groups, size=n_groups
    )

    draws[b] = (
        group_sums[selected].sum(axis=0)
        / group_sizes[selected].sum()
    )

summary = {}

for j, metric in enumerate([
    "delta_adjusted", "delta_percentile"
]):
    low, high = np.quantile(
        draws[:, j], [0.025, 0.975]
    )

    summary[metric] = {
        "mean": float(eligible[metric].mean()),
        "ci_95": [float(low), float(high)],
        "ci_contains_zero": bool(low <= 0 <= high)
    }

# ============================================================
# 6. Save results and report
# ============================================================

data.to_csv(RESULT_PATH, index=False)

report = {
    "experiment": "Experiment05_Step5.8",
    "analysis": "common_7x7_cam_resolution",
    "source": "saved_224x224_cam_maps",
    "downsample": "INTER_AREA",
    "upsample": "INTER_LINEAR",
    "primary_images": 156,
    "eligible_images": 137,
    "risk_groups": n_groups,
    "bootstrap_replicates": N_BOOTSTRAP,
    "bootstrap_seed": SEED,
    "summary": summary,
    "note": (
        "Post-hoc resolution matching of stored CAMs; "
        "not identical to recomputing CAM at native 7x7."
    ),
    "test_split_evaluated": False
}

with open(REPORT_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nSTEP 5.8 - COMMON RESOLUTION SENSITIVITY")
print("Common CAM grid: 7x7")
print("Primary images:", len(data))
print("Eligible images:", len(eligible))
print("Source-risk groups:", n_groups)

for metric, values in summary.items():
    print(f"\n{metric}")
    print("Mean difference:", round(values["mean"], 6))
    print(
        "95% cluster bootstrap CI:",
        tuple(round(x, 6) for x in values["ci_95"])
    )
    print("CI includes zero:", values["ci_contains_zero"])

print("\nRETFOUND HIGHER SPATIAL PERCENTILE")
print(int((eligible["delta_percentile"] > 0).sum()))

print("\nOUTPUT FILES")
print(RESULT_PATH)
print(REPORT_PATH)
print("Test evaluated: False")
print("STEP 5.8 COMPLETED")


# %% [cell 132] Experiment 05 - Step 5.9A

# ============================================================
# Experiment 05 - Step 5.9A
# Publication Figure: Paired Attribution Differences
# ============================================================

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH
OUT = ROOT / "experiment05_results"
FIG_DIR = OUT / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_PATH = OUT / "MA_Cluster_Bootstrap_V1.json"
RESOLUTION_PATH = OUT / "MA_Common_Resolution_Sensitivity_Report_V1.json"

PNG_PATH = FIG_DIR / "Figure_MA_Attribution_Spatial_Control_V1.png"
PDF_PATH = FIG_DIR / "Figure_MA_Attribution_Spatial_Control_V1.pdf"
TABLE_PATH = FIG_DIR / "Table_MA_Attribution_Results_V1.csv"

for path in [BOOTSTRAP_PATH, RESOLUTION_PATH]:
    assert path.is_file(), f"Missing: {path}"

for path in [PNG_PATH, PDF_PATH, TABLE_PATH]:
    assert not path.exists(), f"Output already exists: {path}"

with open(BOOTSTRAP_PATH, encoding="utf-8") as f:
    bootstrap = json.load(f)

with open(RESOLUTION_PATH, encoding="utf-8") as f:
    resolution = json.load(f)

def extract_bootstrap(entry):
    return (
        float(entry["observed_mean"]),
        float(entry["ci_95_lower"]),
        float(entry["ci_95_upper"])
    )

def extract_resolution(entry):
    return (
        float(entry["mean"]),
        float(entry["ci_95"][0]),
        float(entry["ci_95"][1])
    )

raw_all = extract_bootstrap(
    bootstrap["full_raw_sensitivity"]["delta_raw"]
)

raw_eligible = extract_bootstrap(
    bootstrap["main_analysis"]["delta_raw"]
)

adjusted = extract_bootstrap(
    bootstrap["main_analysis"]["delta_adjusted"]
)

matched = extract_resolution(
    resolution["summary"]["delta_adjusted"]
)

labels = [
    "Unadjusted (n=156)",
    "Unadjusted, eligible cohort (n=137)",
    "Retinal spatial control (n=137)",
    "Spatial control, common 7x7 (n=137)"
]

values = np.array([
    raw_all, raw_eligible, adjusted, matched
], dtype=float)

summary = pd.DataFrame({
    "analysis": labels,
    "mean_difference": values[:, 0],
    "ci_lower": values[:, 1],
    "ci_upper": values[:, 2],
    "difference_definition": "RETFound minus EfficientNet",
    "bootstrap_method": "Source-risk-group cluster bootstrap",
    "bootstrap_replicates": 2000
})

summary.to_csv(TABLE_PATH, index=False)

fig, ax = plt.subplots(figsize=(10, 5.5))
y = np.arange(len(labels))

errors = np.vstack([
    values[:, 0] - values[:, 1],
    values[:, 2] - values[:, 0]
])

ax.errorbar(
    values[:, 0],
    y,
    xerr=errors,
    fmt="o",
    markersize=7,
    capsize=5,
    linewidth=1.8,
    color="#254F86",
    ecolor="#6382A5"
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.2,
    color="#777777"
)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()

ax.set_xlabel(
    "Mean paired attribution difference (RETFound - EfficientNet)",
    fontsize=11
)

ax.set_title(
    "Microaneurysm Attribution: Effect of Spatial Controls",
    fontsize=13,
    pad=16
)

ax.grid(axis="x", alpha=0.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(
    0.5, 0.02,
    "Points indicate mean differences; bars indicate 95% "
    "source-risk-group cluster bootstrap intervals.",
    ha="center",
    fontsize=9
)

fig.tight_layout(rect=[0, 0.07, 1, 1])

fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
fig.savefig(PDF_PATH, bbox_inches="tight")

plt.show()
plt.close(fig)

print("STEP 5.9A - PUBLICATION FIGURE")
print(summary.to_string(index=False))
print("\nFiles saved:")
print(PNG_PATH)
print(PDF_PATH)
print(TABLE_PATH)
print("STEP 5.9A COMPLETED")


# %% [cell 133] Experiment 05 - Step 5.9B

# ============================================================
# Experiment 05 - Step 5.9B
# Publication Figure V2: Refined Labels and Caption
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = DR_RESEARCH
FIG_DIR = ROOT / "experiment05_results" / "paper_figures"

TABLE_PATH = FIG_DIR / "Table_MA_Attribution_Results_V1.csv"
PNG_PATH = FIG_DIR / "Figure_MA_Attribution_Spatial_Control_V2.png"
PDF_PATH = FIG_DIR / "Figure_MA_Attribution_Spatial_Control_V2.pdf"

assert TABLE_PATH.is_file()
assert not PNG_PATH.exists()
assert not PDF_PATH.exists()

df = pd.read_csv(TABLE_PATH)

assert len(df) == 4

values = df[
    ["mean_difference", "ci_lower", "ci_upper"]
].to_numpy(dtype=float)

labels = [
    "Unadjusted, full cohort (n=156)",
    "Unadjusted, eligible subset (n=137)",
    "Retinal spatial-null adjusted (n=137)",
    "Spatial-null adjusted, common 7x7 (n=137)"
]

fig, ax = plt.subplots(figsize=(10.5, 5.4))
y = np.arange(4)

errors = np.vstack([
    values[:, 0] - values[:, 1],
    values[:, 2] - values[:, 0]
])

ax.errorbar(
    values[:, 0],
    y,
    xerr=errors,
    fmt="o",
    color="#355D85",
    ecolor="#7893AE",
    markersize=7,
    linewidth=1.6,
    capsize=4
)

ax.axvline(0, linestyle="--", color="#777777", linewidth=1.1)

ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(-0.65, 0.25)

ax.set_xlabel(
    "Mean paired difference in MA attribution enrichment\n"
    "(RETFound minus EfficientNet-B0)"
)

ax.set_title(
    "Microaneurysm Attribution Under Alternative\n"
    "Spatial Reference Conditions",
    fontsize=12,
    pad=15
)

ax.grid(axis="x", alpha=0.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.subplots_adjust(left=0.38, right=0.96, top=0.82, bottom=0.24)

fig.text(
    0.5, 0.055,
    "Rows 1-2: unadjusted enrichment; rows 3-4: retinal spatial-null-adjusted enrichment.\n"
    "Points: paired means; error bars: 95% source-risk-group cluster bootstrap CIs.",
    ha="center",
    va="center",
    fontsize=8.5
)

fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
fig.savefig(PDF_PATH, bbox_inches="tight")

plt.show()
plt.close(fig)

print("STEP 5.9B COMPLETED")
print("PNG:", PNG_PATH)
print("PDF:", PDF_PATH)


# %% [cell 134] Experiment 05 - Step 5.10A

# ============================================================
# Experiment 05 - Step 5.10A
# Publication Results Tables
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    recall_score,
    confusion_matrix
)

ROOT = DR_RESEARCH
OUT = ROOT / "experiment05_results"
VAL_PATH = (
    ROOT / "experiment04_results"
    / "FGADR_Validation_Paired_Predictions_V1.csv"
)
BOOT_PATH = OUT / "MA_Cluster_Bootstrap_V1.json"
RES_PATH = OUT / "MA_Common_Resolution_Sensitivity_Report_V1.json"

TABLE_DIR = OUT / "paper_tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TABLE1 = TABLE_DIR / "Table_1_Validation_Classification_V1.csv"
TABLE2 = TABLE_DIR / "Table_2_MA_Attribution_V1.csv"

for path in [VAL_PATH, BOOT_PATH, RES_PATH]:
    assert path.is_file(), f"Missing input: {path}"

assert not TABLE1.exists()
assert not TABLE2.exists()

validation = pd.read_csv(VAL_PATH)

assert len(validation) == 276
assert validation["image_id"].is_unique

y_true = validation["true_grade"].to_numpy(dtype=int)

assert set(np.unique(y_true)) == {0, 1, 2, 3, 4}

classification_rows = []

for name, column in [
    ("EfficientNet-B0", "efficientnet_pred"),
    ("RETFound ViT-Large", "retfound_pred")
]:
    y_pred = validation[column].to_numpy(dtype=int)

    assert np.isin(y_pred, np.arange(5)).all()

    actual_referable = y_true >= 2
    predicted_referable = y_pred >= 2

    tn, fp, fn, tp = confusion_matrix(
        actual_referable,
        predicted_referable,
        labels=[False, True]
    ).ravel()

    classification_rows.append({
        "Model": name,
        "Validation images": len(validation),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(
            y_true, y_pred, average="macro",
            labels=[0, 1, 2, 3, 4]
        ),
        "Quadratic weighted kappa": cohen_kappa_score(
            y_true, y_pred, weights="quadratic"
        ),
        "Referable DR sensitivity": tp / (tp + fn),
        "Referable DR specificity": tn / (tn + fp)
    })

table1 = pd.DataFrame(classification_rows)

assert np.isclose(
    table1.loc[0, "Referable DR sensitivity"],
    0.895197,
    atol=1e-5
), "Referable DR definition requires verification."

assert np.isclose(
    table1.loc[0, "Macro-F1"],
    0.561555,
    atol=1e-5
), "Saved validation metric mismatch."

with open(BOOT_PATH, encoding="utf-8") as f:
    bootstrap = json.load(f)

with open(RES_PATH, encoding="utf-8") as f:
    resolution = json.load(f)

def bootstrap_row(label, n, entry):
    return {
        "Analysis": label,
        "Images": n,
        "Mean paired difference": entry["observed_mean"],
        "95% CI lower": entry["ci_95_lower"],
        "95% CI upper": entry["ci_95_upper"]
    }

table2 = pd.DataFrame([
    bootstrap_row(
        "Unadjusted, full cohort",
        156,
        bootstrap["full_raw_sensitivity"]["delta_raw"]
    ),
    bootstrap_row(
        "Unadjusted, spatial-null eligible subset",
        137,
        bootstrap["main_analysis"]["delta_raw"]
    ),
    bootstrap_row(
        "Retinal spatial-null adjusted",
        137,
        bootstrap["main_analysis"]["delta_adjusted"]
    ),
    {
        "Analysis": "Retinal spatial-null adjusted, common 7x7",
        "Images": 137,
        "Mean paired difference": resolution["summary"]["delta_adjusted"]["mean"],
        "95% CI lower": resolution["summary"]["delta_adjusted"]["ci_95"][0],
        "95% CI upper": resolution["summary"]["delta_adjusted"]["ci_95"][1]
    }
])

assert len(table1) == 2
assert len(table2) == 4
assert np.isfinite(
    table1.select_dtypes(include="number").to_numpy()
).all()
assert np.isfinite(
    table2.select_dtypes(include="number").to_numpy()
).all()

table1.to_csv(TABLE1, index=False)
table2.to_csv(TABLE2, index=False)

print("STEP 5.10A - PUBLICATION RESULTS TABLES")

print("\nTABLE 1 - VALIDATION CLASSIFICATION")
print(table1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nTABLE 2 - MA ATTRIBUTION")
print(table2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nOUTPUT FILES")
print(TABLE1)
print(TABLE2)
print("Test split evaluated: False")
print("STEP 5.10A COMPLETED")
