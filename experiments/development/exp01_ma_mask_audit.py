# Experiment 01 - FGADR microaneurysm mask audit (phases 1-3) and image inventory
#
# Source: FGADR_MA_Mask_Audit.ipynb (master notebook, 392 cells), cells 0-25, run step by step in Google Colab (L4 GPU).
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

# %% [cell 0]
from pathlib import Path, PurePosixPath
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import platform
import random
import shutil
import stat
import time
import zipfile

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
cv2.setNumThreads(1)

EXPERIMENT_NAME = "fgadr_ma_mask_audit"
CODE_VERSION = "1.0.0"

MASK_ZIP = (LOCAL_DISK / "Microaneurysms_Masks.zip")
GRADE_CSV = (LOCAL_DISK / "DR_Seg_Grading_Label.csv")
OUTPUT_ROOT = (LOCAL_DISK / "fgadr_audit_results")

THRESHOLDS = [0, 32, 64, 127, 192, 254]
TARGET_SIZES = [224, 384, 512]
CONNECTIVITY = 8
EXPECTED_MASK_COUNT = 1842

MAX_ARCHIVE_MEMBERS = 10000
MAX_UNCOMPRESSED_BYTES = 5 * 1024**3

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
RUN_DIR = OUTPUT_ROOT / RUN_ID
REPORT_DIR = RUN_DIR / "reports"
FIGURE_DIR = RUN_DIR / "figures"
EXTRACT_DIR = RUN_DIR / "extracted_input"

for directory in [REPORT_DIR, FIGURE_DIR, EXTRACT_DIR]:
    directory.mkdir(parents=True, exist_ok=False)

CONFIG = {
    "experiment_name": EXPERIMENT_NAME,
    "code_version": CODE_VERSION,
    "run_id": RUN_ID,
    "seed": SEED,
    "threshold_rule": "pixel_value > threshold",
    "thresholds": THRESHOLDS,
    "connectivity": CONNECTIVITY,
    "target_sizes": TARGET_SIZES,
    "resize_interpolation": "cv2.INTER_NEAREST",
    "minimum_component_area_filter": None,
    "expected_mask_count": EXPECTED_MASK_COUNT,
    "environment": {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    },
}

(RUN_DIR / "experiment_config.json").write_text(
    json.dumps(CONFIG, indent=2), encoding="utf-8"
)

print(f"Run directory: {RUN_DIR}")
print(f"Thresholds: {THRESHOLDS}")
print(f"Target sizes: {TARGET_SIZES}")
print("No model training will be performed.")


# %% [cell 1]
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive_path, destination):
    destination = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()

        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("Archive member count exceeds the safety limit.")

        if sum(item.file_size for item in members) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Archive size exceeds the safety limit.")

        validated = []
        seen = set()

        for item in members:
            relative = PurePosixPath(item.filename.replace("\\", "/"))

            if (
                relative.is_absolute()
                or ".." in relative.parts
                or any(":" in part for part in relative.parts)
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe archive member: {item.filename}")

            target = destination.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe extraction path: {item.filename}")

            key = str(target).casefold()
            if key in seen:
                raise ValueError(f"Duplicate archive path: {item.filename}")
            seen.add(key)
            validated.append((item, target))

        for item, target in validated:
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


for path in [MASK_ZIP, GRADE_CSV]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")

INPUT_HASHES = {
    MASK_ZIP.name: sha256_file(MASK_ZIP),
    GRADE_CSV.name: sha256_file(GRADE_CSV),
}

safe_extract(MASK_ZIP, EXTRACT_DIR)

target_folders = [
    path for path in EXTRACT_DIR.rglob("*")
    if path.is_dir() and path.name.casefold() == "microaneurysms_masks"
]

if len(target_folders) > 1:
    raise ValueError("Multiple Microaneurysms_Masks folders were found.")

mask_root = target_folders[0] if target_folders else EXTRACT_DIR

mask_paths = sorted(
    path for path in mask_root.rglob("*")
    if path.is_file()
    and path.suffix.casefold() == ".png"
    and "__MACOSX" not in path.parts
    and not path.name.startswith("._")
)

if not mask_paths:
    raise ValueError("No PNG masks were found.")

if not target_folders and any(p.parent != EXTRACT_DIR for p in mask_paths):
    raise ValueError(
        "The expected mask folder was not found. "
        "Use Microaneurysms_Masks or place MA PNG files at the ZIP root."
    )

if len(mask_paths) != EXPECTED_MASK_COUNT:
    print(
        f"WARNING: Found {len(mask_paths)} masks; "
        f"expected {EXPECTED_MASK_COUNT}. Verify completeness."
    )

(RUN_DIR / "input_hashes.json").write_text(
    json.dumps(INPUT_HASHES, indent=2), encoding="utf-8"
)

print(f"Mask files found: {len(mask_paths)}")
print(json.dumps(INPUT_HASHES, indent=2))


# %% [cell 2]
def read_scalar_mask(path):
    image = cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise ValueError("OpenCV could not decode the PNG.")

    if image.dtype != np.uint8:
        raise ValueError(f"Unsupported pixel dtype: {image.dtype}")

    channels = 1 if image.ndim == 2 else image.shape[2]

    if image.ndim == 2:
        plane = image

    elif image.ndim == 3 and channels in [3, 4]:
        identical = (
            np.array_equal(image[:, :, 0], image[:, :, 1])
            and np.array_equal(image[:, :, 0], image[:, :, 2])
        )
        if not identical:
            raise ValueError("Color channels differ; manual review is required.")

        if channels == 4 and not np.all(image[:, :, 3] == 255):
            raise ValueError("Non-opaque alpha channel requires manual review.")

        plane = image[:, :, 0]

    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.ascontiguousarray(plane), channels


audit_records = []
pixel_histogram = np.zeros(256, dtype=np.int64)
value_presence = np.zeros(256, dtype=np.int64)

for index, path in enumerate(mask_paths, start=1):
    record = {
        "image_id": path.stem,
        "relative_path": path.relative_to(mask_root).as_posix(),
        "file_size_bytes": path.stat().st_size,
        "status": "error",
        "error_message": "",
    }

    try:
        record["sha256"] = sha256_file(path)
        mask, channels = read_scalar_mask(path)
        histogram = np.bincount(mask.ravel(), minlength=256)
        observed_values = np.flatnonzero(histogram)
        intermediate_count = int(histogram[1:255].sum())

        record.update({
            "status": "ok",
            "height": mask.shape[0],
            "width": mask.shape[1],
            "channels": channels,
            "dtype": str(mask.dtype),
            "pixel_count": mask.size,
            "unique_value_count": len(observed_values),
            "unique_values": json.dumps(observed_values.tolist()),
            "zero_pixels": int(histogram[0]),
            "value_255_pixels": int(histogram[255]),
            "intermediate_pixels": intermediate_count,
            "nonzero_pixels": int(mask.size - histogram[0]),
            "is_binary_0_255": intermediate_count == 0,
            "has_intermediate_values": intermediate_count > 0,
            "is_empty": int(histogram[0]) == mask.size,
        })

        pixel_histogram += histogram
        value_presence += histogram > 0

    except Exception as error:
        record["error_message"] = f"{type(error).__name__}: {error}"

    audit_records.append(record)

    if index % 200 == 0 or index == len(mask_paths):
        print(f"Audited masks: {index}/{len(mask_paths)}")

audit_df = pd.DataFrame(audit_records)
audit_df.to_csv(REPORT_DIR / "ma_mask_audit_initial.csv", index=False)

errors_df = audit_df.loc[audit_df["status"].ne("ok")]
errors_df.to_csv(REPORT_DIR / "mask_encoding_errors.csv", index=False)

intensity_df = pd.DataFrame({
    "pixel_value": np.arange(256),
    "pixel_count": pixel_histogram,
    "mask_count": value_presence,
})
total_pixels = int(pixel_histogram.sum())
intensity_df["pixel_fraction"] = (
    intensity_df["pixel_count"] / total_pixels
    if total_pixels else np.nan
)
intensity_df.to_csv(
    REPORT_DIR / "pixel_intensity_distribution.csv", index=False
)

if not errors_df.empty:
    print(errors_df[["image_id", "error_message"]].to_string(index=False))
    raise RuntimeError(
        "Some masks require review. See mask_encoding_errors.csv. "
        "Resolve these issues before continuing the full audit."
    )

assert total_pixels == int(audit_df["pixel_count"].sum())

print(f"Read successfully: {len(audit_df)}")
print(f"Binary masks: {int(audit_df['is_binary_0_255'].sum())}")
print(f"Grayscale masks: {int(audit_df['has_intermediate_values'].sum())}")
print(f"Empty masks: {int(audit_df['is_empty'].sum())}")
print("Empty masks are included in the binary-mask count.")

display(
    audit_df.groupby(["width", "height", "channels"])
    .size()
    .rename("mask_count")
    .reset_index()
)


# %% [cell 3]
def normalize_image_id(value):
    return Path(str(value).strip().replace("\\", "/")).stem


grades_df = pd.read_csv(
    GRADE_CSV,
    dtype=str,
    encoding="utf-8-sig",
)
grades_df.columns = grades_df.columns.str.strip()

required_columns = {"image", "label"}
if not required_columns.issubset(grades_df.columns):
    raise ValueError(
        f"Expected CSV columns {sorted(required_columns)}; "
        f"found {grades_df.columns.tolist()}."
    )

if grades_df[["image", "label"]].isna().any().any():
    raise ValueError("The grade CSV contains missing image IDs or labels.")

grades_df["image_id"] = grades_df["image"].map(normalize_image_id)
numeric_grades = pd.to_numeric(grades_df["label"], errors="coerce")

invalid_grades = ~numeric_grades.isin([0, 1, 2, 3, 4])
if invalid_grades.any():
    grades_df.loc[invalid_grades].to_csv(
        REPORT_DIR / "invalid_grade_rows.csv", index=False
    )
    raise ValueError("Invalid DR grades were found.")

grades_df["dr_grade"] = numeric_grades.astype(int)

duplicate_grade_ids = grades_df[
    grades_df["image_id"].duplicated(keep=False)
]
duplicate_mask_ids = audit_df[
    audit_df["image_id"].duplicated(keep=False)
]

duplicate_grade_ids.to_csv(
    REPORT_DIR / "duplicate_grade_ids.csv", index=False
)
duplicate_mask_ids.to_csv(
    REPORT_DIR / "duplicate_mask_ids.csv", index=False
)

if not duplicate_grade_ids.empty or not duplicate_mask_ids.empty:
    raise ValueError("Duplicate image IDs must be resolved before matching.")

mask_ids = set(audit_df["image_id"])
grade_ids = set(grades_df["image_id"])

missing_grade_ids = sorted(mask_ids - grade_ids)
missing_mask_ids = sorted(grade_ids - mask_ids)

pd.DataFrame({"image_id": missing_grade_ids}).to_csv(
    REPORT_DIR / "masks_without_grades.csv", index=False
)
pd.DataFrame({"image_id": missing_mask_ids}).to_csv(
    REPORT_DIR / "grades_without_masks.csv", index=False
)

if missing_grade_ids or missing_mask_ids:
    raise ValueError(
        "Mask/grade coverage mismatch. Review the exported mismatch reports."
    )

matched_df = audit_df.merge(
    grades_df[["image_id", "dr_grade"]],
    on="image_id",
    how="left",
    validate="one_to_one",
)

matched_df.to_csv(REPORT_DIR / "mask_grade_manifest.csv", index=False)

duplicate_content_df = matched_df[
    matched_df["sha256"].duplicated(keep=False)
].sort_values("sha256")

duplicate_content_df.to_csv(
    REPORT_DIR / "duplicate_mask_content.csv", index=False
)

grade_encoding_df = matched_df.groupby("dr_grade").agg(
    image_count=("image_id", "size"),
    binary_mask_count=("is_binary_0_255", "sum"),
    grayscale_mask_count=("has_intermediate_values", "sum"),
    empty_mask_count=("is_empty", "sum"),
).reindex(range(5))

grade_encoding_df.to_csv(REPORT_DIR / "encoding_by_dr_grade.csv")

print(f"Matched masks: {len(matched_df)}")
print("Identical mask content does not establish patient duplication.")
display(grade_encoding_df)


# %% [cell 4]
def connected_components(binary_mask):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(binary_mask, dtype=np.uint8),
        connectivity=CONNECTIVITY,
        ltype=cv2.CV_32S,
    )
    return labels, stats[1:], centroids[1:]


path_by_id = {path.stem: path for path in mask_paths}
grade_by_id = matched_df.set_index("image_id")["dr_grade"].to_dict()

image_records = []
component_records = []

analysis_started = time.perf_counter()

for index, image_id in enumerate(sorted(path_by_id), start=1):
    mask, _ = read_scalar_mask(path_by_id[image_id])
    height, width = mask.shape
    dr_grade = int(grade_by_id[image_id])
    previous_foreground = mask.size + 1

    for threshold in THRESHOLDS:
        binary = (mask > threshold).astype(np.uint8)
        labels, stats, centroids = connected_components(binary)
        areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
        foreground_pixels = int(binary.sum())

        assert int(areas.sum()) == foreground_pixels
        assert foreground_pixels <= previous_foreground
        previous_foreground = foreground_pixels

        image_records.append({
            "image_id": image_id,
            "dr_grade": dr_grade,
            "threshold": threshold,
            "height": height,
            "width": width,
            "candidate_count": len(areas),
            "foreground_pixels": foreground_pixels,
            "foreground_fraction": foreground_pixels / mask.size,
            "mean_component_area_px": (
                float(areas.mean()) if len(areas) else np.nan
            ),
            "median_component_area_px": (
                float(np.median(areas)) if len(areas) else np.nan
            ),
            "max_component_area_px": (
                int(areas.max()) if len(areas) else np.nan
            ),
            "one_pixel_component_count": int((areas == 1).sum()),
        })

        for component_id, (row, centroid) in enumerate(
            zip(stats, centroids), start=1
        ):
            area = int(row[cv2.CC_STAT_AREA])
            component_records.append({
                "image_id": image_id,
                "dr_grade": dr_grade,
                "threshold": threshold,
                "component_id": component_id,
                "area_px": area,
                "area_fraction": area / mask.size,
                "equivalent_diameter_px": float(
                    np.sqrt(4 * area / np.pi)
                ),
                "bbox_x": int(row[cv2.CC_STAT_LEFT]),
                "bbox_y": int(row[cv2.CC_STAT_TOP]),
                "bbox_width": int(row[cv2.CC_STAT_WIDTH]),
                "bbox_height": int(row[cv2.CC_STAT_HEIGHT]),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
            })

    if index % 200 == 0 or index == len(path_by_id):
        print(f"Component analysis: {index}/{len(path_by_id)}")

image_metrics_df = pd.DataFrame(image_records)

component_columns = [
    "image_id", "dr_grade", "threshold", "component_id",
    "area_px", "area_fraction", "equivalent_diameter_px",
    "bbox_x", "bbox_y", "bbox_width", "bbox_height",
    "centroid_x", "centroid_y",
]
components_df = pd.DataFrame(
    component_records, columns=component_columns
)

image_metrics_df.to_csv(
    REPORT_DIR / "image_metrics_by_threshold.csv", index=False
)
components_df.to_csv(
    REPORT_DIR / "candidate_components.csv", index=False
)

threshold_summary_df = image_metrics_df.groupby("threshold").agg(
    image_count=("image_id", "size"),
    total_candidate_count=("candidate_count", "sum"),
    mean_candidate_count_per_image=("candidate_count", "mean"),
    median_candidate_count_per_image=("candidate_count", "median"),
    total_foreground_pixels=("foreground_pixels", "sum"),
    mean_foreground_fraction=("foreground_fraction", "mean"),
    empty_mask_count=("foreground_pixels", lambda values: int((values == 0).sum())),
)

threshold_summary_df.to_csv(
    REPORT_DIR / "threshold_sensitivity_summary.csv"
)

print(
    f"Component analysis completed in "
    f"{time.perf_counter() - analysis_started:.1f} seconds."
)
print("Component counts are candidate counts, not confirmed lesion counts.")
print("Component counts need not decrease monotonically with threshold.")
display(threshold_summary_df)


# %% [cell 5]
grade_threshold_summary_df = image_metrics_df.groupby(
    ["threshold", "dr_grade"]
).agg(
    image_count=("image_id", "size"),
    total_candidate_count=("candidate_count", "sum"),
    mean_candidate_count_per_image=("candidate_count", "mean"),
    median_candidate_count_per_image=("candidate_count", "median"),
    candidate_count_q25=("candidate_count", lambda x: x.quantile(0.25)),
    candidate_count_q75=("candidate_count", lambda x: x.quantile(0.75)),
    total_foreground_pixels=("foreground_pixels", "sum"),
    mean_foreground_fraction=("foreground_fraction", "mean"),
    median_foreground_fraction=("foreground_fraction", "median"),
    empty_mask_count=("foreground_pixels", lambda x: int((x == 0).sum())),
).reset_index()

grade_threshold_summary_df.to_csv(
    REPORT_DIR / "dr_grade_threshold_summary.csv", index=False
)

component_size_summary_df = components_df.groupby(
    ["threshold", "dr_grade"]
).agg(
    component_count=("area_px", "size"),
    mean_area_px=("area_px", "mean"),
    median_area_px=("area_px", "median"),
    area_q25_px=("area_px", lambda x: x.quantile(0.25)),
    area_q75_px=("area_px", lambda x: x.quantile(0.75)),
    area_p95_px=("area_px", lambda x: x.quantile(0.95)),
    minimum_area_px=("area_px", "min"),
    maximum_area_px=("area_px", "max"),
).reset_index()

component_size_summary_df.to_csv(
    REPORT_DIR / "component_size_by_dr_grade.csv", index=False
)

review_flags_df = image_metrics_df.copy()
review_flags_df["dr0_with_nonzero_mask"] = (
    review_flags_df["dr_grade"].eq(0)
    & review_flags_df["foreground_pixels"].gt(0)
)
review_flags_df["dr_positive_without_ma_candidate"] = (
    review_flags_df["dr_grade"].gt(0)
    & review_flags_df["candidate_count"].eq(0)
)
review_flags_df["has_one_pixel_components"] = (
    review_flags_df["one_pixel_component_count"].gt(0)
)

review_flags_df.to_csv(
    REPORT_DIR / "annotation_review_flags.csv", index=False
)

print("Review flags are not automatically annotation errors.")
print("A positive DR grade does not require an MA annotation in every image.")
display(grade_threshold_summary_df.head(10))


# %% [cell 6]
resize_records = []
resize_started = time.perf_counter()

for index, image_id in enumerate(sorted(path_by_id), start=1):
    mask, _ = read_scalar_mask(path_by_id[image_id])
    height, width = mask.shape
    dr_grade = int(grade_by_id[image_id])

    if height != width:
        raise ValueError(
            f"Non-square mask requires an explicit resize policy: {image_id}"
        )

    for threshold in THRESHOLDS:
        binary = (mask > threshold).astype(np.uint8)
        original_labels, original_stats, _ = connected_components(binary)

        original_count = len(original_stats)
        original_area = int(binary.sum())

        for target_size in TARGET_SIZES:
            resized = cv2.resize(
                binary,
                (target_size, target_size),
                interpolation=cv2.INTER_NEAREST,
            )

            resized_labels, resized_stats, _ = connected_components(resized)

            reconstructed = cv2.resize(
                resized,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

            reconstructed_labels = cv2.resize(
                resized_labels,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

            original_fg = binary.astype(bool)
            reconstructed_fg = reconstructed.astype(bool)

            intersection = int(
                np.count_nonzero(original_fg & reconstructed_fg)
            )
            union = int(
                np.count_nonzero(original_fg | reconstructed_fg)
            )
            reconstructed_area = int(reconstructed.sum())

            surviving_ids = np.unique(
                original_labels[original_fg & reconstructed_fg]
            )
            surviving_ids = surviving_ids[surviving_ids > 0]
            surviving_count = len(surviving_ids)

            overlap = original_fg & reconstructed_fg
            merge_count = 0
            split_count = 0

            if overlap.any():
                overlap_pairs = np.unique(
                    np.column_stack([
                        original_labels[overlap],
                        reconstructed_labels[overlap],
                    ]),
                    axis=0,
                )

                original_degree = Counter(overlap_pairs[:, 0].tolist())
                resized_degree = Counter(overlap_pairs[:, 1].tolist())

                split_count = sum(
                    degree > 1 for degree in original_degree.values()
                )
                merge_count = sum(
                    degree > 1 for degree in resized_degree.values()
                )

            source_pixel_area_per_target_pixel = (
                height * width / target_size**2
            )
            scaled_resized_area = (
                int(resized.sum()) * source_pixel_area_per_target_pixel
            )

            resize_records.append({
                "image_id": image_id,
                "dr_grade": dr_grade,
                "threshold": threshold,
                "target_size": target_size,
                "original_candidate_count": original_count,
                "resized_candidate_count": len(resized_stats),
                "surviving_original_components": surviving_count,
                "lost_original_components": original_count - surviving_count,
                "component_survival_fraction": (
                    surviving_count / original_count
                    if original_count else np.nan
                ),
                "original_foreground_pixels": original_area,
                "resized_foreground_pixels": int(resized.sum()),
                "scaled_resized_area_in_source_pixels": scaled_resized_area,
                "area_retention_ratio": (
                    scaled_resized_area / original_area
                    if original_area else np.nan
                ),
                "foreground_recall": (
                    intersection / original_area
                    if original_area else np.nan
                ),
                "foreground_precision": (
                    intersection / reconstructed_area
                    if reconstructed_area else np.nan
                ),
                "round_trip_iou": (
                    intersection / union if original_area else np.nan
                ),
                "resized_merge_candidate_count": merge_count,
                "original_split_candidate_count": split_count,
            })

            assert surviving_count <= original_count
            assert intersection <= original_area

    if index % 100 == 0 or index == len(path_by_id):
        print(f"Resize analysis: {index}/{len(path_by_id)}")

resize_df = pd.DataFrame(resize_records)
resize_df.to_csv(
    REPORT_DIR / "resize_preservation_per_image.csv", index=False
)

resize_summary_df = resize_df.groupby(
    ["threshold", "target_size"]
).agg(
    image_count=("image_id", "size"),
    nonempty_image_count=("round_trip_iou", "count"),
    mean_round_trip_iou=("round_trip_iou", "mean"),
    median_round_trip_iou=("round_trip_iou", "median"),
    mean_foreground_recall=("foreground_recall", "mean"),
    mean_component_survival=("component_survival_fraction", "mean"),
    mean_area_retention_ratio=("area_retention_ratio", "mean"),
    original_components=("original_candidate_count", "sum"),
    surviving_components=("surviving_original_components", "sum"),
    lost_components=("lost_original_components", "sum"),
    merge_candidates=("resized_merge_candidate_count", "sum"),
    split_candidates=("original_split_candidate_count", "sum"),
).reset_index()

resize_summary_df["pooled_component_survival"] = (
    resize_summary_df["surviving_components"]
    / resize_summary_df["original_components"].replace(0, np.nan)
)

resize_summary_df.to_csv(
    REPORT_DIR / "resize_preservation_summary.csv", index=False
)

resize_grade_summary_df = resize_df.groupby(
    ["threshold", "target_size", "dr_grade"]
).agg(
    image_count=("image_id", "size"),
    nonempty_image_count=("round_trip_iou", "count"),
    mean_round_trip_iou=("round_trip_iou", "mean"),
    mean_foreground_recall=("foreground_recall", "mean"),
    mean_component_survival=("component_survival_fraction", "mean"),
).reset_index()

resize_grade_summary_df.to_csv(
    REPORT_DIR / "resize_preservation_by_dr_grade.csv", index=False
)

print(
    f"Resize analysis completed in "
    f"{time.perf_counter() - resize_started:.1f} seconds."
)
print("These results describe nearest-neighbor geometric preservation only.")
display(resize_summary_df)


# %% [cell 7]
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

grade_colors = plt.get_cmap("tab10")(np.arange(5))


def save_figure(figure, stem):
    figure.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        FIGURE_DIR / f"{stem}.pdf",
        bbox_inches="tight",
    )
    plt.show()
    plt.close(figure)


# Figure 1: Pixel intensity encoding.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

axes[0].bar(np.arange(256), pixel_histogram, width=1)
axes[0].set_yscale("symlog", linthresh=1)
axes[0].set(
    title="Stored pixel intensity distribution",
    xlabel="Pixel intensity",
    ylabel="Pixel count (symmetric log scale)",
)

axes[1].bar(np.arange(1, 255), value_presence[1:255], width=1)
axes[1].set(
    title="Intermediate intensity prevalence",
    xlabel="Pixel intensity",
    ylabel="Number of masks containing the value",
)
save_figure(fig, "pixel_intensity_distribution")


# Figure 2: Threshold sensitivity.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

for grade in range(5):
    subset = grade_threshold_summary_df[
        grade_threshold_summary_df["dr_grade"].eq(grade)
    ].sort_values("threshold")

    axes[0].plot(
        subset["threshold"],
        subset["mean_candidate_count_per_image"],
        marker="o",
        color=grade_colors[grade],
        label=f"DR{grade}",
    )
    axes[1].plot(
        subset["threshold"],
        subset["mean_foreground_fraction"] * 100,
        marker="o",
        color=grade_colors[grade],
        label=f"DR{grade}",
    )

axes[0].set(
    title="Candidate counts across thresholds",
    xlabel="Threshold t in mask > t",
    ylabel="Mean candidate count per image",
)
axes[1].set(
    title="Foreground area across thresholds",
    xlabel="Threshold t in mask > t",
    ylabel="Mean foreground area (%)",
)

for ax in axes:
    ax.set_xticks(THRESHOLDS)
    ax.legend(title="DR grade")
    ax.grid(alpha=0.2)

save_figure(fig, "threshold_sensitivity_by_dr_grade")


# Figure 3: Component area distributions.
fig, axes = plt.subplots(
    2, 3, figsize=(13, 8), constrained_layout=True
)

for ax, threshold in zip(axes.flat, THRESHOLDS):
    for grade in range(5):
        areas = components_df.loc[
            components_df["threshold"].eq(threshold)
            & components_df["dr_grade"].eq(grade),
            "area_px",
        ].to_numpy(dtype=float)

        if len(areas):
            areas = np.sort(areas)
            cumulative = np.arange(1, len(areas) + 1) / len(areas)
            ax.step(
                areas,
                cumulative,
                where="post",
                color=grade_colors[grade],
                label=f"DR{grade} (n={len(areas)})",
            )

    ax.set_xscale("log")
    ax.set(
        title=f"Threshold: mask > {threshold}",
        xlabel="Candidate area at source resolution (pixels)",
        ylabel="Cumulative proportion",
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.2)
    if ax.lines:
        ax.legend(loc="lower right", fontsize=7)

save_figure(fig, "candidate_area_ecdf_by_dr_grade")


# Figure 4: Resize preservation.
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

plot_specs = [
    ("mean_round_trip_iou", "Mean round-trip IoU"),
    ("mean_foreground_recall", "Mean foreground recall"),
    ("pooled_component_survival", "Pooled component survival"),
]

for threshold in THRESHOLDS:
    subset = resize_summary_df[
        resize_summary_df["threshold"].eq(threshold)
    ].sort_values("target_size")

    for ax, (column, label) in zip(axes, plot_specs):
        ax.plot(
            subset["target_size"],
            subset[column],
            marker="o",
            label=f"t = {threshold}",
        )
        ax.set(
            xlabel="Target image size (pixels)",
            ylabel=label,
            ylim=(0, 1.03),
        )
        ax.set_xticks(TARGET_SIZES)
        ax.grid(alpha=0.2)

axes[-1].legend(title="Threshold")
save_figure(fig, "resize_geometric_preservation")

print(f"Figures saved to: {FIGURE_DIR}")


# %% [cell 8]
expected_image_rows = len(matched_df) * len(THRESHOLDS)
expected_resize_rows = expected_image_rows * len(TARGET_SIZES)

assert len(image_metrics_df) == expected_image_rows
assert len(resize_df) == expected_resize_rows

assert not image_metrics_df.duplicated(
    ["image_id", "threshold"]
).any()

assert not resize_df.duplicated(
    ["image_id", "threshold", "target_size"]
).any()

for path in [MASK_ZIP, GRADE_CSV]:
    if sha256_file(path) != INPUT_HASHES[path.name]:
        raise RuntimeError(f"Input file changed during the audit: {path.name}")

sample_ids = random.Random(SEED).sample(
    sorted(path_by_id),
    min(10, len(path_by_id)),
)

metrics_index = image_metrics_df.set_index(["image_id", "threshold"])
reproducibility_records = []

for image_id in sample_ids:
    mask, _ = read_scalar_mask(path_by_id[image_id])

    for threshold in THRESHOLDS:
        binary = (mask > threshold).astype(np.uint8)
        _, stats, _ = connected_components(binary)
        reference = metrics_index.loc[(image_id, threshold)]

        passed = (
            len(stats) == int(reference["candidate_count"])
            and int(binary.sum()) == int(reference["foreground_pixels"])
        )

        reproducibility_records.append({
            "image_id": image_id,
            "threshold": threshold,
            "passed": passed,
        })

reproducibility_df = pd.DataFrame(reproducibility_records)
reproducibility_df.to_csv(
    REPORT_DIR / "reproducibility_checks.csv", index=False
)

if not reproducibility_df["passed"].all():
    raise RuntimeError("Reproducibility checks failed.")

completion_summary = {
    "experiment_name": EXPERIMENT_NAME,
    "code_version": CODE_VERSION,
    "run_id": RUN_ID,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "mask_count": len(matched_df),
    "expected_mask_count": EXPECTED_MASK_COUNT,
    "expected_count_matched": len(matched_df) == EXPECTED_MASK_COUNT,
    "binary_mask_count": int(matched_df["is_binary_0_255"].sum()),
    "grayscale_mask_count": int(matched_df["has_intermediate_values"].sum()),
    "empty_mask_count": int(matched_df["is_empty"].sum()),
    "image_threshold_rows": len(image_metrics_df),
    "component_rows_across_all_thresholds": len(components_df),
    "resize_rows": len(resize_df),
    "reproducibility_checks_passed": True,
    "input_hashes_unchanged": True,
    "interpretation_notes": [
        "No threshold is treated as ground truth.",
        "Connected components are candidate lesions.",
        "Component rows across thresholds are not independent lesions.",
        "Identical masks do not establish patient duplication.",
        "Resize results are specific to nearest-neighbor interpolation.",
        "Empty masks are excluded from undefined overlap metrics.",
        "Pixel areas are not physical areas in square millimeters.",
        "No inferential statistical tests were performed.",
        "No model training was performed.",
    ],
}

(RUN_DIR / "audit_completion_summary.json").write_text(
    json.dumps(completion_summary, indent=2),
    encoding="utf-8",
)

export_files = (
    sorted(REPORT_DIR.glob("*.csv"))
    + sorted(FIGURE_DIR.glob("*"))
    + [
        RUN_DIR / "experiment_config.json",
        RUN_DIR / "input_hashes.json",
        RUN_DIR / "audit_completion_summary.json",
    ]
)

output_manifest_df = pd.DataFrame([
    {
        "relative_path": path.relative_to(RUN_DIR).as_posix(),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    for path in export_files
])

manifest_path = RUN_DIR / "output_manifest.csv"
output_manifest_df.to_csv(manifest_path, index=False)
export_files.append(manifest_path)

RESULTS_ZIP = OUTPUT_ROOT / f"{EXPERIMENT_NAME}_{RUN_ID}.zip"

with zipfile.ZipFile(
    RESULTS_ZIP,
    mode="x",
    compression=zipfile.ZIP_DEFLATED,
) as archive:
    for path in export_files:
        archive.write(path, arcname=path.relative_to(RUN_DIR).as_posix())

print(json.dumps(completion_summary, indent=2))
print(f"\nResults archive: {RESULTS_ZIP}")
print("The results archive does not contain source masks.")


# %% [cell 9]
# (Colab download helper removed)

pass  # files.download(str(RESULTS_ZIP))


# %% [cell 10]


pass  # Google Drive mount removed


# %% [cell 11] identical to cell 13, omitted here


# %% [cell 12]


pass  # Google Drive mount removed


# %% [cell 13]

from pathlib import Path

project_root = DR_RESEARCH

print("Project exists:", project_root.exists())

if project_root.exists():
    print("Project folders:")
    for item in project_root.iterdir():
        print(item.name)


# %% [cell 14]

from pathlib import Path

dataset_root = (DR_RESEARCH / "datasets")

print("Dataset directory exists:", dataset_root.exists())

if dataset_root.exists():
    print("\nTop-level contents:")

    for item in sorted(dataset_root.iterdir()):
        item_type = "DIR" if item.is_dir() else "FILE"
        print(f"[{item_type}] {item.name}")

print("\nDirectory check completed.")


# %% [cell 15]

from pathlib import Path

dr_root = (DR_RESEARCH / "datasets/DR")

print("DR dataset directory exists:", dr_root.exists())

if dr_root.exists():
    print("\nDR dataset folders:")

    for item in sorted(dr_root.iterdir()):
        item_type = "DIR" if item.is_dir() else "FILE"
        print(f"[{item_type}] {item.name}")

print("\nDirectory inspection completed.")


# %% [cell 16]

from pathlib import Path

dr_root = (DR_RESEARCH / "datasets/DR")

fgadr_root = dr_root / "FGADR"

print("FGADR directory exists:", fgadr_root.exists())

if fgadr_root.exists():
    print("\nFGADR top-level contents:")
    for item in sorted(fgadr_root.iterdir()):
        item_type = "DIR" if item.is_dir() else "FILE"
        print(f"[{item_type}] {item.name}")

print("\nFGADR directory inspection completed.")


# %% [cell 17]

from pathlib import Path

phase2_root = (DR_RESEARCH / "phase2_results")

print("Phase 2 directory exists:", phase2_root.exists())

if phase2_root.exists():
    print("\nPhase 2 contents:")
    for item in sorted(phase2_root.iterdir()):
        item_type = "DIR" if item.is_dir() else "FILE"
        print(f"[{item_type}] {item.name}")

print("\nPhase 2 directory inspection completed.")


# %% [cell 18]

from pathlib import Path

phase2_root = (DR_RESEARCH / "phase2_results")

zip_files = sorted(phase2_root.glob("*.zip"))

print("Phase 2 archive count:", len(zip_files))

for file_path in zip_files:
    size_mb = file_path.stat().st_size / (1024 ** 2)
    print(f"Archive: {file_path.name}")
    print(f"Size: {size_mb:.2f} MiB")

if not zip_files:
    print("No Phase 2 ZIP archive found.")
else:
    print("Phase 2 archive is available.")


# %% [cell 19]

from pathlib import Path
import zipfile

archive_path = (DR_RESEARCH / "phase2_results/FGADR_Annotation_Review_Phase2.zip")

print("Archive exists:", archive_path.exists())

if archive_path.exists():
    with zipfile.ZipFile(archive_path, "r") as archive:
        files = archive.namelist()

        print("Total archive entries:", len(files))
        print("\nArchive contents (first 30 entries):")

        for filename in files[:30]:
            print(filename)

        print("\nRequired report files:")

        required_files = [
            "large_component_review.csv",
            "dr0_annotation_review.csv",
            "encoding_image_inventory.csv",
            "prioritized_unresolved_issues.csv",
            "size_stratified_resolution_summary.csv"
        ]

        for required_file in required_files:
            matches = [
                name for name in files
                if Path(name).name == required_file
            ]

            if matches:
                print(f"[FOUND] {required_file}: {matches[0]}")
            else:
                print(f"[MISSING] {required_file}")


# %% [cell 20]

import zipfile
import pandas as pd
from pathlib import Path
from io import BytesIO

archive_path = (DR_RESEARCH / "phase2_results/FGADR_Annotation_Review_Phase2.zip")

report_names = [
    "encoding_image_inventory.csv",
    "large_component_review.csv",
    "dr0_annotation_review.csv",
    "prioritized_unresolved_issues.csv",
    "threshold_sensitive_image_metrics.csv"
]

with zipfile.ZipFile(archive_path, "r") as archive:
    archive_files = archive.namelist()

    for report_name in report_names:
        matches = [
            name for name in archive_files
            if Path(name).name == report_name
        ]

        if len(matches) != 1:
            print(f"Unexpected matches for {report_name}: {len(matches)}")
            continue

        with archive.open(matches[0]) as file:
            dataframe = pd.read_csv(file)

        print(f"\nReport: {report_name}")
        print(f"Rows: {len(dataframe)}")
        print(f"Columns: {dataframe.columns.tolist()}")
        print(dataframe.head(2).to_string(index=False))
        print("-" * 60)

print("\nReport inspection completed.")


# %% [cell 21]

from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone

# Research paths
project_root = DR_RESEARCH

archive_path = (
    project_root
    / "phase2_results"
    / "FGADR_Annotation_Review_Phase2.zip"
)

output_dir = project_root / "phase3_results"
output_dir.mkdir(parents=True, exist_ok=True)

# Exploratory screening parameters
REFERENCE_THRESHOLD = 127
BROAD_AREA_FLAG_PX = 5000
BROAD_COVERAGE_FLAG = 0.01

# Read a unique CSV from the Phase 2 archive
def read_report(archive, filename):
    matches = [
        name for name in archive.namelist()
        if Path(name).name == filename
        and "/reports/" in name
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one {filename}, found {len(matches)}."
        )

    with archive.open(matches[0]) as handle:
        return pd.read_csv(handle)

with zipfile.ZipFile(archive_path, "r") as archive:
    inventory = read_report(
        archive, "encoding_image_inventory.csv"
    )
    components = read_report(
        archive, "reference_component_areas.csv"
    )
    large_review = read_report(
        archive, "large_component_review.csv"
    )
    dr0_review = read_report(
        archive, "dr0_annotation_review.csv"
    )
    issues = read_report(
        archive, "prioritized_unresolved_issues.csv"
    )
    threshold_metrics = read_report(
        archive, "threshold_sensitive_image_metrics.csv"
    )

# Validate image and component identities
assert len(inventory) == 1842
assert inventory["image_id"].is_unique
assert len(components) == 11235
assert components["reference_threshold"].eq(
    REFERENCE_THRESHOLD
).all()
assert not components.duplicated(
    ["image_id", "component_id"]
).any()

assert set(components["image_id"]).issubset(
    set(inventory["image_id"])
)

# Compute complete image-level component statistics
component_stats = (
    components.groupby("image_id")
    .agg(
        candidate_count=("component_id", "size"),
        largest_component_area_px=("original_area_px", "max"),
        total_mask_area_px=("original_area_px", "sum"),
    )
    .reset_index()
)

df = inventory.merge(
    component_stats,
    on="image_id",
    how="left",
    validate="one_to_one"
)

numeric_columns = [
    "candidate_count",
    "largest_component_area_px",
    "total_mask_area_px",
]

df[numeric_columns] = df[numeric_columns].fillna(0)

df["mask_coverage"] = (
    df["total_mask_area_px"]
    / (df["fundus_width"] * df["fundus_height"])
)

df["is_empty_mask"] = df["candidate_count"].eq(0)

# Build a list of known review cases
large_ids = set(large_review["image_id"])
dr0_ids = set(dr0_review["image_id"])
threshold_ids = set(threshold_metrics["image_id"])
issue_ids = set(issues["image_id"]) & set(df["image_id"])

df["known_large_review"] = df["image_id"].isin(large_ids)
df["dr0_nonempty_flag"] = df["image_id"].isin(dr0_ids)
df["threshold_sensitive_flag"] = df["image_id"].isin(
    threshold_ids
)

df["grayscale_encoding_flag"] = df[
    "encoding"
].eq("intermediate_grayscale")

df["broad_area_flag"] = (
    df["largest_component_area_px"] >= BROAD_AREA_FLAG_PX
)

df["broad_coverage_flag"] = (
    df["mask_coverage"] >= BROAD_COVERAGE_FLAG
)

# Known large cases and exploratory broad-region screening
df["broad_region_flag"] = (
    df["known_large_review"]
    | df["broad_area_flag"]
    | df["broad_coverage_flag"]
)

df["manual_review_flag"] = (
    df["dr0_nonempty_flag"]
    | df["threshold_sensitive_flag"]
    | df["grayscale_encoding_flag"]
    | df["image_id"].isin(issue_ids)
)

# Prioritize categories without removing secondary flags
df["review_category"] = np.select(
    [
        df["is_empty_mask"],
        df["broad_region_flag"],
        df["manual_review_flag"],
    ],
    [
        "empty_mask",
        "uncertain_broad_region",
        "requires_manual_review",
    ],
    default="strict_lesion_level_candidate",
)

# Preserve all applicable review reasons
reason_columns = {
    "known_large_review": "previous_large_region_review",
    "broad_area_flag": "large_candidate_component",
    "broad_coverage_flag": "high_mask_coverage",
    "dr0_nonempty_flag": "dr0_with_nonempty_ma_mask",
    "threshold_sensitive_flag": "threshold_sensitive_connectivity",
    "grayscale_encoding_flag": "intermediate_grayscale_encoding",
}

def collect_reasons(row):
    reasons = [
        reason for column, reason in reason_columns.items()
        if bool(row[column])
    ]
    return ";".join(reasons) if reasons else "no_screening_flags"

df["review_reason"] = df.apply(collect_reasons, axis=1)

df["clinical_validation_status"] = "not_clinically_adjudicated"
df["reference_threshold"] = REFERENCE_THRESHOLD

# Check consistency
assert len(df) == 1842
assert df["image_id"].is_unique
assert int(df["is_empty_mask"].sum()) == 418
assert int(df["candidate_count"].sum()) == 11235
assert int(df["total_mask_area_px"].sum()) > 0

# Export complete categorization
output_file = (
    output_dir / "MA_Mask_Quality_Categorization.csv"
)

df.sort_values("image_id").to_csv(
    output_file, index=False
)

# Export review categories
summary = (
    df.groupby("review_category")
    .agg(
        image_count=("image_id", "size"),
    )
    .reset_index()
)

summary.to_csv(
    output_dir / "MA_Quality_Category_Summary.csv",
    index=False
)

# Export flagged cases separately
flagged = df[
    df["review_category"].isin(
        ["uncertain_broad_region", "requires_manual_review"]
    )
].copy()

flagged.to_csv(
    output_dir / "MA_Prioritized_Review_Candidates.csv",
    index=False
)

# Record experiment settings
parameters = {
    "reference_threshold": REFERENCE_THRESHOLD,
    "reference_is_ground_truth": False,
    "broad_area_flag_px": BROAD_AREA_FLAG_PX,
    "broad_coverage_flag": BROAD_COVERAGE_FLAG,
    "categories_are_clinically_validated": False,
    "component_count_represents_confirmed_lesions": False,
    "created_utc": datetime.now(timezone.utc).isoformat(),
}

with open(
    output_dir / "Phase3_Parameters.json",
    "w",
    encoding="utf-8"
) as handle:
    json.dump(parameters, handle, indent=2)

print("PHASE 3 CATEGORIZATION COMPLETE")
print(f"Total images: {len(df)}")
print(f"Empty masks: {df['is_empty_mask'].sum()}")
print(f"Candidate components: {int(df['candidate_count'].sum())}")
print("\nCategory summary:")
print(summary.to_string(index=False))
print(f"\nResults saved to: {output_dir}")


# %% [cell 22]

from pathlib import Path
import pandas as pd

project_root = Path(
    str(DR_RESEARCH)
)

input_path = (
    project_root
    / "phase3_results"
    / "MA_Mask_Quality_Categorization.csv"
)

df = pd.read_csv(input_path)

print("Total images:", len(df))
print("\nCategory distribution by DR grade:")

grade_table = pd.crosstab(
    df["dr_grade"],
    df["review_category"]
)

print(grade_table.to_string())

print("\nPercentage distribution within each DR grade:")

grade_percent = (
    grade_table.div(grade_table.sum(axis=1), axis=0) * 100
)

print(grade_percent.round(2).to_string())

print("\nBroad-region flag counts:")

for column in [
    "known_large_review",
    "broad_area_flag",
    "broad_coverage_flag"
]:
    print(f"{column}: {int(df[column].sum())}")

print("\nAnalysis completed.")


# %% [cell 23]

from pathlib import Path
import pandas as pd

project_root = DR_RESEARCH

input_path = (
    project_root
    / "phase3_results"
    / "MA_Mask_Quality_Categorization.csv"
)

output_dir = project_root / "phase3_results"
df = pd.read_csv(input_path)

area_thresholds = [2500, 5000, 10000, 20000]
coverage_threshold = 0.01

results = []

for threshold in area_thresholds:
    broad_flag = (
        df["known_large_review"].astype(bool)
        | (df["largest_component_area_px"] >= threshold)
        | (df["mask_coverage"] >= coverage_threshold)
    )

    flagged = df.loc[broad_flag]

    results.append({
        "area_threshold_px": threshold,
        "coverage_threshold": coverage_threshold,
        "flagged_images": len(flagged),
        "flagged_percentage": round(
            len(flagged) / len(df) * 100, 2
        ),
        **{
            f"DR{grade}_flagged": int(
                (flagged["dr_grade"] == grade).sum()
            )
            for grade in range(5)
        }
    })

summary = pd.DataFrame(results)

print("Broad-region screening sensitivity:")
print(summary.to_string(index=False))

output_path = (
    output_dir
    / "MA_Broad_Region_Threshold_Sensitivity.csv"
)

summary.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")


# %% [cell 24]

from pathlib import Path
import pandas as pd

PROJECT_ROOT = DR_RESEARCH
FGADR_ROOT = PROJECT_ROOT / "datasets" / "DR" / "FGADR"
OUTPUT_ROOT = PROJECT_ROOT / "phase3_results"

print("EXPERIMENT 02 - STEP 1.1")
print("=" * 60)

assert FGADR_ROOT.is_dir(), "FGADR directory not found."

for directory_name in ["images4classification", "images4seg"]:
    directory = FGADR_ROOT / directory_name

    print(f"\nDirectory: {directory_name}")
    print(f"Exists: {directory.is_dir()}")

    if directory.is_dir():
        for item in sorted(directory.iterdir()):
            item_type = "DIR" if item.is_dir() else "FILE"
            print(f"  [{item_type}] {item.name}")

label_files = [
    FGADR_ROOT / "images4seg" / "DR_Seg_Grading_Label.csv",
    FGADR_ROOT / "images4classification" / "DR_Seg_Grading_Label.csv",
]

print("\nLABEL FILE INSPECTION")
print("=" * 60)

found_labels = 0

for label_path in label_files:
    if not label_path.is_file():
        continue

    found_labels += 1
    labels = pd.read_csv(label_path)

    print(f"\nLabel file: {label_path.relative_to(FGADR_ROOT)}")
    print(f"Rows: {len(labels)}")
    print(f"Columns: {labels.columns.tolist()}")
    print("\nFirst five records:")
    print(labels.head(5).to_string(index=False))
    print("\nMissing values:")
    print(labels.isna().sum().to_string())

if found_labels == 0:
    print("\nNo label CSV found at the expected locations.")
    print("The filename or directory structure must be verified.")

print("\nSTEP 1.1 COMPLETED")


# %% [cell 25]

from pathlib import Path
import pandas as pd
import json
from datetime import datetime, timezone

project_root = DR_RESEARCH
fgadr_root = project_root / "datasets" / "DR" / "FGADR"
classification_root = fgadr_root / "images4classification"
segmentation_root = fgadr_root / "images4seg"
mask_root = segmentation_root / "Microaneurysms_Masks"

label_path = segmentation_root / "DR_Seg_Grading_Label.csv"
phase3_path = (
    project_root
    / "phase3_results"
    / "MA_Mask_Quality_Categorization.csv"
)

output_dir = project_root / "experiment02_results"
output_dir.mkdir(parents=True, exist_ok=True)

print("EXPERIMENT 02 - STEP 1.2")
print("=" * 60)

# Load original labels without changing image identifiers
labels = pd.read_csv(
    label_path,
    dtype={"image": str, "label": "int64"}
)

labels["image_id"] = labels["image"].map(
    lambda value: Path(value).stem
)

assert labels["label"].between(0, 4).all()
assert labels["image"].is_unique
assert labels["image_id"].is_unique

# Inventory classification files in five known folders
image_records = []

for grade in range(5):
    grade_dir = classification_root / str(grade)
    assert grade_dir.is_dir(), f"Missing directory: {grade_dir}"

    for path in grade_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".tif", ".tiff"
        }:
            image_records.append({
                "image": path.name,
                "image_id": path.stem,
                "folder_grade": grade,
                "image_path": str(path),
            })

images = pd.DataFrame(image_records)

# Inventory mask files without scanning other lesion types
mask_records = []

for path in mask_root.iterdir():
    if path.is_file() and path.suffix.lower() in {
        ".png", ".jpg", ".jpeg", ".tif", ".tiff"
    }:
        mask_records.append({
            "image_id": path.stem,
            "mask_path": str(path),
        })

masks = pd.DataFrame(mask_records)

if images.empty or masks.empty:
    raise RuntimeError("Classification images or MA masks are missing.")

# Detect duplicate names and identifiers
duplicate_images = images[
    images.duplicated("image", keep=False)
    | images.duplicated("image_id", keep=False)
].copy()

duplicate_masks = masks[
    masks.duplicated("image_id", keep=False)
].copy()

# Match original labels to classification files
image_check = labels.merge(
    images,
    on=["image", "image_id"],
    how="outer",
    indicator=True,
    validate="one_to_many"
)

missing_images = image_check[
    image_check["_merge"] == "left_only"
].copy()

extra_images = image_check[
    image_check["_merge"] == "right_only"
].copy()

grade_mismatches = image_check[
    (image_check["_merge"] == "both")
    & (image_check["label"] != image_check["folder_grade"])
].copy()

# Match labels to masks
mask_check = labels[["image_id"]].merge(
    masks,
    on="image_id",
    how="outer",
    indicator=True,
    validate="one_to_many"
)

missing_masks = mask_check[
    mask_check["_merge"] == "left_only"
].copy()

extra_masks = mask_check[
    mask_check["_merge"] == "right_only"
].copy()

# Validate Phase 3 mapping
phase3 = pd.read_csv(
    phase3_path,
    dtype={"image_id": str}
)

phase3_ids = set(phase3["image_id"])
label_ids = set(labels["image_id"])

phase3_missing = label_ids - phase3_ids
phase3_extra = phase3_ids - label_ids

# Save diagnostic reports before enforcing validation
diagnostics = {
    "Duplicate_Classification_Images.csv": duplicate_images,
    "Duplicate_MA_Masks.csv": duplicate_masks,
    "Missing_Classification_Images.csv": missing_images,
    "Extra_Classification_Images.csv": extra_images,
    "Grade_Folder_Mismatches.csv": grade_mismatches,
    "Missing_MA_Masks.csv": missing_masks,
    "Extra_MA_Masks.csv": extra_masks,
}

for filename, report in diagnostics.items():
    report.to_csv(output_dir / filename, index=False)

results = {
    "original_label_rows": len(labels),
    "classification_image_files": len(images),
    "ma_mask_files": len(masks),
    "duplicate_classification_records": len(duplicate_images),
    "duplicate_ma_mask_records": len(duplicate_masks),
    "missing_classification_images": len(missing_images),
    "extra_classification_images": len(extra_images),
    "grade_folder_mismatches": len(grade_mismatches),
    "missing_ma_masks": len(missing_masks),
    "extra_ma_masks": len(extra_masks),
    "phase3_missing_ids": len(phase3_missing),
    "phase3_extra_ids": len(phase3_extra),
    "phase3_duplicate_ids": int(
        phase3["image_id"].duplicated().sum()
    ),
}

print("\nINTEGRITY CHECK RESULTS")
for key, value in results.items():
    print(f"{key}: {value}")

all_valid = (
    len(labels) == 1842
    and len(images) == 1842
    and len(masks) == 1842
    and all(
        value == 0
        for key, value in results.items()
        if key not in {
            "original_label_rows",
            "classification_image_files",
            "ma_mask_files",
        }
    )
)

if not all_valid:
    print("\nVALIDATION FAILED")
    print("Diagnostic CSV files have been saved.")
    print("The master manifest was not created.")

else:
    manifest = labels.merge(
        images,
        on=["image", "image_id"],
        how="inner",
        validate="one_to_one"
    )

    manifest = manifest.merge(
        masks,
        on="image_id",
        how="inner",
        validate="one_to_one"
    )

    manifest = manifest.merge(
        phase3[["image_id", "review_category"]],
        on="image_id",
        how="inner",
        validate="one_to_one"
    )

    manifest = manifest.rename(
        columns={
            "label": "dr_grade",
            "review_category": "mask_review_category",
        }
    )

    manifest = manifest[
        [
            "image_id",
            "image",
            "image_path",
            "dr_grade",
            "mask_path",
            "mask_review_category",
        ]
    ].sort_values("image_id").reset_index(drop=True)

    assert len(manifest) == 1842
    assert manifest["image_id"].is_unique

    manifest_path = (
        output_dir / "FGADR_Master_Manifest.csv"
    )

    manifest.to_csv(manifest_path, index=False)

    print("\nVALIDATION PASSED")
    print(f"Master manifest saved: {manifest_path}")

    print("\nDR grade distribution:")
    print(
        manifest["dr_grade"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFirst five manifest records:")
    print(manifest.head(5).to_string(index=False))

print("\nSTEP 1.2 COMPLETED")
