# Experiment 02 - duplicate / same-source detection and the frozen source-risk split
#
# Source: FGADR_MA_Mask_Audit.ipynb (master notebook, 392 cells), cells 26-78, run step by step in Google Colab (L4 GPU).
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

# %% [cell 26]

from pathlib import Path
from collections import Counter
import pandas as pd
import hashlib
import shutil
import json
from datetime import datetime, timezone
from tqdm.auto import tqdm

project_root = DR_RESEARCH
output_dir = project_root / "experiment02_results"
local_root = (LOCAL_DISK / "fgadr_classification")

manifest_path = output_dir / "FGADR_Master_Manifest.csv"
manifest = pd.read_csv(manifest_path, dtype={"image_id": str})

assert len(manifest) == 1842
assert manifest["image_id"].is_unique

local_root.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

def hash_file(file_path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with open(file_path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()

disk = shutil.disk_usage(str(LOCAL_DISK))
print(f"Available local disk: {disk.free / (1024 ** 3):.2f} GiB")

records = []
errors = []

for row in tqdm(
    manifest.itertuples(index=False),
    total=len(manifest),
    desc="Copying and hashing images"
):
    try:
        source_path = Path(row.image_path)
        destination_path = (
            local_root / str(row.dr_grade) / row.image
        )

        destination_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        source_digest = hashlib.sha256()
        source_size = 0

        # Read the source once while copying and hashing
        with open(source_path, "rb") as source:
            with open(destination_path, "wb") as destination:
                while True:
                    chunk = source.read(1024 * 1024)

                    if not chunk:
                        break

                    destination.write(chunk)
                    source_digest.update(chunk)
                    source_size += len(chunk)

        # Independently verify the local copy
        copied_hash = hash_file(destination_path)
        copied_size = destination_path.stat().st_size

        if (
            source_digest.hexdigest() != copied_hash
            or source_size != copied_size
        ):
            raise RuntimeError("Copy verification failed.")

        records.append({
            "image_id": row.image_id,
            "dr_grade": int(row.dr_grade),
            "source_path": str(source_path),
            "local_path": str(destination_path),
            "file_size_bytes": source_size,
            "sha256": copied_hash,
            "copy_verified": True,
        })

    except Exception as exc:
        errors.append({
            "image_id": row.image_id,
            "error": str(exc),
        })

hash_df = pd.DataFrame(records)
error_df = pd.DataFrame(errors)

hash_df.to_csv(
    output_dir / "FGADR_Image_SHA256_Manifest.csv",
    index=False
)

error_df.to_csv(
    output_dir / "FGADR_Copy_Errors.csv",
    index=False
)

print("\nCOPY AND HASH VERIFICATION")
print(f"Expected images: {len(manifest)}")
print(f"Verified images: {len(hash_df)}")
print(f"Failed images: {len(error_df)}")

if len(error_df) > 0:
    print("\nFirst copy errors:")
    print(error_df.head(10).to_string(index=False))
    raise RuntimeError("Image copy verification incomplete.")

assert len(hash_df) == len(manifest)
assert hash_df["image_id"].is_unique
assert hash_df["copy_verified"].all()

# Identify byte-identical duplicate groups
group_sizes = hash_df.groupby("sha256").size()

duplicate_hashes = group_sizes[
    group_sizes > 1
].index

duplicate_records = hash_df[
    hash_df["sha256"].isin(duplicate_hashes)
].copy()

duplicate_records["duplicate_group_size"] = (
    duplicate_records["sha256"].map(group_sizes)
)

duplicate_records = duplicate_records.sort_values(
    ["sha256", "image_id"]
)

# Identify duplicate groups with conflicting DR grades
conflict_hashes = (
    duplicate_records.groupby("sha256")["dr_grade"]
    .nunique()
)

conflict_hashes = conflict_hashes[
    conflict_hashes > 1
].index

conflict_records = duplicate_records[
    duplicate_records["sha256"].isin(conflict_hashes)
].copy()

duplicate_records.to_csv(
    output_dir / "FGADR_Exact_Duplicate_Images.csv",
    index=False
)

conflict_records.to_csv(
    output_dir / "FGADR_Exact_Duplicate_Label_Conflicts.csv",
    index=False
)

print("\nSHA256 DUPLICATE ANALYSIS")
print(f"Unique file hashes: {hash_df['sha256'].nunique()}")
print(f"Duplicate groups: {len(duplicate_hashes)}")
print(f"Images in duplicate groups: {len(duplicate_records)}")
print(f"Conflicting-label groups: {len(conflict_hashes)}")
print(f"Images in conflicting-label groups: {len(conflict_records)}")

if not conflict_records.empty:
    print("\nConflicting-label images:")
    print(
        conflict_records[
            ["image_id", "dr_grade", "sha256"]
        ].to_string(index=False)
    )

print("\nSTEP 2.1 COMPLETED")
print(f"Reports saved to: {output_dir}")


# %% [cell 27]

from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import hashlib

project_root = DR_RESEARCH
output_dir = project_root / "experiment02_results"

hash_manifest_path = (
    output_dir / "FGADR_Image_SHA256_Manifest.csv"
)

hash_manifest = pd.read_csv(
    hash_manifest_path,
    dtype={"image_id": str, "sha256": str}
)

assert len(hash_manifest) == 1842
assert hash_manifest["image_id"].is_unique
assert hash_manifest["sha256"].notna().all()

pixel_records = []
pixel_errors = []

for row in tqdm(
    hash_manifest.itertuples(index=False),
    total=len(hash_manifest),
    desc="Checking decoded pixels"
):
    try:
        image_path = Path(row.local_path)

        if not image_path.is_file():
            raise FileNotFoundError(str(image_path))

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            array = np.ascontiguousarray(
                np.asarray(image, dtype=np.uint8)
            )

        digest = hashlib.sha256()
        digest.update(
            np.asarray(array.shape, dtype="<i8").tobytes()
        )
        digest.update(array.tobytes())

        pixel_records.append({
            "image_id": row.image_id,
            "dr_grade": int(row.dr_grade),
            "sha256": row.sha256,
            "pixel_sha256": digest.hexdigest(),
            "width": int(array.shape[1]),
            "height": int(array.shape[0]),
        })

    except Exception as exc:
        pixel_errors.append({
            "image_id": row.image_id,
            "error": str(exc),
        })

pixels = pd.DataFrame(pixel_records)
errors = pd.DataFrame(pixel_errors)

errors.to_csv(
    output_dir / "FGADR_Pixel_Decode_Errors.csv",
    index=False
)

if len(errors) > 0:
    print(errors.head(10).to_string(index=False))
    raise RuntimeError("Pixel decoding was incomplete.")

assert len(pixels) == len(hash_manifest)
assert pixels["image_id"].is_unique

pixel_group_sizes = pixels.groupby("pixel_sha256").size()

duplicate_pixel_hashes = pixel_group_sizes[
    pixel_group_sizes > 1
].index

pixel_duplicates = pixels[
    pixels["pixel_sha256"].isin(duplicate_pixel_hashes)
].copy()

pixel_duplicates["pixel_group_size"] = (
    pixel_duplicates["pixel_sha256"].map(pixel_group_sizes)
)

pixel_duplicates = pixel_duplicates.sort_values(
    ["pixel_sha256", "image_id"]
)

pixel_conflict_hashes = (
    pixel_duplicates.groupby("pixel_sha256")["dr_grade"]
    .nunique()
)

pixel_conflict_hashes = pixel_conflict_hashes[
    pixel_conflict_hashes > 1
].index

pixel_conflicts = pixel_duplicates[
    pixel_duplicates["pixel_sha256"].isin(
        pixel_conflict_hashes
    )
].copy()

# Separate newly detected pixel duplicates from byte duplicates
new_pixel_groups = (
    pixel_duplicates.groupby("pixel_sha256")["sha256"]
    .nunique()
)

new_pixel_groups = new_pixel_groups[
    new_pixel_groups > 1
].index

new_pixel_duplicates = pixel_duplicates[
    pixel_duplicates["pixel_sha256"].isin(
        new_pixel_groups
    )
].copy()

pixels.to_csv(
    output_dir / "FGADR_Pixel_Hash_Manifest.csv",
    index=False
)

pixel_duplicates.to_csv(
    output_dir / "FGADR_Pixel_Duplicate_Images.csv",
    index=False
)

pixel_conflicts.to_csv(
    output_dir / "FGADR_Pixel_Label_Conflicts.csv",
    index=False
)

new_pixel_duplicates.to_csv(
    output_dir / "FGADR_New_Pixel_Duplicates.csv",
    index=False
)

print("\nPIXEL-LEVEL VERIFICATION")
print(f"Decoded images: {len(pixels)}")
print(f"Decode errors: {len(errors)}")
print(f"Unique pixel hashes: {pixels['pixel_sha256'].nunique()}")
print(f"Pixel duplicate groups: {len(duplicate_pixel_hashes)}")
print(f"Images in pixel duplicate groups: {len(pixel_duplicates)}")
print(f"Groups with conflicting DR labels: {len(pixel_conflict_hashes)}")
print(f"New groups beyond byte-identical duplicates: {len(new_pixel_groups)}")

print("\nIMAGE DIMENSIONS")
print(
    pixels.groupby(["width", "height"])
    .size()
    .to_string()
)

if not new_pixel_duplicates.empty:
    print("\nNEW PIXEL DUPLICATE CANDIDATES")
    print(
        new_pixel_duplicates[
            ["image_id", "dr_grade", "sha256", "pixel_sha256"]
        ].to_string(index=False)
    )

print("\nSTEP 2.2 COMPLETED")
print(f"Results saved to: {output_dir}")


# %% [cell 28]

from pathlib import Path
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import cv2
import json
from tqdm.auto import tqdm

project_root = DR_RESEARCH
output_dir = project_root / "experiment02_results"

manifest_path = output_dir / "FGADR_Image_SHA256_Manifest.csv"
manifest = pd.read_csv(manifest_path, dtype={"image_id": str})

assert len(manifest) == 1842
assert manifest["image_id"].is_unique
assert all(Path(path).is_file() for path in manifest["local_path"])

PHASH_MAX_DISTANCE = 10
COSINE_MIN_SIMILARITY = 0.995
THUMBNAIL_SIZE = 32

def compute_phash(image):
    grayscale = np.asarray(
        image.convert("L").resize(
            (32, 32),
            Image.Resampling.LANCZOS
        ),
        dtype=np.float32
    )

    dct = cv2.dct(grayscale)
    frequencies = dct[:8, :8].flatten()
    threshold = np.median(frequencies[1:])
    bits = frequencies > threshold

    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)

    return result

def compute_thumbnail(image):
    grayscale = image.convert("L").resize(
        (THUMBNAIL_SIZE, THUMBNAIL_SIZE),
        Image.Resampling.LANCZOS
    )

    vector = np.asarray(
        grayscale, dtype=np.float32
    ).flatten()

    vector -= vector.mean()
    norm = np.linalg.norm(vector)

    if norm > 1e-8:
        vector /= norm

    return vector

image_ids = []
phashes = []
thumbnails = []

for row in tqdm(
    manifest.itertuples(index=False),
    total=len(manifest),
    desc="Extracting visual features"
):
    with Image.open(row.local_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        phash = compute_phash(image)
        thumbnail = compute_thumbnail(image)

    image_ids.append(row.image_id)
    phashes.append(phash)
    thumbnails.append(thumbnail)

features = np.stack(thumbnails).astype(np.float32)
similarity_matrix = features @ features.T

candidates = []
n = len(image_ids)

for i in tqdm(range(n), desc="Comparing image pairs"):
    for j in range(i + 1, n):
        if manifest.iloc[i]["sha256"] == manifest.iloc[j]["sha256"]:
            continue

        distance = (phashes[i] ^ phashes[j]).bit_count()
        similarity = float(similarity_matrix[i, j])

        if (
            distance <= PHASH_MAX_DISTANCE
            or similarity >= COSINE_MIN_SIMILARITY
        ):
            candidates.append({
                "image_id_1": image_ids[i],
                "image_id_2": image_ids[j],
                "dr_grade_1": int(manifest.iloc[i]["dr_grade"]),
                "dr_grade_2": int(manifest.iloc[j]["dr_grade"]),
                "phash_distance": distance,
                "thumbnail_cosine_similarity": similarity,
                "phash_candidate": distance <= PHASH_MAX_DISTANCE,
                "cosine_candidate": similarity >= COSINE_MIN_SIMILARITY,
                "review_status": "pending_manual_review",
            })

candidate_columns = [
    "image_id_1",
    "image_id_2",
    "dr_grade_1",
    "dr_grade_2",
    "phash_distance",
    "thumbnail_cosine_similarity",
    "phash_candidate",
    "cosine_candidate",
    "review_status",
]

candidate_df = pd.DataFrame(
    candidates, columns=candidate_columns
)

candidate_df = candidate_df.sort_values(
    ["phash_distance", "thumbnail_cosine_similarity"],
    ascending=[True, False]
).reset_index(drop=True)

output_path = (
    output_dir / "FGADR_Near_Duplicate_Candidates.csv"
)

candidate_df.to_csv(output_path, index=False)

parameters = {
    "phash_max_distance": PHASH_MAX_DISTANCE,
    "cosine_min_similarity": COSINE_MIN_SIMILARITY,
    "thumbnail_size": THUMBNAIL_SIZE,
    "comparison_pairs": n * (n - 1) // 2,
    "confirmed_duplicate_status": "not_adjudicated",
    "automatic_image_exclusion": False,
}

with open(
    output_dir / "Near_Duplicate_Parameters.json",
    "w",
    encoding="utf-8"
) as handle:
    json.dump(parameters, handle, indent=2)

print("\nNEAR-DUPLICATE SCREENING")
print(f"Images analyzed: {n}")
print(f"Total pairs compared: {n * (n - 1) // 2}")
print(f"Candidate pairs: {len(candidate_df)}")

if not candidate_df.empty:
    print("\nTop 20 candidate pairs:")
    print(
        candidate_df.head(20).to_string(index=False)
    )

print(f"\nCandidate report: {output_path}")
print("STEP 2.3 COMPLETED")


# %% [cell 29]

from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import pandas as pd
import numpy as np
import json

project_root = DR_RESEARCH
output_dir = project_root / "experiment02_results"

candidates = pd.read_csv(
    output_dir / "FGADR_Near_Duplicate_Candidates.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

manifest = pd.read_csv(
    output_dir / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

path_lookup = manifest.set_index("image_id")["local_path"].to_dict()

print("STEP 2.4 - CANDIDATE DIAGNOSTICS")
print(f"Original candidate pairs: {len(candidates)}")

rows = []

for phash_limit in [0, 2, 4, 6, 8, 10]:
    for cosine_limit in [0.990, 0.995, 0.998, 0.999]:
        selected = (
            (candidates["phash_distance"] <= phash_limit)
            & (
                candidates["thumbnail_cosine_similarity"]
                >= cosine_limit
            )
        )

        rows.append({
            "phash_max_distance": phash_limit,
            "cosine_min_similarity": cosine_limit,
            "candidate_pairs": int(selected.sum()),
        })

diagnostics = pd.DataFrame(rows)

print("\nJOINT THRESHOLD COUNTS")
print(diagnostics.to_string(index=False))

diagnostics.to_csv(
    output_dir / "FGADR_Near_Duplicate_Threshold_Diagnostics.csv",
    index=False
)

# Sample candidates from different similarity conditions
groups = [
    (
        "very_high_agreement",
        candidates[
            (candidates["phash_distance"] <= 2)
            & (candidates["thumbnail_cosine_similarity"] >= 0.998)
        ].sort_values(
            ["phash_distance", "thumbnail_cosine_similarity"],
            ascending=[True, False]
        ),
    ),
    (
        "moderate_agreement",
        candidates[
            (candidates["phash_distance"].between(3, 6))
            & (
                candidates["thumbnail_cosine_similarity"]
                >= 0.995
            )
        ].sort_values(
            ["thumbnail_cosine_similarity", "phash_distance"],
            ascending=[False, True]
        ),
    ),
    (
        "metric_disagreement",
        candidates[
            (candidates["phash_distance"] <= 2)
            & (
                candidates["thumbnail_cosine_similarity"]
                < 0.990
            )
        ].sort_values(
            ["phash_distance", "thumbnail_cosine_similarity"],
            ascending=[True, False]
        ),
    ),
]

selected_parts = []

for group_name, subset in groups:
    subset = subset.head(12).copy()
    subset["sampling_group"] = group_name
    selected_parts.append(subset)

review_pairs = pd.concat(
    selected_parts, ignore_index=True
)

review_pairs = review_pairs.drop_duplicates(
    ["image_id_1", "image_id_2"]
).reset_index(drop=True)

review_pairs["review_id"] = [
    f"R{i:03d}" for i in range(1, len(review_pairs) + 1)
]

review_pairs["manual_decision"] = "unreviewed"

review_pairs.to_csv(
    output_dir / "FGADR_Near_Duplicate_Visual_Review.csv",
    index=False
)

# Create a compact side-by-side visual review sheet
cell_width = 520
cell_height = 260
image_size = 220
columns = 2
rows_count = int(np.ceil(len(review_pairs) / columns))

sheet = Image.new(
    "RGB",
    (cell_width * columns, cell_height * rows_count),
    "white"
)

draw = ImageDraw.Draw(sheet)

for index, row in enumerate(review_pairs.itertuples(index=False)):
    column = index % columns
    row_index = index // columns

    x0 = column * cell_width
    y0 = row_index * cell_height

    pair_images = []

    for image_id in [row.image_id_1, row.image_id_2]:
        with Image.open(path_lookup[image_id]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((image_size, image_size))
            pair_images.append(image.copy())

    sheet.paste(pair_images[0], (x0 + 10, y0 + 28))
    sheet.paste(pair_images[1], (x0 + 245, y0 + 28))

    draw.text(
        (x0 + 10, y0 + 7),
        f"{row.review_id}: {row.image_id_1} / {row.image_id_2}",
        fill="black"
    )

    draw.text(
        (x0 + 10, y0 + 249),
        f"pHash={row.phash_distance}, "
        f"cosine={row.thumbnail_cosine_similarity:.5f}",
        fill="black"
    )

contact_path = (
    output_dir / "FGADR_Near_Duplicate_Contact_Sheet.png"
)

sheet.save(contact_path)

print(f"\nVisual review pairs: {len(review_pairs)}")
print(f"Contact sheet saved: {contact_path}")
print("STEP 2.4 COMPLETED")


# %% [cell 30]

from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import pandas as pd
import math

root = (DR_RESEARCH / "experiment02_results")

pairs = pd.read_csv(
    root / "FGADR_Near_Duplicate_Candidates.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

manifest = pd.read_csv(
    root / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

paths = manifest.set_index("image_id")["local_path"].to_dict()

high_priority = pairs[
    (pairs["phash_distance"] <= 4)
    & (pairs["thumbnail_cosine_similarity"] >= 0.995)
].copy()

high_priority = high_priority.sort_values(
    ["phash_distance", "thumbnail_cosine_similarity"],
    ascending=[True, False]
).reset_index(drop=True)

high_priority.insert(
    0,
    "review_id",
    [f"H{i:03d}" for i in range(1, len(high_priority) + 1)]
)

high_priority["manual_decision"] = "unreviewed"

high_priority.to_csv(
    root / "FGADR_High_Priority_Review.csv",
    index=False
)

image_size = 350
cell_width = 740
cell_height = 395
batch_size = 9

for start in range(0, len(high_priority), batch_size):
    batch = high_priority.iloc[start:start + batch_size]

    sheet = Image.new(
        "RGB",
        (cell_width, cell_height * len(batch)),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for index, pair in enumerate(batch.itertuples(index=False)):
        y = index * cell_height

        for offset, image_id in enumerate(
            [pair.image_id_1, pair.image_id_2]
        ):
            with Image.open(paths[image_id]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image = image.resize(
                    (image_size, image_size),
                    Image.Resampling.LANCZOS
                )

            sheet.paste(image, (10 + offset * 365, y + 25))

        draw.text(
            (10, y + 5),
            f"{pair.review_id}: {pair.image_id_1} / {pair.image_id_2}",
            fill="black"
        )

        draw.text(
            (10, y + 378),
            f"pHash={pair.phash_distance}, "
            f"cosine={pair.thumbnail_cosine_similarity:.5f}",
            fill="black"
        )

    batch_number = start // batch_size + 1

    output_path = (
        root / f"FGADR_High_Priority_Batch_{batch_number:02d}.png"
    )

    sheet.save(output_path)
    print(f"Saved: {output_path.name}")

print(f"\nHigh-priority candidate pairs: {len(high_priority)}")
print("STEP 2.5 COMPLETED")


# %% [cell 31] Experiment configuration

from pathlib import Path
from datetime import datetime, timezone
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import cv2
import json
import hashlib

# Experiment configuration
PROJECT_ROOT = DR_RESEARCH
OUTPUT_DIR = PROJECT_ROOT / "experiment02_results"

CANDIDATE_FILE = OUTPUT_DIR / "FGADR_Near_Duplicate_Candidates.csv"
MANIFEST_FILE = OUTPUT_DIR / "FGADR_Image_SHA256_Manifest.csv"

PHASH_LIMIT = 10
COSINE_LIMIT = 0.990

IMAGE_SIZE = 640
ORB_FEATURES = 2500
RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 3.0
MIN_MATCHES_FOR_HOMOGRAPHY = 8

cv2.setRNGSeed(42)

# Load data
candidates = pd.read_csv(
    CANDIDATE_FILE,
    dtype={"image_id_1": str, "image_id_2": str}
)

manifest = pd.read_csv(
    MANIFEST_FILE,
    dtype={"image_id": str, "sha256": str}
)

assert len(manifest) == 1842
assert manifest["image_id"].is_unique
assert manifest["sha256"].notna().all()

selected = candidates[
    (candidates["phash_distance"] <= PHASH_LIMIT)
    & (candidates["thumbnail_cosine_similarity"] >= COSINE_LIMIT)
].copy()

selected = selected.drop_duplicates(
    ["image_id_1", "image_id_2"]
).reset_index(drop=True)

print("EXPERIMENT 02 - STEP 2.6")
print(f"Original candidates: {len(candidates)}")
print(f"Selected pairs: {len(selected)}")

assert len(selected) == 299, (
    "Unexpected candidate count. Verify input files and thresholds."
)

path_lookup = manifest.set_index("image_id")["local_path"].to_dict()
hash_lookup = manifest.set_index("image_id")["sha256"].to_dict()

required_ids = (
    set(selected["image_id_1"])
    | set(selected["image_id_2"])
)

assert required_ids.issubset(path_lookup.keys())

# Initialize feature extractor
orb = cv2.ORB_create(
    nfeatures=ORB_FEATURES,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    patchSize=31,
    fastThreshold=15
)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

def extract_features(image_path):
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(f"Image loading failed: {image_path}")

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    image = clahe.apply(image)

    keypoints, descriptors = orb.detectAndCompute(
        image, None
    )

    return keypoints, descriptors

# Extract each image only once
feature_cache = {}
feature_errors = []

for image_id in tqdm(
    sorted(required_ids),
    desc="Extracting ORB features"
):
    try:
        feature_cache[image_id] = extract_features(
            path_lookup[image_id]
        )
    except Exception as exc:
        feature_errors.append({
            "image_id": image_id,
            "error": str(exc)
        })

if feature_errors:
    pd.DataFrame(feature_errors).to_csv(
        OUTPUT_DIR / "FGADR_ORB_Feature_Errors.csv",
        index=False
    )
    raise RuntimeError(
        f"Feature extraction failed for {len(feature_errors)} images."
    )

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def evaluate_pair(image_id_1, image_id_2):
    keypoints_1, descriptors_1 = feature_cache[image_id_1]
    keypoints_2, descriptors_2 = feature_cache[image_id_2]

    result = {
        "keypoints_1": len(keypoints_1),
        "keypoints_2": len(keypoints_2),
        "good_matches": 0,
        "ransac_inliers": 0,
        "inlier_ratio": 0.0,
        "median_reprojection_error_px": np.nan,
        "homography_valid": False,
    }

    if descriptors_1 is None or descriptors_2 is None:
        return result

    if len(descriptors_1) < 2 or len(descriptors_2) < 2:
        return result

    raw_matches = matcher.knnMatch(
        descriptors_1,
        descriptors_2,
        k=2
    )

    good_matches = []

    for neighbors in raw_matches:
        if len(neighbors) != 2:
            continue

        first, second = neighbors

        if first.distance < RATIO_THRESHOLD * second.distance:
            good_matches.append(first)

    result["good_matches"] = len(good_matches)

    if len(good_matches) < MIN_MATCHES_FOR_HOMOGRAPHY:
        return result

    points_1 = np.float32([
        keypoints_1[match.queryIdx].pt
        for match in good_matches
    ]).reshape(-1, 1, 2)

    points_2 = np.float32([
        keypoints_2[match.trainIdx].pt
        for match in good_matches
    ]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(
        points_1,
        points_2,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
        maxIters=2000,
        confidence=0.995
    )

    if H is None or inlier_mask is None:
        return result

    if not np.isfinite(H).all():
        return result

    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())

    result["ransac_inliers"] = inlier_count
    result["inlier_ratio"] = (
        inlier_count / len(good_matches)
    )

    if inlier_count < 4:
        return result

    projected = cv2.perspectiveTransform(
        points_1,
        H
    )

    errors = np.linalg.norm(
        projected.reshape(-1, 2)
        - points_2.reshape(-1, 2),
        axis=1
    )

    result["median_reprojection_error_px"] = float(
        np.median(errors[inliers])
    )

    result["homography_valid"] = True

    return result

# Evaluate all selected pairs
results = []
pair_errors = []

for pair in tqdm(
    selected.itertuples(index=False),
    total=len(selected),
    desc="Running ORB and RANSAC"
):
    try:
        pair_result = evaluate_pair(
            pair.image_id_1,
            pair.image_id_2
        )

        results.append({
            "image_id_1": pair.image_id_1,
            "image_id_2": pair.image_id_2,
            "dr_grade_1": int(pair.dr_grade_1),
            "dr_grade_2": int(pair.dr_grade_2),
            "phash_distance": int(pair.phash_distance),
            "thumbnail_cosine_similarity": float(
                pair.thumbnail_cosine_similarity
            ),
            "same_sha256": (
                hash_lookup[pair.image_id_1]
                == hash_lookup[pair.image_id_2]
            ),
            **pair_result,
            "review_status": "pending_manual_review",
        })

    except Exception as exc:
        pair_errors.append({
            "image_id_1": pair.image_id_1,
            "image_id_2": pair.image_id_2,
            "error": str(exc)
        })

result_df = pd.DataFrame(results)

error_columns = ["image_id_1", "image_id_2", "error"]
error_df = pd.DataFrame(pair_errors, columns=error_columns)

result_path = OUTPUT_DIR / "FGADR_ORB_RANSAC_Results.csv"
error_path = OUTPUT_DIR / "FGADR_ORB_RANSAC_Errors.csv"

result_df.to_csv(result_path, index=False)
error_df.to_csv(error_path, index=False)

# Validate processing completeness
assert len(result_df) + len(error_df) == len(selected)

if not error_df.empty:
    print("\nPair processing errors:")
    print(error_df.head(10).to_string(index=False))
    raise RuntimeError("Some candidate pairs failed evaluation.")

assert not result_df["same_sha256"].any()

# Exploratory ranking: not a duplicate confirmation
result_df = result_df.sort_values(
    [
        "ransac_inliers",
        "inlier_ratio",
        "median_reprojection_error_px",
    ],
    ascending=[False, False, True]
).reset_index(drop=True)

result_df.to_csv(result_path, index=False)

parameters = {
    "experiment": "Experiment02_Step2.6",
    "phash_limit": PHASH_LIMIT,
    "cosine_limit": COSINE_LIMIT,
    "image_size": IMAGE_SIZE,
    "orb_features": ORB_FEATURES,
    "ratio_threshold": RATIO_THRESHOLD,
    "ransac_threshold_px": RANSAC_THRESHOLD,
    "min_matches_for_homography": MIN_MATCHES_FOR_HOMOGRAPHY,
    "opencv_version": cv2.__version__,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "random_seed": 42,
    "automatic_duplicate_confirmation": False,
    "created_utc": datetime.now(timezone.utc).isoformat(),
}

with open(
    OUTPUT_DIR / "FGADR_ORB_RANSAC_Parameters.json",
    "w",
    encoding="utf-8"
) as handle:
    json.dump(parameters, handle, indent=2)

print("\nORB + RANSAC VERIFICATION")
print(f"Pairs evaluated: {len(result_df)}")
print(f"Processing errors: {len(error_df)}")
print(f"Pairs with valid homography: {int(result_df['homography_valid'].sum())}")
print(f"Pairs with at least 10 inliers: {int((result_df['ransac_inliers'] >= 10).sum())}")
print(f"Pairs with at least 20 inliers: {int((result_df['ransac_inliers'] >= 20).sum())}")
print(f"Pairs with at least 40 inliers: {int((result_df['ransac_inliers'] >= 40).sum())}")

print("\nTOP 20 GEOMETRIC MATCH CANDIDATES")
display_columns = [
    "image_id_1",
    "image_id_2",
    "dr_grade_1",
    "dr_grade_2",
    "good_matches",
    "ransac_inliers",
    "inlier_ratio",
    "median_reprojection_error_px",
]

print(result_df[display_columns].head(20).to_string(index=False))

print(f"\nResults saved to: {result_path}")
print("STEP 2.6 COMPLETED")


# %% [cell 32]

from pathlib import Path
import pandas as pd
import numpy as np

root = (DR_RESEARCH / "experiment02_results")

results = pd.read_csv(
    root / "FGADR_ORB_RANSAC_Results.csv",
    dtype={
        "image_id_1": str,
        "image_id_2": str
    }
)

assert len(results) == 299
assert results[["image_id_1", "image_id_2"]].notna().all().all()
assert not results.duplicated(
    ["image_id_1", "image_id_2"]
).any()

results["grade_conflict"] = (
    results["dr_grade_1"] != results["dr_grade_2"]
)

print("GEOMETRIC EVIDENCE DISTRIBUTION")
print(
    results[
        [
            "good_matches",
            "ransac_inliers",
            "inlier_ratio",
            "median_reprojection_error_px"
        ]
    ].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    ).to_string()
)

print("\nINLIER COUNT BANDS")

bands = pd.cut(
    results["ransac_inliers"],
    bins=[-1, 0, 9, 19, 39, 99, 299, float("inf")],
    labels=[
        "0",
        "1-9",
        "10-19",
        "20-39",
        "40-99",
        "100-299",
        "300+"
    ]
)

print(bands.value_counts().sort_index().to_string())

print("\nLABEL CONFLICT ANALYSIS")
print(f"Total grade-conflicting pairs: {results['grade_conflict'].sum()}")

for minimum_inliers in [40, 100, 300, 500]:
    subset = results[
        results["ransac_inliers"] >= minimum_inliers
    ]

    conflicts = subset[subset["grade_conflict"]]

    print(
        f"Inliers >= {minimum_inliers}: "
        f"{len(subset)} pairs, "
        f"{len(conflicts)} grade conflicts"
    )

# Exploratory ranking only, not automatic confirmation
priority = results[
    (results["ransac_inliers"] >= 100)
    & (results["inlier_ratio"] >= 0.70)
    & (results["grade_conflict"])
].copy()

priority = priority.sort_values(
    ["ransac_inliers", "inlier_ratio"],
    ascending=[False, False]
)

priority["review_status"] = "pending_manual_verification"

priority_path = (
    root / "FGADR_Geometric_Label_Conflict_Review.csv"
)

priority.to_csv(priority_path, index=False)

print("\nTOP GRADE-CONFLICT REVIEW CANDIDATES")
print(
    priority[
        [
            "image_id_1",
            "image_id_2",
            "dr_grade_1",
            "dr_grade_2",
            "ransac_inliers",
            "inlier_ratio",
            "median_reprojection_error_px"
        ]
    ].head(25).to_string(index=False)
)

print(f"\nPriority review pairs: {len(priority)}")
print(f"Report saved: {priority_path}")
print("STEP 2.7 COMPLETED")


# %% [cell 33]

from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

root = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    root / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str, "sha256": str}
)

orb = pd.read_csv(
    root / "FGADR_ORB_RANSAC_Results.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

assert len(manifest) == 1842
assert manifest["image_id"].is_unique
assert len(orb) == 299

# Exploratory evidence criteria, not duplicate confirmation
MIN_INLIERS = 100
MIN_INLIER_RATIO = 0.70

parent = {image_id: image_id for image_id in manifest["image_id"]}

def find(node):
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node

def union(a, b):
    root_a = find(a)
    root_b = find(b)
    if root_a != root_b:
        parent[root_b] = root_a

# Confirmed byte-identical relations
exact_edges = []

for sha256, group in manifest.groupby("sha256"):
    ids = sorted(group["image_id"].tolist())

    if len(ids) > 1:
        for image_id in ids[1:]:
            union(ids[0], image_id)
            exact_edges.append((ids[0], image_id))

# Provisional geometric relations
strong = orb[
    (orb["ransac_inliers"] >= MIN_INLIERS)
    & (orb["inlier_ratio"] >= MIN_INLIER_RATIO)
    & (orb["homography_valid"] == True)
].copy()

for row in strong.itertuples(index=False):
    union(row.image_id_1, row.image_id_2)

groups = defaultdict(list)

for image_id in manifest["image_id"]:
    groups[find(image_id)].append(image_id)

grade_lookup = manifest.set_index("image_id")["dr_grade"].to_dict()

group_records = []
member_records = []

for members in groups.values():
    members = sorted(members)
    grades = sorted(set(int(grade_lookup[i]) for i in members))
    group_id = f"PG_{members[0]}"

    group_records.append({
        "provisional_group_id": group_id,
        "image_count": len(members),
        "grade_count": len(grades),
        "grades": ";".join(map(str, grades)),
        "grade_conflict": len(grades) > 1,
        "image_ids": ";".join(members),
        "status": "provisional_not_confirmed"
    })

    for image_id in members:
        member_records.append({
            "image_id": image_id,
            "dr_grade": int(grade_lookup[image_id]),
            "provisional_group_id": group_id,
            "group_size": len(members),
            "group_grade_conflict": len(grades) > 1
        })

group_df = pd.DataFrame(group_records)
member_df = pd.DataFrame(member_records)

group_df = group_df.sort_values(
    ["image_count", "provisional_group_id"],
    ascending=[False, True]
)

group_df.to_csv(
    root / "FGADR_Provisional_Source_Groups.csv",
    index=False
)

member_df.to_csv(
    root / "FGADR_Provisional_Group_Members.csv",
    index=False
)

assert len(member_df) == 1842
assert member_df["image_id"].is_unique
assert sum(group_df["image_count"]) == 1842

print("PROVISIONAL SOURCE-GROUP AUDIT")
print(f"Total images: {len(member_df)}")
print(f"Exact duplicate relations: {len(exact_edges)}")
print(f"Strong geometric candidate relations: {len(strong)}")
print(f"Provisional groups: {len(group_df)}")
print(f"Multi-image groups: {(group_df['image_count'] > 1).sum()}")
print(f"Images in multi-image groups: {group_df.loc[group_df['image_count'] > 1, 'image_count'].sum()}")
print(f"Groups with grade conflicts: {group_df['grade_conflict'].sum()}")
print(f"Images in grade-conflicting groups: {group_df.loc[group_df['grade_conflict'], 'image_count'].sum()}")
print(f"Largest group size: {group_df['image_count'].max()}")

print("\nLARGEST PROVISIONAL GROUPS")
print(group_df.head(15).to_string(index=False))

print("\nSTEP 2.8 COMPLETED")


# %% [cell 34]

from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import pandas as pd
import math

root = (DR_RESEARCH / "experiment02_results")

review_dir = root / "source_group_review"
review_dir.mkdir(parents=True, exist_ok=True)

groups = pd.read_csv(
    root / "FGADR_Provisional_Source_Groups.csv"
)

manifest = pd.read_csv(
    root / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

assert len(manifest) == 1842
assert manifest["image_id"].is_unique

path_lookup = manifest.set_index("image_id")["local_path"].to_dict()
grade_lookup = manifest.set_index("image_id")["dr_grade"].to_dict()

conflict_groups = groups[
    groups["grade_conflict"].astype(bool)
].sort_values(
    ["image_count", "provisional_group_id"],
    ascending=[False, True]
)

assert len(conflict_groups) == 30

review_groups = conflict_groups.head(10)

image_size = 280
cell_width = 310
cell_height = 320
columns = 3

review_records = []

for group in review_groups.itertuples(index=False):
    image_ids = group.image_ids.split(";")
    row_count = math.ceil(len(image_ids) / columns)

    sheet = Image.new(
        "RGB",
        (columns * cell_width, row_count * cell_height + 45),
        "white"
    )

    draw = ImageDraw.Draw(sheet)
    draw.text(
        (12, 12),
        f"{group.provisional_group_id} | "
        f"Images: {len(image_ids)} | Grades: {group.grades}",
        fill="black"
    )

    for index, image_id in enumerate(image_ids):
        image_path = Path(path_lookup[image_id])

        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail(
                (image_size, image_size),
                Image.Resampling.LANCZOS
            )
            image = image.copy()

        col_index = index % columns
        row_index = index // columns

        x = col_index * cell_width + 12
        y = row_index * cell_height + 50

        sheet.paste(image, (x, y))

        draw.text(
            (x, y + 285),
            f"{image_id} | DR{grade_lookup[image_id]}",
            fill="black"
        )

    filename = f"{group.provisional_group_id}.png"
    output_path = review_dir / filename
    sheet.save(output_path)

    review_records.append({
        "provisional_group_id": group.provisional_group_id,
        "image_count": len(image_ids),
        "grades": group.grades,
        "image_ids": group.image_ids,
        "review_image": filename,
        "source_group_decision": "unreviewed",
        "review_notes": "",
    })

review_manifest = pd.DataFrame(review_records)

review_manifest.to_csv(
    review_dir / "Initial_Conflict_Group_Review.csv",
    index=False
)

print("SOURCE-GROUP REVIEW PREPARATION")
print(f"Total conflicting groups: {len(conflict_groups)}")
print(f"Groups prepared for initial review: {len(review_manifest)}")
print(f"Review directory: {review_dir}")

print("\nPREPARED GROUPS")
print(
    review_manifest[
        [
            "provisional_group_id",
            "image_count",
            "grades",
            "review_image",
        ]
    ].to_string(index=False)
)

print("\nSTEP 2.9 COMPLETED")


# %% [cell 35]

from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from collections import defaultdict
from tqdm.auto import tqdm
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import zipfile
import json
import math
import shutil

project_root = DR_RESEARCH
result_root = project_root / "experiment02_results"

local_review_root = (LOCAL_DISK / "FGADR_Source_Group_Review")
figure_dir = local_review_root / "figures"

figure_dir.mkdir(parents=True, exist_ok=True)

groups = pd.read_csv(
    result_root / "FGADR_Provisional_Source_Groups.csv",
    dtype={"provisional_group_id": str}
)

members = pd.read_csv(
    result_root / "FGADR_Provisional_Group_Members.csv",
    dtype={"image_id": str}
)

manifest = pd.read_csv(
    result_root / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str, "sha256": str}
)

orb = pd.read_csv(
    result_root / "FGADR_ORB_RANSAC_Results.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

assert len(manifest) == 1842
assert len(members) == 1842
assert manifest["image_id"].is_unique
assert members["image_id"].is_unique
assert members["provisional_group_id"].notna().all()
assert len(orb) == 299

path_lookup = manifest.set_index("image_id")["local_path"].to_dict()
grade_lookup = manifest.set_index("image_id")["dr_grade"].to_dict()
hash_lookup = manifest.set_index("image_id")["sha256"].to_dict()
group_lookup = members.set_index("image_id")["provisional_group_id"].to_dict()

multi_groups = groups[
    groups["image_count"] > 1
].copy()

assert len(multi_groups) == 90
assert int(multi_groups["image_count"].sum()) == 231

# Preserve geometric evidence separately from confirmed exact duplicates
strong_orb = orb[
    (orb["ransac_inliers"] >= 100)
    & (orb["inlier_ratio"] >= 0.70)
    & (orb["homography_valid"] == True)
].copy()

edge_records = []

for row in strong_orb.itertuples(index=False):
    group_1 = group_lookup[row.image_id_1]
    group_2 = group_lookup[row.image_id_2]

    assert group_1 == group_2

    edge_records.append({
        "group_id": group_1,
        "image_id_1": row.image_id_1,
        "image_id_2": row.image_id_2,
        "evidence_type": "strong_geometric_candidate",
        "ransac_inliers": int(row.ransac_inliers),
        "inlier_ratio": float(row.inlier_ratio),
        "median_reprojection_error_px": float(
            row.median_reprojection_error_px
        ),
        "review_decision": "unreviewed"
    })

for sha256, subset in manifest.groupby("sha256"):
    ids = sorted(subset["image_id"].tolist())

    if len(ids) < 2:
        continue

    for index in range(1, len(ids)):
        edge_records.append({
            "group_id": group_lookup[ids[0]],
            "image_id_1": ids[0],
            "image_id_2": ids[index],
            "evidence_type": "exact_sha256",
            "ransac_inliers": np.nan,
            "inlier_ratio": np.nan,
            "median_reprojection_error_px": np.nan,
            "review_decision": "confirmed_byte_identical"
        })

edge_df = pd.DataFrame(edge_records)

# Validate the relationship graph before producing review materials
assert len(strong_orb) == 168
assert len(edge_df[edge_df["evidence_type"] == "exact_sha256"]) == 14

group_records = []

IMAGE_SIZE = 290
CELL_WIDTH = 320
CELL_HEIGHT = 330
COLUMNS = 3

sorted_groups = multi_groups.sort_values(
    ["grade_conflict", "image_count", "provisional_group_id"],
    ascending=[False, False, True]
)

for group in tqdm(
    sorted_groups.itertuples(index=False),
    total=len(sorted_groups),
    desc="Preparing group review figures"
):
    image_ids = sorted(
        members.loc[
            members["provisional_group_id"] == group.provisional_group_id,
            "image_id"
        ].tolist()
    )

    assert len(image_ids) == int(group.image_count)

    row_count = math.ceil(len(image_ids) / COLUMNS)

    sheet = Image.new(
        "RGB",
        (
            CELL_WIDTH * COLUMNS,
            CELL_HEIGHT * row_count + 55
        ),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    draw.text(
        (12, 12),
        f"{group.provisional_group_id} | "
        f"Images: {len(image_ids)} | "
        f"Grades: {group.grades}",
        fill="black"
    )

    for index, image_id in enumerate(image_ids):
        image_path = Path(path_lookup[image_id])

        if not image_path.is_file():
            raise FileNotFoundError(str(image_path))

        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.LANCZOS
            )
            image = image.copy()

        x = (index % COLUMNS) * CELL_WIDTH + 12
        y = (index // COLUMNS) * CELL_HEIGHT + 50

        sheet.paste(image, (x, y))

        draw.text(
            (x, y + 296),
            f"{image_id} | DR{int(grade_lookup[image_id])}",
            fill="black"
        )

    figure_name = f"{group.provisional_group_id}.png"

    sheet.save(
        figure_dir / figure_name,
        optimize=True
    )

    group_edges = edge_df[
        edge_df["group_id"] == group.provisional_group_id
    ]

    exact_count = int(
        (group_edges["evidence_type"] == "exact_sha256").sum()
    )

    geometric_count = int(
        (
            group_edges["evidence_type"]
            == "strong_geometric_candidate"
        ).sum()
    )

    group_records.append({
        "group_id": group.provisional_group_id,
        "image_count": len(image_ids),
        "grade_count": int(group.grade_count),
        "grades": str(group.grades),
        "grade_conflict": bool(group.grade_conflict),
        "image_ids": ";".join(image_ids),
        "exact_evidence_edges": exact_count,
        "geometric_candidate_edges": geometric_count,
        "figure_filename": figure_name,
        "group_review_decision": "unreviewed",
        "reviewer_id": "",
        "review_date": "",
        "review_notes": "",
    })

review_df = pd.DataFrame(group_records)

review_df.to_csv(
    local_review_root / "FGADR_Source_Group_Review_Manifest.csv",
    index=False
)

edge_df.to_csv(
    local_review_root / "FGADR_Source_Group_Evidence_Edges.csv",
    index=False
)

parameters = {
    "experiment": "Experiment02_Step2.10",
    "source_groups": 90,
    "images_in_multimage_groups": 231,
    "exact_evidence_relations": 14,
    "geometric_candidate_relations": 168,
    "image_size": IMAGE_SIZE,
    "automatic_duplicate_confirmation": False,
    "group_reviews_completed": False,
    "created_utc": datetime.now(timezone.utc).isoformat()
}

with open(
    local_review_root / "Review_Parameters.json",
    "w",
    encoding="utf-8"
) as handle:
    json.dump(parameters, handle, indent=2)

# Bundle all review figures into one archive
zip_path = result_root / "FGADR_Source_Group_Review_Pack.zip"

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=3
) as archive:
    for path in sorted(local_review_root.rglob("*")):
        if path.is_file():
            archive.write(
                path,
                arcname=str(path.relative_to(local_review_root))
            )

# Save immediately accessible review tables
review_df.to_csv(
    result_root / "FGADR_Source_Group_Review_Manifest.csv",
    index=False
)

edge_df.to_csv(
    result_root / "FGADR_Source_Group_Evidence_Edges.csv",
    index=False
)

# Validate the archive
with zipfile.ZipFile(zip_path, "r") as archive:
    bad_member = archive.testzip()
    figure_count = sum(
        name.startswith("figures/") and name.endswith(".png")
        for name in archive.namelist()
    )

assert bad_member is None
assert figure_count == 90
assert len(review_df) == 90
assert len(edge_df) == 182

print("\nSOURCE-GROUP REVIEW PACKAGE")
print(f"Multi-image groups: {len(review_df)}")
print(f"Images covered: {review_df['image_count'].sum()}")
print(f"Grade-conflicting groups: {int(review_df['grade_conflict'].sum())}")
print(f"Exact duplicate evidence edges: {int((edge_df['evidence_type'] == 'exact_sha256').sum())}")
print(f"Geometric candidate edges: {int((edge_df['evidence_type'] == 'strong_geometric_candidate').sum())}")
print(f"Review figures: {figure_count}")
print(f"Archive integrity verified: {bad_member is None}")
print(f"Archive size: {zip_path.stat().st_size / (1024 ** 2):.2f} MiB")
print(f"Archive: {zip_path}")

print("\nSTEP 2.10 COMPLETED")


# %% [cell 36]

from pathlib import Path
import pandas as pd

root = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    root / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

candidates = pd.read_csv(
    root / "FGADR_Near_Duplicate_Candidates.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

orb_results = pd.read_csv(
    root / "FGADR_ORB_RANSAC_Results.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

available_images = sum(
    Path(path).is_file()
    for path in manifest["local_path"]
)

all_pairs = set(
    zip(candidates["image_id_1"], candidates["image_id_2"])
)

checked_pairs = set(
    zip(orb_results["image_id_1"], orb_results["image_id_2"])
)

unchecked_pairs = all_pairs - checked_pairs

print("EXPANDED SCREENING PREFLIGHT")
print(f"Total images: {len(manifest)}")
print(f"Available local images: {available_images}")
print(f"Initial candidate pairs: {len(all_pairs)}")
print(f"ORB-evaluated pairs: {len(checked_pairs)}")
print(f"Remaining candidate pairs: {len(unchecked_pairs)}")

assert len(manifest) == 1842
assert available_images == 1842
assert len(all_pairs) == 43878
assert len(checked_pairs) == 299
assert checked_pairs.issubset(all_pairs)

print("\nPREFLIGHT PASSED")


# %% [cell 37]

from pathlib import Path
from datetime import datetime, timezone
from tqdm.auto import tqdm
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import cv2
import json
import time

ROOT = (DR_RESEARCH / "experiment02_results")
SEED = 42
PILOT_LIMIT = 1000
IMAGE_SIZE = 640
ORB_FEATURES = 2500
RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 3.0

cv2.setRNGSeed(SEED)

candidates = pd.read_csv(
    ROOT / "FGADR_Near_Duplicate_Candidates.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

previous = pd.read_csv(
    ROOT / "FGADR_ORB_RANSAC_Results.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

assert len(candidates) == 43878
assert len(previous) == 299
assert manifest["image_id"].is_unique

def pair_key(a, b):
    return tuple(sorted((str(a), str(b))))

checked = {
    pair_key(a, b)
    for a, b in zip(previous.image_id_1, previous.image_id_2)
}

candidates["pair_key"] = [
    pair_key(a, b)
    for a, b in zip(candidates.image_id_1, candidates.image_id_2)
]

assert candidates["pair_key"].is_unique

remaining = candidates[
    ~candidates["pair_key"].isin(checked)
].copy()

assert len(remaining) == 43579

remaining["phash_band"] = pd.cut(
    remaining["phash_distance"],
    bins=[-1, 2, 4, 6, 8, 10, 64],
    labels=["0-2", "3-4", "5-6", "7-8", "9-10", "11-64"]
)

rng = np.random.default_rng(SEED)
bands = list(remaining["phash_band"].cat.categories)
selected_indices = []
available = remaining.copy()

for band in bands:
    subset = available[available["phash_band"] == band]
    target = PILOT_LIMIT // len(bands)
    count = min(target, len(subset))

    if count:
        chosen = rng.choice(
            subset.index.to_numpy(),
            size=count,
            replace=False
        )
        selected_indices.extend(chosen.tolist())

available = available.drop(index=selected_indices)
additional_count = min(
    PILOT_LIMIT - len(selected_indices),
    len(available)
)

if additional_count:
    chosen = rng.choice(
        available.index.to_numpy(),
        size=additional_count,
        replace=False
    )
    selected_indices.extend(chosen.tolist())

pilot = remaining.loc[selected_indices].copy()
pilot = pilot.sort_values("pair_key").reset_index(drop=True)

assert len(pilot) == PILOT_LIMIT
assert pilot["pair_key"].is_unique

paths = manifest.set_index("image_id")["local_path"].to_dict()

orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

required_ids = sorted(
    set(pilot["image_id_1"]) | set(pilot["image_id_2"])
)

feature_cache = {}

for image_id in tqdm(required_ids, desc="Extracting pilot features"):
    image = cv2.imread(paths[image_id], cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Cannot decode image: {image_id}")

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )
    image = clahe.apply(image)

    feature_cache[image_id] = orb.detectAndCompute(image, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def ratio_matches(first, second):
    matches = matcher.knnMatch(first, second, k=2)
    accepted = {}

    for neighbors in matches:
        if len(neighbors) != 2:
            continue

        best, runner_up = neighbors

        if best.distance < RATIO_THRESHOLD * runner_up.distance:
            accepted[(best.queryIdx, best.trainIdx)] = best

    return accepted

def evaluate_pair(image_id_1, image_id_2):
    kp1, desc1 = feature_cache[image_id_1]
    kp2, desc2 = feature_cache[image_id_2]

    result = {
        "mutual_matches": 0,
        "ransac_inliers": 0,
        "inlier_ratio": 0.0,
        "median_reprojection_error_px": np.nan,
        "homography_valid": False
    }

    if desc1 is None or desc2 is None:
        return result

    if len(desc1) < 2 or len(desc2) < 2:
        return result

    forward = ratio_matches(desc1, desc2)
    backward = ratio_matches(desc2, desc1)

    mutual = [
        match
        for (i, j), match in forward.items()
        if (j, i) in backward
    ]

    result["mutual_matches"] = len(mutual)

    if len(mutual) < 8:
        return result

    points1 = np.float32([
        kp1[m.queryIdx].pt for m in mutual
    ]).reshape(-1, 1, 2)

    points2 = np.float32([
        kp2[m.trainIdx].pt for m in mutual
    ]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
        maxIters=2000,
        confidence=0.995
    )

    if matrix is None or mask is None:
        return result

    if not np.isfinite(matrix).all():
        return result

    inliers = mask.ravel().astype(bool)
    count = int(inliers.sum())

    result["ransac_inliers"] = count
    result["inlier_ratio"] = count / len(mutual)

    if count < 4:
        return result

    projected = cv2.perspectiveTransform(points1, matrix)
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - points2.reshape(-1, 2),
        axis=1
    )

    result["median_reprojection_error_px"] = float(
        np.median(errors[inliers])
    )
    result["homography_valid"] = True

    return result

results = []
errors = []
start_time = time.perf_counter()

for pair in tqdm(
    pilot.itertuples(index=False),
    total=len(pilot),
    desc="Evaluating pilot pairs"
):
    try:
        measurements = evaluate_pair(
            pair.image_id_1,
            pair.image_id_2
        )

        results.append({
            "image_id_1": pair.image_id_1,
            "image_id_2": pair.image_id_2,
            "dr_grade_1": int(pair.dr_grade_1),
            "dr_grade_2": int(pair.dr_grade_2),
            "phash_distance": int(pair.phash_distance),
            "thumbnail_cosine_similarity": float(
                pair.thumbnail_cosine_similarity
            ),
            **measurements,
            "review_status": "not_adjudicated"
        })

    except Exception as exc:
        errors.append({
            "image_id_1": pair.image_id_1,
            "image_id_2": pair.image_id_2,
            "error": str(exc)
        })

elapsed = time.perf_counter() - start_time

result_df = pd.DataFrame(results)
error_df = pd.DataFrame(
    errors,
    columns=["image_id_1", "image_id_2", "error"]
)

result_df.to_csv(
    ROOT / "FGADR_Expanded_ORB_Pilot_Results.csv",
    index=False
)
error_df.to_csv(
    ROOT / "FGADR_Expanded_ORB_Pilot_Errors.csv",
    index=False
)

pilot[
    ["image_id_1", "image_id_2", "phash_band"]
].to_csv(
    ROOT / "FGADR_Expanded_ORB_Pilot_Sampling.csv",
    index=False
)

config = {
    "seed": SEED,
    "pilot_limit": PILOT_LIMIT,
    "image_size": IMAGE_SIZE,
    "orb_features": ORB_FEATURES,
    "ratio_threshold": RATIO_THRESHOLD,
    "ransac_threshold_px": RANSAC_THRESHOLD,
    "mutual_ratio_matching": True,
    "opencv_version": cv2.__version__,
    "automatic_duplicate_confirmation": False,
    "created_utc": datetime.now(timezone.utc).isoformat()
}

with open(
    ROOT / "FGADR_Expanded_ORB_Pilot_Parameters.json",
    "w",
    encoding="utf-8"
) as handle:
    json.dump(config, handle, indent=2)

print("\nEXPANDED ORB PILOT RESULTS")
print(f"Pilot pairs: {len(pilot)}")
print(f"Successful evaluations: {len(result_df)}")
print(f"Processing errors: {len(error_df)}")
print(f"Elapsed seconds: {elapsed:.2f}")
print(f"Seconds per pair: {elapsed / len(pilot):.4f}")
print(f"Pairs with at least 40 inliers: {int((result_df['ransac_inliers'] >= 40).sum())}")
print(f"Pairs with at least 100 inliers: {int((result_df['ransac_inliers'] >= 100).sum())}")

print("\nRESULTS BY PHASH BAND")
pilot_summary = pilot[
    ["image_id_1", "image_id_2", "phash_band"]
].merge(
    result_df[
        ["image_id_1", "image_id_2", "ransac_inliers"]
    ],
    on=["image_id_1", "image_id_2"],
    how="left",
    validate="one_to_one"
)

summary = pilot_summary.groupby(
    "phash_band", observed=True
).agg(
    sampled_pairs=("image_id_1", "size"),
    high_inlier_pairs=(
        "ransac_inliers",
        lambda values: int((values >= 100).sum())
    )
)

print(summary.to_string())
print("\nSTEP 2.11B PILOT COMPLETED")


# %% [cell 38]

from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import cv2
import json
import time
import hashlib

ROOT = (DR_RESEARCH / "experiment02_results")
BATCH_DIR = ROOT / "expanded_orb_batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 3000
IMAGE_SIZE = 640
ORB_FEATURES = 2500
RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 3.0

cv2.setRNGSeed(SEED)

candidates = pd.read_csv(
    ROOT / "FGADR_Near_Duplicate_Candidates.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

assert len(candidates) == 43878
assert len(manifest) == 1842
assert manifest["image_id"].is_unique

candidates = candidates.sort_values(
    ["image_id_1", "image_id_2"]
).reset_index(drop=True)

assert not candidates.duplicated(
    ["image_id_1", "image_id_2"]
).any()

paths = manifest.set_index("image_id")["local_path"].to_dict()

required_ids = sorted(
    set(candidates["image_id_1"])
    | set(candidates["image_id_2"])
)

orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

feature_cache = {}

for image_id in tqdm(
    required_ids,
    desc="Extracting all candidate features"
):
    image = cv2.imread(paths[image_id], cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Cannot decode image: {image_id}")

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    image = clahe.apply(image)
    feature_cache[image_id] = orb.detectAndCompute(image, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def ratio_matches(first, second):
    raw = matcher.knnMatch(first, second, k=2)
    accepted = {}

    for neighbors in raw:
        if len(neighbors) != 2:
            continue

        best, runner_up = neighbors

        if best.distance < RATIO_THRESHOLD * runner_up.distance:
            accepted[(best.queryIdx, best.trainIdx)] = best

    return accepted

def evaluate_pair(a, b):
    kp1, desc1 = feature_cache[a]
    kp2, desc2 = feature_cache[b]

    result = {
        "mutual_matches": 0,
        "ransac_inliers": 0,
        "inlier_ratio": 0.0,
        "median_reprojection_error_px": np.nan,
        "homography_valid": False
    }

    if desc1 is None or desc2 is None:
        return result

    if len(desc1) < 2 or len(desc2) < 2:
        return result

    forward = ratio_matches(desc1, desc2)
    backward = ratio_matches(desc2, desc1)

    mutual = [
        match
        for (i, j), match in forward.items()
        if (j, i) in backward
    ]

    result["mutual_matches"] = len(mutual)

    if len(mutual) < 8:
        return result

    points1 = np.float32([
        kp1[m.queryIdx].pt for m in mutual
    ]).reshape(-1, 1, 2)

    points2 = np.float32([
        kp2[m.trainIdx].pt for m in mutual
    ]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
        maxIters=2000,
        confidence=0.995
    )

    if matrix is None or mask is None:
        return result

    if not np.isfinite(matrix).all():
        return result

    inliers = mask.ravel().astype(bool)
    count = int(inliers.sum())

    result["ransac_inliers"] = count
    result["inlier_ratio"] = count / len(mutual)

    if count < 4:
        return result

    projected = cv2.perspectiveTransform(points1, matrix)

    errors = np.linalg.norm(
        projected.reshape(-1, 2) - points2.reshape(-1, 2),
        axis=1
    )

    result["median_reprojection_error_px"] = float(
        np.median(errors[inliers])
    )

    result["homography_valid"] = True

    return result

config = {
    "version": "expanded_mutual_orb_v1",
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "image_size": IMAGE_SIZE,
    "orb_features": ORB_FEATURES,
    "ratio_threshold": RATIO_THRESHOLD,
    "ransac_threshold": RANSAC_THRESHOLD,
    "opencv_version": cv2.__version__,
    "total_candidate_pairs": len(candidates)
}

config_text = json.dumps(config, sort_keys=True)
config_hash = hashlib.sha256(config_text.encode()).hexdigest()

config_path = BATCH_DIR / "configuration.json"

if config_path.exists():
    existing = json.loads(config_path.read_text())
    assert existing == config, "Existing batch configuration differs."
else:
    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )

print("EXPANDED ORB FULL SCREENING")
print(f"Candidate pairs: {len(candidates)}")
print(f"Unique candidate images: {len(required_ids)}")
print(f"Configuration hash: {config_hash}")

total_batches = (
    len(candidates) + BATCH_SIZE - 1
) // BATCH_SIZE

for batch_index in range(total_batches):
    start = batch_index * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(candidates))

    batch = candidates.iloc[start:end]
    result_path = BATCH_DIR / f"batch_{batch_index:03d}.csv"

    if result_path.exists():
        existing = pd.read_csv(
            result_path,
            dtype={"image_id_1": str, "image_id_2": str}
        )

        if (
            len(existing) == len(batch)
            and existing["image_id_1"].tolist()
                == batch["image_id_1"].tolist()
            and existing["image_id_2"].tolist()
                == batch["image_id_2"].tolist()
            and existing["status"].eq("success").all()
        ):
            print(f"Skipping verified batch {batch_index + 1}")
            continue

        raise RuntimeError(
            f"Existing batch requires inspection: {result_path}"
        )

    records = []

    for pair in tqdm(
        batch.itertuples(index=False),
        total=len(batch),
        desc=f"Batch {batch_index + 1}/{total_batches}"
    ):
        try:
            measurement = evaluate_pair(
                pair.image_id_1,
                pair.image_id_2
            )

            records.append({
                "image_id_1": pair.image_id_1,
                "image_id_2": pair.image_id_2,
                "dr_grade_1": int(pair.dr_grade_1),
                "dr_grade_2": int(pair.dr_grade_2),
                "phash_distance": int(pair.phash_distance),
                "thumbnail_cosine_similarity": float(
                    pair.thumbnail_cosine_similarity
                ),
                **measurement,
                "status": "success",
                "error": ""
            })

        except Exception as exc:
            records.append({
                "image_id_1": pair.image_id_1,
                "image_id_2": pair.image_id_2,
                "dr_grade_1": int(pair.dr_grade_1),
                "dr_grade_2": int(pair.dr_grade_2),
                "phash_distance": int(pair.phash_distance),
                "thumbnail_cosine_similarity": float(
                    pair.thumbnail_cosine_similarity
                ),
                "mutual_matches": np.nan,
                "ransac_inliers": np.nan,
                "inlier_ratio": np.nan,
                "median_reprojection_error_px": np.nan,
                "homography_valid": False,
                "status": "error",
                "error": str(exc)
            })

    batch_df = pd.DataFrame(records)

    if not batch_df["status"].eq("success").all():
        error_path = (
            BATCH_DIR / f"batch_{batch_index:03d}_errors.csv"
        )
        batch_df[batch_df["status"] == "error"].to_csv(
            error_path, index=False
        )
        raise RuntimeError(
            f"Batch {batch_index + 1} has processing errors."
        )

    temporary_path = result_path.with_suffix(".tmp")
    batch_df.to_csv(temporary_path, index=False)
    temporary_path.replace(result_path)

    print(
        f"Saved batch {batch_index + 1}/{total_batches}: "
        f"{len(batch_df)} pairs"
    )

print("\nALL BATCHES COMPLETED")

# Validate all batches before creating the merged report
all_results = []

for batch_index in range(total_batches):
    path = BATCH_DIR / f"batch_{batch_index:03d}.csv"
    assert path.exists(), f"Missing batch: {path}"

    part = pd.read_csv(
        path,
        dtype={"image_id_1": str, "image_id_2": str}
    )

    assert part["status"].eq("success").all()
    all_results.append(part)

full_results = pd.concat(all_results, ignore_index=True)

assert len(full_results) == 43878
assert not full_results.duplicated(
    ["image_id_1", "image_id_2"]
).any()

expected_pairs = set(zip(
    candidates["image_id_1"],
    candidates["image_id_2"]
))

actual_pairs = set(zip(
    full_results["image_id_1"],
    full_results["image_id_2"]
))

assert actual_pairs == expected_pairs

output_path = ROOT / "FGADR_Expanded_ORB_Full_Results.csv"
full_results.to_csv(output_path, index=False)

print("\nFULL SCREENING VERIFICATION")
print(f"Evaluated pairs: {len(full_results)}")
print(f"Processing errors: {(full_results['status'] != 'success').sum()}")
print(
    "Pairs with at least 100 inliers:",
    int((full_results["ransac_inliers"] >= 100).sum())
)
print(
    "Grade-conflicting high-inlier pairs:",
    int((
        (full_results["ransac_inliers"] >= 100)
        & (
            full_results["dr_grade_1"]
            != full_results["dr_grade_2"]
        )
    ).sum())
)
print(f"Results saved: {output_path}")
print("STEP 2.11C COMPLETED")


# %% [cell 39]

from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime, timezone

ROOT = (DR_RESEARCH / "experiment02_results")

results_path = ROOT / "FGADR_Expanded_ORB_Full_Results.csv"
manifest_path = ROOT / "FGADR_Image_SHA256_Manifest.csv"
old_results_path = ROOT / "FGADR_ORB_RANSAC_Results.csv"

results = pd.read_csv(
    results_path,
    dtype={"image_id_1": str, "image_id_2": str}
)

manifest = pd.read_csv(
    manifest_path,
    dtype={"image_id": str, "sha256": str}
)

old_results = pd.read_csv(
    old_results_path,
    dtype={"image_id_1": str, "image_id_2": str}
)

assert len(results) == 43878
assert len(manifest) == 1842
assert results["status"].eq("success").all()
assert not results.duplicated(
    ["image_id_1", "image_id_2"]
).any()

manifest_ids = set(manifest["image_id"])

assert set(results["image_id_1"]).issubset(manifest_ids)
assert set(results["image_id_2"]).issubset(manifest_ids)

grade_lookup = manifest.set_index("image_id")["dr_grade"]

assert (
    results["dr_grade_1"].to_numpy()
    == results["image_id_1"].map(grade_lookup).to_numpy()
).all()

assert (
    results["dr_grade_2"].to_numpy()
    == results["image_id_2"].map(grade_lookup).to_numpy()
).all()

# Quantitative geometric evidence
results["grade_conflict"] = (
    results["dr_grade_1"] != results["dr_grade_2"]
)

inlier_bins = pd.cut(
    results["ransac_inliers"],
    bins=[-1, 0, 9, 19, 39, 99, 199, 299, 499, float("inf")],
    labels=[
        "0", "1-9", "10-19", "20-39",
        "40-99", "100-199", "200-299",
        "300-499", "500+"
    ]
)

inlier_summary = (
    inlier_bins.value_counts()
    .sort_index()
    .rename_axis("inlier_band")
    .reset_index(name="pair_count")
)

high = results[
    (results["ransac_inliers"] >= 100)
    & (results["homography_valid"] == True)
].copy()

high["review_status"] = "pending_geometric_validation"
high["source_relation_confirmed"] = False

# Find previously unreported candidate pairs
def normalize_pair(a, b):
    return tuple(sorted((str(a), str(b))))

old_keys = {
    normalize_pair(a, b)
    for a, b in zip(
        old_results["image_id_1"],
        old_results["image_id_2"]
    )
    if pd.notna(a) and pd.notna(b)
}

high["previously_evaluated"] = [
    normalize_pair(a, b) in old_keys
    for a, b in zip(
        high["image_id_1"],
        high["image_id_2"]
    )
]

high_ids = (
    set(high["image_id_1"])
    | set(high["image_id_2"])
)

candidate_ids = (
    set(results["image_id_1"])
    | set(results["image_id_2"])
)

high = high.sort_values(
    ["ransac_inliers", "inlier_ratio"],
    ascending=[False, False]
)

# Save evidence for subsequent geometric validation
high_path = ROOT / "FGADR_Expanded_High_Geometric_Candidates.csv"
summary_path = ROOT / "FGADR_Expanded_Inlier_Distribution.csv"
audit_path = ROOT / "FGADR_Expanded_ORB_Audit.json"

high.to_csv(high_path, index=False)
inlier_summary.to_csv(summary_path, index=False)

audit = {
    "experiment": "Experiment02_Step2.12",
    "evaluated_pairs": int(len(results)),
    "high_inlier_pairs": int(len(high)),
    "high_inlier_grade_conflicts": int(high["grade_conflict"].sum()),
    "images_in_high_inlier_pairs": len(high_ids),
    "images_without_initial_candidate_pairs": len(
        manifest_ids - candidate_ids
    ),
    "high_inlier_pairs_previously_evaluated": int(
        high["previously_evaluated"].sum()
    ),
    "high_inlier_pairs_not_previously_evaluated": int(
        (~high["previously_evaluated"]).sum()
    ),
    "source_relationships_adjudicated": False,
    "created_utc": datetime.now(timezone.utc).isoformat()
}

with open(audit_path, "w", encoding="utf-8") as file:
    json.dump(audit, file, indent=2)

print("STEP 2.12 - GEOMETRIC EVIDENCE AUDIT")
print(f"Evaluated pairs: {len(results)}")
print(f"Pairs with >=100 inliers: {len(high)}")
print(f"Grade-conflicting high-inlier pairs: {int(high['grade_conflict'].sum())}")
print(f"Images involved in high-inlier pairs: {len(high_ids)}")
print(f"Images without initial candidate pairs: {len(manifest_ids - candidate_ids)}")
print(f"High-inlier pairs previously evaluated: {int(high['previously_evaluated'].sum())}")
print(f"High-inlier pairs not previously evaluated: {int((~high['previously_evaluated']).sum())}")

print("\nINLIER DISTRIBUTION")
print(inlier_summary.to_string(index=False))

print("\nHIGH-INLIER METRIC SUMMARY")
print(
    high[
        [
            "mutual_matches",
            "ransac_inliers",
            "inlier_ratio",
            "median_reprojection_error_px"
        ]
    ].describe().round(3).to_string()
)

print("\nSTEP 2.12 COMPLETED")


# %% [cell 40]

from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import cv2

ROOT = (DR_RESEARCH / "experiment02_results")
IMAGE_SIZE = 640
SEED = 42

cv2.setRNGSeed(SEED)

high = pd.read_csv(
    ROOT / "FGADR_Expanded_High_Geometric_Candidates.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

assert len(high) == 527
assert len(manifest) == 1842

paths = manifest.set_index("image_id")["local_path"].to_dict()
required_ids = sorted(
    set(high["image_id_1"]) | set(high["image_id_2"])
)

orb = cv2.ORB_create(nfeatures=2500)
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

cache = {}

for image_id in tqdm(required_ids, desc="Extracting spatial features"):
    image = cv2.imread(
        paths[image_id],
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(f"Cannot load image: {image_id}")

    image = cv2.resize(
        image, (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )
    image = clahe.apply(image)
    cache[image_id] = orb.detectAndCompute(image, None)

def ratio_matches(a, b):
    matches = matcher.knnMatch(a, b, k=2)
    accepted = {}

    for neighbors in matches:
        if len(neighbors) != 2:
            continue

        first, second = neighbors

        if first.distance < 0.75 * second.distance:
            accepted[(first.queryIdx, first.trainIdx)] = first

    return accepted

def spatial_metrics(image_id_1, image_id_2):
    kp1, desc1 = cache[image_id_1]
    kp2, desc2 = cache[image_id_2]

    empty = {
        "recomputed_inliers": 0,
        "source_hull_coverage": 0.0,
        "target_hull_coverage": 0.0,
        "source_occupied_quadrants": 0,
        "target_occupied_quadrants": 0,
        "spatial_evaluation_status": "insufficient_matches"
    }

    if desc1 is None or desc2 is None:
        return empty

    forward = ratio_matches(desc1, desc2)
    backward = ratio_matches(desc2, desc1)

    mutual = [
        m for (i, j), m in forward.items()
        if (j, i) in backward
    ]

    if len(mutual) < 8:
        return empty

    p1 = np.float32([
        kp1[m.queryIdx].pt for m in mutual
    ]).reshape(-1, 1, 2)

    p2 = np.float32([
        kp2[m.trainIdx].pt for m in mutual
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        p1, p2, cv2.RANSAC, 3.0,
        maxIters=2000, confidence=0.995
    )

    if H is None or mask is None:
        return empty

    if not np.isfinite(H).all():
        return empty

    valid = mask.ravel().astype(bool)
    source = p1.reshape(-1, 2)[valid]
    target = p2.reshape(-1, 2)[valid]

    if len(source) < 4:
        return empty

    def coverage(points):
        hull = cv2.convexHull(
            points.astype(np.float32).reshape(-1, 1, 2)
        )
        return float(
            cv2.contourArea(hull) / (IMAGE_SIZE ** 2)
        )

    def occupied_quadrants(points):
        quadrants = set()

        for x, y in points:
            quadrant = (
                int(x >= IMAGE_SIZE / 2),
                int(y >= IMAGE_SIZE / 2)
            )
            quadrants.add(quadrant)

        return len(quadrants)

    return {
        "recomputed_inliers": len(source),
        "source_hull_coverage": coverage(source),
        "target_hull_coverage": coverage(target),
        "source_occupied_quadrants": occupied_quadrants(source),
        "target_occupied_quadrants": occupied_quadrants(target),
        "spatial_evaluation_status": "evaluated"
    }

records = []

for row in tqdm(
    high.itertuples(index=False),
    total=len(high),
    desc="Evaluating spatial coverage"
):
    try:
        metrics = spatial_metrics(
            row.image_id_1,
            row.image_id_2
        )

        records.append({
            "image_id_1": row.image_id_1,
            "image_id_2": row.image_id_2,
            "original_inliers": int(row.ransac_inliers),
            **metrics,
            "processing_error": ""
        })

    except Exception as exc:
        records.append({
            "image_id_1": row.image_id_1,
            "image_id_2": row.image_id_2,
            "original_inliers": int(row.ransac_inliers),
            "recomputed_inliers": np.nan,
            "source_hull_coverage": np.nan,
            "target_hull_coverage": np.nan,
            "source_occupied_quadrants": np.nan,
            "target_occupied_quadrants": np.nan,
            "spatial_evaluation_status": "error",
            "processing_error": str(exc)
        })

spatial = pd.DataFrame(records)

spatial.to_csv(
    ROOT / "FGADR_Geometric_Spatial_Validation.csv",
    index=False
)

print("\nSTEP 2.13 - SPATIAL VALIDATION")
print(f"Candidate pairs: {len(spatial)}")
print(f"Evaluated: {(spatial['spatial_evaluation_status'] == 'evaluated').sum()}")
print(f"Insufficient matches: {(spatial['spatial_evaluation_status'] == 'insufficient_matches').sum()}")
print(f"Processing errors: {(spatial['spatial_evaluation_status'] == 'error').sum()}")

evaluated = spatial[
    spatial["spatial_evaluation_status"] == "evaluated"
]

print("\nSPATIAL COVERAGE SUMMARY")
print(
    evaluated[
        [
            "recomputed_inliers",
            "source_hull_coverage",
            "target_hull_coverage",
            "source_occupied_quadrants",
            "target_occupied_quadrants"
        ]
    ].describe().round(4).to_string()
)

print("\nLOW SPATIAL COVERAGE CANDIDATES")
low = evaluated[
    (evaluated["source_hull_coverage"] < 0.05)
    | (evaluated["target_hull_coverage"] < 0.05)
].copy()

print(f"Low-coverage pairs: {len(low)}")
print(
    low[
        [
            "image_id_1",
            "image_id_2",
            "recomputed_inliers",
            "source_hull_coverage",
            "target_hull_coverage"
        ]
    ].head(15).to_string(index=False)
)

print("\nSTEP 2.13 COMPLETED")


# %% [cell 41]

from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import pandas as pd
import math

ROOT = (DR_RESEARCH / "experiment02_results")
REVIEW_DIR = ROOT / "geometric_review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

spatial = pd.read_csv(
    ROOT / "FGADR_Geometric_Spatial_Validation.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

high = pd.read_csv(
    ROOT / "FGADR_Expanded_High_Geometric_Candidates.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

paths = manifest.set_index("image_id")["local_path"].to_dict()

low_coverage = spatial[
    (spatial["source_hull_coverage"] < 0.05)
    | (spatial["target_hull_coverage"] < 0.05)
].copy()

low_coverage["review_category"] = "low_spatial_coverage"

conflicts = high[
    high["dr_grade_1"] != high["dr_grade_2"]
].sort_values(
    ["ransac_inliers", "inlier_ratio"],
    ascending=[False, False]
).head(12).copy()

conflicts["review_category"] = "high_evidence_grade_conflict"

review = pd.concat(
    [
        low_coverage[
            ["image_id_1", "image_id_2", "review_category"]
        ],
        conflicts[
            ["image_id_1", "image_id_2", "review_category"]
        ]
    ],
    ignore_index=True
).drop_duplicates(
    ["image_id_1", "image_id_2"]
).reset_index(drop=True)

review["review_id"] = [
    f"V{i:03d}" for i in range(1, len(review) + 1)
]

review["manual_decision"] = "unreviewed"
review["review_notes"] = ""

review.to_csv(
    REVIEW_DIR / "Geometric_Review_Cases.csv",
    index=False
)

IMAGE_SIZE = 480
CELL_HEIGHT = 540
CELL_WIDTH = 1010
BATCH_SIZE = 4

for start in range(0, len(review), BATCH_SIZE):
    batch = review.iloc[start:start + BATCH_SIZE]

    sheet = Image.new(
        "RGB",
        (CELL_WIDTH, CELL_HEIGHT * len(batch)),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(batch.itertuples(index=False)):
        y = index * CELL_HEIGHT

        for side, image_id in enumerate(
            [row.image_id_1, row.image_id_2]
        ):
            with Image.open(paths[image_id]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image = image.resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    Image.Resampling.LANCZOS
                )

            sheet.paste(
                image,
                (10 + side * 500, y + 35)
            )

        draw.text(
            (12, y + 10),
            f"{row.review_id}: {row.image_id_1} / "
            f"{row.image_id_2}",
            fill="black"
        )

        draw.text(
            (12, y + 520),
            row.review_category,
            fill="black"
        )

    batch_index = start // BATCH_SIZE + 1
    output_path = (
        REVIEW_DIR / f"Geometric_Review_Batch_{batch_index:02d}.png"
    )
    sheet.save(output_path, optimize=True)

    print(f"Saved: {output_path.name}")

print("\nGEOMETRIC REVIEW PREPARATION")
print(f"Low-coverage cases: {len(low_coverage)}")
print(f"Grade-conflict priority cases: {len(conflicts)}")
print(f"Unique review pairs: {len(review)}")
print(f"Review directory: {REVIEW_DIR}")
print("STEP 2.14 COMPLETED")


# %% [cell 42]

from pathlib import Path

yolo_root = (DR_RESEARCH / "datasets/DR/FGADR/yolo")

print("YOLO directory exists:", yolo_root.exists())

if yolo_root.is_dir():
    print("\nYOLO top-level contents:")
    for item in sorted(yolo_root.iterdir()):
        item_type = "DIR" if item.is_dir() else "FILE"
        print(f"[{item_type}] {item.name}")

print("\nYOLO directory inspection completed.")


# %% [cell 43]

from pathlib import Path

yolo_root = (DR_RESEARCH / "datasets/DR/FGADR/yolo")

classes_path = yolo_root / "classes.txt"

print("YOLO ANNOTATION INSPECTION")
print("=" * 60)

if classes_path.is_file():
    print("\nClass definitions:")
    print(classes_path.read_text(encoding="utf-8-sig"))
else:
    print("Class definition file not found.")

for directory_name in ["images", "labels", "vis"]:
    directory = yolo_root / directory_name

    print(f"\nDirectory: {directory_name}")
    print(f"Exists: {directory.is_dir()}")

    if directory.is_dir():
        entries = sorted(directory.iterdir())
        print(f"Immediate entries: {len(entries)}")

        for item in entries[:10]:
            kind = "DIR" if item.is_dir() else "FILE"
            print(f"  [{kind}] {item.name}")

labels_root = yolo_root / "labels"

if labels_root.is_dir():
    sample_files = [
        path for path in sorted(labels_root.iterdir())
        if path.is_file() and path.suffix.lower() == ".txt"
    ][:3]

    for label_file in sample_files:
        print(f"\nLabel sample: {label_file.name}")
        content = label_file.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        for line in content[:5]:
            print(line)

print("\nSTEP Y1 COMPLETED")


# %% [cell 44]

from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

ROOT = (DR_RESEARCH / "datasets/DR/FGADR/yolo")

OUTPUT = (DR_RESEARCH / "experiment02_results")

OUTPUT.mkdir(parents=True, exist_ok=True)

image_dir = ROOT / "images"
label_dir = ROOT / "labels"

image_files = sorted(
    path for path in image_dir.iterdir()
    if path.is_file() and path.suffix.lower()
    in {".png", ".jpg", ".jpeg"}
)

label_files = sorted(
    path for path in label_dir.iterdir()
    if path.is_file() and path.suffix.lower() == ".txt"
)

image_ids = {path.stem for path in image_files}
label_ids = {path.stem for path in label_files}

extra_entries = sorted(
    path.name for path in label_dir.iterdir()
    if not (path.is_file() and path.suffix.lower() == ".txt")
)

missing_labels = sorted(image_ids - label_ids)
extra_labels = sorted(label_ids - image_ids)

records = []
issues = []
empty_labels = []

for label_path in label_files:
    lines = label_path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    nonempty_lines = [
        line.strip() for line in lines if line.strip()
    ]

    if not nonempty_lines:
        empty_labels.append(label_path.stem)

    for line_number, line in enumerate(nonempty_lines, start=1):
        tokens = line.split()

        if len(tokens) != 5:
            issues.append({
                "image_id": label_path.stem,
                "line_number": line_number,
                "issue": "invalid_column_count"
            })
            continue

        try:
            values = [float(token) for token in tokens]
        except ValueError:
            issues.append({
                "image_id": label_path.stem,
                "line_number": line_number,
                "issue": "non_numeric_value"
            })
            continue

        class_value, x, y, w, h = values

        if (
            not all(np.isfinite(values))
            or not class_value.is_integer()
            or not 0 <= class_value <= 5
        ):
            issues.append({
                "image_id": label_path.stem,
                "line_number": line_number,
                "issue": "invalid_class_or_nonfinite_value"
            })
            continue

        valid_box = (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < w <= 1
            and 0 < h <= 1
            and x - w / 2 >= -1e-5
            and y - h / 2 >= -1e-5
            and x + w / 2 <= 1 + 1e-5
            and y + h / 2 <= 1 + 1e-5
        )

        if not valid_box:
            issues.append({
                "image_id": label_path.stem,
                "line_number": line_number,
                "issue": "invalid_bbox_geometry"
            })

        records.append({
            "image_id": label_path.stem,
            "class_id": int(class_value),
            "x_center": x,
            "y_center": y,
            "width": w,
            "height": h,
            "box_area_fraction": w * h,
            "geometry_valid": valid_box
        })

boxes = pd.DataFrame(records)
issue_df = pd.DataFrame(
    issues,
    columns=["image_id", "line_number", "issue"]
)

boxes.to_csv(
    OUTPUT / "FGADR_YOLO_Box_Inventory.csv",
    index=False
)

issue_df.to_csv(
    OUTPUT / "FGADR_YOLO_Annotation_Issues.csv",
    index=False
)

print("YOLO ANNOTATION INTEGRITY AUDIT")
print(f"Image files: {len(image_files)}")
print(f"Label TXT files: {len(label_files)}")
print(f"Extra directory entries: {extra_entries}")
print(f"Missing image labels: {len(missing_labels)}")
print(f"Labels without images: {len(extra_labels)}")
print(f"Empty label files: {len(empty_labels)}")
print(f"Parsed annotations: {len(boxes)}")
print(f"Annotation issues: {len(issue_df)}")

print("\nCLASS DISTRIBUTION")
if not boxes.empty:
    print(
        boxes["class_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    ma = boxes[
        (boxes["class_id"] == 3)
        & (boxes["geometry_valid"])
    ]

    print("\nMICROANEURYSM BOX STATISTICS")
    print(f"Valid MA boxes: {len(ma)}")
    print(f"Images containing MA boxes: {ma['image_id'].nunique()}")
    print(
        ma["box_area_fraction"]
        .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .to_string()
    )

print("\nISSUE SAMPLES")
print(issue_df.head(10).to_string(index=False))
print("\nMissing labels:", missing_labels[:10])
print("Extra labels:", extra_labels[:10])

print("\nSTEP Y2 COMPLETED")


# %% [cell 45]

from pathlib import Path
import pandas as pd
import hashlib

PROJECT = DR_RESEARCH
RESULTS = PROJECT / "experiment02_results"
YOLO = PROJECT / "datasets/DR/FGADR/yolo"

boxes = pd.read_csv(
    RESULTS / "FGADR_YOLO_Box_Inventory.csv",
    dtype={"image_id": str}
)

phase3 = pd.read_csv(
    PROJECT / "phase3_results/MA_Mask_Quality_Categorization.csv",
    dtype={"image_id": str}
)

manifest = pd.read_csv(
    RESULTS / "FGADR_Master_Manifest.csv",
    dtype={"image_id": str}
)

def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

print("EXTRA LABEL FILE CHECK")

for name in ["1333_3", "1334_3"]:
    original = YOLO / "labels" / f"{name}.txt"
    duplicate = YOLO / "labels" / f"{name} (1).txt"

    if original.exists() and duplicate.exists():
        print(
            name,
            "same_bytes:",
            file_hash(original) == file_hash(duplicate),
            "original_lines:",
            len(original.read_text(encoding="utf-8-sig").splitlines()),
            "duplicate_lines:",
            len(duplicate.read_text(encoding="utf-8-sig").splitlines())
        )

ma = boxes[
    (boxes["class_id"] == 3)
    & (boxes["geometry_valid"])
].copy()

ma_counts = ma.groupby("image_id").size().rename("yolo_ma_count")

audit = manifest[["image_id", "dr_grade"]].merge(
    phase3[
        [
            "image_id",
            "review_category",
            "candidate_count",
            "is_empty_mask"
        ]
    ],
    on="image_id",
    validate="one_to_one"
)

audit = audit.merge(
    ma_counts,
    left_on="image_id",
    right_index=True,
    how="left",
    validate="one_to_one"
)

audit["yolo_ma_count"] = (
    audit["yolo_ma_count"].fillna(0).astype(int)
)

audit["yolo_ma_present"] = audit["yolo_ma_count"] > 0
audit["mask_ma_present"] = ~audit["is_empty_mask"].astype(bool)

print("\nYOLO AND MASK PRESENCE CROSS-CHECK")
print(
    pd.crosstab(
        audit["mask_ma_present"],
        audit["yolo_ma_present"],
        rownames=["Mask MA present"],
        colnames=["YOLO MA present"]
    ).to_string()
)

disagreement = audit[
    audit["mask_ma_present"] != audit["yolo_ma_present"]
].copy()

print(f"\nImages with presence disagreement: {len(disagreement)}")

disagreement.to_csv(
    RESULTS / "FGADR_YOLO_Mask_Presence_Disagreement.csv",
    index=False
)

ma = ma.merge(
    audit[["image_id", "dr_grade", "review_category"]],
    on="image_id",
    how="left",
    validate="many_to_one"
)

ma["bbox_area_px_1280"] = (
    ma["box_area_fraction"] * 1280 * 1280
)

largest = ma.sort_values(
    "box_area_fraction",
    ascending=False
).head(20)

print("\nLARGEST MA BOUNDING BOXES")
print(
    largest[
        [
            "image_id",
            "dr_grade",
            "review_category",
            "bbox_area_px_1280",
            "box_area_fraction"
        ]
    ].to_string(index=False)
)

print("\nMA BOX COUNTS BY MASK REVIEW CATEGORY")
print(
    ma.groupby("review_category")
    .agg(
        image_count=("image_id", "nunique"),
        box_count=("image_id", "size"),
        maximum_box_area_fraction=("box_area_fraction", "max")
    ).to_string()
)

largest.to_csv(
    RESULTS / "FGADR_YOLO_Largest_MA_Boxes.csv",
    index=False
)

audit.to_csv(
    RESULTS / "FGADR_YOLO_Mask_Image_Level_Audit.csv",
    index=False
)

print("\nSTEP Y3 COMPLETED")


# %% [cell 46]

from pathlib import Path
import pandas as pd

PROJECT = DR_RESEARCH
RESULTS = PROJECT / "experiment02_results"

manifest = pd.read_csv(
    RESULTS / "FGADR_Master_Manifest.csv",
    dtype={"image_id": str}
)

boxes = pd.read_csv(
    RESULTS / "FGADR_YOLO_Box_Inventory.csv",
    dtype={"image_id": str}
)

audit = pd.read_csv(
    RESULTS / "FGADR_YOLO_Mask_Image_Level_Audit.csv",
    dtype={"image_id": str}
)

valid_ids = set(manifest["image_id"])

# Keep only annotations with matching dataset images
valid_boxes = boxes[
    boxes["image_id"].isin(valid_ids)
].copy()

orphan_boxes = boxes[
    ~boxes["image_id"].isin(valid_ids)
].copy()

valid_boxes.to_csv(
    RESULTS / "FGADR_YOLO_Valid_Box_Inventory.csv",
    index=False
)

orphan_boxes.to_csv(
    RESULTS / "FGADR_YOLO_Orphan_Annotations.csv",
    index=False
)

# Use explicit boolean conversion for reproducibility
for column in ["mask_ma_present", "yolo_ma_present"]:
    if audit[column].dtype == object:
        audit[column] = audit[column].map({
            "True": True,
            "False": False
        })

assert len(audit) == 1842
assert audit["image_id"].is_unique
assert audit[
    ["mask_ma_present", "yolo_ma_present"]
].notna().all().all()

disagreement = audit[
    audit["mask_ma_present"]
    & ~audit["yolo_ma_present"]
].copy()

assert len(disagreement) == 58

print("YOLO VALID ANNOTATION AUDIT")
print(f"Original annotations: {len(boxes)}")
print(f"Valid annotations: {len(valid_boxes)}")
print(f"Orphan annotations: {len(orphan_boxes)}")

valid_ma = valid_boxes[
    valid_boxes["class_id"] == 3
]

print(f"Valid MA boxes: {len(valid_ma)}")
print(f"Images containing valid MA boxes: {valid_ma['image_id'].nunique()}")

print("\nORPHAN ANNOTATION COUNTS")
print(
    orphan_boxes.groupby(
        ["image_id", "class_id"]
    ).size().to_string()
)

print("\nDISAGREEMENT BY DR GRADE")
print(
    disagreement["dr_grade"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nDISAGREEMENT BY MASK CATEGORY")
print(
    disagreement["review_category"]
    .value_counts()
    .to_string()
)

print("\nDISAGREEMENT COMPONENT STATISTICS")
print(
    disagreement["candidate_count"]
    .describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    .to_string()
)

print("\nTOP DISAGREEMENT CASES")
print(
    disagreement.sort_values(
        "candidate_count",
        ascending=False
    )[
        [
            "image_id",
            "dr_grade",
            "review_category",
            "candidate_count",
            "yolo_ma_count"
        ]
    ].head(15).to_string(index=False)
)

disagreement.to_csv(
    RESULTS / "FGADR_YOLO_MA_Disagreement_58.csv",
    index=False
)

print("\nSTEP Y4 COMPLETED")


# %% [cell 47]

from pathlib import Path
import pandas as pd

project = DR_RESEARCH
results = project / "experiment02_results"

audit = pd.read_csv(
    results / "FGADR_YOLO_Mask_Image_Level_Audit.csv",
    dtype={"image_id": str}
)

phase3 = pd.read_csv(
    project / "phase3_results" / "MA_Mask_Quality_Categorization.csv",
    dtype={"image_id": str}
)

assert len(audit) == 1842
assert len(phase3) == 1842
assert audit["image_id"].is_unique
assert phase3["image_id"].is_unique

def parse_bool(series):
    normalized = series.astype(str).str.lower()
    assert normalized.isin(["true", "false"]).all()
    return normalized.eq("true")

audit["mask_ma_present"] = parse_bool(audit["mask_ma_present"])
audit["yolo_ma_present"] = parse_bool(audit["yolo_ma_present"])

df = audit.merge(
    phase3[
        [
            "image_id",
            "encoding",
            "filename_suffix",
            "intermediate_pixels"
        ]
    ],
    on="image_id",
    how="inner",
    validate="one_to_one"
)

assert len(df) == 1842

df["presence_disagreement"] = (
    df["mask_ma_present"] & ~df["yolo_ma_present"]
)

print("STEP Y5 - ENCODING ASSOCIATION AUDIT")

print("\nDISAGREEMENT BY MASK ENCODING")
print(pd.crosstab(
    df["encoding"],
    df["presence_disagreement"]
).to_string())

print("\nDISAGREEMENT BY FILENAME SUFFIX")
print(pd.crosstab(
    df["filename_suffix"],
    df["presence_disagreement"]
).to_string())

print("\nGRAYSCALE PIXEL COUNTS IN DISAGREEMENT CASES")
print(
    df.loc[df["presence_disagreement"], "intermediate_pixels"]
    .describe()
    .to_string()
)

print("\nDISAGREEMENT RATES AMONG NONEMPTY MASKS")
nonempty = df[df["mask_ma_present"]].copy()

summary = nonempty.groupby("encoding").agg(
    images=("image_id", "size"),
    disagreement_images=("presence_disagreement", "sum")
)

summary["disagreement_rate"] = (
    summary["disagreement_images"] / summary["images"]
)

print(summary.to_string())

summary.to_csv(
    results / "FGADR_YOLO_Encoding_Disagreement_Summary.csv"
)

print("\nSTEP Y5 COMPLETED")


# %% [cell 48] identical to cell 52, omitted here


# %% [cell 49]

from pathlib import Path

drive_root = DR_RESEARCH.parent
project_root = drive_root / "DR_Research"
results_root = project_root / "experiment02_results"

print("Google Drive mounted:", drive_root.exists())
print("Project directory exists:", project_root.is_dir())
print("Experiment results directory exists:", results_root.is_dir())

if results_root.is_dir():
    print("\nAvailable experiment result files:")
    for item in sorted(results_root.iterdir()):
        if item.is_file():
            print(item.name)

print("\nPATH DIAGNOSTIC COMPLETED")


# %% [cell 50]


pass  # Google Drive mount removed


# %% [cell 51]

from pathlib import Path

root = DR_RESEARCH

required_files = [
    "FGADR_Image_SHA256_Manifest.csv",
    "FGADR_Expanded_ORB_Full_Results.csv",
]

for filename in required_files:
    expected_path = root / "experiment02_results" / filename

    print(f"\nFile: {filename}")
    print(f"Expected path exists: {expected_path.is_file()}")

    if not expected_path.is_file():
        print("Checking immediate project subdirectories...")

        matches = []
        for folder in root.iterdir():
            if folder.is_dir():
                candidate = folder / filename
                if candidate.is_file():
                    matches.append(candidate)

        if matches:
            for match in matches:
                print(f"Found: {match}")
        else:
            print("File not found in immediate project subdirectories.")

print("\nFILE LOCATION CHECK COMPLETED")


# %% [cell 52]

from pathlib import Path
import pandas as pd

root = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    root / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str, "sha256": str}
)

results = pd.read_csv(
    root / "FGADR_Expanded_ORB_Full_Results.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

assert len(manifest) == 1842
assert len(results) == 43878
assert manifest["image_id"].is_unique
assert results["status"].eq("success").all()

# Exploratory geometric evidence, not confirmed source identity
strong = results[
    (results["ransac_inliers"] >= 100)
    & (results["homography_valid"].astype(bool))
].copy()

strong["grade_conflict"] = (
    strong["dr_grade_1"] != strong["dr_grade_2"]
)

# Identify candidate image involvement
strong_ids = (
    set(strong["image_id_1"])
    | set(strong["image_id_2"])
)

# Exact duplicates are independent confirmed evidence
exact = manifest[
    manifest.duplicated("sha256", keep=False)
].copy()

exact_groups = (
    exact.groupby("sha256")
    .agg(
        image_count=("image_id", "size"),
        grade_count=("dr_grade", "nunique")
    )
)

print("STEP 2.15A - SOURCE EVIDENCE INVENTORY")
print(f"Total images: {len(manifest)}")
print(f"Strong geometric candidate pairs: {len(strong)}")
print(f"Images in geometric candidate pairs: {len(strong_ids)}")
print(f"Grade-conflicting candidate pairs: {int(strong['grade_conflict'].sum())}")
print(f"Confirmed exact duplicate groups: {len(exact_groups)}")
print(f"Exact groups with label conflicts: {int((exact_groups['grade_count'] > 1).sum())}")

print("\nGEOMETRIC CANDIDATE GRADE PAIRS")
grade_pairs = strong.copy()
grade_pairs["grade_pair"] = grade_pairs.apply(
    lambda row: "-".join(
        map(str, sorted([
            int(row["dr_grade_1"]),
            int(row["dr_grade_2"])
        ]))
    ),
    axis=1
)

print(
    grade_pairs["grade_pair"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nSTEP 2.15A COMPLETED")


# %% [cell 53]

from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone

ROOT = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str, "sha256": str}
)

orb = pd.read_csv(
    ROOT / "FGADR_Expanded_ORB_Full_Results.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

assert len(manifest) == 1842
assert len(orb) == 43878
assert manifest["image_id"].is_unique
assert orb["status"].eq("success").all()
assert not orb.duplicated(
    ["image_id_1", "image_id_2"]
).any()

image_ids = set(manifest["image_id"])
grade_lookup = manifest.set_index("image_id")["dr_grade"].to_dict()

parent = {image_id: image_id for image_id in image_ids}

def find(image_id):
    while parent[image_id] != image_id:
        parent[image_id] = parent[parent[image_id]]
        image_id = parent[image_id]
    return image_id

def union(first, second):
    root_first = find(first)
    root_second = find(second)

    if root_first != root_second:
        parent[root_second] = root_first

edges = []

# Byte-identical images provide confirmed exact relationships
for sha256, subset in manifest.groupby("sha256"):
    ids = sorted(subset["image_id"].tolist())

    for image_id in ids[1:]:
        union(ids[0], image_id)

        edges.append({
            "image_id_1": ids[0],
            "image_id_2": image_id,
            "evidence_type": "confirmed_exact_duplicate",
            "ransac_inliers": np.nan,
            "inlier_ratio": np.nan,
            "review_status": "confirmed_byte_identical",
        })

# Geometric relationships remain unconfirmed
strong = orb[
    (orb["ransac_inliers"] >= 100)
    & (orb["homography_valid"].astype(str).str.lower() == "true")
].copy()

assert len(strong) == 527

for row in strong.itertuples(index=False):
    union(row.image_id_1, row.image_id_2)

    edges.append({
        "image_id_1": row.image_id_1,
        "image_id_2": row.image_id_2,
        "evidence_type": "strong_geometric_candidate",
        "ransac_inliers": int(row.ransac_inliers),
        "inlier_ratio": float(row.inlier_ratio),
        "review_status": "pending_verification",
    })

components = defaultdict(list)

for image_id in sorted(image_ids):
    components[find(image_id)].append(image_id)

group_rows = []
member_rows = []

for members in components.values():
    members = sorted(members)
    grades = sorted(set(int(grade_lookup[i]) for i in members))
    group_id = f"RG_{members[0]}"

    group_rows.append({
        "risk_group_id": group_id,
        "image_count": len(members),
        "grade_count": len(grades),
        "grades": ";".join(map(str, grades)),
        "grade_conflict": len(grades) > 1,
        "image_ids": ";".join(members),
        "group_status": "provisional_leakage_constraint",
    })

    for image_id in members:
        member_rows.append({
            "image_id": image_id,
            "dr_grade": int(grade_lookup[image_id]),
            "risk_group_id": group_id,
            "risk_group_size": len(members),
            "group_grade_conflict": len(grades) > 1,
        })

group_df = pd.DataFrame(group_rows)
member_df = pd.DataFrame(member_rows)
edge_df = pd.DataFrame(edges)

edge_df["risk_group_id"] = edge_df["image_id_1"].map(
    member_df.set_index("image_id")["risk_group_id"]
)

assert len(member_df) == 1842
assert member_df["image_id"].is_unique
assert group_df["image_count"].sum() == 1842
assert len(edge_df) == 541

group_df = group_df.sort_values(
    ["image_count", "risk_group_id"],
    ascending=[False, True]
)

group_df.to_csv(
    ROOT / "FGADR_Updated_Risk_Groups.csv",
    index=False
)

member_df.to_csv(
    ROOT / "FGADR_Updated_Risk_Group_Members.csv",
    index=False
)

edge_df.to_csv(
    ROOT / "FGADR_Updated_Risk_Evidence_Edges.csv",
    index=False
)

print("STEP 2.15B - UPDATED RISK-GROUP AUDIT")
print(f"Total images: {len(member_df)}")
print(f"Confirmed exact relationships: {int((edge_df['evidence_type'] == 'confirmed_exact_duplicate').sum())}")
print(f"Geometric candidate relationships: {int((edge_df['evidence_type'] == 'strong_geometric_candidate').sum())}")
print(f"Total provisional groups: {len(group_df)}")
print(f"Multi-image groups: {int((group_df['image_count'] > 1).sum())}")
print(f"Images in multi-image groups: {int(group_df.loc[group_df['image_count'] > 1, 'image_count'].sum())}")
print(f"Grade-conflicting groups: {int(group_df['grade_conflict'].sum())}")
print(f"Images in grade-conflicting groups: {int(group_df.loc[group_df['grade_conflict'], 'image_count'].sum())}")
print(f"Largest group size: {int(group_df['image_count'].max())}")

print("\nLARGEST RISK GROUPS")
print(
    group_df[
        ["risk_group_id", "image_count", "grades", "grade_conflict", "image_ids"]
    ].head(12).to_string(index=False)
)

print("\nSTEP 2.15B COMPLETED")


# %% [cell 54]

from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

edges = pd.read_csv(
    ROOT / "FGADR_Updated_Risk_Evidence_Edges.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

groups = pd.read_csv(
    ROOT / "FGADR_Updated_Risk_Groups.csv"
)

graph = nx.Graph()
graph.add_nodes_from(manifest["image_id"])

for row in edges.itertuples(index=False):
    graph.add_edge(
        row.image_id_1,
        row.image_id_2
    )

assert graph.number_of_nodes() == 1842
assert nx.number_connected_components(graph) == len(groups)

bridges = {
    frozenset((a, b))
    for a, b in nx.bridges(graph)
}

records = []

for row in edges.itertuples(index=False):
    key = frozenset(
        (row.image_id_1, row.image_id_2)
    )

    records.append({
        "image_id_1": row.image_id_1,
        "image_id_2": row.image_id_2,
        "risk_group_id": row.risk_group_id,
        "evidence_type": row.evidence_type,
        "ransac_inliers": row.ransac_inliers,
        "inlier_ratio": row.inlier_ratio,
        "is_bridge": key in bridges,
        "review_priority": (
            "high"
            if key in bridges
            and row.evidence_type == "strong_geometric_candidate"
            else "standard"
        )
    })

audit = pd.DataFrame(records)

# Evaluate how much grouping relies on unconfirmed geometry
exact_graph = nx.Graph()
exact_graph.add_nodes_from(manifest["image_id"])

for row in edges.itertuples(index=False):
    if row.evidence_type == "confirmed_exact_duplicate":
        exact_graph.add_edge(
            row.image_id_1,
            row.image_id_2
        )

assert nx.number_connected_components(exact_graph) == 1828

# Compare grouping under exploratory evidence thresholds
scenarios = []

for minimum_inliers, minimum_ratio in [
    (100, 0.0),
    (100, 0.70),
    (150, 0.80),
    (200, 0.85),
]:
    scenario_graph = exact_graph.copy()

    selected = audit[
        (audit["evidence_type"] == "strong_geometric_candidate")
        & (audit["ransac_inliers"] >= minimum_inliers)
        & (audit["inlier_ratio"] >= minimum_ratio)
    ]

    scenario_graph.add_edges_from(
        zip(selected["image_id_1"], selected["image_id_2"])
    )

    components = list(nx.connected_components(scenario_graph))
    multi = [component for component in components if len(component) > 1]

    scenarios.append({
        "minimum_inliers": minimum_inliers,
        "minimum_inlier_ratio": minimum_ratio,
        "geometric_edges": len(selected),
        "risk_groups": len(components),
        "multi_image_groups": len(multi),
        "images_in_multi_image_groups": sum(map(len, multi)),
        "largest_group_size": max(map(len, components))
    })

scenario_df = pd.DataFrame(scenarios)

audit.to_csv(
    ROOT / "FGADR_Graph_Bridge_Audit.csv",
    index=False
)

scenario_df.to_csv(
    ROOT / "FGADR_Risk_Group_Robustness.csv",
    index=False
)

print("STEP 2.15C - GRAPH ROBUSTNESS AUDIT")
print(f"Graph nodes: {graph.number_of_nodes()}")
print(f"Unique graph edges: {graph.number_of_edges()}")
print(f"Graph bridges: {len(bridges)}")
print(
    "Unconfirmed geometric bridge records:",
    int((
        audit["is_bridge"]
        & (audit["evidence_type"] == "strong_geometric_candidate")
    ).sum())
)

print("\nGROUPING ROBUSTNESS")
print(scenario_df.to_string(index=False))

print("\nHIGH PRIORITY BRIDGE EDGES")
print(
    audit[
        audit["review_priority"] == "high"
    ].sort_values(
        ["ransac_inliers", "inlier_ratio"],
        ascending=[True, True]
    ).head(20).to_string(index=False)
)

print("\nSTEP 2.15C COMPLETED")


# %% [cell 55]

from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import cv2
import json
from datetime import datetime, timezone

ROOT = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str, "sha256": str}
).sort_values("image_id").reset_index(drop=True)

candidates = pd.read_csv(
    ROOT / "FGADR_Near_Duplicate_Candidates.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

assert len(manifest) == 1842
assert manifest["image_id"].is_unique
assert len(candidates) == 43878

ids = manifest["image_id"].tolist()
id_to_index = {image_id: i for i, image_id in enumerate(ids)}

hog = cv2.HOGDescriptor(
    (128, 128),
    (32, 32),
    (16, 16),
    (16, 16),
    9
)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

features = []

for row in tqdm(
    manifest.itertuples(index=False),
    total=len(manifest),
    desc="Extracting HOG features"
):
    image = cv2.imread(
        row.local_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(f"Cannot decode image: {row.image_id}")

    image = cv2.resize(
        image,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    image = clahe.apply(image)

    descriptor = hog.compute(image).flatten()
    norm = np.linalg.norm(descriptor)

    if norm <= 1e-12:
        raise RuntimeError(
            f"Invalid HOG descriptor: {row.image_id}"
        )

    features.append(descriptor / norm)

features = np.asarray(features, dtype=np.float32)

similarity = features @ features.T
similarity = np.clip(similarity, -1.0, 1.0)

n = len(ids)
candidate_mask = np.zeros((n, n), dtype=bool)

for row in candidates.itertuples(index=False):
    i = id_to_index[row.image_id_1]
    j = id_to_index[row.image_id_2]
    candidate_mask[i, j] = True
    candidate_mask[j, i] = True

# Record nearest unseen pairs without changing the original candidates
np.fill_diagonal(similarity, -np.inf)

unseen_similarity = similarity.copy()
unseen_similarity[candidate_mask] = -np.inf

upper_i, upper_j = np.triu_indices(n, k=1)
scores = unseen_similarity[upper_i, upper_j]

valid = np.isfinite(scores)
valid_indices = np.flatnonzero(valid)

TOP_PAIRS = 500

selected_indices = valid_indices[
    np.argsort(scores[valid])[::-1][:TOP_PAIRS]
]

records = []

for index in selected_indices:
    i = int(upper_i[index])
    j = int(upper_j[index])

    records.append({
        "image_id_1": ids[i],
        "image_id_2": ids[j],
        "dr_grade_1": int(manifest.iloc[i]["dr_grade"]),
        "dr_grade_2": int(manifest.iloc[j]["dr_grade"]),
        "hog_cosine_similarity": float(similarity[i, j]),
        "source": "independent_hog_retrieval",
        "review_status": "pending_verification",
    })

unseen = pd.DataFrame(records)

unseen.to_csv(
    ROOT / "FGADR_HOG_Unseen_Candidate_Pairs.csv",
    index=False
)

# Check images omitted by the initial candidate generator
candidate_ids = (
    set(candidates["image_id_1"])
    | set(candidates["image_id_2"])
)

omitted_ids = sorted(set(ids) - candidate_ids)

omitted_records = []

for image_id in omitted_ids:
    i = id_to_index[image_id]
    row_scores = similarity[i].copy()

    best_j = int(np.argmax(row_scores))

    omitted_records.append({
        "image_id": image_id,
        "nearest_image_id": ids[best_j],
        "hog_cosine_similarity": float(row_scores[best_j]),
        "was_initial_candidate": bool(
            candidate_mask[i, best_j]
        ),
    })

omitted_df = pd.DataFrame(omitted_records)

omitted_df.to_csv(
    ROOT / "FGADR_HOG_Previously_Omitted_Images.csv",
    index=False
)

# Validate known exact-duplicate relationships
exact_group_checks = []

for digest, group in manifest.groupby("sha256"):
    if len(group) < 2:
        continue

    group_indices = [
        id_to_index[image_id]
        for image_id in group["image_id"]
    ]

    for k in range(1, len(group_indices)):
        a = group_indices[0]
        b = group_indices[k]

        exact_group_checks.append(
            float(similarity[a, b])
        )

assert len(exact_group_checks) == 14
assert min(exact_group_checks) > 0.999

print("STEP 2.15D - INDEPENDENT HOG RETRIEVAL")
print(f"Images analyzed: {n}")
print(f"All possible pairs: {n * (n - 1) // 2}")
print(f"Original candidate pairs: {len(candidates)}")
print(f"Previously unexamined pairs: {int(valid.sum())}")
print(f"New HOG-prioritized pairs: {len(unseen)}")
print(f"Previously omitted images: {len(omitted_df)}")
print(f"Exact duplicate control pairs: {len(exact_group_checks)}")

print("\nTOP 20 NEW HOG CANDIDATES")
print(unseen.head(20).to_string(index=False))

print("\nPREVIOUSLY OMITTED IMAGE SUMMARY")
print(
    omitted_df["hog_cosine_similarity"]
    .describe()
    .round(5)
    .to_string()
)

print("\nSTEP 2.15D COMPLETED")


# %% [cell 56] identical to cell 59, omitted here


# %% [cell 57] identical to cell 60, omitted here


# %% [cell 58]
pass  # Google Drive mount removed


# %% [cell 59]

from pathlib import Path
import pandas as pd

root = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    root / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

first_image = manifest.iloc[0]

local_path = Path(first_image["local_path"])
source_path = Path(first_image["source_path"])

print("IMAGE PATH DIAGNOSTIC")
print("Image ID:", first_image["image_id"])
print("Local path:", local_path)
print("Local file exists:", local_path.is_file())
print("Source path:", source_path)
print("Source file exists:", source_path.is_file())

print("\nFILE AVAILABILITY")
print("Total images:", len(manifest))
print(
    "Available local images:",
    sum(Path(p).is_file() for p in manifest["local_path"])
)
print(
    "Available source images:",
    sum(Path(p).is_file() for p in manifest["source_path"])
)

print("\nPATH DIAGNOSTIC COMPLETED")


# %% [cell 60]

from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import hashlib
import shutil

root = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    root / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str, "sha256": str}
)

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
    desc="Restoring verified local images"
):
    source = Path(row.source_path)
    local = Path(row.local_path)

    try:
        if not source.is_file():
            raise FileNotFoundError(f"Missing source: {source}")

        if local.is_file() and sha256_file(local) == row.sha256:
            already_verified += 1
            continue

        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local)

        if sha256_file(local) != row.sha256:
            raise RuntimeError("SHA256 verification failed.")

        restored += 1

    except Exception as exc:
        failures.append({
            "image_id": row.image_id,
            "error": str(exc)
        })

print("\nLOCAL IMAGE RESTORATION")
print(f"Expected images: {len(manifest)}")
print(f"Restored images: {restored}")
print(f"Previously verified images: {already_verified}")
print(f"Failures: {len(failures)}")

assert len(failures) == 0
assert restored + already_verified == len(manifest)

print("\nRESTORATION VERIFIED")


# %% [cell 61]
from pathlib import Path
import pandas as pd
import cv2

results_root = (DR_RESEARCH / "experiment02_results")
local_image_root = (LOCAL_DISK / "fgadr_classification")

manifest_path = results_root / "FGADR_Master_Manifest.csv"
manifest = pd.read_csv(manifest_path, dtype={"image_id": str})

print("Manifest exists:", manifest_path.exists())
print("Local image root exists:", local_image_root.exists())
print("Manifest rows:", len(manifest))

sample_ids = ["0000_1", "0001_2", "0135_1", "0318_1", "1163_1"]

for image_id in sample_ids:
    local_path = local_image_root / f"{image_id}.png"
    print(f"\nChecking: {image_id}")
    print("Path exists:", local_path.exists())
    if local_path.exists():
        image = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
        if image is None:
            print("Decode status: FAILED")
        else:
            print("Decode status: OK")
            print("Shape:", image.shape)


# %% [cell 62]

from pathlib import Path
import pandas as pd
import cv2

results_root = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    results_root / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

sample_ids = [
    "0000_1",
    "0001_2",
    "0135_1",
    "0318_1",
    "1163_1"
]

print("LOCAL IMAGE VERIFICATION")
print("=" * 50)

assert len(manifest) == 1842
assert manifest["image_id"].is_unique

all_passed = True

for image_id in sample_ids:
    record = manifest.loc[
        manifest["image_id"] == image_id
    ]

    assert len(record) == 1

    image_path = Path(record.iloc[0]["local_path"])

    print(f"\nImage ID: {image_id}")
    print(f"Path: {image_path}")
    print(f"File exists: {image_path.is_file()}")

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        print("Decode status: FAILED")
        all_passed = False
    else:
        print("Decode status: OK")
        print(f"Image shape: {image.shape}")

print("\nVERIFICATION RESULT")
print("All sample images verified:", all_passed)

assert all_passed, "Sample image verification failed."

print("STEP A PASSED")


# %% [cell 63] identical to cell 67, omitted here


# %% [cell 64]

import cv2
import sys
import importlib.metadata as metadata

print("OPENCV ENVIRONMENT DIAGNOSTIC")
print("Python version:", sys.version.split()[0])
print("OpenCV version:", getattr(cv2, "__version__", "UNKNOWN"))
print("OpenCV module path:", getattr(cv2, "__file__", "UNKNOWN"))
print("HOGDescriptor available:", hasattr(cv2, "HOGDescriptor"))
print("CLAHE available:", hasattr(cv2, "createCLAHE"))
print("OpenCV submodule available:", hasattr(cv2, "cv2"))

print("\nINSTALLED OPENCV PACKAGES")

package_names = [
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless"
]

for package_name in package_names:
    try:
        version = metadata.version(package_name)
        print(f"{package_name}: {version}")
    except metadata.PackageNotFoundError:
        print(f"{package_name}: NOT INSTALLED")

print("\nDIAGNOSTIC COMPLETED")


# %% [cell 65]

# %pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless   (see requirements.txt)
# %pip install --no-cache-dir "opencv-contrib-python-headless==4.14.0.94"   (see requirements.txt)


# %% [cell 66]

import cv2
from pathlib import Path

print("OPENCV REPAIR VERIFICATION")
print("OpenCV version:", cv2.__version__)
print("HOGDescriptor available:", hasattr(cv2, "HOGDescriptor"))
print("CLAHE available:", hasattr(cv2, "createCLAHE"))
print("ORB available:", hasattr(cv2, "ORB_create"))

assert hasattr(cv2, "HOGDescriptor")
assert hasattr(cv2, "createCLAHE")
assert hasattr(cv2, "ORB_create")

hog = cv2.HOGDescriptor(
    (128, 128),
    (32, 32),
    (16, 16),
    (16, 16),
    9
)

print("HOG initialization: PASS")

image_path = Path(
    str(LOCAL_DISK / "fgadr_classification/2/0000_1.png")
)

print("Local image available:", image_path.is_file())

if image_path.is_file():
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    print("Image decoding: PASS")
else:
    print("Image decoding: SKIPPED - local cache unavailable")

print("OPENCV REPAIR VERIFICATION COMPLETED")


# %% [cell 67] Experiment 02 - Step 2.15D

from pathlib import Path
from datetime import datetime, timezone
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import cv2
import hashlib
import json

# ============================================================
# Experiment 02 - Step 2.15D
# Independent Full-Dataset HOG Retrieval
# ============================================================

PROJECT_ROOT = DR_RESEARCH
RESULTS_DIR = PROJECT_ROOT / "experiment02_results"

MANIFEST_PATH = RESULTS_DIR / "FGADR_Image_SHA256_Manifest.csv"
CANDIDATE_PATH = RESULTS_DIR / "FGADR_Near_Duplicate_Candidates.csv"

TOP_K = 500
IMAGE_SIZE = 128

# ============================================================
# 1. Load and validate input manifests
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str, "sha256": str}
).sort_values("image_id").reset_index(drop=True)

candidates = pd.read_csv(
    CANDIDATE_PATH,
    dtype={"image_id_1": str, "image_id_2": str}
)

assert len(manifest) == 1842
assert manifest["image_id"].is_unique
assert len(candidates) == 43878

image_ids = manifest["image_id"].tolist()
image_index = {
    image_id: index
    for index, image_id in enumerate(image_ids)
}

assert all(
    Path(path).is_file()
    for path in manifest["local_path"]
), "Some local image files are missing."

candidate_pairs = {
    tuple(sorted((a, b)))
    for a, b in zip(
        candidates["image_id_1"],
        candidates["image_id_2"]
    )
}

assert len(candidate_pairs) == 43878

print("STEP 2.15D - INPUT VALIDATION")
print(f"Images: {len(manifest)}")
print(f"Initial candidate pairs: {len(candidate_pairs)}")
print("Local image availability: PASS")

# ============================================================
# 2. Configure HOG feature extraction
# ============================================================

hog = cv2.HOGDescriptor(
    (128, 128),
    (32, 32),
    (16, 16),
    (16, 16),
    9
)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

features = []
failed_images = []

for row in tqdm(
    manifest.itertuples(index=False),
    total=len(manifest),
    desc="Extracting HOG features"
):
    try:
        image = cv2.imread(
            str(row.local_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError("Image decoding failed.")

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        image = clahe.apply(image)

        descriptor = hog.compute(image).flatten()
        descriptor = descriptor.astype(np.float32)

        norm = np.linalg.norm(descriptor)

        if norm <= 1e-12 or not np.isfinite(norm):
            raise RuntimeError("Invalid feature descriptor.")

        features.append(descriptor / norm)

    except Exception as exc:
        failed_images.append({
            "image_id": row.image_id,
            "error": str(exc)
        })

if failed_images:
    error_path = RESULTS_DIR / "FGADR_HOG_Decode_Errors.csv"

    pd.DataFrame(failed_images).to_csv(
        error_path,
        index=False
    )

    raise RuntimeError(
        f"HOG extraction failed for {len(failed_images)} images."
    )

features = np.stack(features).astype(np.float32)

assert features.shape[0] == 1842
assert np.isfinite(features).all()

print("\nHOG FEATURE EXTRACTION")
print(f"Images processed: {len(features)}")
print(f"Feature dimensions: {features.shape[1]}")
print("Extraction errors: 0")

# ============================================================
# 3. Calculate full-dataset pairwise similarity
# ============================================================

similarity = features @ features.T
similarity = np.clip(similarity, -1.0, 1.0)

n = len(image_ids)
total_pairs = n * (n - 1) // 2

candidate_mask = np.zeros(
    (n, n),
    dtype=bool
)

for a, b in candidate_pairs:
    i = image_index[a]
    j = image_index[b]

    candidate_mask[i, j] = True
    candidate_mask[j, i] = True

np.fill_diagonal(similarity, -np.inf)

# ============================================================
# 4. Verify known exact duplicate pairs
# ============================================================

exact_duplicate_scores = []

for sha256, group in manifest.groupby("sha256"):
    if len(group) < 2:
        continue

    group_indices = [
        image_index[image_id]
        for image_id in group["image_id"]
    ]

    for index in range(1, len(group_indices)):
        i = group_indices[0]
        j = group_indices[index]

        exact_duplicate_scores.append(
            float(similarity[i, j])
        )

assert len(exact_duplicate_scores) == 14

if min(exact_duplicate_scores) < 0.999:
    raise RuntimeError(
        "Exact duplicate positive-control validation failed."
    )

print("\nEXACT DUPLICATE POSITIVE CONTROL")
print(f"Exact duplicate control pairs: {len(exact_duplicate_scores)}")
print(f"Minimum HOG similarity: {min(exact_duplicate_scores):.6f}")
print("Positive-control validation: PASS")

# ============================================================
# 5. Retrieve previously unseen high-similarity pairs
# ============================================================

upper_i, upper_j = np.triu_indices(n, k=1)

scores = similarity[upper_i, upper_j]
unseen_mask = ~candidate_mask[upper_i, upper_j]

unseen_indices = np.flatnonzero(unseen_mask)

top_count = min(TOP_K, len(unseen_indices))

ranking = np.argsort(
    scores[unseen_indices]
)[::-1][:top_count]

selected_indices = unseen_indices[ranking]

records = []

for index in selected_indices:
    i = int(upper_i[index])
    j = int(upper_j[index])

    records.append({
        "image_id_1": image_ids[i],
        "image_id_2": image_ids[j],
        "dr_grade_1": int(manifest.iloc[i]["dr_grade"]),
        "dr_grade_2": int(manifest.iloc[j]["dr_grade"]),
        "hog_cosine_similarity": float(similarity[i, j]),
        "initial_candidate": False,
        "review_status": "pending_verification"
    })

new_candidates = pd.DataFrame(records)

new_candidate_path = (
    RESULTS_DIR / "FGADR_HOG_Unseen_Candidate_Pairs.csv"
)

new_candidates.to_csv(
    new_candidate_path,
    index=False
)

# ============================================================
# 6. Audit images omitted by initial screening
# ============================================================

initial_candidate_ids = (
    set(candidates["image_id_1"])
    | set(candidates["image_id_2"])
)

omitted_ids = sorted(
    set(image_ids) - initial_candidate_ids
)

omitted_records = []

for image_id in omitted_ids:
    i = image_index[image_id]
    nearest_j = int(np.argmax(similarity[i]))

    omitted_records.append({
        "image_id": image_id,
        "nearest_image_id": image_ids[nearest_j],
        "hog_cosine_similarity": float(
            similarity[i, nearest_j]
        ),
        "was_initial_candidate": bool(
            candidate_mask[i, nearest_j]
        )
    })

omitted_df = pd.DataFrame(omitted_records)

omitted_df.to_csv(
    RESULTS_DIR / "FGADR_HOG_Previously_Omitted_Images.csv",
    index=False
)

# ============================================================
# 7. Save reproducibility parameters
# ============================================================

parameters = {
    "experiment": "Experiment02_Step2.15D",
    "method": "HOG_cosine_similarity",
    "image_size": IMAGE_SIZE,
    "top_k": TOP_K,
    "hog_window": [128, 128],
    "hog_block": [32, 32],
    "hog_stride": [16, 16],
    "hog_cell": [16, 16],
    "hog_bins": 9,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": [8, 8],
    "images": n,
    "total_pairs": total_pairs,
    "opencv_version": cv2.__version__,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "automatic_duplicate_confirmation": False,
    "created_utc": datetime.now(timezone.utc).isoformat()
}

with open(
    RESULTS_DIR / "FGADR_HOG_Parameters.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(parameters, file, indent=2)

# ============================================================
# 8. Final integrity checks and results
# ============================================================

assert len(new_candidates) == TOP_K
assert not new_candidates.duplicated(
    ["image_id_1", "image_id_2"]
).any()

assert len(omitted_df) == 142
assert not omitted_df["was_initial_candidate"].any()

print("\nINDEPENDENT HOG RETRIEVAL")
print(f"Images analyzed: {n}")
print(f"All possible pairs: {total_pairs}")
print(f"Original candidate pairs: {len(candidate_pairs)}")
print(f"Previously unseen pairs: {int(unseen_mask.sum())}")
print(f"New HOG-prioritized pairs: {len(new_candidates)}")
print(f"Previously omitted images: {len(omitted_df)}")
print(f"Exact duplicate control pairs: {len(exact_duplicate_scores)}")

print("\nTOP 20 NEW HOG CANDIDATES")
print(
    new_candidates.head(20).to_string(index=False)
)

print("\nPREVIOUSLY OMITTED IMAGE SUMMARY")
print(
    omitted_df["hog_cosine_similarity"]
    .describe()
    .round(6)
    .to_string()
)

print("\nSTEP 2.15D COMPLETED")


# %% [cell 68]

from pathlib import Path
import pandas as pd
from itertools import combinations

ROOT = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str, "sha256": str}
)

initial = pd.read_csv(
    ROOT / "FGADR_Near_Duplicate_Candidates.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

orb = pd.read_csv(
    ROOT / "FGADR_Expanded_ORB_Full_Results.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

hog = pd.read_csv(
    ROOT / "FGADR_HOG_Unseen_Candidate_Pairs.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

edges = pd.read_csv(
    ROOT / "FGADR_Updated_Risk_Evidence_Edges.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

def pair_set(df):
    return {
        tuple(sorted((a, b)))
        for a, b in zip(df["image_id_1"], df["image_id_2"])
    }

initial_pairs = pair_set(initial)
orb_pairs = pair_set(orb)
hog_pairs = pair_set(hog)
risk_pairs = pair_set(edges)

exact_pairs = set()

for _, group in manifest.groupby("sha256"):
    ids = sorted(group["image_id"].tolist())
    exact_pairs.update(combinations(ids, 2))

assert len(exact_pairs) == 14
assert len(initial_pairs) == 43878
assert len(orb_pairs) == 43878
assert len(hog_pairs) == 500

print("STEP 2.15E - CANDIDATE PROVENANCE AUDIT")

print("\nINITIAL VS EXPANDED ORB")
print("Initial candidate pairs:", len(initial_pairs))
print("Expanded ORB pairs:", len(orb_pairs))
print("Shared pairs:", len(initial_pairs & orb_pairs))
print("Initial-only pairs:", len(initial_pairs - orb_pairs))
print("ORB-only pairs:", len(orb_pairs - initial_pairs))
print("Candidate sets identical:", initial_pairs == orb_pairs)

print("\nEXACT DUPLICATE COVERAGE")
print("Confirmed exact pairs:", len(exact_pairs))
print("Present in initial:", len(exact_pairs & initial_pairs))
print("Present in expanded ORB:", len(exact_pairs & orb_pairs))
print("Present in HOG top 500:", len(exact_pairs & hog_pairs))

print("\nHOG TOP 500 PROVENANCE")
print("HOG pairs:", len(hog_pairs))
print("Already in initial:", len(hog_pairs & initial_pairs))
print("Already checked by expanded ORB:", len(hog_pairs & orb_pairs))
print("Already in risk evidence:", len(hog_pairs & risk_pairs))
print("Truly outside expanded ORB:", len(hog_pairs - orb_pairs))
print(
    "Truly new after excluding known exact pairs:",
    len(hog_pairs - orb_pairs - exact_pairs)
)

print("\nEXACT PAIR DETAILS")
for a, b in sorted(exact_pairs):
    print(
        a, b,
        "| initial:", (a, b) in initial_pairs,
        "| ORB:", (a, b) in orb_pairs,
        "| HOG:", (a, b) in hog_pairs
    )

print("\nSTEP 2.15E COMPLETED")


# %% [cell 69]

from pathlib import Path
from itertools import combinations
import pandas as pd

ROOT = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str, "sha256": str}
)

hog = pd.read_csv(
    ROOT / "FGADR_HOG_Unseen_Candidate_Pairs.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

members = pd.read_csv(
    ROOT / "FGADR_Updated_Risk_Group_Members.csv",
    dtype={"image_id": str, "risk_group_id": str}
)

orb = pd.read_csv(
    ROOT / "FGADR_Expanded_ORB_Full_Results.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

def normalize_pair(a, b):
    return tuple(sorted((str(a), str(b))))

exact_pairs = set()

for _, group in manifest.groupby("sha256"):
    ids = sorted(group["image_id"].tolist())
    exact_pairs.update(combinations(ids, 2))

orb_pairs = {
    normalize_pair(a, b)
    for a, b in zip(
        orb["image_id_1"],
        orb["image_id_2"]
    )
}

group_map = members.set_index(
    "image_id"
)["risk_group_id"].to_dict()

grade_map = manifest.set_index(
    "image_id"
)["dr_grade"].to_dict()

assert len(manifest) == 1842
assert len(hog) == 500
assert len(exact_pairs) == 14
assert len(group_map) == 1842

records = []

for row in hog.itertuples(index=False):
    a = row.image_id_1
    b = row.image_id_2
    pair = normalize_pair(a, b)

    if pair in exact_pairs:
        continue

    if pair in orb_pairs:
        raise RuntimeError(
            f"Unexpected previously tested pair: {a}, {b}"
        )

    group_a = group_map[a]
    group_b = group_map[b]

    grade_a = int(grade_map[a])
    grade_b = int(grade_map[b])

    records.append({
        "image_id_1": a,
        "image_id_2": b,
        "dr_grade_1": grade_a,
        "dr_grade_2": grade_b,
        "hog_cosine_similarity": float(
            row.hog_cosine_similarity
        ),
        "risk_group_1": group_a,
        "risk_group_2": group_b,
        "same_existing_risk_group": group_a == group_b,
        "grade_conflict": grade_a != grade_b,
        "geometry_status": "not_tested"
    })

triage = pd.DataFrame(records)

assert len(triage) == 486
assert not triage.duplicated(
    ["image_id_1", "image_id_2"]
).any()

# Rank cross-group candidates first, then grade conflicts,
# followed by HOG appearance similarity.
triage = triage.sort_values(
    [
        "same_existing_risk_group",
        "grade_conflict",
        "hog_cosine_similarity"
    ],
    ascending=[True, False, False]
).reset_index(drop=True)

triage["review_rank"] = range(1, len(triage) + 1)

output_path = (
    ROOT / "FGADR_HOG_486_Provenance_Triage.csv"
)

triage.to_csv(output_path, index=False)

cross = triage[
    ~triage["same_existing_risk_group"]
]

within = triage[
    triage["same_existing_risk_group"]
]

print("STEP 2.15F - HOG CANDIDATE TRIAGE")
print(f"Original HOG candidates: {len(hog)}")
print(f"Known exact duplicate pairs excluded: {len(exact_pairs)}")
print(f"Remaining untested HOG pairs: {len(triage)}")
print(f"Pairs within existing risk groups: {len(within)}")
print(f"Pairs connecting different risk groups: {len(cross)}")
print(f"Grade-conflicting pairs: {int(triage['grade_conflict'].sum())}")
print(
    "Cross-group grade-conflicting pairs:",
    int(cross["grade_conflict"].sum())
)

print("\nHOG SIMILARITY DISTRIBUTION")
print(
    triage["hog_cosine_similarity"]
    .describe()
    .round(6)
    .to_string()
)

print("\nTOP 25 PRIORITY CANDIDATES")
print(
    triage[
        [
            "image_id_1",
            "image_id_2",
            "dr_grade_1",
            "dr_grade_2",
            "hog_cosine_similarity",
            "same_existing_risk_group",
            "grade_conflict"
        ]
    ].head(25).to_string(index=False)
)

print("\nOUTPUT FILE:", output_path)
print("\nSTEP 2.15F COMPLETED")


# %% [cell 70]

from pathlib import Path
import pandas as pd
import numpy as np
import cv2

ROOT = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

orb_reference = pd.read_csv(
    ROOT / "FGADR_Expanded_ORB_Full_Results.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

hog_candidates = pd.read_csv(
    ROOT / "FGADR_HOG_486_Provenance_Triage.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

assert len(manifest) == 1842
assert len(hog_candidates) == 486

print("STEP 2.15G-1 - GEOMETRIC VERIFICATION PREFLIGHT")
print("OpenCV version:", cv2.__version__)
print("Manifest rows:", len(manifest))
print("Reference ORB rows:", len(orb_reference))
print("HOG candidate rows:", len(hog_candidates))

print("\nREFERENCE ORB COLUMNS")
print(orb_reference.columns.tolist())

print("\nHOG CANDIDATE COLUMNS")
print(hog_candidates.columns.tolist())

print("\nREFERENCE STRONG CANDIDATE SAMPLE")
reference_sample = orb_reference.loc[
    orb_reference["ransac_inliers"] >= 100,
    [
        "image_id_1",
        "image_id_2",
        "ransac_inliers",
        "inlier_ratio",
        "homography_valid"
    ]
].sort_values(
    ["ransac_inliers", "image_id_1", "image_id_2"]
).head(10)

print(reference_sample.to_string(index=False))

print("\nGEOMETRIC VERIFICATION PREFLIGHT COMPLETED")


# %% [cell 71] Experiment 02 - Step 2.15G-2

from pathlib import Path
from functools import lru_cache
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import cv2
import json

# ============================================================
# Experiment 02 - Step 2.15G-2
# ORB Reference Reproducibility Test
# ============================================================

ROOT = (DR_RESEARCH / "experiment02_results")

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

reference = pd.read_csv(
    ROOT / "FGADR_Expanded_ORB_Full_Results.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

path_map = manifest.set_index("image_id")["local_path"].to_dict()

IMAGE_SIZE = 640
ORB_FEATURES = 2500
LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 3.0
RANSAC_ITERATIONS = 2000
RANSAC_CONFIDENCE = 0.995
RANDOM_SEED = 42

orb_detector = cv2.ORB_create(
    nfeatures=ORB_FEATURES
)

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

@lru_cache(maxsize=256)
def extract_features(image_id):
    image_path = Path(path_map[image_id])

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Cannot decode image: {image_id}"
        )

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    image = clahe.apply(image)

    keypoints, descriptors = orb_detector.detectAndCompute(
        image,
        None
    )

    return keypoints, descriptors

def ratio_filter(knn_matches):
    accepted = {}

    for neighbors in knn_matches:
        if len(neighbors) < 2:
            continue

        first, second = neighbors

        if first.distance < LOWE_RATIO * second.distance:
            accepted[first.queryIdx] = first

    return accepted

def verify_pair(image_id_1, image_id_2):
    kp1, des1 = extract_features(image_id_1)
    kp2, des2 = extract_features(image_id_2)

    empty_result = {
        "mutual_matches": 0,
        "ransac_inliers": 0,
        "inlier_ratio": 0.0,
        "median_reprojection_error_px": np.nan,
        "homography_valid": False,
        "status": "success"
    }

    if des1 is None or des2 is None:
        return empty_result

    if len(des1) < 4 or len(des2) < 4:
        return empty_result

    forward = ratio_filter(
        matcher.knnMatch(des1, des2, k=2)
    )

    backward = ratio_filter(
        matcher.knnMatch(des2, des1, k=2)
    )

    mutual = []

    for query_idx, match in forward.items():
        reverse = backward.get(match.trainIdx)

        if reverse is not None and reverse.trainIdx == query_idx:
            mutual.append(match)

    mutual.sort(
        key=lambda match: (match.queryIdx, match.trainIdx)
    )

    count = len(mutual)

    if count < 4:
        return {
            **empty_result,
            "mutual_matches": count
        }

    source_points = np.float32([
        kp1[match.queryIdx].pt
        for match in mutual
    ]).reshape(-1, 1, 2)

    target_points = np.float32([
        kp2[match.trainIdx].pt
        for match in mutual
    ]).reshape(-1, 1, 2)

    cv2.setRNGSeed(RANDOM_SEED)

    homography, inlier_mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
        maxIters=RANSAC_ITERATIONS,
        confidence=RANSAC_CONFIDENCE
    )

    if homography is None or inlier_mask is None:
        return {
            **empty_result,
            "mutual_matches": count
        }

    valid_homography = bool(
        np.isfinite(homography).all()
        and np.linalg.matrix_rank(homography) == 3
    )

    if not valid_homography:
        return {
            **empty_result,
            "mutual_matches": count
        }

    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())

    projected = cv2.perspectiveTransform(
        source_points,
        homography
    )

    errors = np.linalg.norm(
        projected.reshape(-1, 2)
        - target_points.reshape(-1, 2),
        axis=1
    )

    median_error = (
        float(np.median(errors[inliers]))
        if inlier_count > 0
        else np.nan
    )

    return {
        "mutual_matches": count,
        "ransac_inliers": inlier_count,
        "inlier_ratio": inlier_count / count,
        "median_reprojection_error_px": median_error,
        "homography_valid": True,
        "status": "success"
    }

# ============================================================
# Select deterministic reference controls
# ============================================================

strong = reference.loc[
    (reference["ransac_inliers"] >= 100)
    & (
        reference["homography_valid"]
        .astype(str)
        .str.lower() == "true"
    )
].copy()

# Include threshold-near and high-inlier reference cases
low_controls = strong.sort_values(
    ["ransac_inliers", "image_id_1", "image_id_2"]
).head(5)

high_controls = strong.sort_values(
    ["ransac_inliers", "image_id_1", "image_id_2"],
    ascending=[False, True, True]
).head(5)

controls = pd.concat(
    [low_controls, high_controls],
    ignore_index=True
)

assert len(controls) == 10

results = []

for row in controls.itertuples(index=False):
    result = verify_pair(
        row.image_id_1,
        row.image_id_2
    )

    results.append({
        "image_id_1": row.image_id_1,
        "image_id_2": row.image_id_2,
        "reference_inliers": int(row.ransac_inliers),
        "new_inliers": result["ransac_inliers"],
        "reference_ratio": float(row.inlier_ratio),
        "new_ratio": result["inlier_ratio"],
        "reference_mutual_matches": int(row.mutual_matches),
        "new_mutual_matches": result["mutual_matches"],
        "new_homography_valid": result["homography_valid"]
    })

control_df = pd.DataFrame(results)

control_df["inlier_difference"] = (
    control_df["new_inliers"]
    - control_df["reference_inliers"]
)

control_df["absolute_inlier_difference"] = (
    control_df["inlier_difference"].abs()
)

control_df.to_csv(
    ROOT / "FGADR_ORB_Version_Reference_Controls.csv",
    index=False
)

parameters = {
    "experiment": "Experiment02_Step2.15G",
    "opencv_version": cv2.__version__,
    "image_size": IMAGE_SIZE,
    "orb_features": ORB_FEATURES,
    "lowe_ratio": LOWE_RATIO,
    "mutual_matching": True,
    "ransac_threshold_px": RANSAC_THRESHOLD,
    "ransac_max_iterations": RANSAC_ITERATIONS,
    "ransac_confidence": RANSAC_CONFIDENCE,
    "random_seed": RANDOM_SEED,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": [8, 8],
    "created_utc": datetime.now(timezone.utc).isoformat()
}

with open(
    ROOT / "FGADR_ORB_HOG_Verification_Parameters.json",
    "w",
    encoding="utf-8"
) as handle:
    json.dump(parameters, handle, indent=2)

print("STEP 2.15G-2 - ORB REFERENCE REPRODUCIBILITY")

print("\nREFERENCE COMPARISON")
print(control_df.to_string(index=False))

print("\nSUMMARY")
print("Control pairs:", len(control_df))
print(
    "Mean absolute inlier difference:",
    round(control_df["absolute_inlier_difference"].mean(), 3)
)
print(
    "Maximum absolute inlier difference:",
    int(control_df["absolute_inlier_difference"].max())
)
print(
    "Controls still above 100 inliers:",
    int((control_df["new_inliers"] >= 100).sum())
)
print(
    "Valid new homographies:",
    int(control_df["new_homography_valid"].sum())
)

print("\nSTEP 2.15G-2 COMPLETED")


# %% [cell 72] Experiment 02 - Step 2.15G-3

# ============================================================
# Experiment 02 - Step 2.15G-3
# HOG Candidate ORB/RANSAC Verification
# ============================================================

from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import cv2
import json
import os
from datetime import datetime, timezone

ROOT = (DR_RESEARCH / "experiment02_results")

INPUT_PATH = ROOT / "FGADR_HOG_486_Provenance_Triage.csv"
OUTPUT_PATH = ROOT / "FGADR_HOG_486_ORB_Verification.csv"
SUMMARY_PATH = ROOT / "FGADR_HOG_486_ORB_Summary.json"

# Confirm that the reference-tested function remains available.
assert callable(verify_pair), (
    "Run Step 2.15G-2 before this cell."
)

assert cv2.__version__ == "4.14.0"

candidates = pd.read_csv(
    INPUT_PATH,
    dtype={
        "image_id_1": str,
        "image_id_2": str
    }
)

assert len(candidates) == 486

pair_keys = [
    tuple(sorted((a, b)))
    for a, b in zip(
        candidates["image_id_1"],
        candidates["image_id_2"]
    )
]

assert len(set(pair_keys)) == 486

result_columns = [
    "image_id_1",
    "image_id_2",
    "dr_grade_1",
    "dr_grade_2",
    "hog_cosine_similarity",
    "same_existing_risk_group",
    "grade_conflict",
    "review_rank",
    "mutual_matches",
    "ransac_inliers",
    "inlier_ratio",
    "median_reprojection_error_px",
    "homography_valid",
    "status",
    "error"
]

# ============================================================
# 1. Restore completed work if available
# ============================================================

if OUTPUT_PATH.is_file():
    results = pd.read_csv(
        OUTPUT_PATH,
        dtype={
            "image_id_1": str,
            "image_id_2": str
        }
    )

    assert not results.duplicated(
        ["image_id_1", "image_id_2"]
    ).any()

    assert set(zip(
        results["image_id_1"],
        results["image_id_2"]
    )).issubset(set(zip(
        candidates["image_id_1"],
        candidates["image_id_2"]
    )))

    records = results.to_dict("records")
else:
    records = []

completed = {
    (row["image_id_1"], row["image_id_2"])
    for row in records
}

pending = candidates.loc[
    ~pd.Series(
        list(zip(
            candidates["image_id_1"],
            candidates["image_id_2"]
        )),
        index=candidates.index
    ).isin(completed)
]

print("STEP 2.15G-3 - BATCH VERIFICATION")
print("Total candidates:", len(candidates))
print("Previously saved:", len(records))
print("Remaining:", len(pending))
print("OpenCV version:", cv2.__version__)

# ============================================================
# 2. Save results atomically
# ============================================================

def save_progress(rows):
    frame = pd.DataFrame(
        rows,
        columns=result_columns
    )

    temporary_path = OUTPUT_PATH.with_suffix(".tmp.csv")

    frame.to_csv(
        temporary_path,
        index=False
    )

    os.replace(temporary_path, OUTPUT_PATH)

# ============================================================
# 3. Verify candidates
# ============================================================

for index, row in enumerate(
    tqdm(
        pending.itertuples(index=False),
        total=len(pending),
        desc="Verifying HOG candidates"
    ),
    start=1
):
    try:
        geometry = verify_pair(
            row.image_id_1,
            row.image_id_2
        )

        status = geometry["status"]
        error = ""

    except Exception as exc:
        geometry = {
            "mutual_matches": 0,
            "ransac_inliers": 0,
            "inlier_ratio": 0.0,
            "median_reprojection_error_px": np.nan,
            "homography_valid": False
        }

        status = "error"
        error = repr(exc)

    records.append({
        "image_id_1": row.image_id_1,
        "image_id_2": row.image_id_2,
        "dr_grade_1": int(row.dr_grade_1),
        "dr_grade_2": int(row.dr_grade_2),
        "hog_cosine_similarity": float(
            row.hog_cosine_similarity
        ),
        "same_existing_risk_group": bool(
            row.same_existing_risk_group
        ),
        "grade_conflict": bool(
            row.grade_conflict
        ),
        "review_rank": int(row.review_rank),
        **geometry,
        "status": status,
        "error": error
    })

    if index % 25 == 0:
        save_progress(records)

save_progress(records)

# ============================================================
# 4. Validate and summarize results
# ============================================================

results = pd.read_csv(
    OUTPUT_PATH,
    dtype={
        "image_id_1": str,
        "image_id_2": str
    }
)

assert len(results) == 486
assert not results.duplicated(
    ["image_id_1", "image_id_2"]
).any()

valid_homography = (
    results["homography_valid"]
    .astype(str)
    .str.lower()
    .eq("true")
)

strong = results.loc[
    results["status"].eq("success")
    & valid_homography
    & results["ransac_inliers"].ge(100)
].copy()

cross_group = (
    ~results["same_existing_risk_group"]
    .astype(str).str.lower().eq("true")
)

strong_cross = strong.loc[
    ~strong["same_existing_risk_group"]
    .astype(str).str.lower().eq("true")
]

strong_cross_path = (
    ROOT / "FGADR_HOG_New_Strong_Cross_Group_Candidates.csv"
)

strong_cross.to_csv(
    strong_cross_path,
    index=False
)

summary = {
    "experiment": "Experiment02_Step2.15G-3",
    "opencv_version": cv2.__version__,
    "total_pairs": len(results),
    "successful_pairs": int(
        results["status"].eq("success").sum()
    ),
    "error_pairs": int(
        results["status"].eq("error").sum()
    ),
    "strong_geometric_pairs": len(strong),
    "strong_cross_group_pairs": len(strong_cross),
    "strong_cross_group_grade_conflicts": int(
        strong_cross["grade_conflict"]
        .astype(str).str.lower().eq("true").sum()
    ),
    "completed_utc": datetime.now(
        timezone.utc
    ).isoformat()
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as handle:
    json.dump(summary, handle, indent=2)

print("\nHOG-ORB VERIFICATION SUMMARY")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nRANSAC INLIER DISTRIBUTION")
print(
    pd.cut(
        results["ransac_inliers"],
        bins=[-1, 0, 9, 39, 99, 199, 299, 499, np.inf],
        labels=[
            "0",
            "1-9",
            "10-39",
            "40-99",
            "100-199",
            "200-299",
            "300-499",
            "500+"
        ]
    ).value_counts(sort=False).to_string()
)

print("\nTOP 20 STRONG CROSS-GROUP CANDIDATES")

if len(strong_cross) > 0:
    print(
        strong_cross.sort_values(
            "ransac_inliers",
            ascending=False
        )[
            [
                "image_id_1",
                "image_id_2",
                "dr_grade_1",
                "dr_grade_2",
                "hog_cosine_similarity",
                "mutual_matches",
                "ransac_inliers",
                "inlier_ratio",
                "median_reprojection_error_px"
            ]
        ].head(20).to_string(index=False)
    )
else:
    print("No strong cross-group candidates found.")

print("\nOUTPUT FILES")
print(OUTPUT_PATH)
print(strong_cross_path)
print(SUMMARY_PATH)

print("\nSTEP 2.15G-3 COMPLETED")


# %% [cell 73] Experiment 02 - Step 2.15G-4

# ============================================================
# Experiment 02 - Step 2.15G-4
# Spatial Validation of New Cross-Group Candidates
# ============================================================

from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import cv2

ROOT = (DR_RESEARCH / "experiment02_results")

INPUT_PATH = ROOT / "FGADR_HOG_New_Strong_Cross_Group_Candidates.csv"
OUTPUT_PATH = ROOT / "FGADR_HOG_Cross_Group_Spatial_Validation.csv"

pairs = pd.read_csv(
    INPUT_PATH,
    dtype={"image_id_1": str, "image_id_2": str}
)

assert len(pairs) == 71
assert callable(extract_features)
assert callable(ratio_filter)
assert IMAGE_SIZE == 640
assert cv2.__version__ == "4.14.0"

def spatial_metrics(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    hull = cv2.convexHull(points)
    hull_area = float(cv2.contourArea(hull))
    coverage = hull_area / float(IMAGE_SIZE ** 2)

    mid = IMAGE_SIZE / 2.0
    quadrant_ids = set()

    for x, y in points:
        quadrant_ids.add(
            (int(x >= mid), int(y >= mid))
        )

    return coverage, len(quadrant_ids)

def validate_spatial_pair(image_id_1, image_id_2):
    kp1, des1 = extract_features(image_id_1)
    kp2, des2 = extract_features(image_id_2)

    if des1 is None or des2 is None:
        raise RuntimeError("Missing feature descriptors.")

    forward = ratio_filter(matcher.knnMatch(des1, des2, k=2))
    backward = ratio_filter(matcher.knnMatch(des2, des1, k=2))

    mutual = []

    for query_idx, match in forward.items():
        reverse = backward.get(match.trainIdx)

        if reverse is not None and reverse.trainIdx == query_idx:
            mutual.append(match)

    mutual.sort(key=lambda m: (m.queryIdx, m.trainIdx))

    if len(mutual) < 4:
        raise RuntimeError("Insufficient mutual matches.")

    src = np.float32([
        kp1[m.queryIdx].pt for m in mutual
    ]).reshape(-1, 1, 2)

    dst = np.float32([
        kp2[m.trainIdx].pt for m in mutual
    ]).reshape(-1, 1, 2)

    cv2.setRNGSeed(RANDOM_SEED)

    homography, mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
        maxIters=RANSAC_ITERATIONS,
        confidence=RANSAC_CONFIDENCE
    )

    if homography is None or mask is None:
        raise RuntimeError("Homography estimation failed.")

    if not np.isfinite(homography).all():
        raise RuntimeError("Non-finite homography.")

    inliers = mask.ravel().astype(bool)
    count = int(inliers.sum())

    src_inliers = src.reshape(-1, 2)[inliers]
    dst_inliers = dst.reshape(-1, 2)[inliers]

    src_coverage, src_quadrants = spatial_metrics(src_inliers)
    dst_coverage, dst_quadrants = spatial_metrics(dst_inliers)

    projected = cv2.perspectiveTransform(src, homography)

    errors = np.linalg.norm(
        projected.reshape(-1, 2) - dst.reshape(-1, 2),
        axis=1
    )

    return {
        "recomputed_inliers": count,
        "source_hull_coverage": src_coverage,
        "target_hull_coverage": dst_coverage,
        "source_quadrants": src_quadrants,
        "target_quadrants": dst_quadrants,
        "median_reprojection_error_px": float(
            np.median(errors[inliers])
        )
    }

records = []

for row in tqdm(
    pairs.itertuples(index=False),
    total=len(pairs),
    desc="Checking spatial coverage"
):
    try:
        metrics = validate_spatial_pair(
            row.image_id_1,
            row.image_id_2
        )

        count_matches = (
            metrics["recomputed_inliers"]
            == int(row.ransac_inliers)
        )

        records.append({
            "image_id_1": row.image_id_1,
            "image_id_2": row.image_id_2,
            "dr_grade_1": int(row.dr_grade_1),
            "dr_grade_2": int(row.dr_grade_2),
            "original_inliers": int(row.ransac_inliers),
            **metrics,
            "inlier_count_consistent": count_matches,
            "status": "success",
            "error": ""
        })

    except Exception as exc:
        records.append({
            "image_id_1": row.image_id_1,
            "image_id_2": row.image_id_2,
            "dr_grade_1": int(row.dr_grade_1),
            "dr_grade_2": int(row.dr_grade_2),
            "original_inliers": int(row.ransac_inliers),
            "recomputed_inliers": np.nan,
            "source_hull_coverage": np.nan,
            "target_hull_coverage": np.nan,
            "source_quadrants": np.nan,
            "target_quadrants": np.nan,
            "median_reprojection_error_px": np.nan,
            "inlier_count_consistent": False,
            "status": "error",
            "error": repr(exc)
        })

audit = pd.DataFrame(records)

audit["low_spatial_coverage"] = (
    audit["source_hull_coverage"].lt(0.05)
    | audit["target_hull_coverage"].lt(0.05)
)

audit.to_csv(OUTPUT_PATH, index=False)

print("STEP 2.15G-4 - SPATIAL VALIDATION")
print("Candidate pairs:", len(audit))
print("Successful validations:", int(audit["status"].eq("success").sum()))
print("Validation errors:", int(audit["status"].eq("error").sum()))
print("Inlier count mismatches:", int((~audit["inlier_count_consistent"]).sum()))
print("Low coverage pairs (<5%):", int(audit["low_spatial_coverage"].sum()))
print("Pairs with both hull coverages >=5%:", int((
    audit["source_hull_coverage"].ge(0.05)
    & audit["target_hull_coverage"].ge(0.05)
).sum()))

print("\nSPATIAL COVERAGE SUMMARY")
print(
    audit[
        [
            "source_hull_coverage",
            "target_hull_coverage",
            "source_quadrants",
            "target_quadrants",
            "median_reprojection_error_px"
        ]
    ].describe().round(5).to_string()
)

print("\nLOWEST COVERAGE PAIRS")
audit["minimum_coverage"] = audit[
    ["source_hull_coverage", "target_hull_coverage"]
].min(axis=1)

print(
    audit.sort_values("minimum_coverage")[
        [
            "image_id_1",
            "image_id_2",
            "original_inliers",
            "source_hull_coverage",
            "target_hull_coverage",
            "source_quadrants",
            "target_quadrants"
        ]
    ].head(15).to_string(index=False)
)

print("\nOUTPUT FILE:", OUTPUT_PATH)
print("STEP 2.15G-4 COMPLETED")


# %% [cell 74] Experiment 02 - Step 2.16A

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone

# ============================================================
# Experiment 02 - Step 2.16A
# Leakage-Constrained FGADR Split V1
# ============================================================

ROOT = (DR_RESEARCH / "experiment02_results")
OUTPUT = ROOT / "FGADR_Split_V1.csv"
GROUP_OUTPUT = ROOT / "FGADR_Split_V1_Groups.csv"
SUMMARY_OUTPUT = ROOT / "FGADR_Split_V1_Summary.json"

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str}
)

old_members = pd.read_csv(
    ROOT / "FGADR_Updated_Risk_Group_Members.csv",
    dtype={"image_id": str, "risk_group_id": str}
)

spatial = pd.read_csv(
    ROOT / "FGADR_HOG_Cross_Group_Spatial_Validation.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)

assert len(manifest) == 1842
assert manifest["image_id"].is_unique
assert len(old_members) == 1842
assert old_members["image_id"].is_unique
assert len(spatial) == 71
assert spatial["status"].eq("success").all()
assert spatial["inlier_count_consistent"].astype(str).str.lower().eq("true").all()
assert spatial["original_inliers"].ge(100).all()

image_ids = sorted(manifest["image_id"])
image_set = set(image_ids)

assert set(old_members["image_id"]) == image_set

parent = {image_id: image_id for image_id in image_ids}

def find(image_id):
    while parent[image_id] != image_id:
        parent[image_id] = parent[parent[image_id]]
        image_id = parent[image_id]
    return image_id

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)

# Preserve every existing provisional risk group.
for _, group in old_members.groupby("risk_group_id"):
    ids = sorted(group["image_id"])
    for image_id in ids[1:]:
        union(ids[0], image_id)

# Add the 71 new spatially checked cross-group candidates.
for row in spatial.itertuples(index=False):
    assert row.image_id_1 in image_set
    assert row.image_id_2 in image_set
    union(row.image_id_1, row.image_id_2)

components = defaultdict(list)
for image_id in image_ids:
    components[find(image_id)].append(image_id)

grade_map = manifest.set_index("image_id")["dr_grade"].astype(int).to_dict()
grades = sorted(set(grade_map.values()))
assert grades == [0, 1, 2, 3, 4]

group_records = []
for members in components.values():
    members = sorted(members)
    counts = np.array(
        [sum(grade_map[i] == grade for i in members) for grade in grades],
        dtype=int
    )
    group_records.append({
        "risk_group_id": "RGV1_" + members[0],
        "members": members,
        "count": len(members),
        "grade_counts": counts,
        "grade_conflict": int(np.count_nonzero(counts) > 1) > 1
    })

# ============================================================
# Stratified group allocation with reproducible restarts
# ============================================================

split_names = ["train", "val", "test"]
fractions = np.array([0.70, 0.15, 0.15])
total_by_grade = np.array(
    [sum(grade_map[i] == g for i in image_ids) for g in grades],
    dtype=float
)
targets = fractions[:, None] * total_by_grade[None, :]
total_targets = fractions * len(image_ids)

def allocate(seed):
    rng = np.random.default_rng(seed)
    order = list(range(len(group_records)))
    rng.shuffle(order)
    order.sort(
        key=lambda k: group_records[k]["count"],
        reverse=True
    )

    allocations = {}
    current = np.zeros((3, len(grades)), dtype=float)
    totals = np.zeros(3, dtype=float)

    # Greedy assignment minimizing grade and size imbalance.
    for idx in order:
        item = group_records[idx]
        scores = []

        for s in range(3):
            trial = current.copy()
            trial_totals = totals.copy()
            trial[s] += item["grade_counts"]
            trial_totals[s] += item["count"]

            grade_loss = np.sum(
                ((trial - targets) / np.maximum(targets, 1.0)) ** 2
            )
            size_loss = np.sum(
                ((trial_totals - total_targets) /
                 np.maximum(total_targets, 1.0)) ** 2
            )

            scores.append(grade_loss + 0.25 * size_loss)

        minimum = min(scores)
        tied = [s for s, value in enumerate(scores)
                if abs(value - minimum) < 1e-12]
        chosen = int(rng.choice(tied))

        allocations[item["risk_group_id"]] = split_names[chosen]
        current[chosen] += item["grade_counts"]
        totals[chosen] += item["count"]

    # Final balance is evaluated independently of processing order.
    final_score = float(
        np.sum(((current - targets) / np.maximum(targets, 1.0)) ** 2)
        + 0.25 * np.sum(
            ((totals - total_targets) /
             np.maximum(total_targets, 1.0)) ** 2
        )
    )
    return final_score, allocations

best_score = np.inf
best_allocation = None
best_seed = None

for seed in range(42, 242):
    score, allocation = allocate(seed)
    if score < best_score:
        best_score = score
        best_allocation = allocation
        best_seed = seed

# ============================================================
# Build outputs
# ============================================================

image_records = []
group_rows = []

for item in group_records:
    group_id = item["risk_group_id"]
    split = best_allocation[group_id]

    group_rows.append({
        "risk_group_id": group_id,
        "split": split,
        "image_count": item["count"],
        "grade_conflict": item["grade_conflict"],
        "image_ids": ";".join(item["members"])
    })

    for image_id in item["members"]:
        image_records.append({
            "image_id": image_id,
            "dr_grade": grade_map[image_id],
            "risk_group_id": group_id,
            "split": split,
            "risk_group_size": item["count"],
            "group_grade_conflict": item["grade_conflict"]
        })

split_df = pd.DataFrame(image_records).sort_values("image_id")
group_df = pd.DataFrame(group_rows).sort_values("risk_group_id")

# ============================================================
# Leakage constraint checks
# ============================================================

assert len(split_df) == 1842
assert split_df["image_id"].is_unique
assert split_df["split"].nunique() == 3
assert split_df.groupby("risk_group_id")["split"].nunique().max() == 1

split_map = split_df.set_index("image_id")["split"].to_dict()

crossing_new_edges = sum(
    split_map[row.image_id_1] != split_map[row.image_id_2]
    for row in spatial.itertuples(index=False)
)
assert crossing_new_edges == 0

# Verify original edge-level constraints as well.
old_edges = pd.read_csv(
    ROOT / "FGADR_Updated_Risk_Evidence_Edges.csv",
    dtype={"image_id_1": str, "image_id_2": str}
)
crossing_old_edges = sum(
    split_map[row.image_id_1] != split_map[row.image_id_2]
    for row in old_edges.itertuples(index=False)
)
assert crossing_old_edges == 0

if OUTPUT.exists() or GROUP_OUTPUT.exists():
    raise FileExistsError(
        "Split V1 already exists. Do not overwrite a frozen split."
    )

split_df.to_csv(OUTPUT, index=False)
group_df.to_csv(GROUP_OUTPUT, index=False)

distribution = pd.crosstab(
    split_df["split"],
    split_df["dr_grade"]
).reindex(index=split_names, columns=grades, fill_value=0)

summary = {
    "split_version": "FGADR_Split_V1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "seed_search": [42, 241],
    "selected_seed": best_seed,
    "allocation_score": best_score,
    "images": len(split_df),
    "risk_groups": len(group_df),
    "multi_image_groups": int((group_df["image_count"] > 1).sum()),
    "largest_group_size": int(group_df["image_count"].max()),
    "grade_conflicting_groups": int(group_df["grade_conflict"].sum()),
    "cross_split_old_edges": crossing_old_edges,
    "cross_split_new_edges": crossing_new_edges,
    "provisional_source_groups_not_patient_ids": True
}

with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("STEP 2.16A - FGADR SPLIT V1")
print(f"Images: {len(split_df)}")
print(f"Risk groups: {len(group_df)}")
print(f"Multi-image groups: {summary['multi_image_groups']}")
print(f"Largest group: {summary['largest_group_size']}")
print(f"Grade-conflicting groups: {summary['grade_conflicting_groups']}")
print(f"Selected allocation seed: {best_seed}")
print(f"Old evidence edges crossing splits: {crossing_old_edges}")
print(f"New evidence edges crossing splits: {crossing_new_edges}")

print("\nSPLIT DISTRIBUTION")
distribution["total"] = distribution.sum(axis=1)
print(distribution.to_string())

print("\nSPLIT PROPORTIONS")
print((distribution["total"] / len(split_df)).round(4).to_string())

print("\nOUTPUT FILES")
print(OUTPUT)
print(GROUP_OUTPUT)
print(SUMMARY_OUTPUT)
print("\nSTEP 2.16A COMPLETED")


# %% [cell 75] Experiment 02 - Step 2.16B

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import json
import shutil
import hashlib

# ============================================================
# Experiment 02 - Step 2.16B
# Split V1 Metadata Correction and Independent Integrity Audit
# ============================================================

ROOT = (DR_RESEARCH / "experiment02_results")

SPLIT_PATH = ROOT / "FGADR_Split_V1.csv"
GROUP_PATH = ROOT / "FGADR_Split_V1_Groups.csv"
SUMMARY_PATH = ROOT / "FGADR_Split_V1_Summary.json"

MANIFEST_PATH = ROOT / "FGADR_Image_SHA256_Manifest.csv"
OLD_EDGES_PATH = ROOT / "FGADR_Updated_Risk_Evidence_Edges.csv"
NEW_EDGES_PATH = ROOT / "FGADR_HOG_Cross_Group_Spatial_Validation.csv"

split_df = pd.read_csv(SPLIT_PATH, dtype={"image_id": str})
group_df = pd.read_csv(GROUP_PATH, dtype={"risk_group_id": str})
manifest = pd.read_csv(MANIFEST_PATH, dtype={"image_id": str})

old_edges = pd.read_csv(
    OLD_EDGES_PATH,
    dtype={"image_id_1": str, "image_id_2": str}
)

new_edges = pd.read_csv(
    NEW_EDGES_PATH,
    dtype={"image_id_1": str, "image_id_2": str}
)

with open(SUMMARY_PATH, "r", encoding="utf-8") as handle:
    summary = json.load(handle)

# ============================================================
# 1. Independent image and label integrity checks
# ============================================================

assert len(split_df) == 1842
assert len(manifest) == 1842
assert split_df["image_id"].is_unique
assert manifest["image_id"].is_unique
assert set(split_df["image_id"]) == set(manifest["image_id"])

assert len(group_df) == 1453
assert group_df["risk_group_id"].is_unique

assert set(split_df["split"]) == {"train", "val", "test"}

source_grades = manifest.set_index("image_id")["dr_grade"].astype(int)
split_grades = split_df.set_index("image_id")["dr_grade"].astype(int)

assert split_grades.sort_index().equals(
    source_grades.sort_index()
), "Split labels do not match the source manifest."

# ============================================================
# 2. Recompute group conflicts from original labels
# ============================================================

group_grade_counts = split_df.groupby(
    "risk_group_id"
)["dr_grade"].nunique()

group_sizes = split_df.groupby(
    "risk_group_id"
)["image_id"].size()

corrected_conflicts = group_grade_counts.gt(1)

assert set(group_df["risk_group_id"]) == set(group_grade_counts.index)

group_df["grade_conflict"] = (
    group_df["risk_group_id"]
    .map(corrected_conflicts)
    .astype(bool)
)

group_df["grade_count"] = (
    group_df["risk_group_id"]
    .map(group_grade_counts)
    .astype(int)
)

split_df["group_grade_conflict"] = (
    split_df["risk_group_id"]
    .map(corrected_conflicts)
    .astype(bool)
)

assert (
    group_df.set_index("risk_group_id")["image_count"]
    .astype(int).sort_index()
    .equals(group_sizes.sort_index())
)

# Validate stored group membership lists.
for row in group_df.itertuples(index=False):
    stored_members = set(str(row.image_ids).split(";"))
    actual_members = set(
        split_df.loc[
            split_df["risk_group_id"] == row.risk_group_id,
            "image_id"
        ]
    )
    assert stored_members == actual_members, (
        f"Membership mismatch: {row.risk_group_id}"
    )

# ============================================================
# 3. Verify leakage constraints independently
# ============================================================

split_map = split_df.set_index("image_id")["split"].to_dict()

assert split_df.groupby(
    "risk_group_id"
)["split"].nunique().max() == 1

def count_crossing_edges(edge_frame):
    return sum(
        split_map[a] != split_map[b]
        for a, b in zip(
            edge_frame["image_id_1"],
            edge_frame["image_id_2"]
        )
    )

old_crossings = count_crossing_edges(old_edges)
new_crossings = count_crossing_edges(new_edges)

assert len(old_edges) == 541
assert len(new_edges) == 71
assert old_crossings == 0
assert new_crossings == 0

# Verify that all exact duplicates remain within one split.
exact_crossings = 0

for _, subset in manifest.groupby("sha256"):
    splits = {
        split_map[image_id]
        for image_id in subset["image_id"]
    }
    if len(splits) > 1:
        exact_crossings += 1

assert exact_crossings == 0

# Existing 80 conflicting groups must remain represented.
conflicting_groups = int(corrected_conflicts.sum())
assert conflicting_groups >= 80

# ============================================================
# 4. Back up original outputs before correcting metadata
# ============================================================

backup_suffix = ".pre_conflict_flag_fix"

for path in [SPLIT_PATH, GROUP_PATH, SUMMARY_PATH]:
    backup = path.with_name(path.name + backup_suffix)

    if not backup.exists():
        shutil.copy2(path, backup)

split_df.to_csv(SPLIT_PATH, index=False)
group_df.to_csv(GROUP_PATH, index=False)

summary["grade_conflicting_groups"] = conflicting_groups
summary["images_in_grade_conflicting_groups"] = int(
    split_df["group_grade_conflict"].sum()
)
summary["cross_split_old_edges"] = old_crossings
summary["cross_split_new_edges"] = new_crossings
summary["cross_split_exact_duplicate_groups"] = exact_crossings
summary["metadata_correction"] = {
    "corrected_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "Corrected grade conflict predicate",
    "split_assignment_changed": False,
    "original_outputs_backed_up": True
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

# ============================================================
# 5. Verify persisted files and report final results
# ============================================================

verified_split = pd.read_csv(SPLIT_PATH, dtype={"image_id": str})
verified_groups = pd.read_csv(GROUP_PATH)

assert len(verified_split) == 1842
assert len(verified_groups) == 1453

print("STEP 2.16B - SPLIT V1 CORRECTION AUDIT")
print("Total images:", len(verified_split))
print("Total risk groups:", len(verified_groups))
print(
    "Grade-conflicting groups:",
    int(verified_groups["grade_conflict"].sum())
)
print(
    "Images in grade-conflicting groups:",
    int(verified_split["group_grade_conflict"].sum())
)
print("Largest risk group:", int(group_sizes.max()))
print("Old evidence edges crossing splits:", old_crossings)
print("New evidence edges crossing splits:", new_crossings)
print("Exact duplicate groups crossing splits:", exact_crossings)

print("\nGRADE CONFLICTS BY SPLIT")
print(
    verified_split.groupby("split")
    .agg(
        total_images=("image_id", "size"),
        conflict_group_images=("group_grade_conflict", "sum")
    )
    .to_string()
)

print("\nGRADE DISTRIBUTION")
print(
    pd.crosstab(
        verified_split["split"],
        verified_split["dr_grade"]
    )
    .reindex(["train", "val", "test"])
    .to_string()
)

print("\nSPLIT ASSIGNMENTS PRESERVED")
print("Metadata corrected without reallocating images.")

print("\nSTEP 2.16B COMPLETED")


# %% [cell 76] Experiment 02 - Step 2.16C

from pathlib import Path
import pandas as pd
from itertools import combinations

# ============================================================
# Experiment 02 - Step 2.16C
# Split Balance and Label Conflict Diagnostic
# ============================================================

ROOT = (DR_RESEARCH / "experiment02_results")

split_df = pd.read_csv(
    ROOT / "FGADR_Split_V1.csv",
    dtype={"image_id": str, "risk_group_id": str}
)

manifest = pd.read_csv(
    ROOT / "FGADR_Image_SHA256_Manifest.csv",
    dtype={"image_id": str, "sha256": str}
)

assert len(split_df) == 1842
assert split_df["image_id"].is_unique

# Recompute conflict indicators independently.
grade_counts = split_df.groupby(
    "risk_group_id"
)["dr_grade"].nunique()

conflicting_groups = set(
    grade_counts[grade_counts > 1].index
)

split_df["audit_conflict"] = (
    split_df["risk_group_id"].isin(conflicting_groups)
)

assert len(conflicting_groups) == 90
assert int(split_df["audit_conflict"].sum()) == 292

print("STEP 2.16C - SPLIT BALANCE DIAGNOSTIC")

print("\nGROUP-LEVEL CONFLICT DISTRIBUTION")

group_summary = (
    split_df.groupby(["split", "risk_group_id"])
    .agg(
        image_count=("image_id", "size"),
        conflict=("audit_conflict", "first")
    )
    .reset_index()
)

group_distribution = (
    group_summary.groupby("split")
    .agg(
        total_groups=("risk_group_id", "size"),
        conflicting_groups=("conflict", "sum")
    )
    .reindex(["train", "val", "test"])
)

print(group_distribution.to_string())

print("\nCONFLICT RATES BY DR GRADE")

grade_summary = (
    split_df.groupby(["split", "dr_grade"])
    .agg(
        total_images=("image_id", "size"),
        conflict_images=("audit_conflict", "sum")
    )
    .reset_index()
)

grade_summary["conflict_rate"] = (
    grade_summary["conflict_images"]
    / grade_summary["total_images"]
)

print(
    grade_summary.sort_values(
        ["split", "dr_grade"]
    ).to_string(index=False)
)

print("\nCONFLICTING GROUP SIZE DISTRIBUTION")

conflict_size_distribution = (
    group_summary.loc[group_summary["conflict"]]
    .groupby(["split", "image_count"])
    .size()
    .rename("group_count")
    .reset_index()
)

print(conflict_size_distribution.to_string(index=False))

print("\nEXACT DUPLICATE LABEL CONFLICTS")

split_lookup = split_df.set_index("image_id")[
    ["split", "dr_grade", "risk_group_id"]
]

exact_conflict_records = []

for digest, group in manifest.groupby("sha256"):
    ids = sorted(group["image_id"])

    if len(ids) < 2:
        continue

    for a, b in combinations(ids, 2):
        grade_a = int(split_lookup.loc[a, "dr_grade"])
        grade_b = int(split_lookup.loc[b, "dr_grade"])

        if grade_a != grade_b:
            assert (
                split_lookup.loc[a, "split"]
                == split_lookup.loc[b, "split"]
            )

            exact_conflict_records.append({
                "image_id_1": a,
                "image_id_2": b,
                "grade_1": grade_a,
                "grade_2": grade_b,
                "split": split_lookup.loc[a, "split"],
                "risk_group_id": split_lookup.loc[
                    a, "risk_group_id"
                ]
            })

exact_conflicts = pd.DataFrame(exact_conflict_records)

assert len(exact_conflicts) == 2

print(exact_conflicts.to_string(index=False))

output_path = (
    ROOT / "FGADR_Split_V1_Conflict_Balance_Diagnostic.csv"
)

grade_summary.to_csv(output_path, index=False)

print("\nOUTPUT FILE:", output_path)
print("STEP 2.16C COMPLETED")


# %% [cell 77] Experiment 02 - Step 2.16D

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

# ============================================================
# Experiment 02 - Step 2.16D
# Leakage-Constrained and Risk-Balanced Split V2
# ============================================================

ROOT = (DR_RESEARCH / "experiment02_results")

INPUT = ROOT / "FGADR_Split_V1.csv"
MANIFEST = ROOT / "FGADR_Image_SHA256_Manifest.csv"
OLD_EDGES = ROOT / "FGADR_Updated_Risk_Evidence_Edges.csv"
NEW_EDGES = ROOT / "FGADR_HOG_Cross_Group_Spatial_Validation.csv"

OUTPUT = ROOT / "FGADR_Split_V2_Candidate.csv"
GROUP_OUTPUT = ROOT / "FGADR_Split_V2_Candidate_Groups.csv"
REPORT_OUTPUT = ROOT / "FGADR_Split_V2_Optimization_Report.json"

for path in [OUTPUT, GROUP_OUTPUT, REPORT_OUTPUT]:
    if path.exists():
        raise FileExistsError(
            f"Existing output will not be overwritten: {path}"
        )

split_v1 = pd.read_csv(
    INPUT,
    dtype={"image_id": str, "risk_group_id": str}
)

manifest = pd.read_csv(
    MANIFEST,
    dtype={"image_id": str, "sha256": str}
)

old_edges = pd.read_csv(
    OLD_EDGES,
    dtype={"image_id_1": str, "image_id_2": str}
)

new_edges = pd.read_csv(
    NEW_EDGES,
    dtype={"image_id_1": str, "image_id_2": str}
)

assert len(split_v1) == 1842
assert split_v1["image_id"].is_unique
assert len(manifest) == 1842
assert len(old_edges) == 541
assert len(new_edges) == 71

SPLITS = ["train", "val", "test"]
FRACTIONS = np.array([0.70, 0.15, 0.15])
GRADES = [0, 1, 2, 3, 4]
RANDOM_SEED = 20261019

# ============================================================
# 1. Reconstruct immutable risk-group features
# ============================================================

grade_nunique = split_v1.groupby(
    "risk_group_id"
)["dr_grade"].nunique()

conflict_map = grade_nunique.gt(1).to_dict()

split_v1["conflict_audit"] = (
    split_v1["risk_group_id"].map(conflict_map)
)

assert len(conflict_map) == 1453
assert sum(conflict_map.values()) == 90
assert int(split_v1["conflict_audit"].sum()) == 292

group_rows = []

for group_id, frame in split_v1.groupby(
    "risk_group_id", sort=True
):
    counts = np.array([
        (frame["dr_grade"] == grade).sum()
        for grade in GRADES
    ], dtype=np.int32)

    conflict_counts = (
        counts.copy()
        if conflict_map[group_id]
        else np.zeros(5, dtype=np.int32)
    )

    original_split = frame["split"].unique()
    assert len(original_split) == 1

    group_rows.append({
        "risk_group_id": group_id,
        "image_count": len(frame),
        "grade_counts": counts,
        "conflict_grade_counts": conflict_counts,
        "conflict_image_count": int(conflict_counts.sum()),
        "grade_conflict": bool(conflict_map[group_id]),
        "v1_split": original_split[0]
    })

group_ids = [g["risk_group_id"] for g in group_rows]
group_sizes = np.array(
    [g["image_count"] for g in group_rows],
    dtype=np.float64
)

grade_features = np.stack(
    [g["grade_counts"] for g in group_rows]
).astype(np.float64)

conflict_features = np.stack(
    [g["conflict_grade_counts"] for g in group_rows]
).astype(np.float64)

conflict_totals = conflict_features.sum(axis=1)

# Each group contributes total size, grade counts,
# conflict totals, and conflict counts by grade.
features = np.column_stack([
    group_sizes,
    grade_features,
    conflict_totals,
    conflict_features
])

feature_totals = features.sum(axis=0)
targets = FRACTIONS[:, None] * feature_totals[None, :]

# Weight conflict balance without allowing it to
# dominate overall size and DR-grade stratification.
weights = np.array([
    5.0,
    2.0, 2.0, 2.0, 2.0, 2.0,
    3.0,
    1.0, 1.0, 1.0, 1.0, 1.0
])

# Use a minimum denominator to avoid excessive penalties
# for small conflict-grade counts.
scales = np.maximum(targets, 8.0)

def objective(totals):
    return float(
        np.sum(
            weights[None, :]
            * ((totals - targets) / scales) ** 2
        )
    )

v1_allocation = np.array([
    SPLITS.index(g["v1_split"])
    for g in group_rows
], dtype=np.int8)

def calculate_totals(allocation):
    result = np.zeros(
        (3, features.shape[1]),
        dtype=np.float64
    )

    for split_index in range(3):
        result[split_index] = features[
            allocation == split_index
        ].sum(axis=0)

    return result

v1_totals = calculate_totals(v1_allocation)
v1_score = objective(v1_totals)

# ============================================================
# 2. Group-level stochastic optimization
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

best_allocation = v1_allocation.copy()
best_totals = v1_totals.copy()
best_score = v1_score

RESTARTS = 4
ITERATIONS_PER_RESTART = 30000

for restart in range(RESTARTS):
    allocation = (
        best_allocation.copy()
        if restart == 0
        else v1_allocation.copy()
    )

    totals = calculate_totals(allocation)
    current_score = objective(totals)

    for iteration in range(ITERATIONS_PER_RESTART):
        first = int(rng.integers(len(group_rows)))
        source_split = int(allocation[first])

        trial = totals.copy()

        if rng.random() < 0.55:
            # Move one complete risk group.
            target_split = int(
                rng.choice([
                    s for s in range(3)
                    if s != source_split
                ])
            )

            trial[source_split] -= features[first]
            trial[target_split] += features[first]

            second = None

        else:
            # Swap two complete risk groups.
            second = int(rng.integers(len(group_rows)))
            target_split = int(allocation[second])

            if target_split == source_split:
                continue

            trial[source_split] += (
                features[second] - features[first]
            )

            trial[target_split] += (
                features[first] - features[second]
            )

        candidate_score = objective(trial)

        # Simulated annealing supports escaping local minima.
        progress = iteration / ITERATIONS_PER_RESTART
        temperature = (
            0.02 * (1.0 - progress) ** 2
            + 0.000001
        )

        delta = candidate_score - current_score

        accept = (
            delta < 0
            or rng.random() < np.exp(
                -min(max(delta / temperature, 0.0), 700)
            )
        )

        if accept:
            allocation[first] = target_split

            if second is not None:
                allocation[second] = source_split

            totals = trial
            current_score = candidate_score

            if current_score < best_score:
                best_score = current_score
                best_allocation = allocation.copy()
                best_totals = totals.copy()

# ============================================================
# 3. Generate candidate split without modifying V1
# ============================================================

allocation_map = {
    group_id: SPLITS[int(best_allocation[i])]
    for i, group_id in enumerate(group_ids)
}

split_v2 = split_v1.copy()

split_v2["split"] = (
    split_v2["risk_group_id"].map(allocation_map)
)

split_v2["group_grade_conflict"] = (
    split_v2["risk_group_id"].map(conflict_map)
)

split_v2 = split_v2.drop(
    columns=["conflict_audit"]
)

group_output = pd.DataFrame({
    "risk_group_id": group_ids,
    "split": [
        allocation_map[group_id]
        for group_id in group_ids
    ],
    "image_count": group_sizes.astype(int),
    "grade_conflict": [
        conflict_map[group_id]
        for group_id in group_ids
    ],
    "conflict_image_count": (
        conflict_totals.astype(int)
    )
})

# ============================================================
# 4. Independent leakage and data integrity checks
# ============================================================

assert len(split_v2) == 1842
assert split_v2["image_id"].is_unique
assert set(split_v2["image_id"]) == set(manifest["image_id"])

assert (
    split_v2.groupby("risk_group_id")["split"]
    .nunique().max() == 1
)

manifest_grades = (
    manifest.set_index("image_id")["dr_grade"]
    .astype(int).sort_index()
)

v2_grades = (
    split_v2.set_index("image_id")["dr_grade"]
    .astype(int).sort_index()
)

assert manifest_grades.equals(v2_grades)

split_lookup = split_v2.set_index("image_id")[
    "split"
].to_dict()

def edge_crossings(edges):
    return sum(
        split_lookup[a] != split_lookup[b]
        for a, b in zip(
            edges["image_id_1"],
            edges["image_id_2"]
        )
    )

old_crossings = edge_crossings(old_edges)
new_crossings = edge_crossings(new_edges)

assert old_crossings == 0
assert new_crossings == 0

exact_crossings = 0

for _, frame in manifest.groupby("sha256"):
    assigned_splits = {
        split_lookup[i]
        for i in frame["image_id"]
    }

    if len(assigned_splits) > 1:
        exact_crossings += 1

assert exact_crossings == 0

assert best_score <= v1_score + 1e-12

# ============================================================
# 5. Save candidate and reproducibility report
# ============================================================

split_v2.to_csv(OUTPUT, index=False)
group_output.to_csv(GROUP_OUTPUT, index=False)

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()

report = {
    "experiment": "Experiment02_Step2.16D",
    "version": "FGADR_Split_V2_Candidate",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "random_seed": RANDOM_SEED,
    "restarts": RESTARTS,
    "iterations_per_restart": ITERATIONS_PER_RESTART,
    "fractions": FRACTIONS.tolist(),
    "feature_weights": weights.tolist(),
    "minimum_feature_scale": 8.0,
    "algorithm": "group_level_simulated_annealing",
    "v1_objective": v1_score,
    "v2_objective": best_score,
    "risk_groups": len(group_ids),
    "old_edge_crossings": old_crossings,
    "new_edge_crossings": new_crossings,
    "exact_duplicate_crossings": exact_crossings,
    "input_split_sha256": sha256_file(INPUT),
    "input_manifest_sha256": sha256_file(MANIFEST),
    "input_old_edges_sha256": sha256_file(OLD_EDGES),
    "input_new_edges_sha256": sha256_file(NEW_EDGES),
    "formal_split_frozen": False
}

with open(
    REPORT_OUTPUT,
    "w",
    encoding="utf-8"
) as handle:
    json.dump(report, handle, indent=2)

# ============================================================
# 6. Compare V1 and V2
# ============================================================

def balance_table(frame):
    result = (
        frame.groupby("split")
        .agg(
            images=("image_id", "size"),
            conflict_images=("group_grade_conflict", "sum")
        )
        .reindex(SPLITS)
    )

    result["conflict_rate"] = (
        result["conflict_images"]
        / result["images"]
    )

    return result

v1_balance = balance_table(split_v1)
v2_balance = balance_table(split_v2)

distribution = pd.crosstab(
    split_v2["split"],
    split_v2["dr_grade"]
).reindex(
    index=SPLITS,
    columns=GRADES,
    fill_value=0
)

changed_images = int(
    (
        split_v1.set_index("image_id")["split"]
        != split_v2.set_index("image_id")["split"]
    ).sum()
)

print("STEP 2.16D - FGADR SPLIT V2 CANDIDATE")
print("Images:", len(split_v2))
print("Risk groups:", len(group_ids))
print("Conflicting groups:", sum(conflict_map.values()))
print("Conflict-group images:", int(conflict_totals.sum()))
print("Changed image assignments:", changed_images)
print("V1 objective:", round(v1_score, 8))
print("V2 objective:", round(best_score, 8))
print("Old edge crossings:", old_crossings)
print("New edge crossings:", new_crossings)
print("Exact duplicate crossings:", exact_crossings)

print("\nV1 BALANCE")
print(v1_balance.round(4).to_string())

print("\nV2 BALANCE")
print(v2_balance.round(4).to_string())

print("\nV2 DR DISTRIBUTION")
print(distribution.to_string())

print("\nV2 CONFLICT RATES BY GRADE")

grade_balance = (
    split_v2.groupby(["split", "dr_grade"])
    .agg(
        total=("image_id", "size"),
        conflict=("group_grade_conflict", "sum")
    )
)

grade_balance["rate"] = (
    grade_balance["conflict"] / grade_balance["total"]
)

print(grade_balance.round(4).to_string())

print("\nOUTPUT FILES")
print(OUTPUT)
print(GROUP_OUTPUT)
print(REPORT_OUTPUT)

print("\nSTEP 2.16D COMPLETED")


# %% [cell 78] Experiment 02 - Step 2.16E

from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
import hashlib
import json
import shutil
import pandas as pd

# ============================================================
# Experiment 02 - Step 2.16E
# Final Split Verification and Freeze
# ============================================================

ROOT = (DR_RESEARCH / "experiment02_results")

CANDIDATE = ROOT / "FGADR_Split_V2_Candidate.csv"
GROUPS = ROOT / "FGADR_Split_V2_Candidate_Groups.csv"
MANIFEST = ROOT / "FGADR_Image_SHA256_Manifest.csv"
OLD_EDGES = ROOT / "FGADR_Updated_Risk_Evidence_Edges.csv"
NEW_EDGES = ROOT / "FGADR_HOG_Cross_Group_Spatial_Validation.csv"

FROZEN = ROOT / "FGADR_Split_V2_Frozen.csv"
FREEZE_REPORT = ROOT / "FGADR_Split_V2_Freeze_Report.json"

assert not FROZEN.exists(), "Frozen split already exists."
assert not FREEZE_REPORT.exists(), "Freeze report already exists."

split_df = pd.read_csv(
    CANDIDATE,
    dtype={"image_id": str, "risk_group_id": str}
)

groups = pd.read_csv(GROUPS, dtype={"risk_group_id": str})
manifest = pd.read_csv(MANIFEST, dtype={"image_id": str, "sha256": str})

old_edges = pd.read_csv(
    OLD_EDGES,
    dtype={"image_id_1": str, "image_id_2": str}
)

new_edges = pd.read_csv(
    NEW_EDGES,
    dtype={"image_id_1": str, "image_id_2": str}
)

# 1. Validate image identity and original grades.
assert len(split_df) == len(manifest) == 1842
assert split_df["image_id"].is_unique
assert manifest["image_id"].is_unique
assert set(split_df["image_id"]) == set(manifest["image_id"])
assert set(split_df["split"]) == {"train", "val", "test"}

original_grades = manifest.set_index("image_id")["dr_grade"].astype(int)
assigned_grades = split_df.set_index("image_id")["dr_grade"].astype(int)

assert original_grades.sort_index().equals(
    assigned_grades.sort_index()
)

# 2. Independently reconstruct risk-group properties.
grouped = split_df.groupby("risk_group_id")
group_sizes = grouped.size()
group_conflicts = grouped["dr_grade"].nunique().gt(1)

assert len(group_sizes) == 1453
assert int(group_conflicts.sum()) == 90
assert int(group_sizes[group_conflicts].sum()) == 292
assert int(group_sizes.max()) == 11
assert grouped["split"].nunique().max() == 1

assert set(groups["risk_group_id"]) == set(group_sizes.index)

group_lookup = groups.set_index("risk_group_id")
assert (
    group_lookup["image_count"].astype(int).sort_index()
    .equals(group_sizes.sort_index())
)

for group_id, frame in grouped:
    declared = bool(group_lookup.loc[group_id, "grade_conflict"])
    assert declared == bool(group_conflicts.loc[group_id])

# 3. Verify every recorded leakage constraint.
split_lookup = split_df.set_index("image_id")["split"].to_dict()

def count_crossings(frame):
    return sum(
        split_lookup[a] != split_lookup[b]
        for a, b in zip(
            frame["image_id_1"],
            frame["image_id_2"]
        )
    )

assert len(old_edges) == 541
assert len(new_edges) == 71

old_crossings = count_crossings(old_edges)
new_crossings = count_crossings(new_edges)

assert old_crossings == 0
assert new_crossings == 0

# 4. Independently identify exact duplicate label conflicts.
exact_pairs = []
conflicting_image_ids = set()
exact_crossings = 0

for digest, frame in manifest.groupby("sha256"):
    ids = sorted(frame["image_id"].tolist())

    for a, b in combinations(ids, 2):
        exact_pairs.append((a, b))

        if split_lookup[a] != split_lookup[b]:
            exact_crossings += 1

        if int(original_grades[a]) != int(original_grades[b]):
            conflicting_image_ids.update([a, b])

assert len(exact_pairs) == 14
assert exact_crossings == 0
assert len(conflicting_image_ids) == 4

split_df["group_grade_conflict"] = (
    split_df["risk_group_id"].map(group_conflicts).astype(bool)
)

split_df["exact_duplicate_label_conflict"] = (
    split_df["image_id"].isin(conflicting_image_ids)
)

# 5. Confirm distribution without reallocating images.
distribution = pd.crosstab(
    split_df["split"], split_df["dr_grade"]
).reindex(
    index=["train", "val", "test"],
    columns=[0, 1, 2, 3, 4],
    fill_value=0
)

assert distribution.sum(axis=1).tolist() == [1290, 276, 276]

conflict_balance = split_df.groupby("split").agg(
    images=("image_id", "size"),
    conflict_images=("group_grade_conflict", "sum"),
    exact_conflict_images=("exact_duplicate_label_conflict", "sum")
).reindex(["train", "val", "test"])

conflict_balance["conflict_rate"] = (
    conflict_balance["conflict_images"] /
    conflict_balance["images"]
)

assert conflict_balance["conflict_images"].tolist() == [204, 44, 44]

# 6. Save immutable-version outputs with cryptographic checksums.
split_df = split_df.sort_values("image_id").reset_index(drop=True)
split_df.to_csv(FROZEN, index=False)

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

report = {
    "version": "FGADR_Split_V2_Frozen",
    "frozen_utc": datetime.now(timezone.utc).isoformat(),
    "images": len(split_df),
    "risk_groups": len(group_sizes),
    "conflicting_risk_groups": int(group_conflicts.sum()),
    "images_in_conflicting_risk_groups": int(
        group_sizes[group_conflicts].sum()
    ),
    "exact_duplicate_pairs": len(exact_pairs),
    "exact_duplicate_label_conflict_images": len(conflicting_image_ids),
    "old_edge_crossings": old_crossings,
    "new_edge_crossings": new_crossings,
    "exact_duplicate_crossings": exact_crossings,
    "candidate_sha256": file_sha256(CANDIDATE),
    "manifest_sha256": file_sha256(MANIFEST),
    "frozen_split_sha256": file_sha256(FROZEN),
    "split_type": "image_source_risk_group_constrained",
    "patient_level_independence_verified": False,
    "source_relationships_clinically_confirmed": False,
    "model_based_split_selection": False
}

with open(FREEZE_REPORT, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

# Verify the saved file is readable.
saved = pd.read_csv(FROZEN, dtype={"image_id": str})
assert len(saved) == 1842
assert saved["image_id"].is_unique

print("STEP 2.16E - FINAL SPLIT FREEZE AUDIT")
print("Images:", len(saved))
print("Risk groups:", len(group_sizes))
print("Grade-conflicting risk groups:", int(group_conflicts.sum()))
print("Images in conflicting groups:", int(group_sizes[group_conflicts].sum()))
print("Exact duplicate pairs:", len(exact_pairs))
print("Exact duplicate label-conflict images:", len(conflicting_image_ids))
print("Old edge crossings:", old_crossings)
print("New edge crossings:", new_crossings)
print("Exact duplicate crossings:", exact_crossings)

print("\nFROZEN DR DISTRIBUTION")
print(distribution.to_string())

print("\nFROZEN CONFLICT BALANCE")
print(conflict_balance.round(4).to_string())

print("\nEXACT DUPLICATE LABEL-CONFLICT IMAGE IDS")
print(sorted(conflicting_image_ids))

print("\nFROZEN SPLIT SHA256")
print(report["frozen_split_sha256"])

print("\nOUTPUT FILES")
print(FROZEN)
print(FREEZE_REPORT)

print("\nSTEP 2.16E COMPLETED")
