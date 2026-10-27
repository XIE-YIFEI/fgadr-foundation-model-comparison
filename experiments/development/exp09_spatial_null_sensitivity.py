# Experiment 09 - cohort selection and spatial-null sensitivity analyses
#
# Source: FGADR_MA_Mask_Audit.ipynb (master notebook, 392 cells), cells 302-331, run step by step in Google Colab (L4 GPU).
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

# %% [cell 302] Experiment 09 - Step 09.01
# ============================================================
# Experiment 09 - Step 09.01
# Discover MA Cohort and Spatial Null Audit Files
# Read-only; no training or image access
# ============================================================

from pathlib import Path

ROOT = DR_RESEARCH

SEARCH_DIRS = [
    ROOT / "experiment04_results",
    ROOT / "experiment05_results",
    ROOT / "experiment06_results",
    ROOT / "experiment07_results",
    ROOT / "experiment08_results"
]

KEYWORDS = [
    "ma_",
    "mask",
    "spatial",
    "null",
    "eligible",
    "exclusion",
    "cohort",
    "gradcam",
    "enrichment",
    "bootstrap"
]

print("EXPERIMENT 09.01 - MA AUDIT FILE DISCOVERY")

candidates = []

for folder in SEARCH_DIRS:
    if not folder.is_dir():
        continue

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".csv", ".json", ".parquet", ".npz"
        }:
            continue

        name = path.name.lower()

        if any(keyword in name for keyword in KEYWORDS):
            candidates.append(path)

print("\nCANDIDATE FILES:", len(candidates))

for path in sorted(candidates):
    relative = path.relative_to(ROOT)
    size_mb = path.stat().st_size / 1024**2

    print(f"{relative} | {size_mb:.2f} MiB")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.01 COMPLETE")


# %% [cell 303] Experiment 09 - Step 09.02
# ============================================================
# Experiment 09 - Step 09.02
# Inspect MA Cohort and Spatial Null Data Structure
# Read-only; no image access or training
# ============================================================

from pathlib import Path
import pandas as pd
import json

ROOT = DR_RESEARCH

files = {
    "MA_COHORT": (
        ROOT / "experiment05_results"
        / "FGADR_Validation_MA_Cohort_V1.csv"
    ),
    "RETFOUND_SPATIAL_NULL": (
        ROOT / "experiment05_results"
        / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
    ),
    "PAIRED_SPATIAL_NULL": (
        ROOT / "experiment07_results"
        / "ViTL_MA_Spatial_Null_Paired_V1.csv"
    ),
    "RAW_ATTRIBUTION": (
        ROOT / "experiment05_results"
        / "MA_Attribution_Paired_V1.csv"
    ),
}

reports = {
    "MA_COHORT_REPORT": (
        ROOT / "experiment05_results"
        / "FGADR_Validation_MA_Cohort_Report_V1.json"
    ),
    "SPATIAL_NULL_REPORT": (
        ROOT / "experiment05_results"
        / "MA_Spatial_Null_Retinal_Constrained_Report_V1.json"
    ),
}

print("EXPERIMENT 09.02 - MA DATA STRUCTURE AUDIT")

for name, path in files.items():
    assert path.is_file(), f"Missing: {path}"

    frame = pd.read_csv(path)

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print("Rows:", len(frame))
    print("Columns:", len(frame.columns))
    print("Column names:", frame.columns.tolist())

    key_columns = [
        column for column in frame.columns
        if any(
            token in column.lower()
            for token in [
                "image_id",
                "risk_group",
                "eligible",
                "excluded",
                "reason",
                "grade",
                "area",
                "shift"
            ]
        )
    ]

    print("\nKEY COLUMN VALUES")

    for column in key_columns:
        print(
            column,
            "| unique:",
            frame[column].nunique(dropna=False),
            "| missing:",
            int(frame[column].isna().sum())
        )

        if (
            frame[column].nunique(dropna=False) <= 12
            and "image_id" not in column.lower()
            and "risk_group" not in column.lower()
        ):
            print(
                frame[column]
                .value_counts(dropna=False)
                .head(12)
                .to_dict()
            )

    print("\nFIRST TWO RECORDS")
    print(frame.head(2).to_string(index=False, max_cols=12))

for name, path in reports.items():
    assert path.is_file(), f"Missing: {path}"

    report = json.loads(path.read_text(encoding="utf-8"))

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print("Top-level keys:", list(report.keys()))

    for key, value in report.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            print(f"{key}: {value}")
        elif isinstance(value, dict):
            print(f"{key}: {list(value.keys())[:15]}")
        elif isinstance(value, list):
            print(f"{key}: list length {len(value)}")

print("\nRESEARCH INTEGRITY")
print("Research files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.02 COMPLETE")


# %% [cell 304] Experiment 09 - Step 09.03
# ============================================================
# Experiment 09 - Step 09.03
# MA Spatial Null Cohort Selection Bias Audit
# Read-only; no image access or training
# ============================================================

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

PAIRED_PATH = (
    ROOT / "experiment07_results"
    / "ViTL_MA_Spatial_Null_Paired_V1.csv"
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
# 1. Load frozen data sources
# ------------------------------------------------------------

for path in [
    COHORT_PATH,
    SPATIAL_PATH,
    PAIRED_PATH,
    INDEX_PATH
]:
    assert path.is_file(), f"Missing file: {path}"

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_SHA256

cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})
spatial = pd.read_csv(SPATIAL_PATH, dtype={"image_id": str})
paired = pd.read_csv(PAIRED_PATH, dtype={"image_id": str})

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

validation = index.loc[
    index["split"] == "val",
    ["image_id", "risk_group_id", "dr_grade"]
].copy()

assert len(validation) == 276
assert len(cohort) == 276
assert len(spatial) == 156
assert len(paired) == 137

for frame in [validation, cohort, spatial, paired]:
    assert frame["image_id"].is_unique

# ------------------------------------------------------------
# 2. Reconcile analysis populations
# ------------------------------------------------------------

primary_ids = set(
    cohort.loc[
        cohort["primary_ma_candidate"] == True,
        "image_id"
    ]
)

spatial_ids = set(spatial["image_id"])

eligible_ids = set(
    spatial.loc[spatial["eligible"] == True, "image_id"]
)

paired_ids = set(paired["image_id"])

assert len(primary_ids) == 156
assert primary_ids == spatial_ids
assert eligible_ids == paired_ids

assert len(eligible_ids) == 137

# ------------------------------------------------------------
# 3. Merge eligibility and annotation characteristics
# ------------------------------------------------------------

cohort_columns = [
    "image_id",
    "true_grade",
    "mask_coverage",
    "candidate_count",
    "largest_component_area_px",
    "total_mask_area_px",
    "review_category",
    "review_reason"
]

spatial_columns = [
    "image_id",
    "eligible",
    "exclusion_reason",
    "observed_coverage",
    "foreground_fraction",
    "possible_shifts",
    "null_shifts"
]

audit = (
    cohort[cohort_columns]
    .merge(
        spatial[spatial_columns],
        on="image_id",
        how="inner",
        validate="one_to_one"
    )
    .merge(
        validation,
        on="image_id",
        how="left",
        validate="one_to_one"
    )
)

assert len(audit) == 156
assert audit["risk_group_id"].notna().all()
assert (audit["true_grade"] == audit["dr_grade"]).all()

audit["selection_group"] = np.where(
    audit["eligible"],
    "Eligible",
    "Excluded"
)

# ------------------------------------------------------------
# 4. Cohort composition
# ------------------------------------------------------------

print("EXPERIMENT 09.03 - MA SELECTION BIAS AUDIT")

print("\nCOHORT FLOW")
print("Total validation images:", len(cohort))
print("Primary MA candidates:", len(audit))
print("Spatial Null eligible:", int(audit["eligible"].sum()))
print("Spatial Null excluded:", int((~audit["eligible"]).sum()))

print("\nEXCLUSION REASONS")
print(
    audit.loc[
        ~audit["eligible"],
        "exclusion_reason"
    ].value_counts(dropna=False).to_string()
)

print("\nDR GRADE DISTRIBUTION")

grade_table = pd.crosstab(
    audit["true_grade"],
    audit["selection_group"]
)

print(grade_table.to_string())

# ------------------------------------------------------------
# 5. Quantitative comparisons
# ------------------------------------------------------------

metrics = [
    "mask_coverage",
    "candidate_count",
    "largest_component_area_px",
    "total_mask_area_px",
    "observed_coverage",
    "foreground_fraction",
    "possible_shifts",
    "null_shifts"
]

print("\nANNOTATION AND SPATIAL CHARACTERISTICS")

for metric in metrics:
    print(f"\n{metric}")

    summary = (
        audit.groupby("selection_group")[metric]
        .agg(["count", "median", "mean", "min", "max"])
        .round(6)
    )

    print(summary.to_string())

# ------------------------------------------------------------
# 6. Source-risk-group composition
# ------------------------------------------------------------

print("\nRISK GROUP COMPOSITION")

for group_name, frame in audit.groupby("selection_group"):
    print(group_name)
    print("Images:", len(frame))
    print("Unique risk groups:", frame["risk_group_id"].nunique())
    print(
        "Largest group size:",
        frame["risk_group_id"].value_counts().max()
    )

# ------------------------------------------------------------
# 7. Check for overlap between groups
# ------------------------------------------------------------

eligible_groups = set(
    audit.loc[audit["eligible"], "risk_group_id"]
)

excluded_groups = set(
    audit.loc[~audit["eligible"], "risk_group_id"]
)

print("\nGROUP OVERLAP")
print(
    "Risk groups appearing in both populations:",
    len(eligible_groups & excluded_groups)
)

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.03 COMPLETE")


# %% [cell 305] Experiment 09 - Step 09.04
# ============================================================
# Experiment 09 - Step 09.04
# Exploratory DR Grade Selection Bias Analysis
# Read-only; no training or image access
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = DR_RESEARCH

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

assert SPATIAL_PATH.is_file()
assert INDEX_PATH.is_file()

spatial = pd.read_csv(
    SPATIAL_PATH,
    dtype={"image_id": str}
)

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

validation = index.loc[
    index["split"] == "val",
    ["image_id", "risk_group_id", "dr_grade"]
]

audit = spatial.drop(
    columns=["risk_group_id"],
    errors="ignore"
).merge(
    validation,
    on="image_id",
    how="left",
    validate="one_to_one"
)

assert len(audit) == 156
assert audit["risk_group_id"].notna().all()
assert (audit["true_grade"] == audit["dr_grade"]).all()
assert audit["eligible"].isin([True, False]).all()

audit["high_grade"] = audit["true_grade"] >= 3

# ------------------------------------------------------------
# 1. Construct the 2 x 2 contingency table
# ------------------------------------------------------------

table = pd.crosstab(
    audit["high_grade"],
    audit["eligible"]
).reindex(
    index=[False, True],
    columns=[False, True],
    fill_value=0
)

excluded_low = int(table.loc[False, False])
eligible_low = int(table.loc[False, True])

excluded_high = int(table.loc[True, False])
eligible_high = int(table.loc[True, True])

counts = np.array([
    [excluded_high, eligible_high],
    [excluded_low, eligible_low]
])

odds_ratio, p_value = fisher_exact(
    counts,
    alternative="two-sided"
)

# ------------------------------------------------------------
# 2. Descriptive effect sizes
# ------------------------------------------------------------

high_exclusion_rate = (
    excluded_high / (excluded_high + eligible_high)
)

low_exclusion_rate = (
    excluded_low / (excluded_low + eligible_low)
)

risk_difference = high_exclusion_rate - low_exclusion_rate

high_share_excluded = (
    excluded_high / (excluded_high + excluded_low)
)

high_share_eligible = (
    eligible_high / (eligible_high + eligible_low)
)

# Standardized difference for binary high-grade composition.
pooled = (high_share_excluded + high_share_eligible) / 2

standardized_difference = (
    (high_share_excluded - high_share_eligible)
    / np.sqrt(pooled * (1 - pooled))
)

print("EXPERIMENT 09.04 - DR GRADE SELECTION BIAS")

print("\nCONTINGENCY TABLE")
print("Rows: High grade (>=3), Low grade (1-2)")
print("Columns: Excluded, Eligible")
print(counts)

print("\nEXCLUSION RATES")
print(
    "High grade exclusion rate:",
    round(high_exclusion_rate, 6)
)
print(
    "Low grade exclusion rate:",
    round(low_exclusion_rate, 6)
)
print(
    "Risk difference:",
    round(risk_difference, 6)
)

print("\nGRADE COMPOSITION")
print(
    "High-grade share among excluded:",
    round(high_share_excluded, 6)
)
print(
    "High-grade share among eligible:",
    round(high_share_eligible, 6)
)
print(
    "Standardized difference:",
    round(standardized_difference, 6)
)

print("\nEXPLORATORY FISHER TEST")
print("Odds ratio:", round(float(odds_ratio), 6))
print("Two-sided p-value:", round(float(p_value), 8))
print(
    "Interpretation:",
    "Exploratory image-level association only"
)

print("\nGROUP DEPENDENCE")
print(
    "Unique source-risk groups:",
    audit["risk_group_id"].nunique()
)
print(
    "Group-level robust inference:",
    "Not performed in this step"
)

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.04 COMPLETE")


# %% [cell 306] Experiment 09 - Step 09.05
# ============================================================
# Experiment 09 - Step 09.05
# Cluster Bootstrap for DR Grade-Related Exclusion
# Read-only; validation cohort only
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

spatial = pd.read_csv(
    SPATIAL_PATH,
    dtype={"image_id": str}
)

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

validation = index.loc[
    index["split"] == "val",
    ["image_id", "risk_group_id", "dr_grade"]
]

audit = spatial.drop(
    columns=["risk_group_id"],
    errors="ignore"
).merge(
    validation,
    on="image_id",
    how="left",
    validate="one_to_one"
)

assert len(audit) == 156
assert audit["risk_group_id"].notna().all()
assert (audit["true_grade"] == audit["dr_grade"]).all()

audit["high_grade"] = audit["true_grade"] >= 3
audit["excluded"] = ~audit["eligible"].astype(bool)

# ------------------------------------------------------------
# 1. Observed exclusion risk difference
# ------------------------------------------------------------

high = audit.loc[audit["high_grade"], "excluded"]
low = audit.loc[~audit["high_grade"], "excluded"]

observed_difference = high.mean() - low.mean()

# ------------------------------------------------------------
# 2. Grouped bootstrap resampling
# ------------------------------------------------------------

grouped = [
    group.copy()
    for _, group in audit.groupby(
        "risk_group_id",
        sort=True
    )
]

n_groups = len(grouped)

assert n_groups == 134

rng = np.random.default_rng(42)

N_BOOTSTRAP = 5000
draws = []
invalid = 0

for _ in range(N_BOOTSTRAP):
    sampled_indices = rng.integers(
        0,
        n_groups,
        size=n_groups
    )

    sampled = pd.concat(
        [grouped[i] for i in sampled_indices],
        ignore_index=True
    )

    high_sample = sampled.loc[
        sampled["high_grade"],
        "excluded"
    ]

    low_sample = sampled.loc[
        ~sampled["high_grade"],
        "excluded"
    ]

    if len(high_sample) == 0 or len(low_sample) == 0:
        invalid += 1
        continue

    draws.append(
        high_sample.mean() - low_sample.mean()
    )

draws = np.asarray(draws, dtype=float)

assert len(draws) > 0.95 * N_BOOTSTRAP

ci_low, ci_high = np.quantile(
    draws,
    [0.025, 0.975]
)

# ------------------------------------------------------------
# 3. Summary
# ------------------------------------------------------------

print("EXPERIMENT 09.05 - GROUPED SELECTION BIAS")

print("\nDATA")
print("Images:", len(audit))
print("Source-risk groups:", n_groups)
print("High-grade images:", len(high))
print("Low-grade images:", len(low))

print("\nOBSERVED EXCLUSION")
print("High-grade exclusion rate:", round(high.mean(), 6))
print("Low-grade exclusion rate:", round(low.mean(), 6))
print("Risk difference:", round(observed_difference, 6))

print("\nCLUSTER BOOTSTRAP")
print("Requested draws:", N_BOOTSTRAP)
print("Valid draws:", len(draws))
print("Invalid draws:", invalid)
print("95% CI lower:", round(float(ci_low), 6))
print("95% CI upper:", round(float(ci_high), 6))
print("CI excludes zero:", bool(ci_low > 0 or ci_high < 0))

print("\nINTERPRETATION")
print("Resampling unit: source-risk group")
print("Confidence interval: percentile bootstrap")
print("Analysis status: exploratory")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Training started: False")
print("Images accessed: 0")
print("Test images accessed: 0")

print("\nSTEP 09.05 COMPLETE")


# %% [cell 307] Experiment 09 - Step 09.06
# ============================================================
# Experiment 09 - Step 09.06
# Grade-Stratified Spatial-Null Attribution Analysis
# Validation only; read-only
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

PAIRED_PATH = (
    ROOT / "experiment07_results"
    / "ViTL_MA_Spatial_Null_Paired_V1.csv"
)

assert PAIRED_PATH.is_file()

df = pd.read_csv(
    PAIRED_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(df) == 137
assert df["image_id"].is_unique
assert df["risk_group_id"].notna().all()
assert df["eligible"].eq(True).all()

required = [
    "true_grade",
    "delta_null_adjusted",
    "delta_spatial_percentile",
    "risk_group_id"
]

assert set(required).issubset(df.columns)

df["grade_group"] = np.where(
    df["true_grade"] >= 3,
    "Grade 3-4",
    "Grade 1-2"
)

# ------------------------------------------------------------
# 1. Observed subgroup effects
# ------------------------------------------------------------

metric = "delta_null_adjusted"

summary = (
    df.groupby("grade_group")
    .agg(
        images=("image_id", "size"),
        risk_groups=("risk_group_id", "nunique"),
        mean_delta=(metric, "mean"),
        median_delta=(metric, "median")
    )
)

low_mean = df.loc[
    df["grade_group"] == "Grade 1-2", metric
].mean()

high_mean = df.loc[
    df["grade_group"] == "Grade 3-4", metric
].mean()

observed_interaction = high_mean - low_mean

# ------------------------------------------------------------
# 2. Cluster bootstrap
# ------------------------------------------------------------

group_arrays = [
    group[["true_grade", metric]].to_numpy(dtype=float)
    for _, group in df.groupby("risk_group_id", sort=True)
]

assert len(group_arrays) == 120

rng = np.random.default_rng(42)

N_BOOTSTRAP = 5000

low_draws = []
high_draws = []
interaction_draws = []

for _ in range(N_BOOTSTRAP):

    sampled_indices = rng.integers(
        0,
        len(group_arrays),
        size=len(group_arrays)
    )

    sample = np.concatenate([
        group_arrays[i] for i in sampled_indices
    ])

    grades = sample[:, 0]
    deltas = sample[:, 1]

    low = deltas[grades <= 2]
    high = deltas[grades >= 3]

    if len(low) == 0 or len(high) == 0:
        continue

    low_effect = float(low.mean())
    high_effect = float(high.mean())

    low_draws.append(low_effect)
    high_draws.append(high_effect)
    interaction_draws.append(high_effect - low_effect)

assert len(interaction_draws) >= 4750

def summarize_ci(values):
    lower, upper = np.quantile(
        values, [0.025, 0.975]
    )
    return round(float(lower), 6), round(float(upper), 6)

# ------------------------------------------------------------
# 3. Display
# ------------------------------------------------------------

print("EXPERIMENT 09.06 - GRADE-STRATIFIED ATTRIBUTION")

print("\nCOHORT")
print("Total images:", len(df))
print("Total risk groups:", df["risk_group_id"].nunique())

print("\nSUBGROUP SUMMARY")
print(summary.round(6).to_string())

print("\nCLUSTER BOOTSTRAP RESULTS")

print("Grade 1-2")
print("Mean delta:", round(float(low_mean), 6))
print("95% CI:", summarize_ci(low_draws))

print("\nGrade 3-4")
print("Mean delta:", round(float(high_mean), 6))
print("95% CI:", summarize_ci(high_draws))

print("\nEXPLORATORY EFFECT MODIFICATION")
print(
    "High-grade minus low-grade delta:",
    round(float(observed_interaction), 6)
)
print(
    "95% CI:",
    summarize_ci(interaction_draws)
)

print("\nINTERPRETATION")
print("Metric: Meta MAE minus RETFound null-adjusted enrichment")
print("Analysis: Exploratory subgroup comparison")
print("Resampling: Source-risk-group cluster bootstrap")
print("Model checkpoints: Original unregularized best models")

print("\nRESEARCH INTEGRITY")
print("Research files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.06 COMPLETE")


# %% [cell 308] Experiment 09 - Step 09.07
# ============================================================
# Experiment 09 - Step 09.07
# Retinal Coverage Threshold Feasibility Audit
# Read-only; no image access or training
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

spatial = pd.read_csv(
    SPATIAL_PATH,
    dtype={"image_id": str}
)

index = pd.read_csv(
    INDEX_PATH,
    dtype={
        "image_id": str,
        "risk_group_id": str
    }
)

validation = index.loc[
    index["split"] == "val",
    ["image_id", "risk_group_id", "dr_grade"]
].copy()

df = spatial.drop(
    columns=["risk_group_id"],
    errors="ignore"
).merge(
    validation,
    on="image_id",
    how="left",
    validate="one_to_one"
)

assert len(df) == 156
assert df["image_id"].is_unique
assert df["risk_group_id"].notna().all()
assert (df["true_grade"] == df["dr_grade"]).all()

assert df["observed_coverage"].between(0, 1).all()
assert df["eligible"].isin([True, False]).all()

# Verify the original eligibility criterion.
original_rule = df["observed_coverage"] >= 0.95

assert np.array_equal(
    original_rule.to_numpy(),
    df["eligible"].to_numpy()
), "Original eligibility has conditions beyond the 95% threshold."

# ------------------------------------------------------------
# 1. Compare candidate coverage thresholds
# ------------------------------------------------------------

thresholds = [0.90, 0.95, 0.98]

records = []

original_ids = set(
    df.loc[df["eligible"], "image_id"]
)

for threshold in thresholds:

    selected = df[
        df["observed_coverage"] >= threshold
    ]

    selected_ids = set(selected["image_id"])

    newly_included = selected_ids - original_ids
    newly_excluded = original_ids - selected_ids

    records.append({
        "coverage_threshold": threshold,
        "candidate_eligible": len(selected),
        "newly_included": len(newly_included),
        "newly_excluded": len(newly_excluded),
        "risk_groups": selected["risk_group_id"].nunique(),
        "grade_1_2": int(
            selected["true_grade"].isin([1, 2]).sum()
        ),
        "grade_3_4": int(
            selected["true_grade"].isin([3, 4]).sum()
        ),
        "minimum_possible_shifts": int(
            selected["possible_shifts"].min()
        )
    })

summary = pd.DataFrame(records)

print("EXPERIMENT 09.07 - COVERAGE THRESHOLD AUDIT")

print("\nTHRESHOLD COMPARISON")
print(summary.to_string(index=False))

# ------------------------------------------------------------
# 2. Inspect potentially recovered images
# ------------------------------------------------------------

potentially_recovered = df.loc[
    (df["observed_coverage"] >= 0.90)
    & (~df["eligible"]),
    [
        "image_id",
        "true_grade",
        "risk_group_id",
        "observed_coverage",
        "possible_shifts",
        "null_shifts",
        "exclusion_reason"
    ]
].copy()

print("\nPOTENTIALLY RECOVERED AT 90%")

print(
    potentially_recovered
    .sort_values("observed_coverage", ascending=False)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 3. Check the 95%-eligible cohort near the boundary
# ------------------------------------------------------------

boundary_cases = df.loc[
    df["eligible"]
    & (df["observed_coverage"] < 0.98),
    [
        "image_id",
        "true_grade",
        "observed_coverage",
        "possible_shifts",
        "null_shifts"
    ]
].copy()

print("\n95%-ELIGIBLE BUT BELOW 98%")
print("Images:", len(boundary_cases))

print(
    boundary_cases
    .sort_values("observed_coverage")
    .head(15)
    .to_string(index=False)
)

print("\nIMPORTANT LIMITATION")
print(
    "This is an eligibility feasibility audit, "
    "not a recomputed spatial-null analysis."
)
print(
    "Null distributions must be recomputed under "
    "any changed analysis protocol."
)

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.07 COMPLETE")


# %% [cell 309] Experiment 09 - Step 09.08
# ============================================================
# Experiment 09 - Step 09.08
# Spatial Null Shift Feasibility and Eligibility Audit
# Read-only; validation data only
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

assert SPATIAL_PATH.is_file()

df = pd.read_csv(
    SPATIAL_PATH,
    dtype={"image_id": str}
)

assert len(df) == 156
assert df["image_id"].is_unique
assert df["eligible"].isin([True, False]).all()

MIN_SHIFTS = 100
THRESHOLDS = [0.90, 0.95, 0.98]

print("EXPERIMENT 09.08 - SHIFT FEASIBILITY AUDIT")

print("\nORIGINAL SPATIAL NULL RECORDS")
print("Images:", len(df))
print("Eligible:", int(df["eligible"].sum()))
print("Excluded:", int((~df["eligible"]).sum()))

print("\nSHIFT COUNT RELATIONSHIPS")

print(
    "Possible shifts < 100:",
    int((df["possible_shifts"] < MIN_SHIFTS).sum())
)

print(
    "Recorded null shifts < 100:",
    int((df["null_shifts"] < MIN_SHIFTS).sum())
)

print(
    "Recorded null shifts = 0:",
    int((df["null_shifts"] == 0).sum())
)

print(
    "Recorded null shifts > possible shifts:",
    int(
        (df["null_shifts"] > df["possible_shifts"]).sum()
    )
)

print("\nORIGINAL ELIGIBILITY CONSISTENCY")

eligible_shift_failure = df.loc[
    df["eligible"] & (df["null_shifts"] < MIN_SHIFTS)
]

print(
    "Eligible images with <100 recorded null shifts:",
    len(eligible_shift_failure)
)

if not eligible_shift_failure.empty:
    print(
        eligible_shift_failure[
            [
                "image_id",
                "observed_coverage",
                "possible_shifts",
                "null_shifts"
            ]
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# Threshold-specific feasibility summaries
# ------------------------------------------------------------

print("\nTHRESHOLD FEASIBILITY")

records = []

for threshold in THRESHOLDS:

    selected = df.loc[
        df["observed_coverage"] >= threshold
    ].copy()

    records.append({
        "threshold": threshold,
        "coverage_candidates": len(selected),
        "recorded_null_ge_100": int(
            (selected["null_shifts"] >= MIN_SHIFTS).sum()
        ),
        "recorded_null_lt_100": int(
            (selected["null_shifts"] < MIN_SHIFTS).sum()
        ),
        "possible_shifts_ge_100": int(
            (selected["possible_shifts"] >= MIN_SHIFTS).sum()
        )
    })

print(pd.DataFrame(records).to_string(index=False))

# ------------------------------------------------------------
# Examine all 90% threshold additions
# ------------------------------------------------------------

new_at_90 = df.loc[
    (df["observed_coverage"] >= 0.90)
    & (df["observed_coverage"] < 0.95)
].copy()

new_at_90["recorded_shift_requirement_met"] = (
    new_at_90["null_shifts"] >= MIN_SHIFTS
)

print("\nNEW CANDIDATES AT 90%")

print(
    new_at_90[
        [
            "image_id",
            "true_grade",
            "observed_coverage",
            "possible_shifts",
            "null_shifts",
            "recorded_shift_requirement_met"
        ]
    ]
    .sort_values("observed_coverage", ascending=False)
    .to_string(index=False)
)

# ------------------------------------------------------------
# Report missing protocol information
# ------------------------------------------------------------

print("\nPROTOCOL CAUTION")
print(
    "Existing shift counts do not establish that a "
    "modified coverage threshold has been recomputed."
)
print(
    "The spatial translation algorithm must be reviewed "
    "before interpreting 90% results."
)

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.08 COMPLETE")


# %% [cell 310] Experiment 09 - Step 09.09
# ============================================================
# Experiment 09 - Step 09.09
# 98% Coverage Subset Robustness Analysis
# Frozen spatial-null estimates; validation only
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

PAIRED_PATH = (
    ROOT / "experiment07_results"
    / "ViTL_MA_Spatial_Null_Paired_V1.csv"
)

spatial = pd.read_csv(
    SPATIAL_PATH,
    dtype={"image_id": str}
)

paired = pd.read_csv(
    PAIRED_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(spatial) == 156
assert len(paired) == 137
assert spatial["image_id"].is_unique
assert paired["image_id"].is_unique
assert paired["risk_group_id"].notna().all()

df = paired.merge(
    spatial[
        [
            "image_id",
            "observed_coverage",
            "possible_shifts"
        ]
    ],
    on="image_id",
    how="left",
    validate="one_to_one"
)

assert len(df) == 137
assert df["observed_coverage"].notna().all()
assert df["observed_coverage"].ge(0.95).all()

# Verify the paired difference against the model-specific columns.
computed_delta = (
    df["meta_mae_null_adjusted_enrichment"]
    - df["retfound_null_adjusted_enrichment"]
)

assert np.allclose(
    computed_delta,
    df["delta_null_adjusted"],
    rtol=1e-6,
    atol=1e-8
)

primary = df.copy()

strict = df.loc[
    df["observed_coverage"] >= 0.98
].copy()

removed = df.loc[
    df["observed_coverage"] < 0.98
].copy()

assert len(primary) == 137
assert len(strict) == 129
assert len(removed) == 8

# ------------------------------------------------------------
# 1. Cluster bootstrap
# ------------------------------------------------------------

rng = np.random.default_rng(42)
N_BOOTSTRAP = 5000

def cluster_bootstrap_mean(frame, n_bootstrap=5000):
    grouped = [
        group["delta_null_adjusted"].to_numpy(dtype=float)
        for _, group in frame.groupby("risk_group_id", sort=True)
    ]

    n_groups = len(grouped)

    draws = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        sampled_indices = rng.integers(
            0, n_groups, size=n_groups
        )

        sampled_values = np.concatenate([
            grouped[i] for i in sampled_indices
        ])

        draws[b] = sampled_values.mean()

    return (
        float(frame["delta_null_adjusted"].mean()),
        np.quantile(draws, [0.025, 0.975])
    )

primary_mean, primary_ci = cluster_bootstrap_mean(
    primary, N_BOOTSTRAP
)

strict_mean, strict_ci = cluster_bootstrap_mean(
    strict, N_BOOTSTRAP
)

# ------------------------------------------------------------
# 2. Paired cohort-level difference in estimates
# ------------------------------------------------------------

# Resample primary groups jointly so membership in the strict
# subset remains fixed within each bootstrap replicate.

group_arrays = [
    group[
        ["delta_null_adjusted", "observed_coverage"]
    ].to_numpy(dtype=float)
    for _, group in primary.groupby("risk_group_id", sort=True)
]

n_groups = len(group_arrays)
difference_draws = []

for _ in range(N_BOOTSTRAP):

    indices = rng.integers(
        0, n_groups, size=n_groups
    )

    sampled = np.concatenate([
        group_arrays[i] for i in indices
    ])

    strict_values = sampled[
        sampled[:, 1] >= 0.98, 0
    ]

    if len(strict_values) == 0:
        continue

    primary_value = sampled[:, 0].mean()
    strict_value = strict_values.mean()

    difference_draws.append(
        strict_value - primary_value
    )

difference_ci = np.quantile(
    difference_draws,
    [0.025, 0.975]
)

# ------------------------------------------------------------
# 3. Report
# ------------------------------------------------------------

print("EXPERIMENT 09.09 - 98% COVERAGE ROBUSTNESS")

print("\nPRIMARY COHORT: 95%")
print("Images:", len(primary))
print("Risk groups:", primary["risk_group_id"].nunique())
print("Mean delta:", round(primary_mean, 6))
print(
    "Cluster bootstrap 95% CI:",
    tuple(np.round(primary_ci, 6))
)

print("\nSTRICT SUBSET: 98%")
print("Images:", len(strict))
print("Risk groups:", strict["risk_group_id"].nunique())
print("Mean delta:", round(strict_mean, 6))
print(
    "Cluster bootstrap 95% CI:",
    tuple(np.round(strict_ci, 6))
)

print("\nSUBSET CHANGE")
print("Removed images:", len(removed))
print(
    "Strict minus primary mean delta:",
    round(strict_mean - primary_mean, 6)
)
print(
    "Joint cluster bootstrap 95% CI:",
    tuple(np.round(difference_ci, 6))
)

print("\nREMOVED IMAGE DETAILS")
print(
    removed[
        [
            "image_id",
            "true_grade",
            "risk_group_id",
            "observed_coverage",
            "delta_null_adjusted"
        ]
    ]
    .sort_values("observed_coverage")
    .to_string(index=False)
)

print("\nMETHOD")
print("Original per-image null distributions retained")
print("98% analysis: fixed-null subset sensitivity")
print("Bootstrap unit: source-risk group")
print("Analysis status: exploratory")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.09 COMPLETE")


# %% [cell 311] Experiment 09 - Step 09.10
# ============================================================
# Experiment 09 - Step 09.10
# MA Mask Geometry and Selection Audit
# Read-only; validation cohort only
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

spatial = pd.read_csv(
    SPATIAL_PATH,
    dtype={"image_id": str}
)

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

validation = index.loc[
    index["split"] == "val",
    ["image_id", "risk_group_id", "dr_grade"]
]

assert len(cohort) == 276
assert len(spatial) == 156

df = (
    spatial[
        ["image_id", "true_grade", "eligible", "observed_coverage"]
    ]
    .merge(
        cohort[
            [
                "image_id",
                "candidate_count",
                "largest_component_area_px",
                "total_mask_area_px",
                "mask_coverage"
            ]
        ],
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        validation,
        on="image_id",
        validate="one_to_one"
    )
)

assert len(df) == 156
assert df["risk_group_id"].notna().all()
assert (df["true_grade"] == df["dr_grade"]).all()

assert df["total_mask_area_px"].gt(0).all()
assert df["candidate_count"].ge(1).all()
assert df["largest_component_area_px"].ge(0).all()
assert (
    df["largest_component_area_px"]
    <= df["total_mask_area_px"]
).all()

# ------------------------------------------------------------
# 1. Derive geometry metrics
# ------------------------------------------------------------

df["component_density_per_1000px"] = (
    1000.0
    * df["candidate_count"]
    / df["total_mask_area_px"]
)

df["largest_component_fraction"] = (
    df["largest_component_area_px"]
    / df["total_mask_area_px"]
)

df["selection_group"] = np.where(
    df["eligible"],
    "Eligible",
    "Excluded"
)

metrics = [
    "component_density_per_1000px",
    "largest_component_fraction",
    "candidate_count",
    "total_mask_area_px"
]

print("EXPERIMENT 09.10 - MA MASK GEOMETRY")

print("\nGROUP SIZES")
print(df["selection_group"].value_counts().to_string())

print("\nDESCRIPTIVE STATISTICS")

for metric in metrics:
    print(f"\n{metric}")

    print(
        df.groupby("selection_group")[metric]
        .agg(["count", "mean", "median", "std"])
        .round(6)
        .to_string()
    )

# ------------------------------------------------------------
# 2. Grouped bootstrap for geometry differences
# ------------------------------------------------------------

rng = np.random.default_rng(42)
N_BOOTSTRAP = 5000

group_arrays = [
    group[
        ["eligible"] + metrics
    ].to_numpy(dtype=float)
    for _, group in df.groupby("risk_group_id", sort=True)
]

assert len(group_arrays) == 134

print("\nCLUSTER BOOTSTRAP DIFFERENCES")
print("Difference definition: Excluded minus Eligible")

for metric_index, metric in enumerate(metrics, start=1):

    excluded_values = df.loc[
        ~df["eligible"], metric
    ]

    eligible_values = df.loc[
        df["eligible"], metric
    ]

    observed = (
        excluded_values.mean()
        - eligible_values.mean()
    )

    draws = []

    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(
            0,
            len(group_arrays),
            size=len(group_arrays)
        )

        sampled = np.concatenate([
            group_arrays[i] for i in indices
        ])

        eligible_flag = sampled[:, 0].astype(bool)
        values = sampled[:, metric_index]

        if (
            eligible_flag.sum() == 0
            or (~eligible_flag).sum() == 0
        ):
            continue

        draws.append(
            values[~eligible_flag].mean()
            - values[eligible_flag].mean()
        )

    assert len(draws) >= 4750

    lower, upper = np.quantile(
        draws,
        [0.025, 0.975]
    )

    print(f"\n{metric}")
    print("Observed difference:", round(float(observed), 6))
    print(
        "95% Cluster Bootstrap CI:",
        (
            round(float(lower), 6),
            round(float(upper), 6)
        )
    )
    print(
        "CI excludes zero:",
        bool(lower > 0 or upper < 0)
    )

print("\nMETHOD")
print("Resampling unit: source-risk group")
print("Analysis status: exploratory")
print("Clinical lesion counts verified: False")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.10 COMPLETE")


# %% [cell 312] Experiment 09 - Step 09.11
# ============================================================
# Experiment 09 - Step 09.11
# MA Geometry vs Retinal Foreground Coverage
# Validation only; read-only
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = DR_RESEARCH

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

cohort = pd.read_csv(
    COHORT_PATH, dtype={"image_id": str}
)

spatial = pd.read_csv(
    SPATIAL_PATH, dtype={"image_id": str}
)

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

validation = index.loc[
    index["split"] == "val",
    ["image_id", "risk_group_id", "dr_grade"]
]

df = (
    spatial[
        ["image_id", "true_grade", "eligible",
         "observed_coverage"]
    ]
    .merge(
        cohort[
            ["image_id", "candidate_count",
             "largest_component_area_px",
             "total_mask_area_px"]
        ],
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        validation,
        on="image_id",
        validate="one_to_one"
    )
)

assert len(df) == 156
assert df["risk_group_id"].notna().all()
assert (df["true_grade"] == df["dr_grade"]).all()
assert df["total_mask_area_px"].gt(0).all()

df["largest_component_fraction"] = (
    df["largest_component_area_px"]
    / df["total_mask_area_px"]
)

df["component_density"] = (
    1000.0 * df["candidate_count"]
    / df["total_mask_area_px"]
)

df["grade_group"] = np.where(
    df["true_grade"] >= 3,
    "Grade 3-4",
    "Grade 1-2"
)

metrics = [
    "candidate_count",
    "largest_component_fraction",
    "component_density",
    "total_mask_area_px"
]

# ------------------------------------------------------------
# 1. Overall and grade-stratified Spearman correlations
# ------------------------------------------------------------

print("EXPERIMENT 09.11 - GEOMETRY COVERAGE ASSOCIATION")

print("\nSPEARMAN CORRELATIONS")
print("Outcome: observed_coverage")

for metric in metrics:
    print(f"\n{metric}")

    for group_name, subset in [
        ("All", df),
        ("Grade 1-2", df[df["grade_group"] == "Grade 1-2"]),
        ("Grade 3-4", df[df["grade_group"] == "Grade 3-4"])
    ]:
        rho, p_value = spearmanr(
            subset[metric],
            subset["observed_coverage"]
        )

        print(
            f"{group_name}: "
            f"n={len(subset)}, "
            f"rho={rho:.6f}, "
            f"naive_p={p_value:.6g}"
        )

# ------------------------------------------------------------
# 2. Source-risk-group bootstrap for overall correlation
# ------------------------------------------------------------

group_arrays = [
    group[
        metrics + ["observed_coverage"]
    ].to_numpy(dtype=float)
    for _, group in df.groupby("risk_group_id", sort=True)
]

assert len(group_arrays) == 134

rng = np.random.default_rng(42)
N_BOOTSTRAP = 3000

print("\nCLUSTER BOOTSTRAP 95% CI")
print("Resampling unit: source-risk group")

for metric_index, metric in enumerate(metrics):

    observed_rho, _ = spearmanr(
        df[metric],
        df["observed_coverage"]
    )

    draws = []

    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(
            0,
            len(group_arrays),
            size=len(group_arrays)
        )

        sampled = np.concatenate([
            group_arrays[i] for i in indices
        ])

        rho, _ = spearmanr(
            sampled[:, metric_index],
            sampled[:, -1]
        )

        if np.isfinite(rho):
            draws.append(float(rho))

    assert len(draws) >= 2850

    lower, upper = np.quantile(
        draws, [0.025, 0.975]
    )

    print(f"\n{metric}")
    print("Observed rho:", round(float(observed_rho), 6))
    print(
        "95% CI:",
        (
            round(float(lower), 6),
            round(float(upper), 6)
        )
    )
    print(
        "CI excludes zero:",
        bool(lower > 0 or upper < 0)
    )

# ------------------------------------------------------------
# 3. Coverage-based descriptive comparison
# ------------------------------------------------------------

df["coverage_band"] = pd.cut(
    df["observed_coverage"],
    bins=[-0.001, 0.90, 0.95, 0.98, 1.001],
    labels=[
        "<90%",
        "90%-<95%",
        "95%-<98%",
        ">=98%"
    ],
    right=False
)

print("\nCOVERAGE BAND COMPOSITION")

print(
    df.groupby(
        "coverage_band",
        observed=True
    )[metrics + ["true_grade"]]
    .agg(["count", "median"])
    .round(6)
    .to_string()
)

print("\nMETHOD")
print("Correlation: Spearman rank correlation")
print("Bootstrap: Source-risk-group resampling")
print("Analysis: Exploratory")
print("Causal interpretation: Not supported")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.11 COMPLETE")


# %% [cell 313] Experiment 09 - Step 09.12
# ============================================================
# Experiment 09 - Step 09.12
# Locate Retinal Foreground and Spatial Null Implementation
# Read-only source-code audit
# ============================================================

from pathlib import Path

ROOT = DR_RESEARCH

SEARCH_DIRS = [
    ROOT / "experiment05_results",
    ROOT / "experiment07_results"
]

TERMS = [
    "observed_coverage",
    "possible_shifts",
    "null_shifts",
    "foreground",
    "morphologyEx",
    "findContours",
    "spatial_null"
]

matches = []

print("EXPERIMENT 09.12 - FOREGROUND IMPLEMENTATION AUDIT")

for folder in SEARCH_DIRS:
    if not folder.is_dir():
        continue

    for path in folder.rglob("*.py"):
        if not path.is_file():
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        found = [
            term for term in TERMS
            if term.lower() in text.lower()
        ]

        if found:
            matches.append((path, found, text))

print("\nMATCHING PYTHON SCRIPTS:", len(matches))

for path, found, text in sorted(
    matches,
    key=lambda item: str(item[0])
):
    print("\n" + "=" * 60)
    print("File:", path.relative_to(ROOT))
    print("Matched terms:", ", ".join(found))

    lines = text.splitlines()

    selected_lines = [
        i for i, line in enumerate(lines)
        if any(
            term.lower() in line.lower()
            for term in TERMS
        )
    ]

    print("Matching line numbers:", [
        i + 1 for i in selected_lines[:30]
    ])

    for i in selected_lines[:8]:
        start = max(0, i - 2)
        end = min(len(lines), i + 3)

        print(f"\nContext around line {i + 1}:")

        for j in range(start, end):
            print(f"{j + 1:04d}: {lines[j]}")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.12 COMPLETE")


# %% [cell 314] Experiment 09 - Step 09.13
# ============================================================
# Experiment 09 - Step 09.13
# Locate Original Spatial Null Code in Colab Notebooks
# Read-only notebook source audit
# ============================================================

from pathlib import Path
import json

ROOT = DR_RESEARCH

SEARCH_DIRS = [
    ROOT,
    (DR_RESEARCH.parent / "Colab Notebooks")
]

TERMS = [
    "observed_coverage",
    "possible_shifts",
    "null_shifts",
    "foreground_method",
    "findContours",
    "morphologyEx",
    "retinal_constrained",
    "spatial_null"
]

print("EXPERIMENT 09.13 - NOTEBOOK SOURCE DISCOVERY")

notebooks = {}
matches = []

# ------------------------------------------------------------
# 1. Find notebook files
# ------------------------------------------------------------

for folder in SEARCH_DIRS:
    if not folder.is_dir():
        print("Missing search directory:", folder)
        continue

    for path in folder.rglob("*.ipynb"):
        if path.is_file():
            notebooks[str(path.resolve())] = path

print("\nNOTEBOOKS DISCOVERED:", len(notebooks))

# ------------------------------------------------------------
# 2. Search code cells only
# ------------------------------------------------------------

for path in sorted(notebooks.values()):

    try:
        with path.open("r", encoding="utf-8") as handle:
            notebook = json.load(handle)

        cells = notebook.get("cells", [])

        for cell_number, cell in enumerate(cells, start=1):

            if cell.get("cell_type") != "code":
                continue

            source = cell.get("source", [])

            if isinstance(source, list):
                code = "".join(source)
            else:
                code = str(source)

            found = [
                term for term in TERMS
                if term.lower() in code.lower()
            ]

            if found:
                matches.append({
                    "path": path,
                    "cell_number": cell_number,
                    "found": found,
                    "code": code
                })

    except Exception as exc:
        print(
            "Notebook read error:",
            path.name,
            type(exc).__name__,
            str(exc)[:150]
        )

print("\nMATCHING CODE CELLS:", len(matches))

# ------------------------------------------------------------
# 3. Rank likely implementation cells
# ------------------------------------------------------------

def relevance_score(item):
    code_lower = item["code"].lower()

    return sum(
        3 if term in [
            "observed_coverage",
            "possible_shifts",
            "findContours",
            "morphologyEx"
        ] else 1
        for term in TERMS
        if term.lower() in code_lower
    )

matches.sort(
    key=relevance_score,
    reverse=True
)

for item in matches[:15]:

    path = item["path"]
    code = item["code"]
    lines = code.splitlines()

    print("\n" + "=" * 65)

    print("Notebook:", path.name)
    print("Cell number:", item["cell_number"])
    print("Matched terms:", item["found"])
    print("Code lines:", len(lines))

    # Display small, relevant excerpts only.
    matching_lines = [
        i for i, line in enumerate(lines)
        if any(
            term.lower() in line.lower()
            for term in TERMS
        )
    ]

    shown = set()

    for i in matching_lines[:5]:

        print(f"\nContext around line {i + 1}:")

        for j in range(
            max(0, i - 2),
            min(len(lines), i + 3)
        ):
            if j not in shown:
                print(f"{j + 1:04d}: {lines[j]}")
                shown.add(j)

print("\nRESEARCH INTEGRITY")
print("Notebook code executed: False")
print("Files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.13 COMPLETE")


# %% [cell 315] Experiment 09 - Step 09.13A
# ============================================================
# Experiment 09 - Step 09.13A
# Targeted Notebook Source Search
# Read-only
# ============================================================

from pathlib import Path
import json

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

TERMS = [
    "observed_coverage",
    "possible_shifts",
    "null_shifts",
    "findContours",
    "morphologyEx",
    "spatial_null",
    "foreground"
]

print("EXPERIMENT 09.13A - TARGETED NOTEBOOK AUDIT")

assert NOTEBOOK.is_file(), f"Notebook not found: {NOTEBOOK}"

with NOTEBOOK.open("r", encoding="utf-8") as f:
    notebook = json.load(f)

print("Notebook:", NOTEBOOK.name)
print("Total cells:", len(notebook.get("cells", [])))

matched = 0

for cell_number, cell in enumerate(
    notebook.get("cells", []), start=1
):
    if cell.get("cell_type") != "code":
        continue

    source = cell.get("source", [])
    code = "".join(source) if isinstance(source, list) else str(source)

    found = [
        term for term in TERMS
        if term.lower() in code.lower()
    ]

    if found:
        matched += 1
        print(f"\nCell {cell_number}")
        print("Matched terms:", found)

        lines = code.splitlines()

        for i, line in enumerate(lines):
            if any(term.lower() in line.lower() for term in TERMS):
                print(f"Line {i + 1}: {line[:180]}")

                if i > 200:
                    break

print("\nMatching code cells:", matched)
print("Files modified: False")
print("Images accessed: 0")
print("Test images accessed: 0")
print("\nSTEP 09.13A COMPLETE")


# %% [cell 316] Experiment 09 - Step 09.14
# ============================================================
# Experiment 09 - Step 09.14
# Extract Original Spatial Null Implementation
# Read-only source-code inspection
# ============================================================

from pathlib import Path
import json
import ast
import hashlib

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert NOTEBOOK.is_file()

with NOTEBOOK.open("r", encoding="utf-8") as handle:
    notebook = json.load(handle)

cells = notebook["cells"]

TARGET_CELLS = {
    127: "Experiment 05 Retinal-Constrained Spatial Null",
    240: "Experiment 07 ViT-Large Paired Spatial Null"
}

FUNCTION_KEYWORDS = [
    "foreground",
    "shift",
    "geometry",
    "fft"
]

CONSTANT_KEYWORDS = [
    "SIZE",
    "COVERAGE",
    "MIN_SHIFTS",
    "KERNEL",
    "EROSION",
    "THRESHOLD"
]

print("EXPERIMENT 09.14 - ORIGINAL SPATIAL NULL CODE AUDIT")

for cell_number, description in TARGET_CELLS.items():

    # Step 09.13A used 1-based cell numbering.
    cell = cells[cell_number - 1]

    assert cell["cell_type"] == "code"

    source = cell["source"]
    code = "".join(source) if isinstance(source, list) else str(source)

    print("\n" + "=" * 72)
    print(description)
    print("Cell number:", cell_number)
    print("Source lines:", len(code.splitlines()))
    print(
        "Source SHA-256:",
        hashlib.sha256(code.encode("utf-8")).hexdigest()
    )

    tree = ast.parse(code)

    lines = code.splitlines()

    print("\nCONFIGURATION CONSTANTS")

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):

            if isinstance(node, ast.Assign):
                names = [
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                ]
            else:
                names = (
                    [node.target.id]
                    if isinstance(node.target, ast.Name)
                    else []
                )

            if any(
                keyword in name.upper()
                for name in names
                for keyword in CONSTANT_KEYWORDS
            ):
                print(
                    "\n".join(
                        lines[node.lineno - 1:node.end_lineno]
                    )
                )

    print("\nCORE FUNCTIONS")

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            keyword in node.name.lower()
            for keyword in FUNCTION_KEYWORDS
        )
    ]

    for function in functions:

        function_code = "\n".join(
            lines[
                function.lineno - 1:
                function.end_lineno
            ]
        )

        print("\n" + "-" * 60)
        print("Function:", function.name)
        print("-" * 60)
        print(function_code)

    print("\nCORE FUNCTION COUNT:", len(functions))

print("\nRESEARCH INTEGRITY")
print("Notebook code executed: False")
print("Files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.14 COMPLETE")


# %% [cell 317] Experiment 09 - Step 09.15
# ============================================================
# Experiment 09 - Step 09.15
# Retinal Foreground Erosion Sensitivity Audit
# Excluded validation images only; no artifact changes
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

ROOT = DR_RESEARCH

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

spatial = pd.read_csv(
    SPATIAL_PATH,
    dtype={"image_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str}
)

excluded = spatial.loc[
    ~spatial["eligible"]
].copy()

assert len(excluded) == 19

validation = index.loc[
    index["split"] == "val",
    ["image_id", "local_path"]
]

audit = (
    excluded[[
        "image_id",
        "true_grade",
        "observed_coverage"
    ]]
    .merge(
        cohort[[
            "image_id",
            "resolved_mask_path"
        ]],
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        validation,
        on="image_id",
        validate="one_to_one"
    )
)

assert len(audit) == 19
assert audit["image_id"].is_unique

SIZE = 224

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (9, 9)
)

records = []

for row in audit.itertuples(index=False):

    image_path = Path(row.local_path)
    mask_path = Path(row.resolved_mask_path)

    assert image_path.is_file()
    assert mask_path.is_file()
    assert not image_path.is_symlink()

    with Image.open(image_path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (SIZE, SIZE),
                Image.Resampling.BICUBIC
            )
        )

    # Reproduce the original retinal foreground pipeline.
    candidate = (
        rgb.max(axis=2) > 20
    ).astype(np.uint8)

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    assert contours

    foreground_before = np.zeros(
        (SIZE, SIZE),
        dtype=np.uint8
    )

    cv2.drawContours(
        foreground_before,
        [max(contours, key=cv2.contourArea)],
        -1,
        1,
        thickness=-1
    )

    foreground_after = cv2.erode(
        foreground_before,
        kernel
    )

    # Diagnostic mask loading: nearest-neighbor, nonzero pixels.
    # Must be checked against original recorded coverage.
    with Image.open(mask_path) as image:
        mask = np.asarray(
            image.convert("L").resize(
                (SIZE, SIZE),
                Image.Resampling.NEAREST
            )
        ) > 0

    assert mask.any()

    before = float(
        (foreground_before * mask).sum()
        / mask.sum()
    )

    after = float(
        (foreground_after * mask).sum()
        / mask.sum()
    )

    records.append({
        "image_id": row.image_id,
        "grade": int(row.true_grade),
        "recorded_coverage": float(row.observed_coverage),
        "before_erosion": before,
        "after_erosion": after,
        "erosion_loss": before - after,
        "reproduction_error": abs(
            after - float(row.observed_coverage)
        )
    })

result = pd.DataFrame(records)

print("EXPERIMENT 09.15 - EROSION SENSITIVITY AUDIT")

print("\nREPRODUCTION CHECK")
print("Images:", len(result))
print(
    "Maximum absolute coverage error:",
    round(result["reproduction_error"].max(), 8)
)
print(
    "Mean absolute coverage error:",
    round(result["reproduction_error"].mean(), 8)
)

print("\nPER-IMAGE AUDIT")
print(
    result.sort_values(
        "erosion_loss",
        ascending=False
    ).round(6).to_string(index=False)
)

print("\nEROSION IMPACT")
print(
    "Mean coverage before erosion:",
    round(result["before_erosion"].mean(), 6)
)
print(
    "Mean coverage after erosion:",
    round(result["after_erosion"].mean(), 6)
)
print(
    "Mean erosion-induced loss:",
    round(result["erosion_loss"].mean(), 6)
)
print(
    "Images meeting 95% before erosion:",
    int((result["before_erosion"] >= 0.95).sum())
)
print(
    "Images meeting 95% after erosion:",
    int((result["after_erosion"] >= 0.95).sum())
)

print("\nREPRODUCTION STATUS")
if result["reproduction_error"].max() <= 1e-5:
    print("PASS: Coverage agrees with original records")
else:
    print("MISMATCH: Mask preprocessing must be audited")
    print("Do not interpret erosion sensitivity yet")

print("\nRESEARCH INTEGRITY")
print("Original analysis modified: False")
print("Training started: False")
print("Validation images examined:", len(result))
print("Test images accessed: 0")

print("\nSTEP 09.15 COMPLETE")


# %% [cell 318] Experiment 09 - Step 09.16
# ============================================================
# Experiment 09 - Step 09.16
# Audit Original MA Mask Loading and Preprocessing
# Read-only source inspection
# ============================================================

from pathlib import Path
import json
import ast

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert NOTEBOOK.is_file()

with NOTEBOOK.open("r", encoding="utf-8") as handle:
    notebook = json.load(handle)

cells = notebook["cells"]

TARGET_CELLS = [127, 240]

KEYWORDS = [
    "mask",
    "resize",
    "resampling",
    "interpolation",
    "threshold",
    "binar",
    "load",
    "image.open",
    "cv2.imread"
]

print("EXPERIMENT 09.16 - ORIGINAL MASK PREPROCESSING AUDIT")

for cell_number in TARGET_CELLS:

    cell = cells[cell_number - 1]
    assert cell["cell_type"] == "code"

    source = cell["source"]
    code = "".join(source) if isinstance(source, list) else str(source)

    tree = ast.parse(code)
    lines = code.splitlines()

    print("\n" + "=" * 72)
    print("CELL:", cell_number)
    print("=" * 72)

    print("\nMASK-RELATED FUNCTIONS")

    for node in tree.body:

        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue

        function_code = "\n".join(
            lines[node.lineno - 1:node.end_lineno]
        )

        function_name = node.name.lower()

        if (
            any(term in function_name for term in KEYWORDS)
            or (
                "mask" in function_code.lower()
                and any(
                    term in function_code.lower()
                    for term in [
                        "resize",
                        "image.open",
                        "cv2.imread",
                        "inter_nearest",
                        "resampling"
                    ]
                )
            )
        ):
            print("\nFUNCTION:", node.name)
            print(function_code[:12000])

    print("\nMASK PREPROCESSING REFERENCES")

    relevant = [
        i for i, line in enumerate(lines)
        if any(
            term in line.lower()
            for term in [
                "mask_path",
                "mask_image",
                "mask =",
                "load_mask",
                "resize(",
                "resampling.",
                "inter_nearest",
                "mask >",
                "mask.astype"
            ]
        )
    ]

    shown = set()

    for i in relevant[:30]:

        start = max(0, i - 3)
        end = min(len(lines), i + 4)

        for j in range(start, end):
            if j not in shown:
                print(f"{j + 1:04d}: {lines[j]}")
                shown.add(j)

print("\nRESEARCH INTEGRITY")
print("Notebook code executed: False")
print("Files modified: False")
print("Images accessed: 0")
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.16 COMPLETE")


# %% [cell 319] Experiment 09 - Step 09.17
# ============================================================
# Experiment 09 - Step 09.17
# Reproduce Original Weighted-Mask Coverage and Erosion
# Excluded validation images only; read-only
# ============================================================

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import cv2
from PIL import Image

ROOT = DR_RESEARCH

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
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
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_SHA256

spatial = pd.read_csv(
    SPATIAL_PATH, dtype={"image_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH, dtype={"image_id": str}
)

index = pd.read_csv(
    INDEX_PATH, dtype={"image_id": str}
)

excluded = spatial.loc[
    spatial["eligible"].eq(False),
    ["image_id", "true_grade", "observed_coverage"]
]

validation = index.loc[
    index["split"].eq("val"),
    ["image_id", "local_path"]
]

audit = (
    excluded
    .merge(
        cohort[
            [
                "image_id",
                "resolved_mask_path",
                "reference_threshold"
            ]
        ],
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        validation,
        on="image_id",
        validate="one_to_one"
    )
)

assert len(audit) == 19
assert audit["image_id"].is_unique

SIZE = 224
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (9, 9)
)

records = []

for row in audit.itertuples(index=False):

    image_path = Path(row.local_path)
    mask_path = Path(row.resolved_mask_path)

    assert image_path.is_file()
    assert not image_path.is_symlink()
    assert mask_path.is_file()

    # Original retinal image preprocessing.
    with Image.open(image_path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (SIZE, SIZE),
                Image.Resampling.BICUBIC
            )
        )

    # Original weighted MA-mask preprocessing.
    with Image.open(mask_path) as image:
        mask = np.asarray(image)

    if mask.ndim == 3:
        assert mask.shape[2] >= 3
        assert np.array_equal(mask[..., 0], mask[..., 1])
        assert np.array_equal(mask[..., 0], mask[..., 2])
        mask = mask[..., 0]

    assert mask.shape == (1280, 1280)

    binary = (
        mask > int(row.reference_threshold)
    ).astype(np.float32)

    weighted = cv2.resize(
        binary,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    ).astype(np.float64)

    assert weighted.sum() > 0

    # Original retinal foreground pipeline.
    intensity = rgb.max(axis=2)

    candidate = (
        intensity > 20
    ).astype(np.uint8)

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    assert contours

    foreground_before = np.zeros(
        (SIZE, SIZE),
        dtype=np.uint8
    )

    cv2.drawContours(
        foreground_before,
        [max(contours, key=cv2.contourArea)],
        -1,
        1,
        thickness=-1
    )

    foreground_after = cv2.erode(
        foreground_before,
        kernel
    )

    before = float(
        (foreground_before * weighted).sum()
        / weighted.sum()
    )

    after = float(
        (foreground_after * weighted).sum()
        / weighted.sum()
    )

    records.append({
        "image_id": row.image_id,
        "grade": int(row.true_grade),
        "recorded_coverage": float(row.observed_coverage),
        "before_erosion": before,
        "after_erosion": after,
        "erosion_loss": before - after,
        "reproduction_error": abs(
            after - float(row.observed_coverage)
        )
    })

result = pd.DataFrame(records)

print("EXPERIMENT 09.17 - WEIGHTED MASK REPRODUCTION")

print("\nREPRODUCTION CHECK")
print("Images audited:", len(result))
print(
    "Maximum absolute error:",
    round(float(result["reproduction_error"].max()), 10)
)
print(
    "Mean absolute error:",
    round(float(result["reproduction_error"].mean()), 10)
)

reproduced = bool(
    result["reproduction_error"].max() <= 1e-5
)

print(
    "Original coverage reproduced:",
    "PASS" if reproduced else "FAIL"
)

if reproduced:

    assert np.all(
        result["before_erosion"]
        >= result["after_erosion"] - 1e-10
    )

    print("\nEROSION IMPACT")
    print(
        "Mean coverage before erosion:",
        round(float(result["before_erosion"].mean()), 6)
    )
    print(
        "Mean coverage after erosion:",
        round(float(result["after_erosion"].mean()), 6)
    )
    print(
        "Mean erosion-induced loss:",
        round(float(result["erosion_loss"].mean()), 6)
    )

    print(
        "Meeting 95% before erosion:",
        int((result["before_erosion"] >= 0.95).sum())
    )
    print(
        "Meeting 95% after erosion:",
        int((result["after_erosion"] >= 0.95).sum())
    )

    print("\nPER-IMAGE RESULTS")
    print(
        result.sort_values(
            "erosion_loss", ascending=False
        ).round(6).to_string(index=False)
    )

    print("\nSTEP 09.17 PASS")

else:

    print("\nREPRODUCTION DIAGNOSTICS")
    print(
        result.sort_values(
            "reproduction_error", ascending=False
        ).head(19).round(8).to_string(index=False)
    )

    print("\nSTOP: Do not interpret erosion effects.")
    print("STEP 09.17 REQUIRES INVESTIGATION")

print("\nRESEARCH INTEGRITY")
print("Frozen index modified: False")
print("Research files modified: False")
print("Validation images accessed:", len(result))
print("Test images accessed: 0")
print("Training started: False")


# %% [cell 320] Experiment 09 - Step 09.18
# ============================================================
# Experiment 09 - Step 09.18
# No-Erosion Spatial Null Eligibility Feasibility Audit
# Original protocol remains frozen; validation images only
# ============================================================

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import cv2
from PIL import Image

ROOT = DR_RESEARCH

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_SHA256

spatial = pd.read_csv(SPATIAL_PATH, dtype={"image_id": str})
cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})
index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})

excluded = spatial.loc[
    ~spatial["eligible"],
    [
        "image_id", "true_grade", "observed_coverage",
        "possible_shifts", "null_shifts"
    ]
].copy()

validation = index.loc[
    index["split"] == "val",
    ["image_id", "local_path"]
]

audit = (
    excluded
    .merge(
        cohort[
            ["image_id", "resolved_mask_path",
             "reference_threshold"]
        ],
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        validation,
        on="image_id",
        validate="one_to_one"
    )
)

assert len(audit) == 19
assert audit["image_id"].is_unique

SIZE = 224
MIN_COVERAGE = 0.95
MIN_NULL = 100

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (9, 9)
)

def compute_geometry(mask, foreground):
    ys, xs = np.nonzero(mask > 0)

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())

    dx = np.arange(-xmin, SIZE - xmax, dtype=int)
    dy = np.arange(-ymin, SIZE - ymax, dtype=int)

    shifts_y, shifts_x = np.meshgrid(
        dy % SIZE, dx % SIZE, indexing="ij"
    )

    coverage_map = np.fft.ifft2(
        np.fft.fft2(foreground.astype(np.float64))
        * np.conj(np.fft.fft2(mask))
    ).real / float(mask.sum())

    permitted = (
        coverage_map[shifts_y, shifts_x] >= MIN_COVERAGE
    )

    permitted &= ~(
        (shifts_y == 0) & (shifts_x == 0)
    )

    observed_coverage = float(
        (foreground * mask).sum() / mask.sum()
    )

    null_shifts = int(permitted.sum())

    return {
        "coverage": observed_coverage,
        "possible_shifts": int(len(dx) * len(dy)),
        "null_shifts": null_shifts,
        "eligible": (
            observed_coverage >= MIN_COVERAGE
            and null_shifts >= MIN_NULL
        )
    }

records = []

for row in audit.itertuples(index=False):

    image_path = Path(row.local_path)
    mask_path = Path(row.resolved_mask_path)

    assert image_path.is_file()
    assert not image_path.is_symlink()
    assert mask_path.is_file()

    with Image.open(image_path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (SIZE, SIZE),
                Image.Resampling.BICUBIC
            )
        )

    with Image.open(mask_path) as image:
        raw_mask = np.asarray(image)

    if raw_mask.ndim == 3:
        assert np.array_equal(raw_mask[..., 0], raw_mask[..., 1])
        assert np.array_equal(raw_mask[..., 0], raw_mask[..., 2])
        raw_mask = raw_mask[..., 0]

    assert raw_mask.shape == (1280, 1280)

    binary = (
        raw_mask > int(row.reference_threshold)
    ).astype(np.float32)

    mask = cv2.resize(
        binary, (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    ).astype(np.float64)

    assert mask.sum() > 0

    candidate = (
        rgb.max(axis=2) > 20
    ).astype(np.uint8)

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    assert contours

    foreground_before = np.zeros(
        (SIZE, SIZE), dtype=np.uint8
    )

    cv2.drawContours(
        foreground_before,
        [max(contours, key=cv2.contourArea)],
        -1, 1, thickness=-1
    )

    foreground_after = cv2.erode(
        foreground_before, kernel
    )

    original = compute_geometry(
        mask, foreground_after
    )

    no_erosion = compute_geometry(
        mask, foreground_before
    )

    # Verify exact original protocol reproduction.
    assert abs(
        original["coverage"] - float(row.observed_coverage)
    ) <= 1e-5, f"Coverage mismatch: {row.image_id}"

    assert (
        original["possible_shifts"] == int(row.possible_shifts)
    ), f"Possible-shift mismatch: {row.image_id}"

    assert (
        original["null_shifts"] == int(row.null_shifts)
    ), f"Null-shift mismatch: {row.image_id}"

    records.append({
        "image_id": row.image_id,
        "grade": int(row.true_grade),
        "original_coverage": original["coverage"],
        "no_erosion_coverage": no_erosion["coverage"],
        "original_null_shifts": original["null_shifts"],
        "no_erosion_null_shifts": no_erosion["null_shifts"],
        "original_eligible": original["eligible"],
        "no_erosion_eligible": no_erosion["eligible"]
    })

result = pd.DataFrame(records)

assert len(result) == 19
assert not result["original_eligible"].any()

print("EXPERIMENT 09.18 - NO-EROSION ELIGIBILITY")

print("\nREPRODUCTION")
print("Original coverage reproduced: PASS")
print("Original possible shifts reproduced: PASS")
print("Original null shifts reproduced: PASS")

print("\nELIGIBILITY COMPARISON")
print("Original eligible:", int(result["original_eligible"].sum()))
print("No-erosion eligible:", int(result["no_erosion_eligible"].sum()))

print(
    "No-erosion coverage >=95%:",
    int((result["no_erosion_coverage"] >= 0.95).sum())
)

print(
    "No-erosion null shifts >=100:",
    int((result["no_erosion_null_shifts"] >= 100).sum())
)

print("\nPER-IMAGE RESULTS")
print(
    result.sort_values(
        "no_erosion_coverage"
    ).round(6).to_string(index=False)
)

print("\nMETHOD")
print("Original 95% protocol: unchanged")
print("Sensitivity condition: erosion removed")
print("CAM enrichment recomputed: False")
print("Analysis status: exploratory feasibility audit")

print("\nRESEARCH INTEGRITY")
print("Research files modified: False")
print("Validation images examined:", len(result))
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.18 COMPLETE")


# %% [cell 321] Experiment 09 - Step 09.19
# ============================================================
# Experiment 09 - Step 09.19
# CAM Availability Audit for No-Erosion Sensitivity
# Read-only; no inference or image access
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

RET_CAM = (
    ROOT / "experiment05_results"
    / "gradcam"
    / "RETFound_MA_CAM_Maps_V1.npz"
)

META_CAM = (
    ROOT / "experiment07_results"
    / "gradcam"
    / "Meta_MAE_MA_CAM_Maps_V1.npz"
)

assert all(
    path.is_file()
    for path in [SPATIAL_PATH, COHORT_PATH, RET_CAM, META_CAM]
)

spatial = pd.read_csv(
    SPATIAL_PATH, dtype={"image_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH, dtype={"image_id": str}
)

excluded = spatial.loc[
    ~spatial["eligible"],
    ["image_id", "true_grade", "observed_coverage"]
].copy()

assert len(excluded) == 19

# The 17 images meeting the no-erosion eligibility rule,
# as verified in Step 09.18.
newly_eligible_ids = {
    "1773_2",
    "0062_1",
    "1214_1",
    "0991_1",
    "0227_3",
    "0043_3",
    "0498_3",
    "0432_3",
    "0295_3",
    "0603_3",
    "0807_3",
    "0641_3",
    "0097_2",
    "0170_2",
    "0870_3",
    "0617_2",
    "1469_3"
}

assert newly_eligible_ids.issubset(
    set(excluded["image_id"])
)
assert len(newly_eligible_ids) == 17

print("EXPERIMENT 09.19 - CAM AVAILABILITY AUDIT")

print("\nNEWLY ELIGIBLE IMAGES:", len(newly_eligible_ids))

def inspect_npz(path, label):
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)

        print(f"\n{label}")
        print("NPZ arrays:", len(keys))

        for key in keys[:12]:
            array = data[key]
            print(
                "Key:", key,
                "| shape:", array.shape,
                "| dtype:", array.dtype
            )

        direct_matches = newly_eligible_ids.intersection(keys)

        print(
            "Direct image-ID key matches:",
            len(direct_matches)
        )

        for key in keys:
            if "id" in key.lower():
                values = np.asarray(data[key])

                if values.ndim == 1:
                    image_ids = set(
                        str(value) for value in values
                    )

                    matches = newly_eligible_ids & image_ids

                    print(
                        "ID array:", key,
                        "| matching images:", len(matches)
                    )

                    print(
                        "Missing IDs:",
                        sorted(newly_eligible_ids - matches)
                    )

inspect_npz(RET_CAM, "RETFOUND CAM")
inspect_npz(META_CAM, "META MAE CAM")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Model inference performed: False")
print("Images accessed: 0")
print("Test images accessed: 0")

print("\nSTEP 09.19 COMPLETE")


# %% [cell 322] Experiment 09 - Step 09.20
# ============================================================
# Experiment 09 - Step 09.20
# Full-Cohort No-Erosion Eligibility Audit
# Validation MA candidates only; read-only
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

ROOT = DR_RESEARCH

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

COHORT_PATH = (
    ROOT / "experiment05_results"
    / "FGADR_Validation_MA_Cohort_V1.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

spatial = pd.read_csv(
    SPATIAL_PATH,
    dtype={"image_id": str}
)

cohort = pd.read_csv(
    COHORT_PATH,
    dtype={"image_id": str}
)

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str}
)

validation = index.loc[
    index["split"] == "val",
    ["image_id", "local_path"]
]

df = (
    spatial[
        [
            "image_id",
            "true_grade",
            "eligible",
            "observed_coverage",
            "possible_shifts",
            "null_shifts"
        ]
    ]
    .merge(
        cohort[
            [
                "image_id",
                "resolved_mask_path",
                "reference_threshold"
            ]
        ],
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        validation,
        on="image_id",
        validate="one_to_one"
    )
)

assert len(df) == 156
assert df["image_id"].is_unique

SIZE = 224
MIN_COVERAGE = 0.95
MIN_NULL = 100

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (9, 9)
)

def geometry(mask, foreground):
    ys, xs = np.nonzero(mask > 0)

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())

    dx = np.arange(-xmin, SIZE - xmax, dtype=int)
    dy = np.arange(-ymin, SIZE - ymax, dtype=int)

    shifts_y, shifts_x = np.meshgrid(
        dy % SIZE,
        dx % SIZE,
        indexing="ij"
    )

    coverage_map = np.fft.ifft2(
        np.fft.fft2(foreground.astype(np.float64))
        * np.conj(np.fft.fft2(mask))
    ).real / float(mask.sum())

    permitted = (
        coverage_map[shifts_y, shifts_x] >= MIN_COVERAGE
    )

    permitted &= ~(
        (shifts_y == 0)
        & (shifts_x == 0)
    )

    observed_coverage = float(
        (foreground * mask).sum()
        / mask.sum()
    )

    null_shifts = int(permitted.sum())

    return {
        "coverage": observed_coverage,
        "possible_shifts": int(len(dx) * len(dy)),
        "null_shifts": null_shifts,
        "eligible": (
            observed_coverage >= MIN_COVERAGE
            and null_shifts >= MIN_NULL
        )
    }

records = []

for row in df.itertuples(index=False):

    with Image.open(row.local_path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (SIZE, SIZE),
                Image.Resampling.BICUBIC
            )
        )

    with Image.open(row.resolved_mask_path) as image:
        original_mask = np.asarray(image)

    if original_mask.ndim == 3:
        assert np.array_equal(
            original_mask[..., 0],
            original_mask[..., 1]
        )
        assert np.array_equal(
            original_mask[..., 0],
            original_mask[..., 2]
        )
        original_mask = original_mask[..., 0]

    assert original_mask.shape == (1280, 1280)

    binary = (
        original_mask > int(row.reference_threshold)
    ).astype(np.float32)

    mask = cv2.resize(
        binary,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    ).astype(np.float64)

    assert mask.sum() > 0

    candidate = (
        rgb.max(axis=2) > 20
    ).astype(np.uint8)

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    assert contours

    foreground_before = np.zeros(
        (SIZE, SIZE),
        dtype=np.uint8
    )

    cv2.drawContours(
        foreground_before,
        [max(contours, key=cv2.contourArea)],
        -1, 1, thickness=-1
    )

    foreground_after = cv2.erode(
        foreground_before,
        kernel
    )

    original = geometry(mask, foreground_after)
    no_erosion = geometry(mask, foreground_before)

    # Reproduce the original frozen records.
    assert abs(
        original["coverage"]
        - float(row.observed_coverage)
    ) <= 1e-5, f"Coverage mismatch: {row.image_id}"

    assert (
        original["possible_shifts"]
        == int(row.possible_shifts)
    ), f"Possible-shift mismatch: {row.image_id}"

    assert (
        original["null_shifts"]
        == int(row.null_shifts)
    ), f"Null-shift mismatch: {row.image_id}"

    assert (
        original["eligible"]
        == bool(row.eligible)
    ), f"Eligibility mismatch: {row.image_id}"

    records.append({
        "image_id": row.image_id,
        "grade": int(row.true_grade),
        "original_eligible": original["eligible"],
        "no_erosion_eligible": no_erosion["eligible"],
        "original_coverage": original["coverage"],
        "no_erosion_coverage": no_erosion["coverage"],
        "original_null_shifts": original["null_shifts"],
        "no_erosion_null_shifts": no_erosion["null_shifts"]
    })

result = pd.DataFrame(records)

original_ids = set(
    result.loc[
        result["original_eligible"],
        "image_id"
    ]
)

new_ids = set(
    result.loc[
        result["no_erosion_eligible"],
        "image_id"
    ]
)

print("EXPERIMENT 09.20 - FULL COHORT ELIGIBILITY")

print("\nORIGINAL PROTOCOL REPRODUCTION")
print("Images verified:", len(result))
print("Original eligible:", len(original_ids))
print("Expected original eligible:", 137)

assert len(original_ids) == 137

print("\nNO-EROSION ELIGIBILITY")
print("Eligible images:", len(new_ids))
print("Newly included:", len(new_ids - original_ids))
print("Previously eligible but removed:", len(original_ids - new_ids))

print("\nGRADE DISTRIBUTION")

selected = result.loc[
    result["no_erosion_eligible"]
]

print(
    selected["grade"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nNULL SHIFT SUMMARY")
print(
    selected["no_erosion_null_shifts"]
    .describe()
    .round(2)
    .to_string()
)

print("\nADDED IMAGE IDS")
print(sorted(new_ids - original_ids))

print("\nREMOVED IMAGE IDS")
print(sorted(original_ids - new_ids))

assert len(new_ids - original_ids) == 17
assert len(original_ids - new_ids) == 0
assert len(new_ids) == 154

print("\nMETHOD")
print("Original 95% primary analysis preserved")
print("No-erosion sensitivity: eligibility only")
print("CAM values recalculated: False")
print("New null distributions calculated: False")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Validation images accessed:", len(result))
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.20 PASS")


# %% [cell 323] Experiment 09 - Step 09.21
# ============================================================
# Experiment 09 - Step 09.21
# Audit Original Spatial-Null Attribution Formulas
# Read-only notebook source inspection
# ============================================================

from pathlib import Path
import json

NOTEBOOK = Path(
    str(DR_RESEARCH.parent / "Colab Notebooks/FGADR_MA_Mask_Audit.ipynb")
)

assert NOTEBOOK.is_file()

with NOTEBOOK.open("r", encoding="utf-8") as handle:
    notebook = json.load(handle)

print("EXPERIMENT 09.21 - ATTRIBUTION FORMULA AUDIT")

TARGET_CELLS = {
    127: "Experiment 05 - Original Retinal Spatial Null",
    240: "Experiment 07 - Paired ViT-Large Spatial Null"
}

TERMS = [
    "enrichment",
    "null_mean",
    "adjusted",
    "cam.mean",
    "mask.sum",
    "fft_shift_values",
    "coords_y",
    "coords_x",
    "spatial_percentile"
]

for cell_number, description in TARGET_CELLS.items():

    cell = notebook["cells"][cell_number - 1]

    assert cell["cell_type"] == "code"

    source = cell["source"]
    code = "".join(source) if isinstance(source, list) else str(source)

    lines = code.splitlines()

    print("\n" + "=" * 70)
    print(description)
    print("Cell number:", cell_number)

    relevant_lines = [
        i for i, line in enumerate(lines)
        if any(term in line.lower() for term in TERMS)
    ]

    intervals = []

    for i in relevant_lines:
        start = max(0, i - 4)
        end = min(len(lines), i + 7)
        intervals.append((start, end))

    merged = []

    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (
                merged[-1][0],
                max(merged[-1][1], end)
            )
        else:
            merged.append((start, end))

    print("Relevant source intervals:", len(merged))

    for start, end in merged:
        print(f"\n--- Lines {start + 1}-{end} ---")

        for j in range(start, end):
            print(f"{j + 1:04d}: {lines[j]}")

print("\nRESEARCH INTEGRITY")
print("Notebook code executed: False")
print("Research files modified: False")
print("Images accessed: 0")
print("Model inference performed: False")
print("Test images accessed: 0")

print("\nSTEP 09.21 COMPLETE")


# %% [cell 324] Experiment 09 - Step 09.22
# ============================================================
# Experiment 09 - Step 09.22
# Reproduce Frozen Experiment 07 Paired Spatial Null
# Validation only; read-only; no model inference
# ============================================================

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import cv2
from PIL import Image

ROOT = DR_RESEARCH

INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
SPATIAL_PATH = ROOT / "experiment05_results/MA_Spatial_Null_Retinal_Constrained_V1.csv"
PAIRED_PATH = ROOT / "experiment07_results/ViTL_MA_Spatial_Null_Paired_V1.csv"

RET_CAM_PATH = ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"
META_CAM_PATH = ROOT / "experiment07_results/gradcam/Meta_MAE_MA_CAM_Maps_V1.npz"

EXPECTED_INDEX_SHA = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

SIZE = 224
MIN_COVERAGE = 0.95
MIN_NULL = 100

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

assert file_sha256(INDEX_PATH) == EXPECTED_INDEX_SHA

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)
cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})
spatial = pd.read_csv(SPATIAL_PATH, dtype={"image_id": str})
reference = pd.read_csv(
    PAIRED_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(reference) == 137
assert reference["image_id"].is_unique
assert reference["eligible"].eq(True).all()

validation = index.loc[
    index["split"].eq("val"),
    ["image_id", "local_path", "risk_group_id", "dr_grade"]
]

rows = (
    reference[["image_id"]]
    .merge(validation, on="image_id", validate="one_to_one")
    .merge(
        cohort[["image_id", "resolved_mask_path", "reference_threshold"]],
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        spatial[
            ["image_id", "observed_coverage", "possible_shifts", "null_shifts"]
        ],
        on="image_id",
        validate="one_to_one"
    )
)

assert len(rows) == 137

with np.load(RET_CAM_PATH, allow_pickle=False) as data:
    ret_ids = [str(v) for v in data["image_ids"]]
    ret_maps = data["retfound_cam"].copy()

with np.load(META_CAM_PATH, allow_pickle=False) as data:
    meta_ids = [str(v) for v in data["image_ids"]]
    meta_maps = data["meta_mae_cam"].copy()

assert len(ret_ids) == len(set(ret_ids)) == 156
assert len(meta_ids) == len(set(meta_ids)) == 156
assert ret_maps.shape == (156, SIZE, SIZE)
assert meta_maps.shape == (156, SIZE, SIZE)

ret_lookup = {key: i for i, key in enumerate(ret_ids)}
meta_lookup = {key: i for i, key in enumerate(meta_ids)}

assert set(rows["image_id"]).issubset(ret_lookup)
assert set(rows["image_id"]).issubset(meta_lookup)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (9, 9)
)

def fft_shift_values(values, mask):
    return np.fft.ifft2(
        np.fft.fft2(values.astype(np.float64))
        * np.conj(np.fft.fft2(mask))
    ).real

def load_weighted_mask(row):
    with Image.open(row.resolved_mask_path) as image:
        raw = np.asarray(image)

    if raw.ndim == 3:
        assert np.array_equal(raw[..., 0], raw[..., 1])
        assert np.array_equal(raw[..., 0], raw[..., 2])
        raw = raw[..., 0]

    assert raw.shape == (1280, 1280)

    binary = (
        raw > int(row.reference_threshold)
    ).astype(np.float32)

    weighted = cv2.resize(
        binary,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    ).astype(np.float64)

    assert weighted.sum() > 0
    return weighted

def load_original_foreground(row):
    path = Path(row.local_path)
    assert path.is_file() and not path.is_symlink()

    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (SIZE, SIZE),
                Image.Resampling.BICUBIC
            )
        )

    candidate = (
        rgb.max(axis=2) > 20
    ).astype(np.uint8)

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel
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

    return cv2.erode(
        foreground, kernel
    ).astype(np.float64)

def original_geometry(mask, foreground):
    ys, xs = np.nonzero(mask > 0)

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())

    dx = np.arange(-xmin, SIZE - xmax, dtype=int)
    dy = np.arange(-ymin, SIZE - ymax, dtype=int)

    shifts_y, shifts_x = np.meshgrid(
        dy % SIZE, dx % SIZE, indexing="ij"
    )

    coverage = (
        fft_shift_values(foreground, mask)
        / float(mask.sum())
    )

    permitted = (
        coverage[shifts_y, shifts_x] >= MIN_COVERAGE
    )

    permitted &= ~(
        (shifts_y == 0) & (shifts_x == 0)
    )

    return {
        "coverage": float((foreground * mask).sum() / mask.sum()),
        "possible_shifts": int(len(dx) * len(dy)),
        "null_shifts": int(permitted.sum()),
        "coords_y": shifts_y[permitted],
        "coords_x": shifts_x[permitted]
    }

def spatial_metrics(cam, mask, geometry):
    scores = fft_shift_values(cam, mask)
    scale = float(mask.sum()) * float(cam.mean())
    assert np.isfinite(scale) and scale > 1e-12

    scores /= scale
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
        "null_adjusted_enrichment": float(observed - null.mean()),
        "spatial_percentile": float(100 * np.mean(null < observed)),
        "upper_tail_fraction": float(
            (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)
        )
    }

metrics = [
    "observed_enrichment",
    "null_mean_enrichment",
    "null_adjusted_enrichment",
    "spatial_percentile",
    "upper_tail_fraction"
]

reproduced = []

print("EXPERIMENT 09.22 - FROZEN PAIRED SPATIAL NULL REPRODUCTION")

for number, row in enumerate(rows.itertuples(index=False), start=1):
    mask = load_weighted_mask(row)
    foreground = load_original_foreground(row)
    geometry = original_geometry(mask, foreground)

    assert abs(
        geometry["coverage"] - float(row.observed_coverage)
    ) <= 1e-5, f"Coverage mismatch: {row.image_id}"

    assert geometry["possible_shifts"] == int(row.possible_shifts)
    assert geometry["null_shifts"] == int(row.null_shifts)

    assert geometry["coverage"] >= MIN_COVERAGE
    assert geometry["null_shifts"] >= MIN_NULL

    ret = spatial_metrics(
        ret_maps[ret_lookup[row.image_id]],
        mask,
        geometry
    )

    meta = spatial_metrics(
        meta_maps[meta_lookup[row.image_id]],
        mask,
        geometry
    )

    record = {"image_id": row.image_id}

    for prefix, result in [
        ("retfound", ret),
        ("meta_mae", meta)
    ]:
        for key, value in result.items():
            record[f"{prefix}_{key}"] = value

    record["delta_null_adjusted"] = (
        meta["null_adjusted_enrichment"]
        - ret["null_adjusted_enrichment"]
    )

    record["delta_spatial_percentile"] = (
        meta["spatial_percentile"]
        - ret["spatial_percentile"]
    )

    reproduced.append(record)

    if number % 25 == 0 or number == len(rows):
        print(f"Verified computations: {number}/{len(rows)}")

actual = pd.DataFrame(reproduced).set_index("image_id")
expected = reference.set_index("image_id").loc[actual.index]

fields = (
    [f"{prefix}_{metric}" for prefix in ["retfound", "meta_mae"]
     for metric in metrics]
    + ["delta_null_adjusted", "delta_spatial_percentile"]
)

print("\nPER-METRIC REPRODUCTION ERRORS")

all_pass = True

for field in fields:
    error = np.abs(
        actual[field].to_numpy(dtype=float)
        - expected[field].to_numpy(dtype=float)
    )

    maximum = float(np.max(error))
    tolerance = 1e-4 if "percentile" in field else 1e-5
    passed = bool(np.isfinite(error).all() and maximum <= tolerance)

    all_pass &= passed

    print(
        f"{field}: max_error={maximum:.10f}, "
        f"tolerance={tolerance}, "
        f"status={'PASS' if passed else 'FAIL'}"
    )

print("\nPRIMARY RESULT")
print(
    "Frozen mean delta:",
    round(float(expected["delta_null_adjusted"].mean()), 6)
)
print(
    "Reproduced mean delta:",
    round(float(actual["delta_null_adjusted"].mean()), 6)
)

print("\nRESEARCH INTEGRITY")
print("Original result files modified: False")
print("New result files written: False")
print("Validation images accessed:", len(rows))
print("Model inference performed: False")
print("Test images accessed: 0")

if all_pass:
    print("\nSTEP 09.22 PASS")
else:
    print("\nSTEP 09.22 REQUIRES INVESTIGATION")
    print("Do not proceed to no-erosion attribution yet.")


# %% [cell 325] Experiment 09 - Step 09.23
# ============================================================
# Experiment 09 - Step 09.23
# No-Erosion Paired Spatial Null Sensitivity Analysis
# Validation only; original experiments preserved
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = DR_RESEARCH

INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
SPATIAL_PATH = ROOT / "experiment05_results/MA_Spatial_Null_Retinal_Constrained_V1.csv"
REFERENCE_PATH = ROOT / "experiment07_results/ViTL_MA_Spatial_Null_Paired_V1.csv"

RET_CAM_PATH = ROOT / "experiment05_results/gradcam/RETFound_MA_CAM_Maps_V1.npz"
META_CAM_PATH = ROOT / "experiment07_results/gradcam/Meta_MAE_MA_CAM_Maps_V1.npz"

OUT = ROOT / "experiment09_results"
RESULT_PATH = OUT / "No_Erosion_Spatial_Null_Paired_V1.csv"
REPORT_PATH = OUT / "No_Erosion_Spatial_Null_Report_V1.json"

EXPECTED_INDEX_SHA = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

SIZE = 224
MIN_COVERAGE = 0.95
MIN_NULL = 100

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

# ------------------------------------------------------------
# 1. Load and validate frozen sources
# ------------------------------------------------------------

assert sha256_file(INDEX_PATH) == EXPECTED_INDEX_SHA
assert not RESULT_PATH.exists(), "Output already exists."
assert not REPORT_PATH.exists(), "Report already exists."

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})
spatial = pd.read_csv(SPATIAL_PATH, dtype={"image_id": str})
reference = pd.read_csv(
    REFERENCE_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(cohort) == 276
assert len(spatial) == 156
assert len(reference) == 137

validation = index.loc[
    index["split"] == "val",
    ["image_id", "local_path", "risk_group_id", "dr_grade"]
]

primary = cohort.loc[
    cohort["primary_ma_candidate"].eq(True),
    ["image_id", "resolved_mask_path", "reference_threshold"]
]

rows = (
    spatial[
        [
            "image_id",
            "true_grade",
            "eligible",
            "observed_coverage",
            "possible_shifts",
            "null_shifts"
        ]
    ]
    .merge(primary, on="image_id", validate="one_to_one")
    .merge(validation, on="image_id", validate="one_to_one")
    .sort_values("image_id")
    .reset_index(drop=True)
)

assert len(rows) == 156
assert rows["image_id"].is_unique
assert rows["risk_group_id"].notna().all()
assert (rows["true_grade"] == rows["dr_grade"]).all()

with np.load(RET_CAM_PATH, allow_pickle=False) as data:
    ret_ids = [str(x) for x in data["image_ids"]]
    ret_maps = data["retfound_cam"].copy()

with np.load(META_CAM_PATH, allow_pickle=False) as data:
    meta_ids = [str(x) for x in data["image_ids"]]
    meta_maps = data["meta_mae_cam"].copy()

assert len(set(ret_ids)) == len(ret_ids) == 156
assert len(set(meta_ids)) == len(meta_ids) == 156
assert ret_maps.shape == (156, SIZE, SIZE)
assert meta_maps.shape == (156, SIZE, SIZE)

ret_lookup = {image_id: i for i, image_id in enumerate(ret_ids)}
meta_lookup = {image_id: i for i, image_id in enumerate(meta_ids)}

assert set(rows["image_id"]) == set(ret_lookup)
assert set(rows["image_id"]) == set(meta_lookup)

# ------------------------------------------------------------
# 2. Reproduce original mask and foreground preprocessing
# ------------------------------------------------------------

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (9, 9)
)

def load_mask(row):
    with Image.open(row.resolved_mask_path) as image:
        raw = np.asarray(image)

    if raw.ndim == 3:
        assert np.array_equal(raw[..., 0], raw[..., 1])
        assert np.array_equal(raw[..., 0], raw[..., 2])
        raw = raw[..., 0]

    assert raw.shape == (1280, 1280)

    binary = (
        raw > int(row.reference_threshold)
    ).astype(np.float32)

    mask = cv2.resize(
        binary,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    ).astype(np.float64)

    assert mask.sum() > 0
    return mask

def load_foregrounds(row):
    path = Path(row.local_path)
    assert path.is_file() and not path.is_symlink()

    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (SIZE, SIZE),
                Image.Resampling.BICUBIC
            )
        )

    candidate = (
        rgb.max(axis=2) > 20
    ).astype(np.uint8)

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    assert contours

    no_erosion = np.zeros(
        (SIZE, SIZE),
        dtype=np.uint8
    )

    cv2.drawContours(
        no_erosion,
        [max(contours, key=cv2.contourArea)],
        -1, 1, thickness=-1
    )

    original = cv2.erode(no_erosion, kernel)

    return (
        original.astype(np.float64),
        no_erosion.astype(np.float64)
    )

# ------------------------------------------------------------
# 3. Original spatial null geometry
# ------------------------------------------------------------

def fft_shift_values(values, mask):
    return np.fft.ifft2(
        np.fft.fft2(values.astype(np.float64))
        * np.conj(np.fft.fft2(mask))
    ).real

def geometry(mask, foreground):
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

    return {
        "coverage": observed_coverage,
        "possible_shifts": int(len(dx) * len(dy)),
        "null_shifts": int(len(coords_y)),
        "coords_y": coords_y,
        "coords_x": coords_x,
        "eligible": bool(
            observed_coverage >= MIN_COVERAGE
            and len(coords_y) >= MIN_NULL
        )
    }

# ------------------------------------------------------------
# 4. Reuse exactly the original attribution formula
# ------------------------------------------------------------

def spatial_metrics(cam, mask, geo):
    scores = fft_shift_values(cam, mask)

    scale = float(mask.sum()) * float(cam.mean())
    assert np.isfinite(scale) and scale > 1e-12

    scores /= scale

    observed = float(scores[0, 0])

    null = scores[
        geo["coords_y"],
        geo["coords_x"]
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

# ------------------------------------------------------------
# 5. Compute all 156 geometries, then 154 paired results
# ------------------------------------------------------------

print("EXPERIMENT 09.23 - NO-EROSION PAIRED SPATIAL NULL")

records = []
original_ids = set()
new_ids = set()

for number, row in enumerate(rows.itertuples(index=False), 1):
    mask = load_mask(row)
    old_foreground, new_foreground = load_foregrounds(row)

    old_geo = geometry(mask, old_foreground)
    new_geo = geometry(mask, new_foreground)

    # Audit against the complete frozen geometry records.
    assert abs(
        old_geo["coverage"] - float(row.observed_coverage)
    ) <= 1e-5, f"Coverage mismatch: {row.image_id}"

    assert old_geo["possible_shifts"] == int(row.possible_shifts)
    assert old_geo["null_shifts"] == int(row.null_shifts)
    assert old_geo["eligible"] == bool(row.eligible)

    if old_geo["eligible"]:
        original_ids.add(row.image_id)

    if new_geo["eligible"]:
        new_ids.add(row.image_id)

        ret = spatial_metrics(
            ret_maps[ret_lookup[row.image_id]],
            mask,
            new_geo
        )

        meta = spatial_metrics(
            meta_maps[meta_lookup[row.image_id]],
            mask,
            new_geo
        )

        record = {
            "image_id": row.image_id,
            "true_grade": int(row.true_grade),
            "risk_group_id": row.risk_group_id,
            "eligible": True,
            "original_eligible": bool(old_geo["eligible"]),
            "observed_coverage": new_geo["coverage"],
            "null_shifts": new_geo["null_shifts"]
        }

        for prefix, values in [
            ("retfound", ret),
            ("meta_mae", meta)
        ]:
            for key, value in values.items():
                record[f"{prefix}_{key}"] = value

        record["delta_null_adjusted"] = (
            meta["null_adjusted_enrichment"]
            - ret["null_adjusted_enrichment"]
        )

        record["delta_spatial_percentile"] = (
            meta["spatial_percentile"]
            - ret["spatial_percentile"]
        )

        records.append(record)

    if number % 25 == 0 or number == len(rows):
        print(f"Processed: {number}/{len(rows)}")

# ------------------------------------------------------------
# 6. Verify expected population and result integrity
# ------------------------------------------------------------

result = pd.DataFrame(records).sort_values(
    "image_id"
).reset_index(drop=True)

assert len(original_ids) == 137
assert len(new_ids) == 154
assert len(new_ids - original_ids) == 17
assert len(original_ids - new_ids) == 0
assert len(result) == 154
assert result["image_id"].is_unique
assert result["risk_group_id"].notna().all()
assert result["null_shifts"].ge(MIN_NULL).all()
assert result["observed_coverage"].ge(MIN_COVERAGE).all()

numeric_cols = [
    column for column in result.columns
    if any(term in column for term in [
        "enrichment", "percentile", "fraction",
        "delta_null_adjusted"
    ])
]

assert np.isfinite(
    result[numeric_cols].to_numpy(dtype=float)
).all()

primary_delta = float(
    reference["delta_null_adjusted"].mean()
)

new_delta = float(
    result["delta_null_adjusted"].mean()
)

overlap = result.loc[
    result["original_eligible"]
]

added = result.loc[
    ~result["original_eligible"]
]

print("\nELIGIBILITY")
print("Original cohort:", len(original_ids))
print("No-erosion cohort:", len(result))
print("Newly included:", len(added))
print("Risk groups:", result["risk_group_id"].nunique())

print("\nPAIRED NULL-ADJUSTED RESULTS")
print("Original 95% mean delta:", round(primary_delta, 6))
print("No-erosion 154-image mean delta:", round(new_delta, 6))
print(
    "No-erosion 137-image overlap mean delta:",
    round(float(overlap["delta_null_adjusted"].mean()), 6)
)
print(
    "No-erosion 17-image addition mean delta:",
    round(float(added["delta_null_adjusted"].mean()), 6)
)

for model in ["retfound", "meta_mae"]:
    print(
        f"{model} no-erosion adjusted mean:",
        round(float(
            result[
                f"{model}_null_adjusted_enrichment"
            ].mean()
        ), 6)
    )

print("\nGRADE DISTRIBUTION")
print(
    result["true_grade"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ------------------------------------------------------------
# 7. Preserve a new independent research artifact
# ------------------------------------------------------------

report = {
    "experiment": "Experiment09_No_Erosion_Spatial_Null_V1",
    "analysis_status": "exploratory_sensitivity",
    "dataset_split": "val",
    "original_primary_analysis_modified": False,
    "foreground_variant": "no_erosion",
    "foreground_threshold": 20,
    "morphological_closing_kernel": "ellipse_9x9",
    "foreground_contour": "largest_external_contour",
    "erosion_applied": False,
    "min_coverage": MIN_COVERAGE,
    "min_null_shifts": MIN_NULL,
    "mask_interpolation": "cv2_INTER_AREA",
    "mask_threshold": "per_image_reference_threshold",
    "attribution_formula": (
        "observed_enrichment_minus_mean_permitted_shift_enrichment"
    ),
    "paired_difference": "Meta_MAE_minus_RETFound",
    "total_primary_ma_candidates": 156,
    "original_eligible": 137,
    "no_erosion_eligible": 154,
    "newly_included": 17,
    "risk_groups": int(result["risk_group_id"].nunique()),
    "original_mean_delta": primary_delta,
    "no_erosion_mean_delta": new_delta,
    "training_index_sha256": EXPECTED_INDEX_SHA,
    "ret_cam_sha256": sha256_file(RET_CAM_PATH),
    "meta_cam_sha256": sha256_file(META_CAM_PATH),
    "frozen_paired_csv_sha256": sha256_file(REFERENCE_PATH),
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "test_split_evaluated": False
}

OUT.mkdir(parents=True, exist_ok=True)

# Exclusive creation prevents overwriting existing artifacts.
with RESULT_PATH.open("x", encoding="utf-8") as handle:
    result.to_csv(handle, index=False)

report["result_csv_sha256"] = sha256_file(RESULT_PATH)

with REPORT_PATH.open("x", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("\nOUTPUT VERIFICATION")
print("Results saved:", RESULT_PATH.is_file())
print("Report saved:", REPORT_PATH.is_file())
print("CSV SHA-256:", report["result_csv_sha256"])

print("\nRESEARCH INTEGRITY")
print("Original Experiment 05 modified: False")
print("Original Experiment 07 modified: False")
print("Model inference performed: False")
print("Test images accessed: 0")

print("\nSTEP 09.23 PASS")


# %% [cell 326] Experiment 09 - Step 09.24
# ============================================================
# Experiment 09 - Step 09.24
# Joint Cluster Bootstrap for No-Erosion Sensitivity
# Validation only; read-only
# ============================================================

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

ORIGINAL_PATH = (
    ROOT / "experiment07_results"
    / "ViTL_MA_Spatial_Null_Paired_V1.csv"
)

NO_EROSION_PATH = (
    ROOT / "experiment09_results"
    / "No_Erosion_Spatial_Null_Paired_V1.csv"
)

EXPECTED_NO_EROSION_SHA256 = (
    "ade86c5827d77319bef18f11efff46718738112bc198cf6bf5e7257168142552"
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

assert ORIGINAL_PATH.is_file()
assert NO_EROSION_PATH.is_file()
assert sha256_file(NO_EROSION_PATH) == EXPECTED_NO_EROSION_SHA256

original = pd.read_csv(
    ORIGINAL_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

no_erosion = pd.read_csv(
    NO_EROSION_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(original) == 137
assert len(no_erosion) == 154

assert original["image_id"].is_unique
assert no_erosion["image_id"].is_unique
assert original["risk_group_id"].notna().all()
assert no_erosion["risk_group_id"].notna().all()

assert original["risk_group_id"].nunique() == 120
assert no_erosion["risk_group_id"].nunique() == 133

original_ids = set(original["image_id"])
new_ids = set(no_erosion["image_id"])

assert original_ids.issubset(new_ids)
assert len(new_ids - original_ids) == 17

assert no_erosion["original_eligible"].sum() == 137

# ------------------------------------------------------------
# 1. Match both conditions by image ID
# ------------------------------------------------------------

primary = original[
    ["image_id", "risk_group_id", "true_grade",
     "delta_null_adjusted"]
].rename(columns={
    "risk_group_id": "original_risk_group_id",
    "true_grade": "original_grade",
    "delta_null_adjusted": "original_delta"
})

joint = no_erosion.merge(
    primary,
    on="image_id",
    how="left",
    validate="one_to_one"
)

assert len(joint) == 154

overlap = joint["original_eligible"].astype(bool)

assert overlap.sum() == 137
assert joint.loc[overlap, "original_delta"].notna().all()
assert joint.loc[~overlap, "original_delta"].isna().all()

assert (
    joint.loc[overlap, "risk_group_id"]
    == joint.loc[overlap, "original_risk_group_id"]
).all()

assert (
    joint.loc[overlap, "true_grade"]
    == joint.loc[overlap, "original_grade"]
).all()

assert np.isfinite(
    joint["delta_null_adjusted"].to_numpy(dtype=float)
).all()

assert np.isfinite(
    joint.loc[overlap, "original_delta"].to_numpy(dtype=float)
).all()

# ------------------------------------------------------------
# 2. Construct image-level contribution columns
# ------------------------------------------------------------

joint["primary_count"] = overlap.astype(int)

joint["primary_sum"] = np.where(
    overlap,
    joint["original_delta"].fillna(0.0),
    0.0
)

joint["overlap_new_sum"] = np.where(
    overlap,
    joint["delta_null_adjusted"],
    0.0
)

joint["full_new_sum"] = joint["delta_null_adjusted"]
joint["full_new_count"] = 1

contribution_columns = [
    "primary_sum",
    "primary_count",
    "overlap_new_sum",
    "full_new_sum",
    "full_new_count"
]

grouped = joint.groupby(
    "risk_group_id",
    sort=True
)[contribution_columns].sum()

assert len(grouped) == 133

group_values = grouped.to_numpy(dtype=float)

# ------------------------------------------------------------
# 3. Observed estimates
# ------------------------------------------------------------

def compute_effects(totals):

    primary_sum = totals[..., 0]
    primary_count = totals[..., 1]

    overlap_new_sum = totals[..., 2]

    full_new_sum = totals[..., 3]
    full_new_count = totals[..., 4]

    primary_mean = primary_sum / primary_count
    overlap_new_mean = overlap_new_sum / primary_count
    full_new_mean = full_new_sum / full_new_count

    foreground_effect = overlap_new_mean - primary_mean
    cohort_effect = full_new_mean - overlap_new_mean
    total_change = full_new_mean - primary_mean

    return np.stack([
        primary_mean,
        overlap_new_mean,
        full_new_mean,
        foreground_effect,
        cohort_effect,
        total_change
    ], axis=-1)

labels = [
    "Original 137 mean delta",
    "No-erosion overlap 137 mean delta",
    "No-erosion full 154 mean delta",
    "Foreground effect (same 137)",
    "Cohort expansion effect",
    "Total change"
]

observed = compute_effects(
    group_values.sum(axis=0)
)

assert np.isclose(observed[0], -0.056152, atol=1e-5)
assert np.isclose(observed[1], -0.016659, atol=1e-5)
assert np.isclose(observed[2], -0.006105, atol=1e-5)

assert np.isclose(
    observed[3] + observed[4],
    observed[5],
    atol=1e-12
)

# ------------------------------------------------------------
# 4. Joint source-risk-group cluster bootstrap
# ------------------------------------------------------------

N_BOOTSTRAP = 10000
N_GROUPS = len(group_values)

rng = np.random.default_rng(42)

sampled_indices = rng.integers(
    0,
    N_GROUPS,
    size=(N_BOOTSTRAP, N_GROUPS)
)

sampled_totals = group_values[
    sampled_indices
].sum(axis=1)

valid = (
    (sampled_totals[:, 1] > 0)
    & (sampled_totals[:, 4] > 0)
)

assert valid.sum() >= 0.99 * N_BOOTSTRAP

draws = compute_effects(
    sampled_totals[valid]
)

ci = np.quantile(
    draws,
    [0.025, 0.975],
    axis=0
)

# ------------------------------------------------------------
# 5. Report
# ------------------------------------------------------------

print("EXPERIMENT 09.24 - JOINT CLUSTER BOOTSTRAP")

print("\nCOHORT INTEGRITY")
print("Original images:", len(original))
print("No-erosion images:", len(no_erosion))
print("Shared images:", int(overlap.sum()))
print("Newly included images:", int((~overlap).sum()))
print("Joint risk groups:", N_GROUPS)
print("Source file SHA-256: PASS")

print("\nJOINT BOOTSTRAP")
print("Requested draws:", N_BOOTSTRAP)
print("Valid draws:", int(valid.sum()))
print("Resampling unit: source-risk group")

print("\nESTIMATES AND 95% CONFIDENCE INTERVALS")

for i, label in enumerate(labels):
    lower = float(ci[0, i])
    upper = float(ci[1, i])

    print(f"\n{label}")
    print("Observed:", round(float(observed[i]), 6))
    print(
        "95% CI:",
        (round(lower, 6), round(upper, 6))
    )
    print(
        "CI excludes zero:",
        bool(lower > 0 or upper < 0)
    )

print("\nDECOMPOSITION CHECK")
print(
    "Foreground + cohort effect:",
    round(float(observed[3] + observed[4]), 6)
)
print(
    "Total change:",
    round(float(observed[5]), 6)
)

print("\nINTERPRETATION")
print("Primary analysis: original 137-image frozen cohort")
print("Sensitivity analysis: no-erosion 154-image cohort")
print("Inference: paired source-risk-group bootstrap")
print("Analysis status: exploratory")
print("Patient-level independence established: False")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images accessed: 0")
print("Model inference performed: False")
print("Test images accessed: 0")

print("\nSTEP 09.24 COMPLETE")


# %% [cell 327] Experiment 09 - Step 09.25
# ============================================================
# Experiment 09 - Step 09.25
# Decompose Foreground Sensitivity into Null Baseline Changes
# Validation only; read-only
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

ORIGINAL_PATH = (
    ROOT / "experiment07_results"
    / "ViTL_MA_Spatial_Null_Paired_V1.csv"
)

NEW_PATH = (
    ROOT / "experiment09_results"
    / "No_Erosion_Spatial_Null_Paired_V1.csv"
)

original = pd.read_csv(
    ORIGINAL_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

new = pd.read_csv(
    NEW_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

assert len(original) == 137
assert len(new) == 154
assert original["image_id"].is_unique
assert new["image_id"].is_unique

shared = original.merge(
    new,
    on="image_id",
    suffixes=("_original", "_no_erosion"),
    validate="one_to_one"
)

assert len(shared) == 137

assert (
    shared["risk_group_id_original"]
    == shared["risk_group_id_no_erosion"]
).all()

# ------------------------------------------------------------
# 1. Validate the observed enrichment invariance
# ------------------------------------------------------------

print("EXPERIMENT 09.25 - NULL BASELINE DECOMPOSITION")

print("\nOBSERVED ENRICHMENT INVARIANCE")

for model in ["retfound", "meta_mae"]:
    old = shared[
        f"{model}_observed_enrichment_original"
    ].to_numpy(dtype=float)

    new_values = shared[
        f"{model}_observed_enrichment_no_erosion"
    ].to_numpy(dtype=float)

    maximum_error = float(
        np.max(np.abs(new_values - old))
    )

    print(
        f"{model}: maximum absolute difference = "
        f"{maximum_error:.12f}"
    )

    assert maximum_error <= 1e-5

# ------------------------------------------------------------
# 2. Decompose each model's adjusted enrichment
# ------------------------------------------------------------

for model in ["retfound", "meta_mae"]:

    old_null = shared[
        f"{model}_null_mean_enrichment_original"
    ].to_numpy(dtype=float)

    new_null = shared[
        f"{model}_null_mean_enrichment_no_erosion"
    ].to_numpy(dtype=float)

    old_adj = shared[
        f"{model}_null_adjusted_enrichment_original"
    ].to_numpy(dtype=float)

    new_adj = shared[
        f"{model}_null_adjusted_enrichment_no_erosion"
    ].to_numpy(dtype=float)

    null_change = new_null - old_null
    adjusted_change = new_adj - old_adj

    assert np.allclose(
        adjusted_change,
        -null_change,
        atol=1e-5,
        rtol=1e-5
    )

    shared[f"{model}_null_change"] = null_change
    shared[f"{model}_adjusted_change"] = adjusted_change

    print(f"\n{model.upper()}")
    print("Original null mean:", round(old_null.mean(), 6))
    print("No-erosion null mean:", round(new_null.mean(), 6))
    print("Mean null change:", round(null_change.mean(), 6))
    print(
        "Mean adjusted change:",
        round(adjusted_change.mean(), 6)
    )

# ------------------------------------------------------------
# 3. Verify paired-difference decomposition
# ------------------------------------------------------------

shared["delta_change"] = (
    shared["delta_null_adjusted_no_erosion"]
    - shared["delta_null_adjusted_original"]
)

shared["null_difference_change"] = (
    shared["meta_mae_null_change"]
    - shared["retfound_null_change"]
)

assert np.allclose(
    shared["delta_change"],
    -shared["null_difference_change"],
    atol=1e-5,
    rtol=1e-5
)

observed_change = float(
    shared["delta_change"].mean()
)

assert np.isclose(
    observed_change,
    0.039493,
    atol=1e-5
)

print("\nPAIRED DECOMPOSITION")
print("Mean adjusted delta change:", round(observed_change, 6))
print(
    "Negative mean differential null change:",
    round(
        -float(shared["null_difference_change"].mean()),
        6
    )
)
print("Algebraic consistency: PASS")

# ------------------------------------------------------------
# 4. Cluster bootstrap of model-specific changes
# ------------------------------------------------------------

columns = [
    "retfound_null_change",
    "meta_mae_null_change",
    "retfound_adjusted_change",
    "meta_mae_adjusted_change",
    "delta_change"
]

groups = [
    frame[columns].to_numpy(dtype=float)
    for _, frame in shared.groupby(
        "risk_group_id_original",
        sort=True
    )
]

assert len(groups) == 120

rng = np.random.default_rng(42)
N_BOOTSTRAP = 5000

draws = np.empty(
    (N_BOOTSTRAP, len(columns)),
    dtype=float
)

for b in range(N_BOOTSTRAP):
    indices = rng.integers(
        0, len(groups), size=len(groups)
    )

    sample = np.concatenate([
        groups[i] for i in indices
    ])

    draws[b] = sample.mean(axis=0)

print("\nCLUSTER BOOTSTRAP 95% CI")

for j, column in enumerate(columns):
    lower, upper = np.quantile(
        draws[:, j], [0.025, 0.975]
    )

    print(f"\n{column}")
    print(
        "Observed:",
        round(float(shared[column].mean()), 6)
    )
    print(
        "95% CI:",
        (
            round(float(lower), 6),
            round(float(upper), 6)
        )
    )

print("\nMETHOD")
print("Images:", len(shared))
print("Source-risk groups:", len(groups))
print("Resampling: Paired cluster bootstrap")
print("Analysis status: Exploratory")
print("Original CAM values: Unchanged")
print("Original primary protocol: Unchanged")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images accessed: 0")
print("Model inference performed: False")
print("Test images accessed: 0")

print("\nSTEP 09.25 COMPLETE")


# %% [cell 328] Experiment 09 - Step 09.26
# ============================================================
# Experiment 09 - Step 09.26
# Targeted Visual Audit of Retinal Foreground Boundaries
# Validation images only; display only
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

ROOT = DR_RESEARCH

INDEX_PATH = ROOT / "experiment03_results/FGADR_Training_Index_V1.csv"
COHORT_PATH = ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv"
SPATIAL_PATH = ROOT / "experiment05_results/MA_Spatial_Null_Retinal_Constrained_V1.csv"

TARGET_IDS = ["1214_1", "0617_2", "0432_3"]
SIZE = 224

index = pd.read_csv(INDEX_PATH, dtype={"image_id": str})
cohort = pd.read_csv(COHORT_PATH, dtype={"image_id": str})
spatial = pd.read_csv(SPATIAL_PATH, dtype={"image_id": str})

validation = index.loc[
    index["split"].eq("val"),
    ["image_id", "local_path"]
]

rows = (
    spatial.loc[
        spatial["image_id"].isin(TARGET_IDS),
        ["image_id", "true_grade", "eligible", "observed_coverage"]
    ]
    .merge(
        validation,
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        cohort[
            ["image_id", "resolved_mask_path", "reference_threshold"]
        ],
        on="image_id",
        validate="one_to_one"
    )
    .set_index("image_id")
    .loc[TARGET_IDS]
)

assert len(rows) == 3
assert rows["eligible"].eq(False).all()

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (9, 9)
)

fig, axes = plt.subplots(
    len(TARGET_IDS), 4,
    figsize=(15, 12),
    constrained_layout=True
)

for i, image_id in enumerate(TARGET_IDS):

    row = rows.loc[image_id]
    image_path = Path(row["local_path"])
    mask_path = Path(row["resolved_mask_path"])

    assert image_path.is_file()
    assert not image_path.is_symlink()
    assert mask_path.is_file()

    with Image.open(image_path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (SIZE, SIZE),
                Image.Resampling.BICUBIC
            )
        )

    with Image.open(mask_path) as image:
        raw_mask = np.asarray(image)

    if raw_mask.ndim == 3:
        assert np.array_equal(raw_mask[..., 0], raw_mask[..., 1])
        assert np.array_equal(raw_mask[..., 0], raw_mask[..., 2])
        raw_mask = raw_mask[..., 0]

    assert raw_mask.shape == (1280, 1280)

    binary = (
        raw_mask > int(row["reference_threshold"])
    ).astype(np.float32)

    mask = cv2.resize(
        binary,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    ).astype(np.float64)

    assert mask.sum() > 0

    candidate = (rgb.max(axis=2) > 20).astype(np.uint8)

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    assert contours

    no_erosion = np.zeros((SIZE, SIZE), dtype=np.uint8)

    cv2.drawContours(
        no_erosion,
        [max(contours, key=cv2.contourArea)],
        -1, 1, thickness=-1
    )

    original = cv2.erode(no_erosion, kernel)

    before_coverage = float(
        (no_erosion * mask).sum() / mask.sum()
    )

    after_coverage = float(
        (original * mask).sum() / mask.sum()
    )

    assert abs(
        after_coverage - float(row["observed_coverage"])
    ) <= 1e-5

    # Panel 1: Original retinal image.
    axes[i, 0].imshow(rgb)
    axes[i, 0].set_title(f"{image_id} | DR Grade {int(row['true_grade'])}")

    # Panel 2: Weighted MA annotation overlay.
    axes[i, 1].imshow(rgb)
    axes[i, 1].imshow(
        np.ma.masked_where(mask <= 0, mask),
        cmap="autumn",
        alpha=0.85,
        vmin=0,
        vmax=1
    )
    axes[i, 1].set_title("Weighted MA Mask")

    # Panel 3: Foreground before erosion.
    axes[i, 2].imshow(rgb)
    axes[i, 2].contour(
        no_erosion,
        levels=[0.5],
        colors=["lime"],
        linewidths=1.5
    )
    axes[i, 2].imshow(
        np.ma.masked_where(mask <= 0, mask),
        cmap="autumn",
        alpha=0.75
    )
    axes[i, 2].set_title(
        f"No Erosion | Coverage {before_coverage:.2%}"
    )

    # Panel 4: Original eroded foreground.
    axes[i, 3].imshow(rgb)
    axes[i, 3].contour(
        original,
        levels=[0.5],
        colors=["cyan"],
        linewidths=1.5
    )
    axes[i, 3].imshow(
        np.ma.masked_where(mask <= 0, mask),
        cmap="autumn",
        alpha=0.75
    )
    axes[i, 3].set_title(
        f"9x9 Erosion | Coverage {after_coverage:.2%}"
    )

    for ax in axes[i]:
        ax.axis("off")

plt.show()
plt.close(fig)

print("EXPERIMENT 09.26 - TARGETED VISUAL AUDIT")
print("Images displayed:", len(TARGET_IDS))
print("Original coverage reproduced: PASS")
print("Files modified: False")
print("Test images accessed: 0")
print("STEP 09.26 COMPLETE")


# %% [cell 329] Experiment 09 - Step 09.27
# ============================================================
# Experiment 09 - Step 09.27
# Magnified MA Boundary and Erosion-Loss Audit
# Validation only; display only
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

ROOT = DR_RESEARCH

index = pd.read_csv(
    ROOT / "experiment03_results/FGADR_Training_Index_V1.csv",
    dtype={"image_id": str}
)

cohort = pd.read_csv(
    ROOT / "experiment05_results/FGADR_Validation_MA_Cohort_V1.csv",
    dtype={"image_id": str}
)

spatial = pd.read_csv(
    ROOT / "experiment05_results/MA_Spatial_Null_Retinal_Constrained_V1.csv",
    dtype={"image_id": str}
)

TARGET_IDS = ["1214_1", "0617_2", "0432_3"]

validation = index.loc[
    index["split"] == "val",
    ["image_id", "local_path"]
]

rows = (
    validation
    .merge(
        cohort[[
            "image_id",
            "resolved_mask_path",
            "reference_threshold"
        ]],
        on="image_id",
        validate="one_to_one"
    )
    .merge(
        spatial[[
            "image_id",
            "observed_coverage",
            "eligible"
        ]],
        on="image_id",
        validate="one_to_one"
    )
    .set_index("image_id")
)

SIZE = 224
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (9, 9)
)

fig, axes = plt.subplots(
    3, 3,
    figsize=(15, 14),
    constrained_layout=True
)

print("EXPERIMENT 09.27 - MAGNIFIED MA BOUNDARY AUDIT")

for i, image_id in enumerate(TARGET_IDS):

    row = rows.loc[image_id]
    assert not bool(row["eligible"])

    image_path = Path(row["local_path"])
    mask_path = Path(row["resolved_mask_path"])

    assert image_path.is_file()
    assert mask_path.is_file()
    assert not image_path.is_symlink()

    with Image.open(image_path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (SIZE, SIZE),
                Image.Resampling.BICUBIC
            )
        )

    with Image.open(mask_path) as image:
        raw = np.asarray(image)

    if raw.ndim == 3:
        assert np.array_equal(raw[..., 0], raw[..., 1])
        assert np.array_equal(raw[..., 0], raw[..., 2])
        raw = raw[..., 0]

    assert raw.shape == (1280, 1280)

    binary = (
        raw > int(row["reference_threshold"])
    ).astype(np.float32)

    mask = cv2.resize(
        binary,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    ).astype(np.float64)

    assert mask.sum() > 0

    candidate = (
        rgb.max(axis=2) > 20
    ).astype(np.uint8)

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    assert contours

    before = np.zeros((SIZE, SIZE), dtype=np.uint8)

    cv2.drawContours(
        before,
        [max(contours, key=cv2.contourArea)],
        -1, 1, thickness=-1
    )

    after = cv2.erode(before, kernel)

    # Identify MA weight removed specifically by erosion.
    lost_weight = mask * (
        before.astype(np.float64)
        - after.astype(np.float64)
    )

    assert (lost_weight >= -1e-12).all()

    total_weight = float(mask.sum())
    lost_fraction = float(
        lost_weight.sum() / total_weight
    )

    reproduced = float(
        (after * mask).sum() / total_weight
    )

    assert abs(
        reproduced - float(row["observed_coverage"])
    ) <= 1e-5

    # Crop around the actual weighted MA support.
    ys, xs = np.nonzero(mask > 0)

    padding = 18
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(SIZE, int(xs.max()) + padding + 1)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(SIZE, int(ys.max()) + padding + 1)

    # Use the same crop for all panels.
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    crop_lost = lost_weight[y0:y1, x0:x1]

    crop_before = before[y0:y1, x0:x1]
    crop_after = after[y0:y1, x0:x1]

    axes[i, 0].imshow(rgb)

    # Mark the complete mask bounding box.
    rectangle = plt.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        edgecolor="yellow",
        linewidth=1.8
    )

    axes[i, 0].add_patch(rectangle)
    axes[i, 0].set_title(
        f"{image_id} | Full Image + MA Region"
    )

    axes[i, 1].imshow(
        crop_rgb,
        interpolation="nearest"
    )

    axes[i, 1].imshow(
        np.ma.masked_where(crop_mask <= 0, crop_mask),
        cmap="autumn",
        alpha=0.85,
        interpolation="nearest"
    )

    axes[i, 1].contour(
        crop_before,
        levels=[0.5],
        colors=["lime"],
        linewidths=1.2
    )

    axes[i, 1].set_title(
        "Weighted MA + No-Erosion Boundary"
    )

    axes[i, 2].imshow(
        crop_rgb,
        interpolation="nearest"
    )

    axes[i, 2].imshow(
        np.ma.masked_where(crop_lost <= 0, crop_lost),
        cmap="cool",
        alpha=0.9,
        interpolation="nearest"
    )

    axes[i, 2].contour(
        crop_after,
        levels=[0.5],
        colors=["cyan"],
        linewidths=1.2
    )

    axes[i, 2].set_title(
        f"Erosion-Lost MA Weight: {lost_fraction:.2%}"
    )

    for ax in axes[i]:
        ax.axis("off")

    print(f"\n{image_id}")
    print("Total weighted MA mass:", round(total_weight, 6))
    print("MA support pixels at 224:", int((mask > 0).sum()))
    print("Erosion-lost mass fraction:", round(lost_fraction, 6))
    print("Original coverage reproduced:", round(reproduced, 6))
    print("Crop bounds:", (x0, y0, x1, y1))

plt.show()
plt.close(fig)

print("\nRESEARCH INTEGRITY")
print("Original research files modified: False")
print("Validation images accessed:", len(TARGET_IDS))
print("Training started: False")
print("Test images accessed: 0")

print("\nSTEP 09.27 COMPLETE")


# %% [cell 330] Experiment 09 - Step 09.28
# ============================================================
# Experiment 09 - Step 09.28
# Grade-Related Exclusion Before and After Erosion Removal
# Read-only; validation cohort only
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

SPATIAL_PATH = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

NEW_PATH = (
    ROOT / "experiment09_results"
    / "No_Erosion_Spatial_Null_Paired_V1.csv"
)

INDEX_PATH = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

spatial = pd.read_csv(SPATIAL_PATH, dtype={"image_id": str})
new = pd.read_csv(NEW_PATH, dtype={"image_id": str})

index = pd.read_csv(
    INDEX_PATH,
    dtype={"image_id": str, "risk_group_id": str}
)

validation = index.loc[
    index["split"].eq("val"),
    ["image_id", "risk_group_id", "dr_grade"]
]

audit = spatial[
    ["image_id", "true_grade", "eligible"]
].merge(
    validation,
    on="image_id",
    validate="one_to_one"
)

assert len(audit) == 156
assert audit["risk_group_id"].notna().all()
assert (audit["true_grade"] == audit["dr_grade"]).all()

original_ids = set(
    audit.loc[audit["eligible"], "image_id"]
)

no_erosion_ids = set(new["image_id"])

assert len(original_ids) == 137
assert len(no_erosion_ids) == 154
assert original_ids.issubset(no_erosion_ids)

audit["high_grade"] = audit["true_grade"].ge(3)
audit["original_excluded"] = ~audit["eligible"]
audit["no_erosion_excluded"] = ~audit["image_id"].isin(
    no_erosion_ids
)

assert int(audit["original_excluded"].sum()) == 19
assert int(audit["no_erosion_excluded"].sum()) == 2

def risk_difference(frame, column):
    high = frame.loc[frame["high_grade"], column]
    low = frame.loc[~frame["high_grade"], column]

    return float(high.mean() - low.mean())

original_rd = risk_difference(
    audit, "original_excluded"
)

no_erosion_rd = risk_difference(
    audit, "no_erosion_excluded"
)

change = no_erosion_rd - original_rd

print("EXPERIMENT 09.28 - SELECTION BIAS SENSITIVITY")

print("\nEXCLUSION COUNTS")

for condition in [
    "original_excluded",
    "no_erosion_excluded"
]:
    print(f"\n{condition}")

    print(
        pd.crosstab(
            audit["high_grade"],
            audit[condition]
        ).to_string()
    )

print("\nRISK DIFFERENCES")
print("Original:", round(original_rd, 6))
print("No erosion:", round(no_erosion_rd, 6))
print("Change:", round(change, 6))

# Joint bootstrap preserves both conditions within groups.
group_arrays = [
    group[
        [
            "high_grade",
            "original_excluded",
            "no_erosion_excluded"
        ]
    ].to_numpy(dtype=float)
    for _, group in audit.groupby("risk_group_id", sort=True)
]

assert len(group_arrays) == 134

rng = np.random.default_rng(42)
N_BOOTSTRAP = 10000

draws = []

for _ in range(N_BOOTSTRAP):
    indices = rng.integers(
        0, len(group_arrays), size=len(group_arrays)
    )

    sample = np.concatenate([
        group_arrays[i] for i in indices
    ])

    high = sample[:, 0].astype(bool)
    low = ~high

    if not high.any() or not low.any():
        continue

    old_rd = (
        sample[high, 1].mean()
        - sample[low, 1].mean()
    )

    new_rd = (
        sample[high, 2].mean()
        - sample[low, 2].mean()
    )

    draws.append([old_rd, new_rd, new_rd - old_rd])

draws = np.asarray(draws)

assert len(draws) >= 9900

print("\nJOINT CLUSTER BOOTSTRAP")

for i, label in enumerate([
    "Original risk difference",
    "No-erosion risk difference",
    "Change in risk difference"
]):
    lower, upper = np.quantile(
        draws[:, i], [0.025, 0.975]
    )

    print(f"\n{label}")
    print(
        "95% CI:",
        (
            round(float(lower), 6),
            round(float(upper), 6)
        )
    )

print("\nREMAINING EXCLUDED IMAGES")
print(
    audit.loc[
        audit["no_erosion_excluded"],
        ["image_id", "true_grade", "risk_group_id"]
    ].to_string(index=False)
)

print("\nMETHOD")
print("Resampling unit: source-risk group")
print("Analysis status: exploratory")
print("Original primary cohort modified: False")

print("\nRESEARCH INTEGRITY")
print("Files modified: False")
print("Images accessed: 0")
print("Model inference performed: False")
print("Test images accessed: 0")

print("\nSTEP 09.28 COMPLETE")


# %% [cell 331] Experiment 09 - Step 09.29
# ============================================================
# Experiment 09 - Step 09.29
# Freeze Experiment 09 Research Summary
# Validation only; preserve all original artifacts
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = DR_RESEARCH

ORIGINAL = (
    ROOT / "experiment07_results"
    / "ViTL_MA_Spatial_Null_Paired_V1.csv"
)

NEW = (
    ROOT / "experiment09_results"
    / "No_Erosion_Spatial_Null_Paired_V1.csv"
)

SPATIAL = (
    ROOT / "experiment05_results"
    / "MA_Spatial_Null_Retinal_Constrained_V1.csv"
)

INDEX = (
    ROOT / "experiment03_results"
    / "FGADR_Training_Index_V1.csv"
)

OUTPUT = (
    ROOT / "experiment09_results"
    / "Experiment09_Final_Audit_Summary_V1.json"
)

EXPECTED_NEW_SHA256 = (
    "ade86c5827d77319bef18f11efff46718738112bc198cf6bf5e7257168142552"
)

EXPECTED_INDEX_SHA256 = (
    "9352577b7ed8279912392d9349a4bcd67f98ba8a67356e8d6fada0839e2082d3"
)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()

assert not OUTPUT.exists(), "Summary already exists."
assert sha256_file(NEW) == EXPECTED_NEW_SHA256
assert sha256_file(INDEX) == EXPECTED_INDEX_SHA256

original = pd.read_csv(
    ORIGINAL, dtype={"image_id": str, "risk_group_id": str}
)
new = pd.read_csv(
    NEW, dtype={"image_id": str, "risk_group_id": str}
)
spatial = pd.read_csv(
    SPATIAL, dtype={"image_id": str}
)
index = pd.read_csv(
    INDEX, dtype={"image_id": str, "risk_group_id": str}
)

assert len(original) == 137
assert len(new) == 154
assert len(spatial) == 156

for frame in [original, new, spatial]:
    assert frame["image_id"].is_unique

original_ids = set(original["image_id"])
new_ids = set(new["image_id"])

assert original_ids.issubset(new_ids)
assert len(new_ids - original_ids) == 17

primary_mean = float(
    original["delta_null_adjusted"].mean()
)
new_mean = float(
    new["delta_null_adjusted"].mean()
)

overlap_mean = float(
    new.loc[
        new["image_id"].isin(original_ids),
        "delta_null_adjusted"
    ].mean()
)

assert np.isclose(primary_mean, -0.056152, atol=1e-5)
assert np.isclose(overlap_mean, -0.016659, atol=1e-5)
assert np.isclose(new_mean, -0.006105, atol=1e-5)

validation = index.loc[
    index["split"].eq("val"),
    ["image_id", "risk_group_id", "dr_grade"]
]

audit = spatial[
    ["image_id", "eligible"]
].merge(
    validation,
    on="image_id",
    validate="one_to_one"
)

assert len(audit) == 156

audit["high_grade"] = audit["dr_grade"] >= 3
audit["old_excluded"] = ~audit["eligible"].astype(bool)
audit["new_excluded"] = ~audit["image_id"].isin(new_ids)

def grade_risk_difference(column):
    return float(
        audit.loc[audit["high_grade"], column].mean()
        - audit.loc[~audit["high_grade"], column].mean()
    )

old_rd = grade_risk_difference("old_excluded")
new_rd = grade_risk_difference("new_excluded")

assert np.isclose(old_rd, 1 / 6, atol=1e-10)
assert np.isclose(new_rd, 0.0, atol=1e-10)

summary = {
    "experiment": "Experiment09_Foreground_Selection_Audit_V1",
    "status": "completed_exploratory_validation_analysis",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_split": "val",
    "primary_protocol_preserved": True,
    "source_risk_groups_are_verified_patient_ids": False,
    "independent_clinical_mask_revalidation": False,

    "cohort": {
        "ma_candidates": 156,
        "original_eligible": 137,
        "no_erosion_eligible": 154,
        "newly_included": 17,
        "original_risk_groups": int(
            original["risk_group_id"].nunique()
        ),
        "no_erosion_risk_groups": int(
            new["risk_group_id"].nunique()
        )
    },

    "attribution": {
        "delta_definition": "Meta_MAE_minus_RETFound",
        "original_mean": primary_mean,
        "no_erosion_overlap_mean": overlap_mean,
        "no_erosion_full_mean": new_mean,
        "foreground_change": overlap_mean - primary_mean,
        "cohort_expansion_change": new_mean - overlap_mean,
        "total_change": new_mean - primary_mean,

        "step09_24_reported_cluster_ci95": {
            "original": [-0.175640, 0.061284],
            "no_erosion_overlap": [-0.135562, 0.100575],
            "no_erosion_full": [-0.117118, 0.104199],
            "foreground_change": [0.031248, 0.048100],
            "cohort_expansion_change": [-0.020734, 0.045456],
            "total_change": [0.017913, 0.084543]
        }
    },

    "selection": {
        "original_excluded": int(audit["old_excluded"].sum()),
        "no_erosion_excluded": int(audit["new_excluded"].sum()),
        "original_grade_risk_difference": old_rd,
        "no_erosion_grade_risk_difference": new_rd,
        "change_in_grade_risk_difference": new_rd - old_rd,

        "step09_28_reported_cluster_ci95": {
            "original_risk_difference": [0.062087, 0.272865],
            "no_erosion_risk_difference": [-0.034884, 0.036594],
            "change": [-0.265326, -0.075095]
        }
    },

    "methods": {
        "primary_foreground": "ellipse_9x9_erosion",
        "sensitivity_foreground": "no_erosion",
        "coverage_requirement": 0.95,
        "minimum_null_shifts": 100,
        "bootstrap_unit": "source_risk_group",
        "bootstrap_ci_provenance": (
            "Transcribed from completed notebook outputs "
            "Step09.24 and Step09.28; not recomputed here"
        )
    },

    "source_sha256": {
        "original_paired": sha256_file(ORIGINAL),
        "no_erosion_paired": sha256_file(NEW),
        "spatial_eligibility": sha256_file(SPATIAL),
        "frozen_index": sha256_file(INDEX)
    },

    "limitations": [
        "Exploratory sensitivity analysis",
        "No verified patient-level independence",
        "No independent clinical revalidation of MA masks",
        "No equivalence inference from confidence intervals spanning zero",
        "No-erosion foreground is not established as clinically superior"
    ],

    "test_split_accessed": False
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("x", encoding="utf-8") as handle:
    json.dump(
        summary,
        handle,
        indent=2,
        ensure_ascii=False
    )

with OUTPUT.open("r", encoding="utf-8") as handle:
    verified = json.load(handle)

assert verified["cohort"]["no_erosion_eligible"] == 154
assert verified["test_split_accessed"] is False

print("EXPERIMENT 09.29 - FINAL AUDIT SUMMARY")

print("\nCOHORT")
print("Original eligible:", len(original))
print("No-erosion eligible:", len(new))
print("Newly included:", len(new_ids - original_ids))

print("\nATTRIBUTION")
print("Original mean delta:", round(primary_mean, 6))
print("No-erosion mean delta:", round(new_mean, 6))
print(
    "Foreground effect:",
    round(overlap_mean - primary_mean, 6)
)

print("\nGRADE SELECTION")
print("Original risk difference:", round(old_rd, 6))
print("No-erosion risk difference:", round(new_rd, 6))

print("\nARCHIVE")
print("Summary saved:", OUTPUT.is_file())
print("Summary SHA-256:", sha256_file(OUTPUT))

print("\nRESEARCH INTEGRITY")
print("Original analysis files modified: False")
print("Test images accessed: 0")

print("\nSTEP 09.29 PASS")
