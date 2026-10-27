"""Paths and data tables shared by all experiment scripts.

The experiments were originally run in Google Colab, where every file lived
under /content/drive/MyDrive/DR_Research. All of those locations now come
from config.yaml (or the file named by the DR_CONFIG environment variable),
so the scripts run on any machine without editing code.

Mapping from the original Colab layout:

    /content/drive/MyDrive/DR_Research              -> ROOT        (results_root)
    DR_Research/model_checkpoints                   -> PRETRAINED  (pretrained_dir)
    /content  (local Colab disk, caches)            -> CACHE       (cache_dir)
    DR_Research/datasets/DR/FGADR/images4classification -> fgadr_image_dir
    DR_Research/datasets/DR/FGADR/images4seg/Microaneurysms_Masks -> fgadr_ma_mask_dir
    experiment03_results/FGADR_Training_Index_V1.csv -> load_index()
    experiment02_results/FGADR_Master_Manifest.csv   -> load_master()
"""

import os
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("DR_CONFIG", REPO / "config.yaml"))
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(
        f"{CONFIG_PATH} not found. Copy config.example.yaml to config.yaml and fill in your paths.")
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _path(key, required=True):
    value = CONFIG.get(key)
    if value in (None, ""):
        if required:
            raise KeyError(f"'{key}' is missing from {CONFIG_PATH}")
        return None
    p = Path(os.path.expanduser(str(value)))
    return p if p.is_absolute() else (REPO / p).resolve()


ROOT = _path("results_root")              # experiment outputs (was DR_Research on Drive)
PRETRAINED = _path("pretrained_dir")      # pretrained encoder weights (was DR_Research/model_checkpoints)
CACHE = _path("cache_dir")                # fast local scratch (was the local Colab disk /content)
FGADR_IMAGES = _path("fgadr_image_dir")
FGADR_MA_MASKS = _path("fgadr_ma_mask_dir")
IMAGE_LAYOUT = CONFIG.get("fgadr_image_layout", "by_grade")   # "by_grade" or "flat"
IDRID_ROOT = _path("idrid_root", required=False)

for _d in (ROOT, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

SPLIT_CSV = REPO / "data_splits/FGADR_Split_V2_Frozen.csv"
SHA256_CSV = REPO / "data_splits/FGADR_Image_SHA256.csv"
MASK_AUDIT_CSV = REPO / "mask_audit/FGADR_MA_Mask_Audit.csv"
EXP10_MANIFEST_CSV = REPO / "mask_audit/Experiment10_Validation_Manifest_V1.csv"


def fgadr_image_path(image_id, grade):
    """Location of one FGADR Seg-set fundus image."""
    if IMAGE_LAYOUT == "by_grade":
        return FGADR_IMAGES / str(grade) / f"{image_id}.png"
    return FGADR_IMAGES / f"{image_id}.png"


def load_index():
    """Rebuilds experiment03_results/FGADR_Training_Index_V1.csv.

    Same rows, columns and order as the original; only local_path and
    source_path point to this machine instead of Colab / Google Drive.
    """
    df = pd.read_csv(SPLIT_CSV, dtype={"image_id": str, "risk_group_id": str})
    sha = pd.read_csv(SHA256_CSV, dtype=str)
    df = df.merge(sha, on="image_id", how="left", validate="one_to_one")
    df.insert(df.columns.get_loc("sha256"), "local_path",
              [str(CACHE / "fgadr_classification" / str(g) / f"{i}.png") for i, g in zip(df.image_id, df.dr_grade)])
    df.insert(df.columns.get_loc("sha256"), "source_path",
              [str(fgadr_image_path(i, g)) for i, g in zip(df.image_id, df.dr_grade)])
    return df


def load_master():
    """Rebuilds the columns of experiment02_results/FGADR_Master_Manifest.csv used by the scripts."""
    audit = pd.read_csv(MASK_AUDIT_CSV, dtype={"image_id": str})
    return pd.DataFrame({
        "image_id": audit.image_id,
        "image": audit.image_id + ".png",
        "image_path": [str(fgadr_image_path(i, g)) for i, g in zip(audit.image_id, audit.dr_grade)],
        "dr_grade": audit.dr_grade,
        "mask_path": [str(FGADR_MA_MASKS / f"{i}.png") for i in audit.image_id],
        "mask_review_category": audit.review_category,
    })


def load_exp10_manifest():
    """Rebuilds experiment10_results/Experiment10_Full_Validation_Manifest_V1.csv (156 validation MA images)."""
    m = pd.read_csv(EXP10_MANIFEST_CSV, dtype={"image_id": str, "risk_group_id": str})
    m.insert(m.columns.get_loc("image_sha256_from_frozen_index") + 1, "resolved_image_path",
             [str(fgadr_image_path(i, g)) for i, g in zip(m.image_id, m.true_grade)])
    return m
