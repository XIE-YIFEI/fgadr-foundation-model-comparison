# Retinal vs. generic self-supervised pretraining for diabetic retinopathy grading (FGADR / IDRiD)

Code, frozen data splits, mask audits and pre-specified analysis plans for the paper

> **Retina-specific foundation pretraining trades fine-grained lesion features for global fundus semantics: a matched comparison for diabetic retinopathy and microaneurysms**. Yifei Xie, Doreen Ying Ying Sim. Under submission to *Frontiers in Medicine* (Ophthalmology section).

The study compares retinal-specific foundation models (RETFound-MAE, RETFound-DINOv2) with
generic counterparts trained with the same objective (Meta MAE, DINOv2) under matched
fine-tuning conditions, a frozen test split that was opened once, and analysis plans that
were written and hashed before any test image was read.

## What is in this repository

| Folder | Contents |
| --- | --- |
| `experiments/` | One script per experiment, named by experiment number (Exp 10–16) |
| `experiments/development/` | Exp 01–10 as run step by step in the original notebook, one file per experiment |
| `common/paths.py` | Reads `config.yaml`; rebuilds the image index and mask manifest used by the scripts |
| `data_splits/` | Frozen FGADR split (image ID, DR grade, source-risk group, split) and image SHA256 values |
| `mask_audit/` | Per-image audit of the FGADR microaneurysm masks and the validation MA cohort |
| `analysis_plans/` | Pre-specified analysis plans and protocols (byte-identical originals) with their SHA256 values |
| `scripts/verify_checksums.py` | Confirms that plans, split and (optionally) your images match the ones used in the paper |

## What is not included

| Item | Where to get it |
| --- | --- |
| FGADR images and masks | Request access from the dataset authors: https://csyizhou.github.io/FGADR/ |
| IDRiD images and masks | https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid |
| RETFound-MAE weights (`RETFound_mae_natureCFP.pth`) | https://github.com/rmaphoh/RETFound_MAE (Hugging Face: `YukunZhou/RETFound_mae_natureCFP`) |
| Meta MAE ViT-L/16 weights (`mae_pretrain_vit_large.pth`) | https://github.com/facebookresearch/mae (https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth) |
| DINOv2 ViT-L/14 (`timm/vit_large_patch14_dinov2.lvd142m`) | Downloaded automatically by `exp16_dinov2_pair.py` |
| RETFound-DINOv2 (`YukunZhou/RETFound_dinov2_meh`) | Downloaded automatically by `exp16_dinov2_pair.py` (gated: accept the licence on Hugging Face and set `HF_TOKEN`) |
| Fine-tuned checkpoints (about 1.2 GB each) | Available from the corresponding author on request |

The images and pretrained weights are distributed by their original authors under their own
terms, so they are linked rather than copied here.

## Setup

```bash
pip install -r requirements.txt          # install the matching PyTorch build first
cp config.example.yaml config.yaml       # then edit the paths
python scripts/verify_checksums.py --images
```

Pretrained weights go under `pretrained_dir`:

```
pretrained/
├── RETFound_pretrained/RETFound_mae_natureCFP.pth     sha256 e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485
├── Meta_MAE_pretrained/mae_pretrain_vit_large.pth     sha256 d2ceff4892889d0c0f5418fa8bcb02e6a25a37fc3bdc0619c3cacd7d3d05c39f
└── DINOv2_pair/                                       filled by exp16_dinov2_pair.py
```

Set `HF_TOKEN` in your environment before running Experiment 16. Experiment 14 reads the
IDRiD zip files from `idrid_root`.

All original runs used Google Colab with one NVIDIA L4 GPU (bfloat16 autocast). Every script
is a cell from the original notebooks with only the Colab and Google Drive specifics replaced
(drive mounting, `/content/...` paths, Colab secrets). The training, evaluation and statistics
code is unchanged. `common/paths.py` rebuilds `FGADR_Training_Index_V1.csv` and
`FGADR_Master_Manifest.csv` exactly as they were (checked row by row against the originals).

Run each script from the repository root, for example:

```bash
python -m experiments.exp11_multiseed
```

Outputs are written to `results_root` with the same folder names as in the original runs
(`experiment11_multiseed/`, `experiment13_test/`, ...), so later experiments find the outputs
of earlier ones.

## Experiments and run order

| Exp | Script | Purpose | Needs |
| --- | --- | --- | --- |
| 01–10 | `development/exp01_…` to `development/exp10_…` | Mask audit, source-risk split, EfficientNet-B0 baseline, RETFound and Meta MAE seed-42 fine-tuning, Grad-CAM attribution, YOLO detection, drop-path ablation, spatial-null sensitivity, randomisation pilot (see below) | original Drive layout |
| 10 | `exp10_gradcam_randomization.py` | Cascading parameter-randomisation sanity check for Grad-CAM (validation MA cohort) | Exp 04/05/07 outputs |
| 11 | `exp11_multiseed.py` | RETFound-MAE vs Meta MAE, seeds 1042 and 2042 (matched protocol) | pretrained weights |
| 12 | `exp12_official_recipe.py` | Sensitivity analysis: RETFound official fine-tuning recipe for both encoders | pretrained weights |
| 15 | `exp15_frozen_probe.py` | Frozen-feature probes: image-level grade probe and patch-level MA probe | pretrained weights |
| 15B | `exp15b_scale_sweep.py` | MA patch probe across input scales 224–1344 | pretrained weights |
| 16 | `exp16_dinov2_pair.py` | Second controlled pair: DINOv2 vs RETFound-DINOv2, 3 seeds | `HF_TOKEN` |
| 16B | `exp16b_dinov2_probes.py` | Frozen probes for the DINOv2 pair | Exp 16 weights |
| 13A | `exp13a_test_classification.py` | **Single test-set evaluation** of all fine-tuned models | Exp 03/04/07/11/12/16 checkpoints |
| 13B | `exp13b_test_frozen_probes.py` | Test-set replication of the frozen probes | pretrained weights |
| 13C | `exp13c_test_attribution.py` | Test-set replication of the Grad-CAM attribution analysis, both pairs, 3 seeds | Exp 04/07/11/16 checkpoints |
| 14 | `exp14_external_idrid.py` | External validation on IDRiD: fine-tuned models and frozen probes | `idrid_root`, all checkpoints |
| 14B | `exp14b_idrid_diagnostics.py` | Diagnostics for the IDRiD classification drop (uses saved predictions) | Exp 14 outputs |
| 14C | `exp14c_colour_normalisation.py` | Post-hoc colour-normalisation sensitivity on IDRiD | Exp 14 outputs |

### Development scripts (Exp 01–10)

Experiments 01–10 were run as several hundred small steps in one Colab notebook
(audits, smoke tests, the runs themselves, re-runs after runtime restarts, and fixes).
`experiments/development/` keeps those cells in their original order, one file per
experiment, with `# %% [cell N]` markers that open as cells in VS Code or Jupyter.
They are a record of how the results were produced and are meant to be read or run
cell by cell, not top to bottom. Paths start at `DR_RESEARCH` (`results_root`, laid
out like the original `DR_Research` folder on Google Drive) and `LOCAL_DISK`
(`cache_dir`). Exact duplicate cells are listed by number but not repeated. The FGADR
data are expected under `datasets/DR/FGADR/` (`images4classification/`, `images4seg/`,
`yolo/`), and the 8-class annotation release, used only in the Exp 06 inventory, under
`datasets/DR/FGADR_8class/`.

The final protocols that later experiments depend on are frozen in the files of this
repository: the split in `data_splits/`, the mask categories and validation MA cohort
in `mask_audit/`, and the Exp 08 and Exp 10 protocols in `analysis_plans/`. The full
Exp 10 run over the validation MA cohort is `experiments/exp10_gradcam_randomization.py`.

No model was trained, tuned or evaluated on the test split before Experiment 13A. Experiments
10–12, 15 and 16 select only training and validation rows and record `test_images_used: 0`.

## Pre-specified analysis plans

Each test-set script writes its plan to disk before any test image is loaded, and records the
plan's SHA256 in a run-info file when it finishes. The files in `analysis_plans/` are the
originals, unchanged, so they still contain the Colab paths of the time.

| Plan | Created (UTC) | SHA256 | Recorded in |
| --- | --- | --- | --- |
| `Exp13_Analysis_Plan.json` | 2026-10-27 03:22 | `834b8d7c19a07617…` | `Exp13A_run_info.json` |
| `Exp13B_Probe_Plan.json` | 2026-10-27 03:45 | `3668f9f15a28b452…` | `Exp13B_run_info.json` |
| `Exp13C_Attribution_Plan.json` | 2026-10-27 08:07 | `1d3b49e1f8ddc8dd…` | `Exp13C_run_info.json` |
| `Exp14_External_Plan.json` | 2026-10-27 14:03 | `30bf5bcd0e4d9e05…` | `Exp14_run_info.json` |
| `Exp14C_Plan.json` (post hoc) | 2026-10-27 15:28 | `9eb7063781c959fe…` | – |

Validation-stage protocols (Exp 08 regularisation ablation, Exp 10 randomisation protocol and
execution plan) are in the same folder. Full hashes are in `analysis_plans/SHA256SUMS.txt`, and
`python scripts/verify_checksums.py` checks all of them.

## Data splits

`data_splits/FGADR_Split_V2_Frozen.csv` (SHA256 `a3d83242847834ef…`, frozen 2026-10-19) assigns
the 1,842 FGADR Seg-set images to train (1,290), validation (276) and test (276). Images that
are exact or near duplicates, or that share a likely source, are grouped into source-risk groups,
and every group lies entirely in one split. Risk groups are image-similarity groupings, not
verified patient identities.

| Column | Meaning |
| --- | --- |
| `image_id` | FGADR file name without `.png` |
| `dr_grade` | DR grade 0–4 from the FGADR labels |
| `risk_group_id` | Source-risk group |
| `split` | `train`, `val` or `test` |
| `risk_group_size` | Images in the group |
| `group_grade_conflict` | Group contains more than one DR grade |
| `exact_duplicate_label_conflict` | Image is a pixel-identical duplicate with a different grade |

## Mask audit

`mask_audit/FGADR_MA_Mask_Audit.csv` describes every FGADR microaneurysm mask (encoding,
connected components, area, coverage, screening flags) and assigns one review category:
`strict_lesion_level_candidate` (1,024), `empty_mask` (418), `requires_manual_review` (221) or
`uncertain_broad_region` (179). Only `strict_lesion_level_candidate` masks are used for
lesion-level analyses. The categories come from automatic screening rules
(`Phase3_Parameters.json`) and are not clinically adjudicated.

## Licence

Code: MIT (see `LICENSE`). Split and audit files describe FGADR and IDRiD; use of those
datasets remains subject to their own terms.

## Citation

If you use this code, please cite the paper above and the archived release: [10.5281/zenodo.23699463](https://doi.org/10.5281/zenodo.23699463).
