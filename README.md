# Retina-specific versus generic self-supervised pretraining for diabetic retinopathy grading and microaneurysm representation

Code, frozen data splits, mask audits and pre-specified analysis plans for the paper

> **Retina-specific versus generic self-supervised pretraining for diabetic retinopathy grading and microaneurysm representation: a matched comparison**. Yifei Xie, Doreen Ying Ying Sim. Submitted to *PeerJ Computer Science* (AI Application).

Archived release: Zenodo, DOI [10.5281/zenodo.23986342](https://doi.org/10.5281/zenodo.23986342).

## Contents

1. [Description](#description)
2. [Dataset information](#dataset-information)
3. [Code information](#code-information)
4. [Requirements](#requirements)
5. [Usage instructions](#usage-instructions)
6. [Methodology](#methodology)
7. [Citations](#citations)
8. [License and contribution guidelines](#license-and-contribution-guidelines)

## Description

The study compares retinal-specific foundation models (RETFound-MAE, RETFound-DINOv2) with
generic counterparts trained with the same objective (ImageNet MAE, DINOv2) in two matched
ViT-Large pairs. Within each pair the architecture and self-supervised objective are the same
and the two members differ in the released pretraining stage. The encoders are compared on
five-grade diabetic retinopathy (DR) classification after fine-tuning, on frozen-feature
probes for image-level grading and for patch-level microaneurysm (MA) separability across
input scales, on Grad-CAM attribution scored against a retina-constrained spatial null and
checked by parameter randomisation, and on external validation with IDRiD.

The repository contains every script used to produce the results, the frozen
train/validation/test split, the per-image audit of the FGADR microaneurysm masks, and the
analysis plans that were written and checksummed before the test partition was first
evaluated. Result tables for every analysis are provided with the paper as Data S1–S9.

## Dataset information

No images are redistributed here. Both datasets are third-party and must be obtained from
their providers.

| Dataset | Use in the study | Access and licence |
| --- | --- | --- |
| **FGADR** (Zhou et al., 2021), Seg-set: 1,842 colour fundus photographs, 1,280 × 1,280 px, image-level DR grade 0–4 and pixel-level lesion masks | Development (training and validation) and held-out test evaluation | Distributed by its authors on request: https://csyizhou.github.io/FGADR/. May not be redistributed. |
| **IDRiD** (Porwal et al., 2018): 516 graded images (Disease Grading), 81 with microaneurysm masks (Segmentation) | External validation only (Exp 14, 14B, 14C) | CC BY 4.0: https://idrid.grand-challenge.org/Data/ ; IEEE DataPort, DOI [10.21227/H25W98](https://doi.org/10.21227/H25W98) |

Files in this repository that describe the datasets (no pixel data):

| File | Contents |
| --- | --- |
| `data_splits/FGADR_Split_V2_Frozen.csv` | Image ID, DR grade, source-risk group and split (train/val/test) for all 1,842 images |
| `data_splits/FGADR_Split_V2_Freeze_Report.json` | Freeze date and SHA256 of the split file |
| `data_splits/FGADR_Image_SHA256.csv` | SHA256 of every FGADR image used, so you can check that your copy is identical |
| `mask_audit/FGADR_MA_Mask_Audit.csv` | Per-image audit and review category of every FGADR MA mask |
| `mask_audit/FGADR_Validation_MA_Cohort_V1.csv` | Validation images used for the attribution analyses |
| `mask_audit/*.json`, other CSVs | Screening parameters, mask resampling protocol, YOLO mask audit, Exp 10 manifest |

Pretrained encoder weights are also third-party and are linked, not copied:

| Item | Where to get it |
| --- | --- |
| RETFound-MAE weights (`RETFound_mae_natureCFP.pth`) | https://github.com/rmaphoh/RETFound_MAE (Hugging Face: `YukunZhou/RETFound_mae_natureCFP`) |
| ImageNet MAE ViT-L/16 weights (`mae_pretrain_vit_large.pth`) | https://github.com/facebookresearch/mae (https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth) |
| DINOv2 ViT-L/14 (`timm/vit_large_patch14_dinov2.lvd142m`) | Downloaded automatically by `exp16_dinov2_pair.py` |
| RETFound-DINOv2 (`YukunZhou/RETFound_dinov2_meh`) | Downloaded automatically by `exp16_dinov2_pair.py` (gated: accept the licence on Hugging Face and set `HF_TOKEN`) |
| Fine-tuned checkpoints (about 1.2 GB each) | Available from the corresponding author on request |

### Data splits

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

### Mask audit

`mask_audit/FGADR_MA_Mask_Audit.csv` describes every FGADR microaneurysm mask (encoding,
connected components, area, coverage, screening flags) and assigns one review category:
`strict_lesion_level_candidate` (1,024), `empty_mask` (418), `requires_manual_review` (221) or
`uncertain_broad_region` (179). Only `strict_lesion_level_candidate` masks are used for
lesion-level analyses. The categories come from automatic screening rules
(`Phase3_Parameters.json`) and are not clinically adjudicated.

## Code information

| Path | Contents |
| --- | --- |
| `experiments/` | One script per experiment, named by experiment number (Exp 10–18) |
| `experiments/development/` | Exp 01–10 as run step by step in the original notebook, one file per experiment |
| `experiments/exp17_encoders.py`, `experiments/_smoke.py` | Encoder loading for Exp 17; synthetic data and a tiny ViT for CPU smoke tests of Exp 17/18 |
| `figures/` | Scripts that draw Figure 3 (from Exp 17 output) and Figure S1 (from Exp 18 output) |
| `common/paths.py` | Reads `config.yaml`; rebuilds the image index and mask manifest used by the scripts |
| `config.example.yaml` | Template for local paths (images, masks, IDRiD, weights, outputs) |
| `data_splits/`, `mask_audit/` | See [Dataset information](#dataset-information) |
| `analysis_plans/` | Pre-specified analysis plans and protocols (byte-identical originals) with their SHA256 values |
| `scripts/verify_checksums.py` | Confirms that plans, split and (optionally) your images match the ones used in the paper |
| `requirements.txt` | Python dependencies (minimum versions) |

All original runs used Google Colab. Every script is a cell from the original notebooks with
only the Colab and Google Drive specifics replaced (drive mounting, `/content/...` paths, Colab
secrets). The training, evaluation and statistics code is unchanged. `common/paths.py` rebuilds
`FGADR_Training_Index_V1.csv` and `FGADR_Master_Manifest.csv` exactly as they were (checked row
by row against the originals).

### Experiments and run order

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
| 17 | `exp17_probe_sensitivity.py` | Post-hoc sensitivity analyses of the test-set MA patch probe: lenient and stricter mask sets, all cells, hard negatives, per-image AUROC, cells read from centred overlapping crops (uses `exp17_encoders.py`) | pretrained weights |
| 18 | `exp18_attribution_occlusion.py` | Post-hoc attribution analyses on the test MA cohort: Grad-CAM for the predicted class, correct vs misclassified images, lesion occlusion with a lesion-free control occlusion | Exp 04/07/11/16 checkpoints |

Exp 17 and 18 were added after the test results were known. Both can be smoke-tested on a CPU without data or weights: `EXP17_SMOKE=1 python -m experiments.exp17_probe_sensitivity` and `EXP18_SMOKE=1 python -m experiments.exp18_attribution_occlusion` (synthetic images and a tiny ViT from `experiments/_smoke.py`; the numbers are meaningless).

#### Development scripts (Exp 01–10)

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

## Requirements

The experiments were run on Google Colab (Linux runtime) in October 2026 on a single NVIDIA L4
GPU (24 GB) with CUDA and bfloat16 mixed precision (autocast). A CUDA GPU with at least 24 GB
of memory is needed to repeat the ViT-Large fine-tuning and the native-resolution probes at the
original batch sizes; the smoke tests of Exp 17 and 18 run on a CPU.

- Python 3 with PyTorch ≥ 2.4, torchvision ≥ 0.19 and timm ≥ 1.0.9
- safetensors, huggingface_hub, NumPy, pandas, scikit-learn, scikit-image, SciPy, Pillow,
  tqdm, PyYAML, Matplotlib
- OpenCV (`opencv-contrib-python-headless`); version 4.14.0.94 was used for the HOG/ORB
  duplicate search in Exp 02
- Ultralytics (YOLO11), only for the detection reference in Exp 06

Exact versions of the other packages were not recorded at run time. `requirements.txt` lists
the minimum versions, and the archived release on Zenodo is the reference implementation.
Install the PyTorch build that matches your CUDA version first
(https://pytorch.org/get-started/locally/), then the rest:

```bash
pip install -r requirements.txt
```

## Usage instructions

1. **Get the data and weights.** Request FGADR from its authors, download IDRiD (only for
   Exp 14) and the pretrained weights listed under [Dataset information](#dataset-information).
   Pretrained weights go under `pretrained_dir`:

   ```
   pretrained/
   ├── RETFound_pretrained/RETFound_mae_natureCFP.pth     sha256 e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485
   ├── Meta_MAE_pretrained/mae_pretrain_vit_large.pth     sha256 d2ceff4892889d0c0f5418fa8bcb02e6a25a37fc3bdc0619c3cacd7d3d05c39f
   └── DINOv2_pair/                                       filled by exp16_dinov2_pair.py
   ```

2. **Configure paths.** Copy the template and edit the paths for your machine:

   ```bash
   cp config.example.yaml config.yaml
   ```

   Set `HF_TOKEN` in your environment before running Exp 16. Exp 14 reads the IDRiD zip
   files from `idrid_root`.

3. **Check the files.** Confirm that the analysis plans and split are unchanged and that your
   FGADR images are byte-identical to the ones used in the paper:

   ```bash
   python scripts/verify_checksums.py --images
   ```

4. **Run the experiments** from the repository root, in the order of the table above, for
   example:

   ```bash
   python -m experiments.exp11_multiseed
   ```

   Outputs are written to `results_root` with the same folder names as in the original runs
   (`experiment11_multiseed/`, `experiment13_test/`, ...), so later experiments find the
   outputs of earlier ones.

5. **Smoke test without data** (CPU, synthetic images, numbers meaningless):

   ```bash
   EXP17_SMOKE=1 python -m experiments.exp17_probe_sensitivity
   EXP18_SMOKE=1 python -m experiments.exp18_attribution_occlusion
   ```

6. **Draw the figures** from the Exp 17 and Exp 18 outputs:

   ```bash
   python figures/figure3_probe_sensitivity.py \
       <results_root>/experiment17_probe_sensitivity/exp17_paired_differences.csv \
       "Figure 3.png" figure3_values.csv

   # Figure S1: put the three example images and their MA masks in
   # <workdir>/img/<image_id>.png and <workdir>/mask/<image_id>.png
   python figures/figureS1_occlusion_prepare.py <workdir> 1165_1 0606_1 1740_3
   python figures/figureS1_occlusion_plot.py <workdir> \
       <results_root>/experiment18_attribution_occlusion/exp18_per_image.csv "Figure S1.png"
   ```

## Methodology

The full methods are in the paper (Materials & Methods). In short:

**Data quality control.** Each FGADR microaneurysm mask was binarised at 127, decomposed
into 8-connected components and assigned by rule to one review category; only the 1,024
lesion-level masks were used as localisation ground truth (`mask_audit/`). Exact duplicates
were found by SHA256 and near-duplicates by perceptual hashing and thumbnail similarity,
verified with ORB features and RANSAC (Exp 02); linked images form source-risk groups.

**Partition.** Source-risk groups were split 70/15/15 into training (1,290), validation (276)
and test (276), so that no group spans two partitions. The split was frozen and checksummed
(`data_splits/`).

**Preprocessing.** For fine-tuning and image-level probes, images were resized to 224 × 224
(bicubic) and normalised with ImageNet channel statistics; horizontal flipping and rotation up
to 10° were applied to training images only. For patch-level probes, images were zero-padded
to 1,344 × 1,344, resized to 224, 448 or 896 px or left unresized, and tiled into 224 × 224
crops. Masks were resampled to 224 px by area averaging for attribution. IDRiD images were
cropped to the fundus, padded to a square and resized to 1,280 × 1,280. No other enhancement or
colour normalisation was applied, except in the post hoc colour-transfer analysis (Exp 14C).
Probe features were standardised with training-data statistics only.

**Models.** Each encoder received a global-average-pooling head with layer normalisation and
a five-way linear classifier, and was fine-tuned with AdamW, class-weighted cross-entropy and
early stopping on validation macro-F1, with seeds 42, 1042 and 2042 (Exp 04, 07, 11, 16). The
RETFound fine-tuning recipe (Exp 12) and an EfficientNet-B0 (Exp 03) served as sensitivity
analysis and convolutional reference. Linear probes on frozen features measured image-level
grading and patch-level MA separability (Exp 15, 15B, 16B). Grad-CAM attribution was scored
against a retina-constrained spatial null (Exp 05, 09, 13C) and checked by cascading parameter
randomisation (Exp 10).

**Evaluation.** All development decisions (checkpoint selection, probe regularisation, best
layer) used the training and validation partitions only. The primary and secondary
comparisons were written into an analysis plan and checksummed before the held-out test
partition was evaluated once (Exp 13A–C); models and probes were then applied unchanged to
IDRiD under a separate plan (Exp 14). Analyses designed after the test results were known
(Exp 14C, 17, 18) are exploratory. Within each pair the two encoders were compared on the
same images as a paired difference (retinal minus generic).

**Metrics.** Classification: macro-F1 (primary), quadratic weighted kappa, grade-1 recall,
referable-DR AUC, accuracy, multiclass Brier score and expected calibration error (15 bins).
Probes: AUROC (primary) and average precision. Detection reference: lesion-level recall at
fixed false positives per image. Attribution: MA enrichment, null-adjusted enrichment and its
percentile in the spatial null; randomisation agreement by Pearson, Spearman and SSIM;
occlusion by the drop in reference-grade probability and in expected grade. 95% intervals come
from a cluster bootstrap over source-risk groups (2,000 resamples for classification and
attribution, 1,000 for probes).

### Pre-specified analysis plans

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

## Citations

If you use this code, please cite the paper above and the archived release:

> Xie Y, Sim DYY. 2026. fgadr-foundation-model-comparison. Zenodo. DOI [10.5281/zenodo.23986342](https://doi.org/10.5281/zenodo.23986342)

Datasets and models used:

- Zhou Y, Wang B, Huang L, Cui S, Shao L. 2021. A benchmark for studying diabetic retinopathy: segmentation, grading, and transferability. *IEEE Transactions on Medical Imaging* 40(3):818–828. (FGADR)
- Porwal P, Pachade S, Kamble R, Kokare M, Deshmukh G, Sahasrabuddhe V, et al. 2018. Indian Diabetic Retinopathy Image Dataset (IDRiD): a database for diabetic retinopathy screening research. *Data* 3(3):25. DOI 10.3390/data3030025. Dataset DOI 10.21227/H25W98.
- Zhou Y, Chia MA, Wagner SK, et al. 2023. A foundation model for generalizable disease detection from retinal images. *Nature* 622(7981):156–163. (RETFound)
- He K, Chen X, Xie S, Li Y, Dollár P, Girshick R. 2022. Masked autoencoders are scalable vision learners. *CVPR 2022*, 16000–16009. (MAE)
- Oquab M, Darcet T, Moutakanni T, et al. 2024. DINOv2: learning robust visual features without supervision. *Transactions on Machine Learning Research*. (DINOv2)
- Wightman R. 2019. PyTorch Image Models. DOI 10.5281/zenodo.4414861. (timm)

## License and contribution guidelines

Code: MIT (see `LICENSE`). Split and audit files describe FGADR and IDRiD; use of those
datasets remains subject to their own terms, and pretrained weights remain subject to the
licences of their releases.

This repository is the record of a completed study and is archived on Zenodo, so the
experiment scripts and analysis plans will not be changed. Bug reports and questions are
welcome as GitHub issues. Fixes that do not change results can be proposed as pull requests;
please describe the change and how you tested it.

AI assistance: Claude (Anthropic) was used to assist with code preparation, as stated in the
paper's Acknowledgements. The authors reviewed and verified all code.
