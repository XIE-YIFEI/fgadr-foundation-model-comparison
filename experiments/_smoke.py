"""Tiny synthetic inputs so Exp 17 / Exp 18 can be checked on a CPU without
data or weights. Numbers produced in smoke mode are meaningless."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer

DEVICE = torch.device("cpu")
N_TRAIN, N_TEST, SIZE = 8, 6, 448


def _tiny_vit(patch, num_classes=0, seed=0):
    torch.manual_seed(seed)
    return VisionTransformer(img_size=224, patch_size=patch, embed_dim=32, depth=24, num_heads=2,
                             num_classes=num_classes, global_pool="avg").eval()


def _images(root):
    rng = np.random.default_rng(0); rows = []
    yy, xx = np.mgrid[:SIZE, :SIZE]; disc = (yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2 < (SIZE * 0.46) ** 2
    for k in range(N_TRAIN + N_TEST):
        iid = f"{k:04d}_1"
        img = (rng.random((SIZE, SIZE, 3)) * 120 + 60).astype(np.uint8) * disc[..., None]
        mask = np.zeros((SIZE, SIZE), np.uint8)
        for _ in range(rng.integers(2, 8)):
            cy, cx = rng.integers(140, SIZE - 140, 2); mask[cy - 2:cy + 3, cx - 2:cx + 3] = 255
            img[cy - 2:cy + 3, cx - 2:cx + 3] = (30, 10, 10)
        ip, mp = root / f"{iid}.png", root / f"{iid}_mask.png"
        Image.fromarray(img).save(ip)
        if k % 5 == 4:                       # a grey-level encoded mask, as in the audit
            mask = (mask // 255 * 200).astype(np.uint8)
        Image.fromarray(mask).save(mp)
        rows.append(dict(image_id=iid, split="train" if k < N_TRAIN else "test", risk_group_id=f"g{k % 4}",
                         dr_grade=int(rng.integers(0, 5)), source_path=str(ip), local_path=str(ip), mask_path=str(mp),
                         mask_review_category="requires_manual_review" if k % 5 == 4 else "strict_lesion_level_candidate",
                         review_reason="intermediate_grayscale_encoding" if k % 5 == 4 else "no_screening_flags",
                         largest_component_area_px=float(25 if k % 3 else 2000), total_mask_area_px=100.0))
    return pd.DataFrame(rows)


def make_smoke_inputs(kind):
    root = Path(tempfile.mkdtemp(prefix=f"{kind}_smoke_")); out = root / "out"; out.mkdir()
    data = _images(root)
    if kind == "exp17":
        encoders = {"retfound": ("mae", "tiny16", 14, None), "meta_mae": ("mae", "tiny16", 14, None),
                    "retfound_dinov2": ("dinov2", "tiny14", 16, None), "dinov2_generic": ("dinov2", "tiny14", 16, None)}
        build = lambda name: _tiny_vit(16 if encoders[name][1] == "tiny16" else 14, seed=sum(map(ord, name))).requires_grad_(False)
        return out, data, encoders, build, DEVICE
    if kind == "exp18":
        models = {f"{p}_{r}_s{s}": (p, r, s, "tiny16" if p == "mae" else "tiny14", f"{p}_{r}_{s}")
                  for p in ["mae", "dinov2"] for r in ["retinal", "generic"] for s in [42, 1042]}
        build = lambda arch, ckpt: _tiny_vit(16 if arch == "tiny16" else 14, num_classes=5, seed=sum(map(ord, ckpt))).requires_grad_(False)
        cohort = data[data.split == "test"].reset_index(drop=True)
        return out, cohort, models, build, DEVICE
    raise ValueError(kind)
