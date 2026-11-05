"""Frozen ViT-L encoders for Exp 17. Copied unchanged from Exp 13B (section 2)."""
import argparse

import torch
import timm
from timm.layers import resample_abs_pos_embed
from safetensors.torch import load_file as load_safetensors

from common.paths import PRETRAINED as CK

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V16, D14 = "vit_large_patch16_224", "vit_large_patch14_dinov2"
ENCODERS = {   # name: (pair, architecture, tokens per side of a 224 tile, weight file)
    "retfound":        ("mae",    V16, 14, CK / "RETFound_pretrained/RETFound_mae_natureCFP.pth"),
    "meta_mae":        ("mae",    V16, 14, CK / "Meta_MAE_pretrained/mae_pretrain_vit_large.pth"),
    "retfound_dinov2": ("dinov2", D14, 16, CK / "DINOv2_pair/RETFound_dinov2_meh.pth"),
    "dinov2_generic":  ("dinov2", D14, 16, CK / "DINOv2_pair/model.safetensors"),
}
for _v in ENCODERS.values():
    assert _v[3].is_file(), f"Missing weights: {_v[3]}"


def build_encoder(name):
    pair, arch, nt, path = ENCODERS[name]
    if arch == V16:
        model = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="avg")
        with torch.serialization.safe_globals([argparse.Namespace]):
            ck = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        enc = {k: v for k, v in ck["model"].items() if k.startswith(("patch_embed.", "blocks.")) or k in {"cls_token", "pos_embed"}}
        assert len(enc) == 292; assert not model.load_state_dict(enc, strict=False).unexpected_keys
    else:
        model = timm.create_model(arch, pretrained=False, img_size=224, num_classes=0, global_pool="avg")
        target = model.state_dict()
        if path.suffix == ".safetensors":
            sd = load_safetensors(str(path))
        else:
            sd = torch.load(path, map_location="cpu", weights_only=False)
            for key in ["teacher", "model", "state_dict", "student"]:
                if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict): sd = sd[key]; break
        clean = {}
        for k, v in sd.items():
            for pre in ["module.", "backbone.", "encoder.", "model."]:
                if k.startswith(pre): k = k[len(pre):]
            clean[k.replace("mlp.w12.", "mlp.fc1.").replace("mlp.w3.", "mlp.fc2.")] = v
        if clean["pos_embed"].shape != target["pos_embed"].shape:
            clean["pos_embed"] = resample_abs_pos_embed(clean["pos_embed"], new_size=(16, 16), num_prefix_tokens=1)
        keys = [k for k in target if not k.startswith(("head.", "fc_norm.", "norm."))]
        loaded = {k: clean[k] for k in keys if k in clean and clean[k].shape == target[k].shape}
        assert len(loaded) == len(keys), f"{name}: loaded {len(loaded)}/{len(keys)}"
        model.load_state_dict(loaded, strict=False)
    return model.eval().requires_grad_(False).to(DEVICE)
