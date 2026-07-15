import torch
import torch.nn as nn
import logging

def load_pretrained_weights(quant_model: nn.Module, float_model: nn.Module):
    """
    Maps weights from a floating-point model to a quantized Brevitas model.

    Handles automatic reshaping of depthwise conv weights that may be
    flattened to 1D in the source checkpoint.
    """
    logging.info("Mapping pretrained floating-point weights to quantized model...")
    float_state_dict = float_model.state_dict()
    quant_state_dict = quant_model.state_dict()

    filtered_dict = {}
    for k, v in float_state_dict.items():
        if k not in quant_state_dict:
            continue
        target_shape = quant_state_dict[k].shape
        # Auto-reshape flattened depthwise conv weights (1D -> 4D)
        if v.dim() == 1 and len(target_shape) == 4:
            if v.numel() == target_shape.numel():
                v = v.reshape(target_shape)
            else:
                logging.warning(f"Skipping {k}: size mismatch ({v.shape} vs {target_shape})")
                continue
        # Skip any remaining shape mismatches (e.g. classifier head with different num_classes)
        if v.shape != target_shape:
            logging.warning(f"Skipping {k}: shape mismatch ({v.shape} vs {target_shape})")
            continue
        filtered_dict[k] = v

    missing_keys, unexpected_keys = quant_model.load_state_dict(filtered_dict, strict=False)

    logging.info(f"Successfully mapped {len(filtered_dict)} tensors.")
    if missing_keys:
        logging.debug(f"Missing keys (expected for Brevitas): {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected keys: {unexpected_keys}")

    return quant_model


# ---------------------------------------------------------------------------
# timm-specific remapping for MobileNet architectures
# ---------------------------------------------------------------------------

# MobileNetV1: timm uses staged blocks.{stage}.{idx}; our model uses flat blocks.{i}
# Staging: [1, 2, 2, 6, 2] = 13 blocks total
_MV1_STAGING = [
    (0, 0),                                         # block 0
    (1, 0), (1, 1),                                 # blocks 1-2
    (2, 0), (2, 1),                                 # blocks 3-4
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),  # blocks 5-10
    (4, 0), (4, 1),                                 # blocks 11-12
]

# MobileNetV2: timm staging [1, 2, 3, 4, 3, 3, 1] = 17 blocks
# Block 0 has no expansion conv in timm (expand_ratio=1); our model has it (identity).
_MV2_STAGING = [
    (0, 0),                                         # features.3  (no expand conv in timm)
    (1, 0), (1, 1),                                 # features.4-5
    (2, 0), (2, 1), (2, 2),                         # features.6-8
    (3, 0), (3, 1), (3, 2), (3, 3),                 # features.9-12
    (4, 0), (4, 1), (4, 2),                         # features.13-15
    (5, 0), (5, 1), (5, 2),                         # features.16-18
    (6, 0),                                         # features.19
]

_BN_SUFFIXES = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")


def _remap_timm_mobilenetv1_sd(timm_sd: dict) -> dict:
    out = {}
    out["stem.0.weight"] = timm_sd["conv_stem.weight"]
    for sfx in _BN_SUFFIXES:
        if f"bn1.{sfx}" in timm_sd:
            out[f"stem.1.{sfx}"] = timm_sd[f"bn1.{sfx}"]
    for flat_i, (s, idx) in enumerate(_MV1_STAGING):
        tp = f"blocks.{s}.{idx}"
        op = f"blocks.{flat_i}"
        out[f"{op}.dw.weight"] = timm_sd[f"{tp}.conv_dw.weight"]
        for sfx in _BN_SUFFIXES:
            if f"{tp}.bn1.{sfx}" in timm_sd:
                out[f"{op}.bn_dw.{sfx}"] = timm_sd[f"{tp}.bn1.{sfx}"]
        out[f"{op}.pw.weight"] = timm_sd[f"{tp}.conv_pw.weight"]
        for sfx in _BN_SUFFIXES:
            if f"{tp}.bn2.{sfx}" in timm_sd:
                out[f"{op}.bn_pw.{sfx}"] = timm_sd[f"{tp}.bn2.{sfx}"]
    out["fc.weight"] = timm_sd["classifier.weight"]
    out["fc.bias"] = timm_sd["classifier.bias"]
    return out


def _remap_timm_mobilenetv2_sd(timm_sd: dict) -> dict:
    out = {}
    out["features.0.weight"] = timm_sd["conv_stem.weight"]
    for sfx in _BN_SUFFIXES:
        if f"bn1.{sfx}" in timm_sd:
            out[f"features.1.{sfx}"] = timm_sd[f"bn1.{sfx}"]

    for flat_i, (s, idx) in enumerate(_MV2_STAGING):
        tp = f"blocks.{s}.{idx}"
        feat_i = flat_i + 3  # our features start at index 3
        op = f"features.{feat_i}.conv"

        if flat_i == 0:
            # expand_ratio=1: neither timm nor our model has an expansion conv, so the
            # block is a 5-slot Sequential (no leading pw-expand):
            #   dw → conv.0, bn1 → conv.1, conv_pw → conv.3, bn2 → conv.4
            out[f"{op}.0.weight"] = timm_sd[f"{tp}.conv_dw.weight"]
            for sfx in _BN_SUFFIXES:
                if f"{tp}.bn1.{sfx}" in timm_sd:
                    out[f"{op}.1.{sfx}"] = timm_sd[f"{tp}.bn1.{sfx}"]
            out[f"{op}.3.weight"] = timm_sd[f"{tp}.conv_pw.weight"]
            for sfx in _BN_SUFFIXES:
                if f"{tp}.bn2.{sfx}" in timm_sd:
                    out[f"{op}.4.{sfx}"] = timm_sd[f"{tp}.bn2.{sfx}"]
        else:
            # Full inverted residual: conv_pw → conv.0, bn1 → conv.1,
            #                         conv_dw → conv.3, bn2 → conv.4,
            #                         conv_pwl → conv.6, bn3 → conv.7
            out[f"{op}.0.weight"] = timm_sd[f"{tp}.conv_pw.weight"]
            for sfx in _BN_SUFFIXES:
                if f"{tp}.bn1.{sfx}" in timm_sd:
                    out[f"{op}.1.{sfx}"] = timm_sd[f"{tp}.bn1.{sfx}"]
            out[f"{op}.3.weight"] = timm_sd[f"{tp}.conv_dw.weight"]
            for sfx in _BN_SUFFIXES:
                if f"{tp}.bn2.{sfx}" in timm_sd:
                    out[f"{op}.4.{sfx}"] = timm_sd[f"{tp}.bn2.{sfx}"]
            out[f"{op}.6.weight"] = timm_sd[f"{tp}.conv_pwl.weight"]
            for sfx in _BN_SUFFIXES:
                if f"{tp}.bn3.{sfx}" in timm_sd:
                    out[f"{op}.7.{sfx}"] = timm_sd[f"{tp}.bn3.{sfx}"]

    out["features.20.weight"] = timm_sd["conv_head.weight"]
    for sfx in _BN_SUFFIXES:
        if f"bn2.{sfx}" in timm_sd:
            out[f"features.21.{sfx}"] = timm_sd[f"bn2.{sfx}"]
    out["classifier.1.weight"] = timm_sd["classifier.weight"]
    out["classifier.1.bias"] = timm_sd["classifier.bias"]
    return out


def load_timm_weights(quant_model: nn.Module, timm_model: nn.Module, arch: str) -> nn.Module:
    """
    Load weights from a timm model into our quantized model.

    For ResNets, timm uses the same key names as torchvision so the generic
    load_pretrained_weights() path is used.  For MobileNets, a bespoke
    remapping is applied first.

    Args:
        quant_model: our Brevitas-wrapped model
        timm_model:  timm model created with pretrained=True
        arch:        one of 'resnet18', 'resnet50', 'mobilenetv1', 'mobilenetv2'
    """
    if arch in ("resnet18", "resnet50"):
        return load_pretrained_weights(quant_model, timm_model)

    timm_sd = timm_model.state_dict()
    if arch == "mobilenetv1":
        remapped = _remap_timm_mobilenetv1_sd(timm_sd)
    elif arch == "mobilenetv2":
        remapped = _remap_timm_mobilenetv2_sd(timm_sd)
    else:
        raise ValueError(f"Unknown arch for timm remapping: {arch}")

    our_sd = quant_model.state_dict()
    filtered = {}
    skipped = []
    for k, v in remapped.items():
        if k not in our_sd:
            skipped.append(f"{k} (not in model)")
            continue
        if v.shape != our_sd[k].shape:
            skipped.append(f"{k}: shape {v.shape} vs {our_sd[k].shape}")
            continue
        filtered[k] = v

    n_total = len([k for k in our_sd if not any(x in k for x in ("quant", "search", "annealing"))])
    print(f"[pretrained] Loaded {len(filtered)}/{n_total} weight tensors from timm {arch}.")
    if skipped:
        print(f"[pretrained] Skipped ({len(skipped)}): {skipped}")

    quant_model.load_state_dict(filtered, strict=False)
    return quant_model
