"""
BatchNorm fusion ("BN folding"): bake a BatchNorm layer's affine transform
into the weight and bias of the Conv/Linear layer immediately preceding it,
then replace the BatchNorm module with nn.Identity().

Why: PTQ calibration should see the same weight distribution the deployed
(BN-folded) model will actually use. Calibrating LSBs against un-fused
weights and then folding BN afterwards would silently invalidate the search.

Works generically across this repo's model files (models/resnet_quant.py,
models/mobilenetv1_quant.py, models/mobilenetv2_quant.py) without any
per-architecture special-casing: every block in this codebase declares its
conv then its BatchNorm immediately after, either as named attributes
(self.conv1 / self.bn1) or as adjacent entries in an nn.Sequential. Since
nn.Module preserves child registration order, walking each module's direct
children and looking for adjacent (Conv*d/Linear, BatchNorm*d) pairs finds
every fusable pair regardless of how the model is structured.

Brevitas's QuantConv2d/QuantLinear subclass the corresponding plain torch
layer and store .weight/.bias as ordinary nn.Parameter tensors — the
quantizer is only applied dynamically inside forward(). So fusing directly
against those float tensors is safe and is exactly what a deployment-time
BN fold should do; it requires no Brevitas-specific handling.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_FOLDABLE_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _fuse_conv_bn_(conv: nn.Module, bn: nn.Module) -> None:
    """
    In-place: fold bn's affine transform into conv.weight / conv.bias.

    conv must expose .weight and .bias (possibly None) like nn.Conv*d or
    nn.Linear. If conv.bias is None (the common case in this repo — every
    conv feeding a BatchNorm is constructed with bias=False), a new bias
    parameter is created.
    """
    w = conv.weight.data
    out_channels = w.shape[0]

    if conv.bias is not None:
        b = conv.bias.data.clone()
    else:
        b = torch.zeros(out_channels, dtype=w.dtype, device=w.device)

    bn_rm  = bn.running_mean.to(dtype=w.dtype, device=w.device)
    bn_rv  = bn.running_var.to(dtype=w.dtype, device=w.device)
    bn_eps = bn.eps
    bn_w   = (bn.weight.data if bn.weight is not None
              else torch.ones_like(bn_rm)).to(dtype=w.dtype, device=w.device)
    bn_b   = (bn.bias.data if bn.bias is not None
              else torch.zeros_like(bn_rm)).to(dtype=w.dtype, device=w.device)

    scale = bn_w / torch.sqrt(bn_rv + bn_eps)

    # Broadcast scale (one value per output channel) over the remaining
    # weight dims — works for Conv*d (out, in, k1, k2, ...) and Linear (out, in).
    broadcast_shape = [out_channels] + [1] * (w.dim() - 1)
    new_w = w * scale.reshape(broadcast_shape)
    new_b = (b - bn_rm) * scale + bn_b

    conv.weight.data.copy_(new_w)
    if conv.bias is None:
        conv.bias = nn.Parameter(new_b)
    else:
        conv.bias.data.copy_(new_b)


def fuse_bn_into_conv(model: nn.Module) -> int:
    """
    Recursively fold every BatchNorm layer into its immediately preceding
    Conv/Linear layer, in place, replacing the BatchNorm module with
    nn.Identity() so the forward pass is unchanged (up to floating-point
    fusion error) but BatchNorm is no longer a separate op.

    The model's BatchNorm running statistics must already reflect real data
    (pretrained or trained weights) — fusing freshly initialized BatchNorm
    (running_mean=0, running_var=1) is a no-op modulo the BN's own affine
    weight/bias and will not raise an error, but is rarely useful.

    Returns the number of conv-bn pairs fused.
    """
    n_fused = 0
    for module in model.modules():
        children = list(module.named_children())
        for i in range(len(children) - 1):
            _, child_a = children[i]
            name_b, child_b = children[i + 1]
            if isinstance(child_a, _FOLDABLE_TYPES) and isinstance(child_b, _BN_TYPES):
                _fuse_conv_bn_(child_a, child_b)
                setattr(module, name_b, nn.Identity())
                n_fused += 1
    return n_fused
