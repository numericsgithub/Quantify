"""
Regression test: examples/find_perfect_lsbs_imagenet_ptq.py --fuse-bn folds
BatchNorm into the preceding conv/linear (conv gains a bias, BatchNorm becomes
nn.Identity) before saving its checkpoint. examples/train_imagenet_qat.py's
_load_ptq_checkpoint used to load that checkpoint's model_state_dict straight
into a freshly built model that still has separate, randomly-initialized
BatchNorm layers — a structural mismatch (missing bn*.weight/bias/running_*
keys, unexpected conv*.bias keys) that left BatchNorm untrained and produced
near-random accuracy after "loading" a fully PTQ-calibrated checkpoint.

The fix: the checkpoint already stores extra["fuse_bn"] (set by
find_perfect_lsbs_imagenet_ptq.py at save time). _load_ptq_checkpoint now
checks it and fuses this model's BatchNorm the same way before loading, so
the module structures match.
"""

import pytest
import torch
import torch.nn as nn

from quantizers.manager import QuantizerManager
from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorWeightQuant,
    FixedPointPerTensorActivationQuant,
)
from models.resnet_quant import QuantResNet18
from utils.bn_fusion import fuse_bn_into_conv
from examples.train_imagenet_qat import _load_ptq_checkpoint


class _WQ(FixedPointPerTensorWeightQuant):
    bit_width = 8


class _AQ(FixedPointPerTensorActivationQuant):
    bit_width = 8


@pytest.fixture(autouse=True)
def reset_manager():
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


def _randomize_bn_stats(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            with torch.no_grad():
                m.running_mean.copy_(torch.randn_like(m.running_mean) * 2.0)
                m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
                m.weight.copy_(torch.randn_like(m.weight) * 0.5 + 1.0)
                m.bias.copy_(torch.randn_like(m.bias) * 0.5)


def _build_fused_checkpoint(tmp_path):
    """Stand-in for find_perfect_lsbs_imagenet_ptq.py --fuse-bn: build a
    model, fuse its BatchNorm, calibrate every quantizer, and save a
    checkpoint with extra.fuse_bn=True — exactly what that script produces."""
    QuantizerManager().reset()
    model = QuantResNet18(num_classes=10, weight_quant=_WQ, act_quant=_AQ)
    _randomize_bn_stats(model)
    fuse_bn_into_conv(model)

    for q in QuantizerManager().quantizers.values():
        q.search_result_lsb.fill_(-8)
        q.search_done.fill_(True)
        q.annealing_alpha.data.fill_(1.0)

    payload = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "metrics": {"final_val_acc": 56.1},
        "extra": {
            "ptq_search_mode": "activations",
            "bit_width": 8,
            "role_bit_widths": {"weight": 8, "activation": 8},
            "fuse_bn": True,
        },
    }
    path = tmp_path / "fused_ckpt.pt"
    torch.save(payload, path)
    return str(path)


def _build_unfused_checkpoint(tmp_path):
    """Same as above but WITHOUT --fuse-bn — extra.fuse_bn is False, the
    model_state_dict still has separate BatchNorm layers."""
    QuantizerManager().reset()
    model = QuantResNet18(num_classes=10, weight_quant=_WQ, act_quant=_AQ)
    _randomize_bn_stats(model)

    for q in QuantizerManager().quantizers.values():
        q.search_result_lsb.fill_(-8)
        q.search_done.fill_(True)
        q.annealing_alpha.data.fill_(1.0)

    payload = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "metrics": {},
        "extra": {
            "ptq_search_mode": "activations",
            "bit_width": 8,
            "role_bit_widths": {"weight": 8, "activation": 8},
            "fuse_bn": False,
        },
    }
    path = tmp_path / "unfused_ckpt.pt"
    torch.save(payload, path)
    return str(path)


class TestLoadPtqCheckpointBnFusion:

    def test_fused_checkpoint_triggers_matching_fusion_on_target_model(self, tmp_path):
        ckpt_path = _build_fused_checkpoint(tmp_path)

        QuantizerManager().reset()
        target = QuantResNet18(num_classes=10, weight_quant=_WQ, act_quant=_AQ)
        assert isinstance(target.bn1, nn.BatchNorm2d), "target starts unfused"

        _load_ptq_checkpoint(target, ckpt_path)

        assert isinstance(target.bn1, nn.Identity), (
            "_load_ptq_checkpoint must fuse BatchNorm on the target model "
            "when the checkpoint's extra.fuse_bn is True, to match its "
            "module structure before load_state_dict"
        )
        assert target.conv1.bias is not None

    def test_fused_checkpoint_loads_with_no_missing_or_unexpected_keys(self, tmp_path):
        ckpt_path = _build_fused_checkpoint(tmp_path)

        QuantizerManager().reset()
        target = QuantResNet18(num_classes=10, weight_quant=_WQ, act_quant=_AQ)
        payload = torch.load(ckpt_path, map_location="cpu")

        fuse_bn_into_conv(target)
        incompatible = target.load_state_dict(payload["model_state_dict"], strict=False)
        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []

    def test_fused_checkpoint_loaded_model_runs_without_error(self, tmp_path):
        ckpt_path = _build_fused_checkpoint(tmp_path)

        QuantizerManager().reset()
        target = QuantResNet18(num_classes=10, weight_quant=_WQ, act_quant=_AQ)
        _load_ptq_checkpoint(target, ckpt_path)

        target.eval()
        with torch.no_grad():
            out = target(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 10)
        assert torch.isfinite(out).all()

    def test_unfused_checkpoint_does_not_trigger_fusion(self, tmp_path):
        ckpt_path = _build_unfused_checkpoint(tmp_path)

        QuantizerManager().reset()
        target = QuantResNet18(num_classes=10, weight_quant=_WQ, act_quant=_AQ)
        _load_ptq_checkpoint(target, ckpt_path)

        assert isinstance(target.bn1, nn.BatchNorm2d), (
            "a checkpoint without extra.fuse_bn must not trigger fusion on the target"
        )
