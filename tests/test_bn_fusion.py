"""
Tests for utils/bn_fusion.py — folding BatchNorm into the preceding
Conv/Linear layer before quantizer calibration.

Verifies:
  - Numerical equivalence: model(x) before and after fusion must match
    (within float tolerance) when BatchNorm is in eval mode (using its
    running statistics, not batch statistics).
  - The BatchNorm module is replaced with nn.Identity() after fusion.
  - Works on plain nn.Conv2d/nn.BatchNorm2d as well as Brevitas QuantConv2d.
  - Works whether the conv/bn pair sits in an nn.Sequential or as named
    attributes on a custom block (the two patterns used across models/).
  - Correct fusion count is returned, including on the real model
    architectures defined in this repo (QuantResNet18, QuantMobileNetV1,
    QuantMobileNetV2).
  - Idempotent / safe when there is nothing to fuse.
  - A conv that already has bias=True is handled correctly (no double
    counting of the existing bias).
"""

import copy

import pytest
import torch
import torch.nn as nn
import brevitas.nn as qnn

from utils.bn_fusion import fuse_bn_into_conv
from quantizers.manager import QuantizerManager
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorWeightQuant
from models.resnet_quant import QuantResNet18, QuantBasicBlock
from models.mobilenetv1_quant import QuantMobileNetV1
from models.mobilenetv2_quant import QuantMobileNetV2


@pytest.fixture(autouse=True)
def reset_manager():
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


def _randomize_bn_stats(model: nn.Module) -> None:
    """Give every BatchNorm layer non-trivial running stats so fusion is
    actually exercised (fresh BN has running_mean=0, running_var=1, which
    would make fusion trivially a no-op for the mean/var part)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            with torch.no_grad():
                m.running_mean.copy_(torch.randn_like(m.running_mean) * 2.0)
                m.running_var.copy_(torch.rand_like(m.running_var) * 3.0 + 0.5)
                if m.weight is not None:
                    m.weight.copy_(torch.rand_like(m.weight) * 2.0 + 0.5)
                if m.bias is not None:
                    m.bias.copy_(torch.randn_like(m.bias))


# ---------------------------------------------------------------------------
# Numerical equivalence
# ---------------------------------------------------------------------------

class TestNumericalEquivalence:

    def test_plain_conv_bn_equivalence(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, bias=False),
            nn.BatchNorm2d(8),
        )
        _randomize_bn_stats(model)
        model.eval()

        x = torch.randn(4, 3, 16, 16)
        with torch.no_grad():
            out_before = model(x)

        n_fused = fuse_bn_into_conv(model)
        assert n_fused == 1

        with torch.no_grad():
            out_after = model(x)

        assert torch.allclose(out_before, out_after, atol=1e-4, rtol=1e-4), (
            f"max diff = {(out_before - out_after).abs().max().item()}"
        )

    def test_quant_conv_bn_equivalence(self):
        torch.manual_seed(0)

        class _WQ(FixedPointPerTensorWeightQuant):
            bit_width = 8

        model = nn.Sequential(
            qnn.QuantConv2d(3, 8, 3, bias=False, weight_quant=None),
            nn.BatchNorm2d(8),
        )
        _randomize_bn_stats(model)
        model.eval()

        x = torch.randn(4, 3, 16, 16)
        with torch.no_grad():
            out_before = model(x)

        n_fused = fuse_bn_into_conv(model)
        assert n_fused == 1

        with torch.no_grad():
            out_after = model(x)

        assert torch.allclose(out_before, out_after, atol=1e-4, rtol=1e-4)

    def test_conv_with_existing_bias_equivalence(self):
        """A conv that already has bias=True must still fuse correctly —
        the BN fold must combine with the existing bias, not overwrite it
        incorrectly or double-count it."""
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, bias=True),
            nn.BatchNorm2d(8),
        )
        with torch.no_grad():
            model[0].bias.copy_(torch.randn(8))
        _randomize_bn_stats(model)
        model.eval()

        x = torch.randn(4, 3, 16, 16)
        with torch.no_grad():
            out_before = model(x)

        fuse_bn_into_conv(model)

        with torch.no_grad():
            out_after = model(x)

        assert torch.allclose(out_before, out_after, atol=1e-4, rtol=1e-4)

    def test_linear_bn1d_equivalence(self):
        """Generic support for Linear + BatchNorm1d (not used by this repo's
        models today, but the function should handle it correctly)."""
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(16, 32, bias=False),
            nn.BatchNorm1d(32),
        )
        _randomize_bn_stats(model)
        model.eval()

        x = torch.randn(5, 16)
        with torch.no_grad():
            out_before = model(x)

        n_fused = fuse_bn_into_conv(model)
        assert n_fused == 1

        with torch.no_grad():
            out_after = model(x)

        assert torch.allclose(out_before, out_after, atol=1e-4, rtol=1e-4)

    def test_named_attribute_block_equivalence(self):
        """conv/bn declared as named attributes (self.conv1/self.bn1), not
        inside an nn.Sequential — the pattern used by QuantBasicBlock etc."""
        torch.manual_seed(0)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(4)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.bn1(self.conv1(x)))

        model = Block()
        _randomize_bn_stats(model)
        model.eval()

        x = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            out_before = model(x)

        n_fused = fuse_bn_into_conv(model)
        assert n_fused == 1
        assert isinstance(model.bn1, nn.Identity)

        with torch.no_grad():
            out_after = model(x)

        assert torch.allclose(out_before, out_after, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

class TestStructuralChanges:

    def test_bn_replaced_with_identity(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3, bias=False), nn.BatchNorm2d(4))
        fuse_bn_into_conv(model)
        assert isinstance(model[1], nn.Identity)

    def test_bias_created_when_missing(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3, bias=False), nn.BatchNorm2d(4))
        assert model[0].bias is None
        fuse_bn_into_conv(model)
        assert model[0].bias is not None
        assert model[0].bias.shape == (4,)

    def test_no_fusion_when_no_bn_present(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3, bias=False), nn.ReLU())
        n_fused = fuse_bn_into_conv(model)
        assert n_fused == 0
        assert isinstance(model[1], nn.ReLU)  # untouched

    def test_standalone_bn_without_preceding_conv_untouched(self):
        """A BatchNorm with no conv/linear immediately before it must be
        left alone (no adjacent foldable layer to fuse into)."""
        model = nn.Sequential(nn.ReLU(), nn.BatchNorm2d(4))
        n_fused = fuse_bn_into_conv(model)
        assert n_fused == 0
        assert isinstance(model[1], nn.BatchNorm2d)

    def test_multiple_conv_bn_pairs_in_sequential(self):
        """conv, bn, relu, conv, bn — the MobileNetV2 inverted-residual
        pattern — must fuse both pairs without skipping or double-counting."""
        model = nn.Sequential(
            nn.Conv2d(3, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 1, bias=False),
            nn.BatchNorm2d(4),
        )
        n_fused = fuse_bn_into_conv(model)
        assert n_fused == 2
        assert isinstance(model[1], nn.Identity)
        assert isinstance(model[4], nn.Identity)


# ---------------------------------------------------------------------------
# Real model architectures
# ---------------------------------------------------------------------------

class TestRealModels:

    def test_resnet18_fuses_expected_count_and_matches_output(self):
        torch.manual_seed(0)
        model = QuantResNet18(num_classes=10)
        n_bn_before = sum(
            1 for m in model.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
        )
        assert n_bn_before > 0
        _randomize_bn_stats(model)
        model.eval()

        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out_before = model(x)

        n_fused = fuse_bn_into_conv(model)
        assert n_fused == n_bn_before

        n_bn_after = sum(
            1 for m in model.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
        )
        assert n_bn_after == 0

        with torch.no_grad():
            out_after = model(x)

        assert torch.allclose(out_before, out_after, atol=1e-3, rtol=1e-3), (
            f"max diff = {(out_before - out_after).abs().max().item()}"
        )

    def test_resnet18_downsample_path_fused(self):
        """layer2[0].downsample is an nn.Sequential([QuantConv2d, BatchNorm2d])
        — verify the downsample BN specifically gets folded."""
        model = QuantResNet18(num_classes=10)
        _randomize_bn_stats(model)
        block = model.layer2[0]
        assert any(isinstance(m, nn.BatchNorm2d) for m in block.downsample.modules())

        fuse_bn_into_conv(model)

        assert not any(isinstance(m, nn.BatchNorm2d) for m in block.downsample.modules())

    def test_mobilenetv1_fuses_all_bn(self):
        torch.manual_seed(0)
        model = QuantMobileNetV1(num_classes=10)
        n_bn_before = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        assert n_bn_before > 0
        _randomize_bn_stats(model)
        model.eval()

        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out_before = model(x)

        n_fused = fuse_bn_into_conv(model)
        assert n_fused == n_bn_before
        assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())

        with torch.no_grad():
            out_after = model(x)
        assert torch.allclose(out_before, out_after, atol=1e-3, rtol=1e-3)

    def test_mobilenetv2_fuses_all_bn(self):
        """QuantMobileNetV2 always builds a real (default) weight quantizer
        internally even when weight_quant=None is passed in, so disable
        quantization to isolate the BN-fold equivalence check from the
        separate concern of quantizer calibration."""
        torch.manual_seed(0)
        model = QuantMobileNetV2(num_classes=10)
        QuantizerManager().disable_quantization()
        n_bn_before = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        assert n_bn_before > 0
        _randomize_bn_stats(model)
        model.eval()

        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out_before = model(x)

        n_fused = fuse_bn_into_conv(model)
        assert n_fused == n_bn_before
        assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())

        with torch.no_grad():
            out_after = model(x)
        assert torch.allclose(out_before, out_after, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Idempotency / safety
# ---------------------------------------------------------------------------

class TestIdempotency:

    def test_running_twice_second_pass_is_noop(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3, bias=False), nn.BatchNorm2d(4))
        _randomize_bn_stats(model)
        first = fuse_bn_into_conv(model)
        second = fuse_bn_into_conv(model)
        assert first == 1
        assert second == 0

    def test_deepcopy_preserves_fusability(self):
        """Sanity: fusion operates on the model instance passed in, not a
        cached reference — deep-copied models fuse independently."""
        model_a = nn.Sequential(nn.Conv2d(3, 4, 3, bias=False), nn.BatchNorm2d(4))
        _randomize_bn_stats(model_a)
        model_b = copy.deepcopy(model_a)

        fuse_bn_into_conv(model_a)
        assert isinstance(model_a[1], nn.Identity)
        assert isinstance(model_b[1], nn.BatchNorm2d)
