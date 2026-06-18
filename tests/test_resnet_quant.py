"""
Tests for QuantResNet architecture correctness.

Verifies:
  - Pre-add activation quantization (QuantIdentity) is present in BasicBlock / Bottleneck
  - Downsample skip path is quantized with QuantIdentity when act_quant is set
  - nn.Flatten module is used instead of inline x.flatten(1)
  - Forward passes produce correct shapes and don't crash
  - All quantizers are calibrated after a single train forward
"""

import copy

import pytest
import torch
import torch.nn as nn
import brevitas.nn as qnn

from models.resnet_quant import (
    QuantBasicBlock,
    QuantBottleneck,
    QuantResNet18,
    QuantResNet50,
)
from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorBiasQuant,
    FixedPointPerTensorWeightQuant,
)
from quantizers.manager import QuantizerManager


# ---------------------------------------------------------------------------
# Local fixed-width injector subclasses (avoid sharing state across tests)
# ---------------------------------------------------------------------------

class _WQ(FixedPointPerTensorWeightQuant):
    bit_width = 8

class _AQ(FixedPointPerTensorActivationQuant):
    bit_width = 8

class _BQ(FixedPointPerTensorBiasQuant):
    bit_width = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_manager():
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


@pytest.fixture
def rn18_full():
    return QuantResNet18(num_classes=10, weight_quant=_WQ, act_quant=_AQ, bias_quant=_BQ)


@pytest.fixture
def rn50_full():
    return QuantResNet50(num_classes=10, weight_quant=_WQ, act_quant=_AQ, bias_quant=_BQ)


@pytest.fixture
def rn18_float():
    """Float model — no quant injectors."""
    return QuantResNet18(num_classes=10)


@pytest.fixture
def x_small():
    return torch.randn(2, 3, 64, 64)


# ---------------------------------------------------------------------------
# 1. BasicBlock structural tests
# ---------------------------------------------------------------------------

class TestBasicBlockStructure:

    def test_pre_add_quant_present_with_act_quant(self):
        """BasicBlock must expose a QuantIdentity pre_add_quant when act_quant is given."""
        block = QuantBasicBlock(64, 64, act_quant=_AQ)
        assert hasattr(block, "pre_add_quant"), (
            "QuantBasicBlock is missing 'pre_add_quant' attribute when act_quant is set. "
            "The pre-add activation is not quantized before the residual addition."
        )
        assert block.pre_add_quant is not None
        assert isinstance(block.pre_add_quant, qnn.QuantIdentity), (
            f"Expected qnn.QuantIdentity for pre_add_quant, got {type(block.pre_add_quant)}"
        )

    def test_pre_add_quant_absent_without_act_quant(self):
        """Without act_quant, BasicBlock.pre_add_quant must be None (no spurious quantizer)."""
        block = QuantBasicBlock(64, 64)
        quant = getattr(block, "pre_add_quant", None)
        assert quant is None, (
            "pre_add_quant should be None when no act_quant is given"
        )

    def test_relu1_is_quant_relu_with_act_quant(self):
        block = QuantBasicBlock(64, 64, act_quant=_AQ)
        assert isinstance(block.relu1, qnn.QuantReLU)

    def test_relu2_is_quant_relu_with_act_quant(self):
        block = QuantBasicBlock(64, 64, act_quant=_AQ)
        assert isinstance(block.relu2, qnn.QuantReLU)


# ---------------------------------------------------------------------------
# 2. Bottleneck structural tests
# ---------------------------------------------------------------------------

class TestBottleneckStructure:

    def test_pre_add_quant_present_with_act_quant(self):
        """Bottleneck must expose a QuantIdentity pre_add_quant when act_quant is given."""
        block = QuantBottleneck(64, 16, act_quant=_AQ)
        assert hasattr(block, "pre_add_quant"), (
            "QuantBottleneck is missing 'pre_add_quant' attribute when act_quant is set."
        )
        assert block.pre_add_quant is not None
        assert isinstance(block.pre_add_quant, qnn.QuantIdentity)

    def test_pre_add_quant_absent_without_act_quant(self):
        block = QuantBottleneck(64, 16)
        quant = getattr(block, "pre_add_quant", None)
        assert quant is None

    def test_all_three_relus_are_quant_relu(self):
        block = QuantBottleneck(64, 16, act_quant=_AQ)
        assert isinstance(block.relu1, qnn.QuantReLU)
        assert isinstance(block.relu2, qnn.QuantReLU)
        assert isinstance(block.relu3, qnn.QuantReLU)


# ---------------------------------------------------------------------------
# 3. Downsample quantization tests
# ---------------------------------------------------------------------------

class TestDownsampleQuantization:

    def _has_quant_identity(self, module):
        return any(isinstance(m, qnn.QuantIdentity) for m in module.modules())

    def test_layer2_downsample_has_quant_identity(self, rn18_full):
        block = rn18_full.layer2[0]
        assert block.downsample is not None
        assert self._has_quant_identity(block.downsample), (
            "layer2[0].downsample should contain a QuantIdentity to quantize the skip path"
        )

    def test_layer3_downsample_has_quant_identity(self, rn18_full):
        block = rn18_full.layer3[0]
        assert block.downsample is not None
        assert self._has_quant_identity(block.downsample), (
            "layer3[0].downsample should contain a QuantIdentity"
        )

    def test_layer4_downsample_has_quant_identity(self, rn18_full):
        block = rn18_full.layer4[0]
        assert block.downsample is not None
        assert self._has_quant_identity(block.downsample), (
            "layer4[0].downsample should contain a QuantIdentity"
        )

    def test_float_model_downsample_has_no_quant_identity(self, rn18_float):
        """Float model downsample must NOT contain QuantIdentity (no act_quant)."""
        block = rn18_float.layer2[0]
        assert block.downsample is not None
        assert not self._has_quant_identity(block.downsample), (
            "Float model downsample should not contain QuantIdentity"
        )

    def test_layer1_has_no_downsample(self, rn18_full):
        """layer1 in ResNet-18 has no downsample (same channels, stride=1)."""
        assert rn18_full.layer1[0].downsample is None


# ---------------------------------------------------------------------------
# 4. Flatten module test
# ---------------------------------------------------------------------------

class TestFlattenModule:

    def test_resnet18_has_flatten_module(self, rn18_full):
        """QuantResNet18 must use an nn.Flatten module, not inline .flatten()."""
        has_flatten = any(isinstance(m, nn.Flatten) for m in rn18_full.modules())
        assert has_flatten, (
            "QuantResNet18 should contain an nn.Flatten module before QuantLinear. "
            "Using x.flatten(1) in forward() is fragile with QuantTensors."
        )

    def test_resnet18_float_has_flatten_module(self, rn18_float):
        has_flatten = any(isinstance(m, nn.Flatten) for m in rn18_float.modules())
        assert has_flatten

    def test_resnet50_has_flatten_module(self, rn50_full):
        has_flatten = any(isinstance(m, nn.Flatten) for m in rn50_full.modules())
        assert has_flatten


# ---------------------------------------------------------------------------
# 5. Forward pass / shape tests (regression — must pass before and after fix)
# ---------------------------------------------------------------------------

class TestQuantResNetForward:

    def test_resnet18_train_output_shape(self, rn18_full, x_small):
        rn18_full.train()
        out = rn18_full(x_small)
        assert out.shape == (2, 10)

    def test_resnet18_float_train_output_shape(self, rn18_float, x_small):
        rn18_float.train()
        out = rn18_float(x_small)
        assert out.shape == (2, 10)

    def test_resnet50_train_output_shape(self, rn50_full, x_small):
        rn50_full.train()
        out = rn50_full(x_small)
        assert out.shape == (2, 10)

    def test_resnet18_eval_after_calibration(self, rn18_full, x_small):
        """After calibration in train mode, eval forward must not raise."""
        rn18_full.train()
        rn18_full(x_small)  # calibration pass
        rn18_full.eval()
        with torch.no_grad():
            out = rn18_full(x_small)
        assert out.shape == (2, 10)

    def test_resnet50_eval_after_calibration(self, rn50_full, x_small):
        rn50_full.train()
        rn50_full(x_small)
        rn50_full.eval()
        with torch.no_grad():
            out = rn50_full(x_small)
        assert out.shape == (2, 10)

    def test_state_dict_roundtrip(self, rn18_full, x_small):
        """state_dict save + load with strict=True must work."""
        rn18_full.train()
        rn18_full(x_small)  # calibrate so search_done buffers are set
        sd = copy.deepcopy(rn18_full.state_dict())
        rn18_full.load_state_dict(sd, strict=True)

    def test_float_checkpoint_loads_with_strict_false(self, rn18_full):
        """Loading a float checkpoint (missing quant keys) must work with strict=False."""
        sd = rn18_full.state_dict()
        float_sd = {
            k: v for k, v in sd.items()
            if "search_result" not in k
            and "search_done" not in k
            and "annealing" not in k
        }
        rn18_full.load_state_dict(float_sd, strict=False)


# ---------------------------------------------------------------------------
# 6. Quantizer calibration coverage
# ---------------------------------------------------------------------------

class TestQuantizerCoverage:

    def test_quantizers_registered_at_construction(self, rn18_full):
        mgr = QuantizerManager()
        assert len(mgr.quantizers) > 0, "No quantizers registered after model construction"

    def test_all_quantizers_calibrated_after_train_forward(self, rn18_full, x_small):
        """After one train forward, every registered quantizer must be calibrated."""
        rn18_full.train()
        rn18_full(x_small)
        mgr = QuantizerManager()
        uncalibrated = [
            qid for qid, q in mgr.quantizers.items()
            if not q.search_done.item()
        ]
        assert len(uncalibrated) == 0, (
            f"Quantizers still uncalibrated after train forward: {uncalibrated}"
        )

    def test_quant_model_registers_more_quantizers_than_weight_only(self):
        """act_quant adds QuantIdentity layers, so more quantizers must be registered."""
        QuantizerManager().reset()
        QuantResNet18(num_classes=10, weight_quant=_WQ)
        n_weight_only = len(QuantizerManager().quantizers)

        QuantizerManager().reset()
        QuantResNet18(num_classes=10, weight_quant=_WQ, act_quant=_AQ)
        n_with_act = len(QuantizerManager().quantizers)

        assert n_with_act > n_weight_only, (
            f"Expected more quantizers with act_quant ({n_with_act}) "
            f"than weight-only ({n_weight_only}). "
            "QuantIdentity layers for pre-add and downsample paths may be missing."
        )
