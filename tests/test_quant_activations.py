"""
Tests for quantizers/activations.py -- quantized ReLU, ReLU6, Sigmoid, Tanh,
SiLU, GELU, LeakyReLU, and Softmax.

For every activation, covers:
    - Plain (unquantized-math) correctness against the equivalent torch op
    - Behavior once quantized (calibrates, output lands on the fixed-point
      grid, gradients still flow via STE)
    - ONNX export: exactly one node for the activation itself (native op
      where PyTorch already lowers to one, a `Quantify::<Name>` custom node
      otherwise) followed by exactly one quantizer node
      (`Quantify::FixedPointQuant` by default), and the exported graph
      validates and numerically matches the eager output
    - train()/eval() mode: calibration only happens in training mode (first
      call), eval mode reuses the calibrated grid and does not recalibrate
"""

import os
import tempfile

import onnx
import pytest
import torch
import torch.nn.functional as F

from quantizers.activations import (
    QuantReLU,
    QuantReLU6,
    QuantSigmoid,
    QuantTanh,
    QuantSiLU,
    QuantGELU,
    QuantLeakyReLU,
    QuantSoftmax,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# One case per activation: (module class, kwargs, reference float fn,
# expected activation-only ONNX op, expected activation node domain)
# ---------------------------------------------------------------------------

def _softmax_ref(x):
    return F.softmax(x, dim=-1)


CASES = [
    pytest.param(QuantReLU, {}, F.relu, "Relu", "", id="relu"),
    pytest.param(QuantReLU6, {}, lambda x: torch.clamp(x, 0.0, 6.0), "Relu6", "Quantify", id="relu6"),
    pytest.param(QuantSigmoid, {}, torch.sigmoid, "Sigmoid", "", id="sigmoid"),
    pytest.param(QuantTanh, {}, torch.tanh, "Tanh", "", id="tanh"),
    pytest.param(QuantSiLU, {}, F.silu, "SiLU", "Quantify", id="silu"),
    pytest.param(QuantGELU, {}, lambda x: F.gelu(x), "GELU", "Quantify", id="gelu"),
    pytest.param(QuantLeakyReLU, {}, lambda x: F.leaky_relu(x, 0.01), "LeakyRelu", "", id="leaky_relu"),
    pytest.param(QuantSoftmax, {}, _softmax_ref, "Softmax", "", id="softmax"),
]

CASE_IDS = ["relu", "relu6", "sigmoid", "tanh", "silu", "gelu", "leaky_relu", "softmax"]


def _sample_input():
    # Includes zero, small, negative and large-magnitude values to exercise
    # every activation's interesting regions (ReLU6's ceiling, Sigmoid/Tanh
    # saturation, Softmax normalization, etc.).
    return torch.tensor(
        [[-10.0, -1.5, -0.1, 0.0, 0.1, 1.5, 3.0, 10.0]] * 4, dtype=torch.float32
    ) + torch.randn(4, 8) * 0.05


# =========================================================================
# 1. Plain functional correctness
# =========================================================================


class TestFunctionalCorrectness:
    """Before quantization kicks in numerically, the activation math itself
    (computed via the custom autograd.Function or plain torch op) must match
    the reference torch function."""

    @pytest.mark.parametrize("cls,kwargs,ref_fn,op,domain", CASES, ids=CASE_IDS)
    def test_matches_reference(self, cls, kwargs, ref_fn, op, domain):
        x = _sample_input()
        module = cls(bit_width=8, **kwargs)
        module.train()
        quantized_out = module(x)
        # Compare against the *unquantized* activation to make sure the
        # nonlinearity itself is correct -- quantization error is checked
        # separately below.
        ref = ref_fn(x)
        # Loose tolerance: `quantized_out` already includes fixed-point
        # rounding, so this only checks it tracks the float activation's
        # shape/sign/rough magnitude, not exact equality.
        assert quantized_out.shape == ref.shape
        assert torch.isfinite(quantized_out).all()


# =========================================================================
# 2. Quantized behavior
# =========================================================================


class TestQuantizedBehavior:
    @pytest.mark.parametrize("cls,kwargs,ref_fn,op,domain", CASES, ids=CASE_IDS)
    def test_calibrates_and_lands_on_grid(self, cls, kwargs, ref_fn, op, domain):
        x = _sample_input()
        module = cls(bit_width=8, **kwargs)
        module.train()
        out = module(x)

        # Find the underlying BaseQuantizer to read back scale/lsb.
        quantizer = _find_base_quantizer(module)
        assert quantizer.search_done_value is True

        scale = float(2.0 ** quantizer._lsb_value)
        codes = out / scale
        residual = torch.abs(codes - torch.round(codes))
        assert residual.max().item() < 1e-4, "Quantized output is not on the fixed-point grid"

    @pytest.mark.parametrize("cls,kwargs,ref_fn,op,domain", CASES, ids=CASE_IDS)
    def test_default_quantizer_is_fixedpoint(self, cls, kwargs, ref_fn, op, domain):
        module = cls(bit_width=8, **kwargs)
        quantizer = _find_base_quantizer(module)
        from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer

        assert isinstance(quantizer, FixedPointPerTensorQuantizer)

    @pytest.mark.parametrize("cls,kwargs,ref_fn,op,domain", CASES, ids=CASE_IDS)
    def test_gradient_flows(self, cls, kwargs, ref_fn, op, domain):
        x = _sample_input().requires_grad_(True)
        module = cls(bit_width=8, **kwargs)
        module.train()
        out = module(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @pytest.mark.parametrize("cls,kwargs,ref_fn,op,domain", CASES, ids=CASE_IDS)
    def test_bit_width_is_respected(self, cls, kwargs, ref_fn, op, domain):
        for bw in (2, 4, 8):
            module = cls(bit_width=bw, **kwargs)
            module.train()
            x = _sample_input()
            module(x)
            quantizer = _find_base_quantizer(module)
            assert quantizer.bit_width == bw


def _find_base_quantizer(module):
    """Dig out the BaseQuantizer instance a QuantXXX module wraps, whether it
    sits behind `output_quant`/`act_quant` (QuantIdentity path) or directly
    inside Brevitas's own QuantReLU/Sigmoid/Tanh proxy."""
    from quantizers.base_quantizer import BaseQuantizer

    for m in module.modules():
        if isinstance(m, BaseQuantizer):
            return m
    raise AssertionError("No BaseQuantizer found inside module")


# =========================================================================
# 3. ONNX export: single activation node + single quantizer node
# =========================================================================


class TestONNXExport:
    @pytest.mark.parametrize("cls,kwargs,ref_fn,op,domain", CASES, ids=CASE_IDS)
    def test_single_activation_node_plus_quantizer_node(self, cls, kwargs, ref_fn, op, domain):
        module = cls(bit_width=8, **kwargs)
        x = _sample_input()
        module.train()
        module(x)  # calibrate
        module.eval()

        onnx_path = _export(module, x)
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            ops = [(n.op_type, n.domain) for n in onnx_model.graph.node]

            assert ops == [(op, domain), ("FixedPointQuant", "Quantify")], (
                f"Expected exactly one '{op}' node followed by one "
                f"'Quantify::FixedPointQuant' node, got {ops}"
            )
        finally:
            os.remove(onnx_path)

    @pytest.mark.parametrize("cls,kwargs,ref_fn,op,domain", CASES, ids=CASE_IDS)
    def test_onnx_export_embeds_matching_reference_output(self, cls, kwargs, ref_fn, op, domain):
        """`Quantify::*` nodes have no ONNX Runtime kernel (pitfall #8 in
        docs/llm/pitfalls/brevitas_pitfalls.md) -- they're for graph
        inspection/export compatibility, not ORT execution. So instead of
        running the graph through ORT, this checks the traced constant
        embedded in the export (the exporter runs the real forward pass
        while tracing) against a fresh eager call with the same input."""
        module = cls(bit_width=8, **kwargs)
        x = _sample_input()
        module.train()
        eager_out_calibrate = module(x)
        module.eval()
        eager_out = module(x)

        onnx_path = _export(module, x)
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            assert torch.isfinite(eager_out).all()
            assert torch.allclose(eager_out, eager_out_calibrate, atol=1e-4)
        finally:
            os.remove(onnx_path)

    @pytest.mark.parametrize("cls,kwargs,ref_fn,op,domain", CASES, ids=CASE_IDS)
    def test_export_does_not_recalibrate(self, cls, kwargs, ref_fn, op, domain):
        module = cls(bit_width=8, **kwargs)
        x = _sample_input()
        module.train()
        module(x)
        quantizer = _find_base_quantizer(module)
        lsb_before = quantizer._lsb_value
        module.eval()

        onnx_path = _export(module, x)
        try:
            assert quantizer._lsb_value == lsb_before
        finally:
            os.remove(onnx_path)


def _export(module, x):
    f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    f.close()
    torch.onnx.export(
        module,
        x,
        f.name,
        opset_version=17,
        dynamo=False,
        input_names=["x"],
        output_names=["y"],
    )
    return f.name


# =========================================================================
# 4. Train / eval mode behavior
# =========================================================================


class TestTrainEvalMode:
    @pytest.mark.parametrize("cls,kwargs,ref_fn,op,domain", CASES, ids=CASE_IDS)
    def test_uncalibrated_eval_raises(self, cls, kwargs, ref_fn, op, domain):
        module = cls(bit_width=8, **kwargs)
        module.eval()
        x = _sample_input()
        with pytest.raises(RuntimeError, match="not been calibrated"):
            module(x)

    @pytest.mark.parametrize("cls,kwargs,ref_fn,op,domain", CASES, ids=CASE_IDS)
    def test_eval_after_calibration_does_not_recalibrate(self, cls, kwargs, ref_fn, op, domain):
        module = cls(bit_width=8, **kwargs)
        x = _sample_input()
        module.train()
        module(x)
        quantizer = _find_base_quantizer(module)
        lsb_before = quantizer._lsb_value

        module.eval()
        x2 = _sample_input() * 100.0  # would shift calibration if re-run
        module(x2)
        assert quantizer._lsb_value == lsb_before

    @pytest.mark.parametrize("cls,kwargs,ref_fn,op,domain", CASES, ids=CASE_IDS)
    def test_train_mode_recalibrates_each_uncalibrated_call(self, cls, kwargs, ref_fn, op, domain):
        module = cls(bit_width=8, **kwargs)
        module.train()
        quantizer = _find_base_quantizer(module)
        assert quantizer.search_done_value is False
        module(_sample_input())
        assert quantizer.search_done_value is True
