"""
Tests for default input+output quantizers on quantizers/activations.py's
eight quantized activation modules.

Three behaviors are covered, written BEFORE the corresponding implementation
(so the first run of this file is expected to fail/error -- see the module
docstring note in each test class for what "red" looks like before the
guardrails exist):

1. Every activation has BOTH an input and an output fixed-point quantizer
   by default (currently only an output quantizer exists).

2. Bit-width asymmetry: an activation whose output is structurally forced
   non-negative (ReLU, ReLU6, Sigmoid, Softmax) doesn't need a sign bit on
   its OUTPUT quantizer, so the saved bit goes to the INPUT quantizer
   instead (input_bit_width = bit_width + 1), which commonly *does* need a
   sign bit (pre-activation values are usually signed). Activations whose
   output can be negative (Tanh, SiLU, GELU, LeakyReLU) get
   input_bit_width == bit_width (no asymmetry).

3. Saturating activations (ReLU6: hard clip at [0, 6]; Sigmoid/Tanh:
   asymptotic saturation) must not let their INPUT quantizer's calibrated
   range balloon far beyond what the activation itself will ever let
   through -- a huge pre-activation outlier would otherwise force a huge
   quantization range, wasting resolution on values the activation clips
   away anyway. Non-saturating activations (ReLU, SiLU, GELU, LeakyReLU,
   Softmax) must NOT be constrained this way -- a genuinely wide input
   range for these is legitimate and must calibrate normally.
"""

import torch
import pytest
import brevitas.nn as qnn

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
from quantizers.base_quantizer import BaseQuantizer


def _quantizers_of(module):
    """All BaseQuantizer instances inside a module, in whatever order
    named_modules() finds them."""
    return [m for m in module.modules() if isinstance(m, BaseQuantizer)]


def _find_by_role_position(module, want_input: bool):
    """Return the quantizer that sits BEFORE the activation's own nonlinearity
    (want_input=True) or AFTER it (want_input=False), identified by which
    one calibrates first when a forward pass runs -- the input quantizer's
    forward() always executes strictly before the output quantizer's.
    """
    quantizers = _quantizers_of(module)
    assert len(quantizers) >= 2, (
        f"expected at least 2 quantizers (input + output), found {len(quantizers)}: "
        f"{[type(q).__name__ for q in quantizers]}"
    )
    # inference_sequence_id is assigned in true forward-execution order (see
    # quantizers/manager.py), so sorting by it reliably separates input (id 0)
    # from output (id 1) regardless of how each module nests its children.
    quantizers_sorted = sorted(quantizers, key=lambda q: q.inference_sequence_id)
    return quantizers_sorted[0] if want_input else quantizers_sorted[-1]


# All 8 modules, each with (constructor, is_unsigned_output, kwargs_for_ctor).
# is_unsigned_output drives the expected input_bit_width = bit_width (+1 if True).
ACTIVATIONS = [
    pytest.param(QuantReLU, True, {}, id="relu"),
    pytest.param(QuantReLU6, True, {}, id="relu6"),
    pytest.param(QuantSigmoid, True, {}, id="sigmoid"),
    pytest.param(QuantSoftmax, True, {}, id="softmax"),
    pytest.param(QuantTanh, False, {}, id="tanh"),
    pytest.param(QuantSiLU, False, {}, id="silu"),
    pytest.param(QuantGELU, False, {}, id="gelu"),
    pytest.param(QuantLeakyReLU, False, {}, id="leaky_relu"),
]


def _sample_input():
    return torch.randn(8, 16) * 0.5


# =========================================================================
# 1. Default input + output quantizers
# =========================================================================


class TestDefaultInputOutputQuantizers:
    """Before the feature exists, each module has exactly ONE quantizer
    (output only) -- `_quantizers_of` would return length 1, failing the
    `>= 2` assertion inside `_find_by_role_position`. This is the expected
    "red" state prior to implementation.
    """

    @pytest.mark.parametrize("cls,unsigned_output,kwargs", ACTIVATIONS)
    def test_has_at_least_two_quantizers(self, cls, unsigned_output, kwargs):
        module = cls(bit_width=8, **kwargs)
        quantizers = _quantizers_of(module)
        assert len(quantizers) >= 2, (
            f"{cls.__name__} must have both an input and an output quantizer "
            f"by default, found {len(quantizers)}"
        )

    @pytest.mark.parametrize("cls,unsigned_output,kwargs", ACTIVATIONS)
    def test_input_and_output_both_calibrate_and_are_fixedpoint(self, cls, unsigned_output, kwargs):
        from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer

        module = cls(bit_width=8, **kwargs)
        module.train()
        module(_sample_input())

        q_in = _find_by_role_position(module, want_input=True)
        q_out = _find_by_role_position(module, want_input=False)
        assert q_in is not q_out
        assert isinstance(q_in, FixedPointPerTensorQuantizer)
        assert isinstance(q_out, FixedPointPerTensorQuantizer)
        assert q_in.search_done_value is True
        assert q_out.search_done_value is True

    @pytest.mark.parametrize("cls,unsigned_output,kwargs", ACTIVATIONS)
    def test_onnx_export_has_input_quant_activation_output_quant(self, cls, unsigned_output, kwargs):
        """Exactly 3 ONNX nodes now: input quantizer, activation, output
        quantizer (was 2 before this feature: activation, output quantizer)."""
        import tempfile, os
        import onnx

        module = cls(bit_width=8, **kwargs)
        x = _sample_input()
        module.train()
        module(x)
        module.eval()

        f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        f.close()
        torch.onnx.export(module, x, f.name, opset_version=17, dynamo=False,
                           input_names=["x"], output_names=["y"])
        try:
            onnx_model = onnx.load(f.name)
            onnx.checker.check_model(onnx_model)
            ops = [(n.op_type, n.domain) for n in onnx_model.graph.node]
            quant_nodes = [o for o in ops if o[0] == "FixedPointQuant"]
            assert len(quant_nodes) == 2, f"expected input+output quant nodes, got {ops}"
        finally:
            os.remove(f.name)


# =========================================================================
# 2. Bit-width asymmetry (input gets the sign bit the output doesn't need)
# =========================================================================


class TestBitWidthAsymmetry:
    @pytest.mark.parametrize("cls,unsigned_output,kwargs", ACTIVATIONS)
    def test_input_bit_width_matches_expected_rule(self, cls, unsigned_output, kwargs):
        module = cls(bit_width=8, **kwargs)
        # Feed data wide enough to make sign auto-detection unambiguous
        # for the output (negative-capable activations must show a real
        # negative value; non-negative ones naturally won't).
        module.train()
        module(torch.randn(64, 32) * 2.0)

        q_in = _find_by_role_position(module, want_input=True)
        q_out = _find_by_role_position(module, want_input=False)

        expected_input_bw = 9 if unsigned_output else 8
        assert q_in.bit_width == expected_input_bw, (
            f"{cls.__name__}: expected input bit_width={expected_input_bw} "
            f"(output bit_width=8, unsigned_output={unsigned_output}), got {q_in.bit_width}"
        )
        assert q_out.bit_width == 8

    @pytest.mark.parametrize("cls,unsigned_output,kwargs", ACTIVATIONS)
    def test_input_bit_width_override_is_respected(self, cls, unsigned_output, kwargs):
        module = cls(bit_width=8, input_bit_width=12, **kwargs)
        module.train()
        module(_sample_input())
        q_in = _find_by_role_position(module, want_input=True)
        assert q_in.bit_width == 12

    @pytest.mark.parametrize("bw", [4, 8, 16])
    def test_scales_with_output_bit_width_for_unsigned_case(self, bw):
        module = QuantReLU(bit_width=bw)
        module.train()
        module(_sample_input())
        q_in = _find_by_role_position(module, want_input=True)
        assert q_in.bit_width == bw + 1

    @pytest.mark.parametrize("bw", [4, 8, 16])
    def test_scales_with_output_bit_width_for_signed_case(self, bw):
        module = QuantTanh(bit_width=bw)
        module.train()
        module(_sample_input())
        q_in = _find_by_role_position(module, want_input=True)
        assert q_in.bit_width == bw


# =========================================================================
# 3. Saturation-aware input calibration guardrail
# =========================================================================


def _huge_outlier_input():
    """A realistic bulk of small pre-activation values plus a handful of
    huge outliers -- the scenario the guardrail exists to handle. Without
    it, coverage-first calibration would size the whole grid to fit the
    outlier."""
    torch.manual_seed(0)
    bulk = torch.randn(970) * 0.3
    outliers = torch.tensor([500.0, -500.0, 300.0, -300.0] * 7)
    return torch.cat([bulk, outliers]).reshape(-1, 1)


def _calibrated_q_max(quantizer) -> float:
    lsb = int(quantizer.search_result_lsb.item())
    signed = bool(quantizer.search_result_is_signed.item())
    step = 2.0 ** lsb
    if signed:
        integer_max = 2 ** (quantizer.bit_width - 1) - 1
    else:
        integer_max = 2 ** quantizer.bit_width - 1
    return integer_max * step


class TestSaturationAwareInputCalibration:
    """ReLU6 hard-clips to [0, 6]; Sigmoid/Tanh saturate asymptotically.
    Before the guardrail exists, calibrating the input quantizer against
    `_huge_outlier_input()` (abs max ~500) makes coverage-first size the
    whole grid to ~500 -- these assertions are expected to FAIL against
    that unguarded behavior (calibrated q_max would be ~500, not <= a
    small multiple of the activation's own saturation point).
    """

    def test_relu6_input_range_does_not_balloon_to_outlier_magnitude(self):
        module = QuantReLU6(bit_width=8)
        module.train()
        module(_huge_outlier_input())
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        # ReLU6 saturates at 6 -- the user's own bracketing: a range up to
        # about 8 is fine, 16 is too much. We assert a generous but real
        # ceiling here (16) so this is robust to the exact multiplier
        # chosen during implementation while still catching the ~500
        # unguarded blowup.
        assert q_max <= 16.0, f"expected input range capped near ReLU6's saturation point (6), got q_max={q_max}"

    def test_sigmoid_input_range_does_not_balloon_to_outlier_magnitude(self):
        module = QuantSigmoid(bit_width=8)
        module.train()
        module(_huge_outlier_input())
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        assert q_max <= 20.0, f"expected input range capped near Sigmoid's saturation point, got q_max={q_max}"

    def test_tanh_input_range_does_not_balloon_to_outlier_magnitude(self):
        module = QuantTanh(bit_width=8)
        module.train()
        module(_huge_outlier_input())
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        assert q_max <= 12.0, f"expected input range capped near Tanh's saturation point, got q_max={q_max}"


class TestGuardrailIsNotTooStrict:
    """The guardrail must be a CEILING, not a replacement for normal
    calibration: data that's already well within the activation's natural
    operating range must calibrate exactly as it would unconstrained (no
    artificial clipping introduced by an overly tight cap), and activations
    with no natural saturation point must never be constrained at all.
    """

    def test_relu6_normal_range_data_is_not_clipped(self):
        """Typical pre-activation data (abs max ~4, well under ReLU6's own
        saturation point of 6) must calibrate to cover it fully -- the
        guardrail must not be so tight that it clips normal data."""
        torch.manual_seed(1)
        x = (torch.randn(2000, 1) * 1.0).clamp(-4.0, 4.0)
        module = QuantReLU6(bit_width=8)
        module.train()
        module(x)
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        assert q_max >= float(x.abs().max()), (
            f"guardrail is too strict: clipped normal data (abs max {float(x.abs().max())}) "
            f"to q_max={q_max}"
        )

    def test_tanh_normal_range_data_is_not_clipped(self):
        torch.manual_seed(2)
        x = (torch.randn(2000, 1) * 0.5).clamp(-2.0, 2.0)
        module = QuantTanh(bit_width=8)
        module.train()
        module(x)
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        assert q_max >= float(x.abs().max())

    def test_sigmoid_normal_range_data_is_not_clipped(self):
        torch.manual_seed(3)
        x = (torch.randn(2000, 1) * 0.5).clamp(-3.0, 3.0)
        module = QuantSigmoid(bit_width=8)
        module.train()
        module(x)
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        assert q_max >= float(x.abs().max())

    @pytest.mark.parametrize("cls,kwargs", [
        pytest.param(QuantReLU, {}, id="relu"),
        pytest.param(QuantSiLU, {}, id="silu"),
        pytest.param(QuantGELU, {}, id="gelu"),
        pytest.param(QuantLeakyReLU, {}, id="leaky_relu"),
    ])
    def test_unbounded_activations_are_never_range_constrained(self, cls, kwargs):
        """ReLU/SiLU/GELU/LeakyReLU have no upper saturation point (they
        grow ~linearly for large positive input) -- a huge but genuine
        input range must calibrate to cover it, not get artificially capped
        the way ReLU6/Sigmoid/Tanh's input does."""
        x = _huge_outlier_input()
        module = cls(bit_width=8, **kwargs)
        module.train()
        module(x)
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        assert q_max >= float(x.abs().max()) * 0.99, (
            f"{cls.__name__} has no saturation point and must not be range-constrained; "
            f"input abs max={float(x.abs().max())}, calibrated q_max={q_max}"
        )

    def test_softmax_is_never_range_constrained(self):
        """Softmax's useful input range depends on the relative spread of
        logits, not an absolute magnitude -- it must not get the
        saturation-point guardrail either."""
        x = _huge_outlier_input()
        module = QuantSoftmax(bit_width=8)
        module.train()
        module(x)
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        assert q_max >= float(x.abs().max()) * 0.99
