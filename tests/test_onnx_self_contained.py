"""
Tests that exported Quantify ONNX models are self-contained: loadable and
executable with plain `onnxruntime.InferenceSession`, with ZERO custom op
registration -- see pitfall #22 in docs/llm/pitfalls/brevitas_pitfalls.md
and utils/onnx_self_contained.py.

Every custom `Quantify::*` node this project emits
(FixedPointQuant, QuantSiLU, Relu6, SiLU, GELU, CoefficientQuant) is
covered: the model is exported, loaded with raw `onnx.load` +
`onnx.checker.check_model`, run through `onnxruntime.InferenceSession`
(explicitly only `CPUExecutionProvider`, no custom ops registered), and the
result compared against the eager PyTorch output.
"""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import brevitas.nn as qnn

from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorWeightQuant,
    FixedPointPerTensorActivationQuant,
    RoundingMode,
)
from quantizers import QuantSiLUActivationQuant, CoefficientPerTensorWeightQuant
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
from utils.onnx_export import export_onnx_with_io
from utils.onnx_self_contained import embed_self_contained_functions, _used_quantify_op_types

torch.manual_seed(0)


def _export_and_run(module: torch.nn.Module, x: torch.Tensor):
    """Export, load raw, check, run via plain ORT (no custom ops), return
    (onnx_model, ort_output) -- the caller compares against its own eager
    reference."""
    f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    f.close()
    try:
        export_onnx_with_io(
            module, x, f.name, opset_version=17, custom_opsets={"Quantify": 1},
            dynamo=False, input_names=["x"], output_names=["y"],
        )
        onnx_model = onnx.load(f.name)
        onnx.checker.check_model(onnx_model)
        sess = ort.InferenceSession(f.name, providers=["CPUExecutionProvider"])
        (out,) = sess.run(None, {"x": x.numpy()})
        return onnx_model, out
    finally:
        os.remove(f.name)


def _function_names(onnx_model) -> set:
    return {fn.name for fn in onnx_model.functions}


# =========================================================================
# 1. Quantify::FixedPointQuant -- weights, across rounding modes / narrow_range
# =========================================================================


class TestFixedPointQuantSelfContained:
    @pytest.mark.parametrize("rm", [
        RoundingMode.FLOOR, RoundingMode.ROUND, RoundingMode.ROUND_TO_NEAREST_EVEN,
    ])
    def test_weight_quant_matches_eager_across_rounding_modes(self, rm):
        # NOTE: the class attribute below must not be named `rm` (or
        # `rounding_mode`) too -- inside a class body, `rounding_mode =
        # rounding_mode` resolves the RHS in the (not-yet-populated)
        # class namespace first and raises NameError, since class bodies
        # don't close over the enclosing function's locals like nested
        # functions do.
        class WQ(FixedPointPerTensorWeightQuant):
            bit_width = 6
            rounding_mode = rm

        layer = qnn.QuantConv2d(3, 4, 3, padding=1, bias=False, weight_quant=WQ)
        x = torch.randn(1, 3, 8, 8) * 3
        layer.train(); layer(x); layer.eval()
        with torch.no_grad():
            eager = layer(x)

        onnx_model, out = _export_and_run(layer, x)
        assert "FixedPointQuant" in _function_names(onnx_model)
        assert np.allclose(out, eager.numpy(), atol=1e-5)

    def test_narrow_range_weight_quant_matches_eager(self):
        class WQ(FixedPointPerTensorWeightQuant):
            bit_width = 4
            narrow_range = True

        layer = qnn.QuantConv2d(3, 4, 3, padding=1, bias=False, weight_quant=WQ)
        x = torch.randn(1, 3, 8, 8) * 3
        layer.train(); layer(x); layer.eval()
        with torch.no_grad():
            eager = layer(x)

        onnx_model, out = _export_and_run(layer, x)
        assert np.allclose(out, eager.numpy(), atol=1e-5)

    def test_unsigned_activation_quant_matches_eager(self):
        layer = qnn.QuantConv2d(
            3, 4, 3, padding=1, bias=False,
            weight_quant=FixedPointPerTensorWeightQuant,
            output_quant=FixedPointPerTensorActivationQuant,
        )
        x = torch.randn(1, 3, 8, 8)
        layer.train(); layer(x); layer.eval()
        with torch.no_grad():
            eager = layer(x)

        onnx_model, out = _export_and_run(layer, x)
        assert np.allclose(out, eager.numpy(), atol=1e-5)


# =========================================================================
# 2. The 8 quantizers/activations.py modules -- input quant + activation +
#    output quant, all self-contained
# =========================================================================


ACTIVATION_CASES = [
    pytest.param(QuantReLU, {}, id="relu"),
    pytest.param(QuantReLU6, {}, id="relu6"),
    pytest.param(QuantSigmoid, {}, id="sigmoid"),
    pytest.param(QuantTanh, {}, id="tanh"),
    pytest.param(QuantSiLU, {}, id="silu"),
    pytest.param(QuantGELU, {"approximate": "none"}, id="gelu_none"),
    pytest.param(QuantGELU, {"approximate": "tanh"}, id="gelu_tanh"),
    pytest.param(QuantLeakyReLU, {}, id="leaky_relu"),
    pytest.param(QuantSoftmax, {}, id="softmax"),
]


class TestActivationsSelfContained:
    @pytest.mark.parametrize("cls,kwargs", ACTIVATION_CASES)
    def test_matches_eager(self, cls, kwargs):
        module = cls(bit_width=8, **kwargs)
        x = torch.tensor([[-10.0, -1.5, -0.1, 0.0, 0.1, 1.5, 3.0, 10.0]] * 4) + torch.randn(4, 8) * 0.05
        module.train()
        module(x)  # calibrate
        module.eval()
        with torch.no_grad():
            eager = module(x)

        onnx_model, out = _export_and_run(module, x)
        assert np.allclose(out, eager.numpy(), atol=1e-4), f"{cls.__name__}{kwargs}: mismatch"
        # Every Quantify::* op_type actually used must have gotten a function.
        used = _used_quantify_op_types(onnx_model.graph)
        assert used <= _function_names(onnx_model)


# =========================================================================
# 3. Quantify::QuantSiLU (legacy fused silu_quant.py)
# =========================================================================


class TestQuantSiLUSelfContained:
    def test_function_body_matches_eager_in_isolation(self):
        """`SiLUTensorQuant`'s actual export (via any wrapping -- a bare
        DummyModel, same as tests/test_silu_quant.py, or
        QuantConv2d(output_quant=QuantSiLUActivationQuant)) currently hits
        a pre-existing, unrelated legacy-ONNX-exporter bug: the same
        "outerNode->outputs().size() == node->inputs().size()" INTERNAL
        ASSERT already failing 3 tests in tests/test_silu_quant.py. That
        bug is in PyTorch's own TorchScript-based exporter handling of this
        Function's symbolic(), not in the embedded FunctionProto -- so this
        test validates the FunctionProto's math directly, by hand-building
        an isolated ONNX graph that calls `Quantify::QuantSiLU` with
        concrete attributes (bypassing torch.onnx.export entirely) and
        comparing against the eager SiLU+fixed-point-quantize computation.
        See test_full_export_is_blocked_by_pre_existing_exporter_bug below
        for the xfail documenting the blocked path.
        """
        from onnx import helper, TensorProto
        from quantizers.fixedpoint_per_tensor import quantize_fixed_point, RoundingMode
        from utils.onnx_self_contained import _build_quant_silu_function, DOMAIN

        fn = _build_quant_silu_function()
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])
        y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])
        node = helper.make_node(
            "QuantSiLU", ["x"], ["y"], domain=DOMAIN,
            scale=0.0625, integer_min=0.0, integer_max=255.0, rounding_mode_code=2,
            lsb=-4, bit_width=8, signed=0, rounding_mode="round_to_nearest_even",
        )
        graph = helper.make_graph([node], "g", [x_info], [y_info])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(DOMAIN, 1)],
            functions=[fn],
        )
        model.ir_version = 9
        onnx.checker.check_model(model)

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        x = torch.linspace(-5, 5, 11)
        (out,) = sess.run(None, {"x": x.numpy().astype(np.float32)})

        eager = quantize_fixed_point(
            torch.nn.functional.silu(x), lsb=-4, bit_width=8, signed=False,
            rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN,
        )
        assert np.allclose(out, eager.numpy(), atol=1e-5)

    @pytest.mark.xfail(
        reason="SiLUTensorQuant's ONNX export hits a pre-existing, unrelated "
               "legacy-exporter bug (same as 3 failures in test_silu_quant.py) -- "
               "not caused by the self-contained-functions feature; see "
               "test_function_body_matches_eager_in_isolation above for the "
               "function-body-level verification instead.",
        strict=True,
    )
    def test_full_export_is_blocked_by_pre_existing_exporter_bug(self):
        from quantizers.silu_quant import SiLUTensorQuant

        class DummyModel(torch.nn.Module):
            def __init__(self, quantizer):
                super().__init__()
                self.quantizer = quantizer

            def forward(self, x):
                q, s, z, b = self.quantizer(x)
                return q

        quantizer = SiLUTensorQuant(bit_width=8)
        x = torch.randn(1, 3, 8, 8) * 3
        model = DummyModel(quantizer)
        model.train(); model(x); model.eval()
        with torch.no_grad():
            eager = model(x)

        onnx_model, out = _export_and_run(model, x)
        assert "QuantSiLU" in _function_names(onnx_model)
        assert np.allclose(out, eager.numpy(), atol=1e-4)


# =========================================================================
# 4. Quantify::CoefficientQuant
# =========================================================================


class TestCoefficientQuantSelfContained:
    def test_matches_eager(self, tmp_path):
        coeffs_path = tmp_path / "coeffs.txt"
        coeffs_path.write_text("-1.0 -0.5 0.0 0.5 1.0\n")

        class CQ(CoefficientPerTensorWeightQuant):
            filepath = str(coeffs_path)

        layer = qnn.QuantConv2d(3, 4, 3, padding=1, bias=False, weight_quant=CQ)
        x = torch.randn(1, 3, 8, 8) * 3
        layer.train(); layer(x); layer.eval()
        with torch.no_grad():
            eager = layer(x)

        onnx_model, out = _export_and_run(layer, x)
        assert "CoefficientQuant" in _function_names(onnx_model)
        assert np.allclose(out, eager.numpy(), atol=1e-4)


# =========================================================================
# 5. embed_self_contained_functions() behavior directly
# =========================================================================


class TestEmbedSelfContainedFunctions:
    def test_idempotent(self):
        layer = qnn.QuantConv2d(3, 4, 3, padding=1, bias=False, weight_quant=FixedPointPerTensorWeightQuant)
        x = torch.randn(1, 3, 8, 8)
        layer.train(); layer(x); layer.eval()

        f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        f.close()
        try:
            export_onnx_with_io(
                layer, x, f.name, opset_version=17, custom_opsets={"Quantify": 1},
                dynamo=False, self_contained=False,
            )
            m = onnx.load(f.name)
            assert len(m.functions) == 0

            embed_self_contained_functions(m)
            n_after_first = len(m.functions)
            assert n_after_first == 1

            embed_self_contained_functions(m)  # must not duplicate
            assert len(m.functions) == n_after_first
        finally:
            os.remove(f.name)

    def test_self_contained_false_leaves_model_without_functions(self):
        layer = qnn.QuantConv2d(3, 4, 3, padding=1, bias=False, weight_quant=FixedPointPerTensorWeightQuant)
        x = torch.randn(1, 3, 8, 8)
        layer.train(); layer(x); layer.eval()

        f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        f.close()
        try:
            export_onnx_with_io(
                layer, x, f.name, opset_version=17, custom_opsets={"Quantify": 1},
                dynamo=False, self_contained=False,
            )
            m = onnx.load(f.name)
            assert len(m.functions) == 0
            # Still a valid ONNX file -- just not runnable by plain ORT for
            # the custom nodes (pitfall #8).
            onnx.checker.check_model(m)
        finally:
            os.remove(f.name)

    def test_model_with_no_custom_ops_is_untouched(self):
        """A model using only standard Brevitas quantizers (no Quantify
        custom nodes at all) must come back with zero embedded functions --
        embed_self_contained_functions must not add anything unneeded."""
        import torch.nn as nn

        layer = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        x = torch.randn(2, 4)
        f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        f.close()
        try:
            torch.onnx.export(layer, x, f.name, opset_version=17, dynamo=False)
            m = onnx.load(f.name)
            embed_self_contained_functions(m)
            assert len(m.functions) == 0
        finally:
            os.remove(f.name)
