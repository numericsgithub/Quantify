"""ONNX exports are always float64 (no float32 rounding vs. bit-exact hardware)."""
import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
from onnx import TensorProto

from quantizers.layers import QuantConv2d
from quantizers.activations import QuantGELU, QuantHardSigmoid, QuantReLU
from quantizers import FixedPointPerTensorWeightQuant
from tests.onnx_run import run_onnx
from utils.onnx_export import export_onnx_with_io, load_embedded_io

WQ = dict(weight_quant=FixedPointPerTensorWeightQuant, bias=False)


def _all_tensor_types(m: onnx.ModelProto):
    """Every float-ish dtype appearing in the graph (io, initializers, constants)."""
    types = {i.type.tensor_type.elem_type for i in list(m.graph.input) + list(m.graph.output)}
    types |= {t.data_type for t in m.graph.initializer}
    for n in m.graph.node:
        for a in n.attribute:
            if a.HasField("t"):
                types.add(a.t.data_type)
    return {t for t in types if t in (TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.FLOAT16)}


def _calibrated(model, x):
    model.train()
    model(x)
    model.eval()
    return model


def _export(model, x, path):
    return export_onnx_with_io(model, x, str(path), input_names=["x"], output_names=["y"],
                               canonical_checks="off")


def test_graph_is_float64_everywhere(tmp_path):
    model = nn.Sequential(QuantConv2d(1, 2, 3, padding="same", **WQ), QuantGELU(), QuantHardSigmoid())
    x = torch.randn(1, 1, 8, 8)
    _calibrated(model, x)
    m = _export(model, x, tmp_path / "m.onnx")
    assert _all_tensor_types(m) == {TensorProto.DOUBLE}


def test_embedded_reference_io_is_float64(tmp_path):
    model = _calibrated(nn.Sequential(QuantConv2d(1, 2, 3, **WQ), QuantReLU()), torch.randn(1, 1, 6, 6))
    x = torch.randn(1, 1, 6, 6)
    _export(model, x, tmp_path / "m.onnx")
    inp, out = load_embedded_io(str(tmp_path / "m.onnx"))
    assert inp.dtype == np.float64 and out.dtype == np.float64
    assert np.array_equal(inp, x.double().numpy())


def test_live_model_dtypes_restored(tmp_path):
    model = _calibrated(nn.Sequential(QuantConv2d(1, 2, 3, **WQ), QuantReLU()), torch.randn(1, 1, 6, 6))
    before = {k: v.dtype for k, v in model.state_dict().items()}
    x = torch.randn(1, 1, 6, 6)
    _export(model, x, tmp_path / "m.onnx")
    assert {k: v.dtype for k, v in model.state_dict().items()} == before
    assert model(x).dtype == torch.float32  # still usable as a float32 model


@pytest.mark.parametrize("act", [QuantGELU, QuantHardSigmoid, QuantReLU])
def test_onnx_output_equals_float64_eager_exactly(tmp_path, act):
    """The float64 graph reproduces the float64 eager model bit for bit,
    including at quantizer rounding ties (where float32 constants used to flip it)."""
    torch.manual_seed(0)
    model = nn.Sequential(QuantConv2d(1, 3, 3, padding="same", **WQ), act())
    x = torch.randn(4, 1, 12, 12) * 3
    _calibrated(model, x)
    _export(model, x, tmp_path / "m.onnx")
    (out,) = run_onnx(str(tmp_path / "m.onnx"), {"x": x.numpy()})
    # Reference: the same model evaluated in float64 eager mode.
    ref = model.double()(x.double()).detach().numpy()
    assert out.dtype == np.float64
    assert np.array_equal(out, ref)
