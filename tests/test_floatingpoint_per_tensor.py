import math
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from brevitas.nn import QuantLinear
from quantizers import (
    FloatingPointPerTensorActivationQuant,
    FloatingPointPerTensorBiasQuant,
    FloatingPointPerTensorQuantizer,
    FloatingPointPerTensorWeightQuant,
    quantize_floating_point,
)
from quantizers.floatingpoint_per_tensor import flopoco_format_info
from tests.onnx_run import run_onnx
from utils.onnx_export import export_onnx_with_io


def reference(x: float, wE: int, wF: int, overflow: str = "inf") -> float:
    """Independent exact-arithmetic model of the FloPoCo rules (Fraction)."""
    if math.isnan(x):
        return math.nan
    if math.isinf(x):
        return math.copysign(math.inf, x)
    if x == 0:
        return x
    bias = (1 << (wE - 1)) - 1
    emax = (1 << wE) - 1 - bias
    f = abs(Fraction(x))
    e = f.numerator.bit_length() - f.denominator.bit_length()
    if Fraction(2) ** e > f:
        e -= 1
    ulp = Fraction(2) ** (e - wF)
    q = f / ulp
    n = math.floor(q)
    rem = q - n
    n += 1 if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2) else 0
    y = n * ulp
    if y < Fraction(2) ** (-bias):
        return math.copysign(0.0, x)
    if y >= Fraction(2) ** (emax + 1):
        if overflow == "inf":
            return math.copysign(math.inf, x)
        return math.copysign(float((2 - Fraction(1, 2 ** wF)) * 2 ** emax), x)
    return math.copysign(float(y), x)


FORMATS = [(3, 4), (2, 1), (4, 3), (5, 10), (8, 23)]


@pytest.mark.parametrize("wE,wF", FORMATS)
@pytest.mark.parametrize("overflow", ["inf", "saturate"])
def test_matches_exact_reference(wE, wF, overflow):
    rng = np.random.default_rng(0)
    info = flopoco_format_info(wE, wF)
    span = 2.0 ** (info["emax"] + 2)
    xs = np.concatenate([
        rng.normal(size=300) * rng.choice([1e-3, 1, 30, span], size=300),
        # exact ties / boundaries
        [info["max_value"], info["max_value"] * (1 + 2.0 ** -(wF + 1)), info["min_value"],
         info["min_value"] * 0.75, info["min_value"] * 0.49, 0.0, -1.0, 1 + 2.0 ** -(wF + 1),
         1 + 3 * 2.0 ** -(wF + 1)],
    ])
    xs = np.concatenate([xs, -xs])
    got = quantize_floating_point(torch.tensor(xs, dtype=torch.float64), wE, wF, overflow).numpy()
    want = np.array([reference(float(v), wE, wF, overflow) for v in xs])
    np.testing.assert_array_equal(got, want)
    assert np.array_equal(np.signbit(got), np.signbit(want))


def test_documented_examples():
    q = lambda v, ov="inf": float(quantize_floating_point(torch.tensor(v, dtype=torch.float64), 3, 4, ov))
    assert q(8 + 1 / 4) == 8.0          # tie -> even
    assert q(8 + 3 / 4) == 9.0
    assert q(8 + 3 / 8) == 8.5
    assert q(31.5) == math.inf          # ties up to 32 -> overflow
    assert q(31.5, "saturate") == 31.0
    assert q(31.0) == 31.0              # largest finite value
    assert q(-0.0078) == 0.0 and math.copysign(1, q(-0.0078)) == -1  # flush keeps sign
    assert q(0.124) == 0.125            # within half an ulp: rounds up to the smallest normal, survives
    assert q(float("nan")) != q(float("nan"))


def test_dtype_preserved_and_nan_passthrough():
    x = torch.tensor([0.3, float("nan"), float("inf")], dtype=torch.float32)
    y = quantize_floating_point(x, 5, 10, "saturate")
    assert y.dtype == torch.float32 and torch.isnan(y[1]) and y[2] == flopoco_format_info(5, 10)["max_value"]


def test_quantizer_tuple_and_no_calibration_needed():
    qz = FloatingPointPerTensorQuantizer(exponent_bits=4, fraction_bits=3)
    assert qz.search_done_value
    qz.eval()  # no calibration pass required, even in eval mode
    out, scale, zp, bw = qz(torch.randn(5, 5))
    assert scale.item() == 1.0 and zp.item() == 0.0 and bw.item() == 4 + 3 + 3


def test_invalid_args():
    for kw in (dict(exponent_bits=1), dict(exponent_bits=11), dict(fraction_bits=0), dict(overflow="wrap")):
        with pytest.raises(ValueError):
            FloatingPointPerTensorQuantizer(**kw)


def test_ste_gradients():
    wE, wF = 3, 4  # max 31
    x = torch.tensor([0.5, 20.0, 40.0, -40.0, 0.01], requires_grad=True)

    def grad(mode):
        x.grad = None
        qz = FloatingPointPerTensorQuantizer(exponent_bits=wE, fraction_bits=wF, clipped_ste=mode)
        qz(x)[0].sum().backward()
        return x.grad.tolist()

    assert grad("not_clip") == [1, 1, 1, 1, 1]
    assert grad("clip") == [1, 1, 0, 0, 1]  # overflow masked; underflow (0.01) is not
    x.grad = None
    qz = FloatingPointPerTensorQuantizer(exponent_bits=wE, fraction_bits=wF, clipped_ste="single_direction_clip")
    (qz(x)[0] * torch.tensor([1.0, 1, 1, 1, 1])).sum().backward()  # descent pushes values down
    assert x.grad.tolist() == [1, 1, 1, 0, 1]  # +40 may come back in (grad>0); -40 pushed out


def test_brevitas_layer_integration():
    class W(FloatingPointPerTensorWeightQuant):
        exponent_bits, fraction_bits = 4, 3

    layer = QuantLinear(8, 4, bias=True, weight_quant=W, bias_quant=FloatingPointPerTensorBiasQuant)
    out = layer(torch.randn(2, 8))
    assert out.shape == (2, 4)
    w = layer.quant_weight().value
    assert torch.equal(w, quantize_floating_point(layer.weight, 4, 3, "saturate"))
    out.sum().backward()
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_activation_injector():
    from brevitas.nn import QuantIdentity
    act = QuantIdentity(act_quant=FloatingPointPerTensorActivationQuant, return_quant_tensor=False)
    x = torch.randn(4, 6)
    y = act(x)
    qz = act.act_quant.fused_activation_quant_proxy.tensor_quant
    assert qz.bit_width == 16 and qz.search_done_value
    assert torch.equal(y, quantize_floating_point(x, qz.exponent_bits, qz.fraction_bits, "saturate"))


@pytest.mark.parametrize("overflow", ["saturate", "inf"])
def test_onnx_export_runs_and_matches(tmp_path, overflow):
    OVF = overflow
    class W(FloatingPointPerTensorWeightQuant):
        exponent_bits, fraction_bits = 3, 4
        overflow = OVF

    layer = QuantLinear(6, 3, bias=False, weight_quant=W)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(3, 6) * 20)  # some overflow in (3,4)
        layer.weight[0, 0] = 31.5
        layer.weight[1, 1] = 0.01
    layer.eval()
    x = torch.randn(2, 6)
    m = export_onnx_with_io(layer, x, str(tmp_path / "fp.onnx"), input_names=["x"], output_names=["y"],
                            canonical_checks="off")
    ops = [(n.domain, n.op_type) for n in m.graph.node]
    assert ("Quantify", "FloatingPointQuant") in ops
    node = next(n for n in m.graph.node if n.op_type == "FloatingPointQuant")
    names = {a.name for a in node.attribute}
    assert {"exponent_bits", "fraction_bits", "exponent_bias", "emax", "overflow_inf", "min_value",
            "max_value"} <= names

    got = run_onnx(m, {"x": x.numpy()})[0]
    ref_w = quantize_floating_point(layer.weight.detach().double(), 3, 4, overflow)
    want = (x.double() @ ref_w.T).numpy()
    if overflow == "inf":
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
        np.testing.assert_array_equal(got[~np.isnan(got)], want[~np.isnan(want)])
    else:
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize("wE,wF", [(3, 4), (5, 10), (8, 23)])
@pytest.mark.parametrize("overflow", ["saturate", "inf"])
def test_onnx_function_equals_torch_on_many_values(wE, wF, overflow):
    """The embedded function body vs. quantize_floating_point, incl. specials."""
    import onnx
    from onnx import TensorProto, helper
    from utils.onnx_self_contained import embed_self_contained_functions

    info = flopoco_format_info(wE, wF)
    rng = np.random.default_rng(1)
    xs = np.concatenate([
        rng.normal(size=500) * 10.0 ** rng.integers(-6, 6, size=500),
        2.0 ** np.arange(-60, 60, dtype=np.float64),
        [0.0, np.inf, -np.inf, info["max_value"], info["min_value"], 31.5, 1 + 2.0 ** -(wF + 1)],
    ])
    xs = np.concatenate([xs, -xs])
    node = helper.make_node(
        "FloatingPointQuant", ["x"], ["y"], domain="Quantify",
        exponent_bits=wE, fraction_bits=wF, exponent_bias=info["bias"], emax=info["emax"],
        overflow_inf=int(overflow == "inf"))
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [len(xs)])],
        [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [len(xs)])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("Quantify", 1)])
    embed_self_contained_functions(model)
    got = run_onnx(model, {"x": xs})[0]
    want = quantize_floating_point(torch.tensor(xs), wE, wF, overflow).numpy()
    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------- search

def _search(x, **kw):
    qz = FloatingPointPerTensorQuantizer(**kw)
    assert not qz.search_done_value
    qz.train()
    qz(x)
    assert qz.search_done_value
    return qz


def test_search_picks_exponent_width_from_data_range():
    """Wide dynamic range needs more exponent bits than a narrow one."""
    narrow = _search(torch.randn(4000) * 0.5, bit_width=12, quantizer_role="weight")
    wide = _search(torch.randn(4000).double() * 10.0 ** torch.randint(-12, 12, (4000,)),
                   bit_width=12, quantizer_role="weight")
    assert narrow.exponent_bits < wide.exponent_bits
    for qz in (narrow, wide):
        assert qz.exponent_bits + qz.fraction_bits + 3 == 12


def test_search_is_optimal_among_allowed_candidates():
    torch.manual_seed(0)
    x = torch.randn(2000) * 3
    qz = _search(x, bit_width=13, quantizer_role="weight")
    best = None
    for we in range(2, 10):
        wf = 13 - 3 - we
        q = quantize_floating_point(x.double(), we, wf, "saturate")
        clip = (quantize_floating_point(x.double(), we, wf, "inf").abs() > flopoco_format_info(we, wf)["max_value"]).double().mean() * 100
        if clip <= 15.0:
            sad = (x.double() - q).abs().sum().item()
            if best is None or sad < best[0]:
                best = (sad, we)
    assert qz.exponent_bits == best[1]


def test_search_respects_clip_ceiling():
    # 20% of the values are huge: with a 15% weight ceiling the chosen format must cover them,
    # even though fewer exponent bits would give lower SAD on the small bulk.
    x = torch.cat([torch.randn(800) * 0.01, torch.full((200,), 1e4)])
    qz = _search(x, bit_width=14, quantizer_role="weight")
    assert qz.format_info["max_value"] >= 1e4
    # an activation (0% ceiling) must not clip either
    qa = _search(x, bit_width=14, quantizer_role="activation")
    assert qa.format_info["max_value"] >= 1e4


def test_search_falls_back_to_least_overflowing():
    # bit_width 9 -> wE in 2..5, max value 2^17 at best: 10% of the values (1e6) overflow
    # every candidate, but only wE=5 covers the 40% at 1e3 -> least-overflowing wins.
    x = torch.cat([torch.full((10,), 1e6), torch.full((40,), 1e3), torch.full((50,), 0.5)])
    qz = _search(x, bit_width=9, quantizer_role="bias")  # 0% ceiling is unreachable
    assert qz.exponent_bits == 5


def test_pinned_exponent_or_fraction_and_fixed_format():
    qz = FloatingPointPerTensorQuantizer(bit_width=14, exponent_bits=4)
    assert qz.search_done_value and (qz.exponent_bits, qz.fraction_bits) == (4, 7)
    qz = FloatingPointPerTensorQuantizer(bit_width=14, fraction_bits=6)
    assert qz.search_done_value and (qz.exponent_bits, qz.fraction_bits) == (5, 6)
    qz = FloatingPointPerTensorQuantizer(bit_width=99, exponent_bits=3, fraction_bits=4)
    assert qz.search_done_value and qz.bit_width == 10  # both pinned: bit_width derived


def test_search_result_survives_state_dict_and_export(tmp_path):
    torch.manual_seed(1)
    layer = QuantLinear(16, 4, bias=False, weight_quant=FloatingPointPerTensorWeightQuant)
    layer.train()
    layer(torch.randn(2, 16))
    qz = layer.weight_quant.tensor_quant
    we, wf = qz.exponent_bits, qz.fraction_bits
    assert we + wf + 3 == 16
    layer.eval()
    x = torch.randn(2, 16)
    m = export_onnx_with_io(layer, x, str(tmp_path / "s.onnx"), input_names=["x"], output_names=["y"],
                            canonical_checks="off")
    node = next(n for n in m.graph.node if n.op_type == "FloatingPointQuant")
    attrs = {a.name: a.i for a in node.attribute if a.name in ("exponent_bits", "fraction_bits")}
    assert attrs == {"exponent_bits": we, "fraction_bits": wf}
    sd = {k: v for k, v in layer.state_dict().items() if "search_result_exponent_bits" in k}
    assert sd and list(sd.values())[0].item() == we


def test_eval_before_calibration_raises_like_fixed_point():
    qz = FloatingPointPerTensorQuantizer(bit_width=12)
    qz.eval()
    with pytest.raises(RuntimeError, match="not been calibrated"):
        qz(torch.randn(4))
