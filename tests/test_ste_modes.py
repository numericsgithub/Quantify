"""
Tests for the three STE clipping modes selected by a quantizer's `clipped_ste`
setting:

    "not_clip" (False)       plain STE: slope 1 everywhere, even where the
                             forward clamp saturated
    "clip" (True)            clipped STE: slope 0 for saturated inputs
    "single_direction_clip"  slope 0 for saturated inputs only when the
                             gradient pushes them further out; a gradient
                             pointing back into the range passes

The quantizer under test is pinned to a known grid (10-bit signed, lsb=-7 ->
range [-4.0, 3.9921875]) instead of calibrated, so the range limits are exact.
Upstream gradients are injected directly via L = sum(g * q(w)), so dL/dq == g
and the expected w.grad per element is a plain truth table.
"""
import os

import pytest
import torch
import brevitas.nn as qnn

from quantizers import FixedPointPerTensorWeightQuant
from quantizers.base_quantizer import BaseQuantizer, STE_MODES, normalize_ste_mode
from quantizers.coefficient_per_tensor_weights import CoefficientPerTensorWeightQuantizer
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer
from quantizers.manager import QuantizerManager
from quantizers.silu_quant import SiLUTensorQuant

BIT_WIDTH = 10
LSB = -7
Q_MIN = -(2 ** (BIT_WIDTH - 1)) * 2.0 ** LSB      # -4.0
Q_MAX = (2 ** (BIT_WIDTH - 1) - 1) * 2.0 ** LSB   # 3.9921875

# below range, lower limit, in range, upper limit, above range
W = [-6.0, Q_MIN, 1.0, Q_MAX, 6.0]
SIDE = [-1, 0, 0, 0, +1]


@pytest.fixture(autouse=True)
def reset_manager():
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


def _pinned_quantizer(mode) -> FixedPointPerTensorQuantizer:
    q = FixedPointPerTensorQuantizer(bit_width=BIT_WIDTH, signed=True, clipped_ste=mode,
                                     quantizer_role="weight")
    q.search_result_lsb.fill_(LSB)
    q.search_result_is_signed.fill_(True)
    q.signed = True
    q.set_search_done(True)
    return q.train()


def _expected_grad(mode, side, g):
    if side == 0 or mode == "not_clip":
        return g
    if mode == "clip":
        return 0.0
    # single_direction_clip: pass only if descent (w -= lr*g) moves it inward
    return g if (side > 0 and g > 0) or (side < 0 and g < 0) else 0.0


# ---------------------------------------------------------------------------
# Mode parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, mode", [
    (False, "not_clip"), (True, "clip"),
    ("not_clip", "not_clip"), ("clip", "clip"),
    ("single_direction_clip", "single_direction_clip"),
])
def test_normalize_ste_mode(value, mode):
    assert normalize_ste_mode(value) == mode
    q = FixedPointPerTensorQuantizer(clipped_ste=value)
    assert q.ste_mode == mode
    # Backward-compatible bool view: True for either clipping mode.
    assert q.clipped_ste == (mode != "not_clip")


@pytest.mark.parametrize("value", ["single_direction", "CLIP", 1, None])
def test_invalid_ste_mode_rejected(value):
    with pytest.raises(ValueError, match="clipped_ste"):
        FixedPointPerTensorQuantizer(clipped_ste=value)


def test_mode_can_be_changed_by_attribute_assignment():
    q = FixedPointPerTensorQuantizer()
    assert q.ste_mode == "not_clip"
    q.clipped_ste = "single_direction_clip"
    assert q.ste_mode == "single_direction_clip"
    q.clipped_ste = True
    assert q.ste_mode == "clip"
    with pytest.raises(ValueError):
        q.clipped_ste = "nope"


def test_all_modes_listed():
    assert STE_MODES == ("not_clip", "clip", "single_direction_clip")


# ---------------------------------------------------------------------------
# Backward truth table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("g", [+1.0, -1.0], ids=["grad_pos", "grad_neg"])
@pytest.mark.parametrize("mode", STE_MODES)
def test_backward_truth_table(mode, g):
    """Per element: in range (incl. both limits) every mode passes g;
    outside, not_clip passes g, clip passes 0, single_direction_clip passes g
    only when it points back into the range."""
    q = _pinned_quantizer(mode)
    w = torch.tensor(W, requires_grad=True)
    qw = q(w)[0]
    assert qw.tolist() == [Q_MIN, Q_MIN, 1.0, Q_MAX, Q_MAX]  # forward identical in every mode
    (qw * g).sum().backward()
    expected = [_expected_grad(mode, s, g) for s in SIDE]
    assert w.grad.tolist() == expected


def test_forward_values_identical_across_modes():
    w = torch.linspace(-7.0, 7.0, 57)
    outs = [_pinned_quantizer(mode)(w)[0] for mode in STE_MODES]
    for out in outs[1:]:
        assert torch.equal(out, outs[0])


# ---------------------------------------------------------------------------
# The training story from examples/ste_clipping_demo.py, as assertions
# ---------------------------------------------------------------------------

def _train_up_then_down(mode, steps_up=30, steps_down=60, lr=0.02):
    """y = q(w) + b, SGD on MSE: target +20 (pushes w past Q_MAX), then -20."""
    q = _pinned_quantizer(mode)
    w = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=lr)
    history = []
    for target, n in ((20.0, steps_up), (-20.0, steps_down)):
        for _ in range(n):
            opt.zero_grad()
            loss = (q(w)[0] + b - target) ** 2
            loss.backward()
            opt.step()
            history.append(w.item())
    return history[:steps_up], history[steps_up:]


def test_not_clip_drifts_outward_and_lags_back():
    up, down = _train_up_then_down("not_clip")
    assert up[-1] > Q_MAX + 5.0, "plain STE should let the clipped float weight drift far out"
    # On the way back it needs several steps just to re-enter the range.
    steps_to_reenter = next(i for i, v in enumerate(down) if v <= Q_MAX)
    assert steps_to_reenter >= 5


def test_clip_gets_stuck_for_good():
    up, down = _train_up_then_down("clip")
    stuck_at = up[-1]
    assert Q_MAX < stuck_at < Q_MAX + 1.0, "clipped STE stops the drift right past the limit"
    assert all(v == stuck_at for v in down), "clipped STE never lets the weight back in"


def test_single_direction_clip_no_drift_and_recovers_immediately():
    up, down = _train_up_then_down("single_direction_clip")
    stopped_at = up[-1]
    assert Q_MAX < stopped_at < Q_MAX + 1.0, "outward drift must be stopped like 'clip'"
    assert len(set(up[3:])) == 1, "no outward movement once clipped"
    assert down[0] < Q_MAX, "first inward gradient must bring it straight back into range"
    # And it does not run off the other end either: stopped just past Q_MIN.
    assert Q_MIN - 1.0 < down[-1] <= Q_MIN


# ---------------------------------------------------------------------------
# Wiring: Brevitas injector, other quantizers, ONNX export
# ---------------------------------------------------------------------------

def _single_weight_layer(mode):
    weight_quant = type("W10", (FixedPointPerTensorWeightQuant,),
                        {"bit_width": BIT_WIDTH, "clipped_ste": mode})
    layer = qnn.QuantLinear(1, 1, bias=True, weight_quant=weight_quant, return_quant_tensor=False)
    (q,) = [m for m in layer.modules() if isinstance(m, BaseQuantizer)]
    q.search_result_lsb.fill_(LSB)
    q.search_result_is_signed.fill_(True)
    q.signed = True
    q.set_search_done(True)
    return layer.train(), q


@pytest.mark.parametrize("g, expected", [(+1.0, 1.0), (-1.0, 0.0)])
def test_mode_set_on_brevitas_injector(g, expected):
    layer, q = _single_weight_layer("single_direction_clip")
    assert q.ste_mode == "single_direction_clip"
    with torch.no_grad():
        layer.weight.fill_(6.0)  # above Q_MAX
    (layer(torch.ones(1, 1)) * g).sum().backward()
    assert layer.weight.grad.item() == expected


def test_invalid_mode_on_injector_rejected():
    with pytest.raises(ValueError, match="clipped_ste"):
        _single_weight_layer("sideways")


def _coefficient_quantizer(mode, tmp_path):
    path = tmp_path / "coeffs.txt"
    path.write_text("0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5\n")
    return CoefficientPerTensorWeightQuantizer(filepath=str(path), bit_width=3, clipped_ste=mode)


@pytest.mark.parametrize("name", ["coefficient", "silu"])
@pytest.mark.parametrize("g, expected", [(+1.0, 1.0), (-1.0, 0.0)])
def test_other_quantizers_single_direction(name, g, expected, tmp_path):
    """Above-range input: the gradient passes only if it points back inward."""
    if name == "coefficient":
        q = _coefficient_quantizer("single_direction_clip", tmp_path)
    else:
        q = SiLUTensorQuant(bit_width=3, clipped_ste="single_direction_clip")
    q.train()
    with torch.no_grad():
        q(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]))  # calibrate
    assert q.search_done_value
    w = torch.tensor(10.0, requires_grad=True)  # far above both quantizers' range
    (q(w)[0] * g).backward()
    if name == "silu":
        expected *= torch.sigmoid(w) * (1 + w * (1 - torch.sigmoid(w)))  # silu'(10)
        expected = expected.item()
    assert w.grad.item() == pytest.approx(expected)


@pytest.mark.parametrize("mode", STE_MODES)
def test_onnx_export_unaffected(mode, tmp_path):
    from utils.onnx_export import export_onnx_with_io
    layer, _ = _single_weight_layer(mode)
    layer.eval()
    export_onnx_with_io(layer, torch.ones(1, 1), str(tmp_path / f"{mode}.onnx"))
    assert os.path.getsize(tmp_path / f"{mode}.onnx") > 0
