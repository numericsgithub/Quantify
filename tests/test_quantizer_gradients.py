"""
Gradient-correctness tests for the Straight-Through Estimator (STE) in
FixedPointPerTensorQuantizer.

These are NOT training-runs. They use the smallest possible model,

    y_hat = quantizer(w) * x + b
    L     = (y_hat - y) ** 2

with hand-derived ground-truth gradients, to check whether the quantizer's
backward pass actually implements straight-through (local slope = 1) in
each of its three operating states: off (annealing_alpha=0), fully on
(annealing_alpha=1), and mid-anneal (annealing_alpha=0.5).

Hand-derived ground truth (do not "fix" these numbers if a test fails --
report the actual gradient instead, see module docstring in the PR):

    w = 2.0, b = 1.0, x = 3.0, y = 10.0
    y_hat = w * x + b = 7.0
    error = y_hat - y = -3.0
    L = error ** 2
    dL/dy_hat = 2 * error = -6.0
    dL/dw = dL/dy_hat * x = -6.0 * 3.0 = -18.0   (only holds if the
                                                    quantizer's local slope is 1,
                                                    i.e. a true STE)
    dL/db = dL/dy_hat * 1.0 = -6.0               (b never touches the quantizer,
                                                    so this must hold in every state)
"""
import pytest
import torch
import torch.nn as nn
import brevitas.nn as qnn

from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer, FixedPointPerTensorWeightQuant
from quantizers.manager import QuantizerManager

EXPECTED_W_GRAD = -18.0
EXPECTED_B_GRAD = -6.0
EXPECTED_Y_HAT = 7.0
TOL = 1e-4

# Two-layer formula: h = q1(w_1)*x + b_1 ; y_hat = h*q2(w_2) + b_2
# w_1=2, b_1=1, x=3, w_2=2, b_2=0, y=20
# h=7, y_hat=14, L=36, dL/dy_hat=-12
# dL/dw_2 = -12 * h   = -84   dL/db_2 = -12
# dL/dh   = -12 * 2   = -24   dL/dw_1 = -24 * x = -72   dL/db_1 = -24
TWO_LAYER_W1_GRAD = -72.0
TWO_LAYER_B1_GRAD = -24.0
TWO_LAYER_W2_GRAD = -84.0
TWO_LAYER_B2_GRAD = -12.0
TWO_LAYER_Y_HAT = 14.0

def print_graph(fn, depth=0, seen=None, input_idx=None):
    if seen is None:
        seen = set()
    if fn is None:
        return
    prefix = f"{depth:2d}: {'  ' * depth}"
    label = type(fn).__name__
    if input_idx is not None:
        label = f"[input {input_idx}] {label}"

    if fn in seen:
        print(prefix + label + "  (repeated, see above)")
        return
    seen.add(fn)

    # AccumulateGrad wraps a leaf tensor, show its actual identity
    if hasattr(fn, "variable"):
        var = fn.variable
        print(f"{prefix}{label}  shape={tuple(var.shape)} dtype={var.dtype} "
              f"requires_grad={var.requires_grad} id={id(var)}")
    else:
        print(prefix + label)

    for idx, (next_fn, output_nr) in enumerate(fn.next_functions):
        print_graph(next_fn, depth + 1, seen, input_idx=idx)

@pytest.fixture(autouse=True)
def reset_manager():
    """QuantizerManager is a singleton, so quantizers from a previous test
    would otherwise still be registered and get hit by enable_quantization()/
    disable_quantization() calls in this test."""
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


def _make_calibrated_quantizer(bit_width: int = 8) -> FixedPointPerTensorQuantizer:
    """Build a quantizer and run one warmup forward pass (in train mode, on
    data unrelated to the actual w/b/x/y test values) so that search_done
    becomes True before the gradient-measuring forward pass. The warmup
    data spans 0..4 (multiple unique values, required for search_done to
    flip True -- a single repeated value has only 1 unique quantized value
    and _save_calibration() never sets search_done in that case) and
    includes 2.0 exactly, so the resulting fixed-point grid represents
    w=2.0 exactly -- this keeps the forward value at 7.0 in every state,
    so a test failure can only be about gradients, not quantization
    rounding error.
    """
    q = FixedPointPerTensorQuantizer(bit_width=bit_width, signed=True)
    q.train()
    with torch.no_grad():
        q(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))  # triggers _calibrate() -> sets search_done=True
    assert q.search_done.item(), "calibration did not set search_done=True"
    return q


def test_sanity_baseline_no_quantizer():
    """Baseline with no quantizer at all: plain y_hat = w * x + b.
    This proves the test harness itself (loss, backward, expected numbers)
    is correct before trusting it to judge the quantizer."""
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(3.0)
    y = torch.tensor(10.0)

    y_hat = w * x + b
    assert torch.isclose(y_hat, torch.tensor(EXPECTED_Y_HAT), atol=TOL)

    loss = (y_hat - y) ** 2
    loss.backward()

    # x = torch.randn(4, requires_grad=True)
    # x_round = torch.round(x)
    # x_q = x + (x_round - x).detach()

    print("PRINT GRAPH")
    print_graph(loss.grad_fn)

    # No quantizer in the graph at all, so this must be exact.
    assert torch.isclose(w.grad, torch.tensor(EXPECTED_W_GRAD), atol=TOL)
    assert torch.isclose(b.grad, torch.tensor(EXPECTED_B_GRAD), atol=TOL)


def test_ste_off_annealing_alpha_zero():
    """annealing_alpha = 0, set via manager.disable_quantization(): the
    quantizer's output collapses to the pure-float pass-through branch
    `result = (1 - alpha) * x + alpha * quantized` with alpha=0, i.e. just
    `x`. The quantized branch isn't used at all, so the gradient must be
    exactly -18, same as the no-quantizer baseline."""
    q = _make_calibrated_quantizer()
    q.quantizer_manager.disable_quantization()
    assert q.annealing_alpha.item() == 0.0

    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(3.0)
    y = torch.tensor(10.0)

    quantized_w, _scale, _zero_point, _bit_width = q(w)
    y_hat = quantized_w * x + b
    assert torch.isclose(y_hat, torch.tensor(EXPECTED_Y_HAT), atol=TOL)

    loss = (y_hat - y) ** 2
    loss.backward()

    print(f"quantized_w: {quantized_w}")
    print("loss", loss)
    print("w.grad", w.grad)

    # x = torch.randn(4, requires_grad=True)
    # x_round = torch.round(x)
    # x_q = x + (x_round - x).detach()

    print("PRINT GRAPH")
    print_graph(loss.grad_fn)

    assert torch.isclose(w.grad, torch.tensor(EXPECTED_W_GRAD), atol=TOL), (
        f"OFF state: expected w.grad={EXPECTED_W_GRAD}, got {w.grad.item()}"
    )
    assert torch.isclose(b.grad, torch.tensor(EXPECTED_B_GRAD), atol=TOL)


def test_ste_on_annealing_alpha_one():
    """annealing_alpha = 1, set via manager.enable_quantization(): the
    quantizer's output is fully quantized (`result = quantized`). A correct
    Straight-Through Estimator makes the *local* gradient of the quantizer
    equal to 1 (the rounding/clamping is treated as identity for backprop
    purposes), so the gradient must still be exactly -18.

    If the quantizer instead lets autograd trace straight through
    torch.round()/torch.clamp() with no STE override, the local gradient
    collapses toward 0 and this assertion fails -- that is a real STE bug,
    not a tolerance issue."""
    q = _make_calibrated_quantizer()
    q.quantizer_manager.enable_quantization()
    assert q.annealing_alpha.item() == 1.0

    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(3.0)
    y = torch.tensor(10.0)

    quantized_w, _scale, _zero_point, _bit_width = q(w)
    y_hat = quantized_w * x + b
    assert torch.isclose(y_hat, torch.tensor(EXPECTED_Y_HAT), atol=TOL)

    loss = (y_hat - y) ** 2
    loss.backward()

    print(f"quantized_w: {quantized_w}")
    print("loss", loss)
    print("w.grad", w.grad)

    # x = torch.randn(4, requires_grad=True)
    # x_round = torch.round(x)
    # x_q = x + (x_round - x).detach()

    print("PRINT GRAPH")
    print_graph(loss.grad_fn)

    assert torch.isclose(w.grad, torch.tensor(EXPECTED_W_GRAD), atol=TOL), (
        f"ON state (STE check): expected w.grad={EXPECTED_W_GRAD}, "
        f"got {w.grad.item()} -- gradients are not flowing straight-through "
        f"the quantizer in the non-ONNX-export code path."
    )
    assert torch.isclose(b.grad, torch.tensor(EXPECTED_B_GRAD), atol=TOL)


def test_ste_annealing_alpha_half():
    """annealing_alpha = 0.5, set directly on the buffer (mid-anneal):
    `result = (1 - alpha) * x + alpha * quantized`. If the quantized
    branch has a correct STE local slope of 1, the blended slope is still
    1 (0.5 * 1 + 0.5 * 1), so the gradient must STILL be exactly -18.

    If the quantized branch instead has zero local gradient (no STE), the
    blended slope becomes 0.5 and the gradient becomes -9 instead of -18.
    That is a real bug -- this test reports the actual value rather than
    relaxing the expected number to match it."""
    q = _make_calibrated_quantizer()
    q.annealing_alpha.data.fill_(0.5)
    q.annealing_alpha_step = 0.0  # keep alpha pinned at 0.5 during this forward

    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(3.0)
    y = torch.tensor(10.0)

    quantized_w, _scale, _zero_point, _bit_width = q(w)
    y_hat = quantized_w * x + b
    assert torch.isclose(y_hat, torch.tensor(EXPECTED_Y_HAT), atol=TOL)

    loss = (y_hat - y) ** 2
    loss.backward()

    print(f"quantized_w: {quantized_w}")
    print("loss", loss)
    print("w.grad", w.grad)

    # x = torch.randn(4, requires_grad=True)
    # x_round = torch.round(x)
    # x_q = x + (x_round - x).detach()

    print("PRINT GRAPH")
    print_graph(loss.grad_fn)

    actual_w_grad = w.grad.item()
    assert torch.isclose(w.grad, torch.tensor(EXPECTED_W_GRAD), atol=TOL), (
        f"ANNEALING state (alpha=0.5): expected w.grad={EXPECTED_W_GRAD}, "
        f"got {actual_w_grad} -- a value around -9.0 means the quantized "
        f"branch contributes zero local gradient (no STE), so the blend "
        f"only passes through half of the true straight-through slope."
    )
    assert torch.isclose(b.grad, torch.tensor(EXPECTED_B_GRAD), atol=TOL)


# ---------------------------------------------------------------------------
# Two-layer disconnection tests
#
# These verify that a quantizer in layer 2 cannot sever the gradient path
# back to layer 1's weights (graph disconnection).  A disconnected graph
# would produce w_1.grad == None or w_1.grad ≈ 0.
#
# Formula:  h = q1(w_1) * x + b_1
#           y_hat = h * q2(w_2) + b_2
#           L = (y_hat - y) ** 2
# ---------------------------------------------------------------------------

def test_two_layer_no_quantizer():
    """Baseline with no quantizers: confirms the two-layer math before
    trusting it to judge quantizer-induced disconnection."""
    w_1 = torch.tensor(2.0, requires_grad=True)
    b_1 = torch.tensor(1.0, requires_grad=True)
    w_2 = torch.tensor(2.0, requires_grad=True)
    b_2 = torch.tensor(0.0, requires_grad=True)
    x   = torch.tensor(3.0)
    y   = torch.tensor(20.0)

    h     = w_1 * x + b_1
    y_hat = h * w_2 + b_2
    assert torch.isclose(y_hat, torch.tensor(TWO_LAYER_Y_HAT), atol=TOL)

    loss = (y_hat - y) ** 2
    loss.backward()

    print("PRINT GRAPH (two-layer baseline, no quantizer)")
    print_graph(loss.grad_fn)

    assert torch.isclose(w_1.grad, torch.tensor(TWO_LAYER_W1_GRAD), atol=TOL)
    assert torch.isclose(b_1.grad, torch.tensor(TWO_LAYER_B1_GRAD), atol=TOL)
    assert torch.isclose(w_2.grad, torch.tensor(TWO_LAYER_W2_GRAD), atol=TOL)
    assert torch.isclose(b_2.grad, torch.tensor(TWO_LAYER_B2_GRAD), atol=TOL)


def test_two_layer_ste_on():
    """annealing_alpha=1 (fully quantized): both q1 and q2 are active.
    Gradients must flow from y_hat all the way back through q2 and q1 to
    both w_1 and w_2 via the STE.

    A disconnection bug in q2 would leave w_1.grad as None or 0,
    because no gradient path reaches layer 1."""
    q1 = _make_calibrated_quantizer()
    q2 = _make_calibrated_quantizer()
    q1.quantizer_manager.enable_quantization()  # sets alpha=1 on both q1 and q2

    w_1 = torch.tensor(2.0, requires_grad=True)
    b_1 = torch.tensor(1.0, requires_grad=True)
    w_2 = torch.tensor(2.0, requires_grad=True)
    b_2 = torch.tensor(0.0, requires_grad=True)
    x   = torch.tensor(3.0)
    y   = torch.tensor(20.0)

    qw_1, *_ = q1(w_1)
    h         = qw_1 * x + b_1
    qw_2, *_ = q2(w_2)
    y_hat     = h * qw_2 + b_2
    assert torch.isclose(y_hat, torch.tensor(TWO_LAYER_Y_HAT), atol=TOL)

    loss = (y_hat - y) ** 2
    loss.backward()

    print(f"w_1.grad={w_1.grad}  b_1.grad={b_1.grad}")
    print(f"w_2.grad={w_2.grad}  b_2.grad={b_2.grad}")
    print("PRINT GRAPH (two-layer, both quantizers ON)")
    print_graph(loss.grad_fn)

    assert w_1.grad is not None, (
        "w_1.grad is None -- q2 disconnected the graph between layer 2 and layer 1"
    )
    assert torch.isclose(w_1.grad, torch.tensor(TWO_LAYER_W1_GRAD), atol=TOL), (
        f"w_1.grad={w_1.grad.item()}, expected {TWO_LAYER_W1_GRAD}"
    )
    assert torch.isclose(b_1.grad, torch.tensor(TWO_LAYER_B1_GRAD), atol=TOL), (
        f"b_1.grad={b_1.grad.item()}, expected {TWO_LAYER_B1_GRAD}"
    )
    assert torch.isclose(w_2.grad, torch.tensor(TWO_LAYER_W2_GRAD), atol=TOL), (
        f"w_2.grad={w_2.grad.item()}, expected {TWO_LAYER_W2_GRAD}"
    )
    assert torch.isclose(b_2.grad, torch.tensor(TWO_LAYER_B2_GRAD), atol=TOL), (
        f"b_2.grad={b_2.grad.item()}, expected {TWO_LAYER_B2_GRAD}"
    )


def test_two_layer_ste_annealing_alpha_half():
    """annealing_alpha=0.5 in both quantizers: the blended slope for each
    STE branch is 0.5*1 + 0.5*1 = 1.0, so gradients must be identical to
    the no-quantizer baseline.

    A value of w_1.grad ≈ -36 (half of -72) means one quantizer's
    quantized branch contributes zero local gradient (no STE)."""
    q1 = _make_calibrated_quantizer()
    q2 = _make_calibrated_quantizer()
    for q in (q1, q2):
        q.annealing_alpha.data.fill_(0.5)
        q.annealing_alpha_step = 0.0

    w_1 = torch.tensor(2.0, requires_grad=True)
    b_1 = torch.tensor(1.0, requires_grad=True)
    w_2 = torch.tensor(2.0, requires_grad=True)
    b_2 = torch.tensor(0.0, requires_grad=True)
    x   = torch.tensor(3.0)
    y   = torch.tensor(20.0)

    qw_1, *_ = q1(w_1)
    h         = qw_1 * x + b_1
    qw_2, *_ = q2(w_2)
    y_hat     = h * qw_2 + b_2
    assert torch.isclose(y_hat, torch.tensor(TWO_LAYER_Y_HAT), atol=TOL)

    loss = (y_hat - y) ** 2
    loss.backward()

    print(f"w_1.grad={w_1.grad}  b_1.grad={b_1.grad}")
    print(f"w_2.grad={w_2.grad}  b_2.grad={b_2.grad}")
    print("PRINT GRAPH (two-layer, alpha=0.5)")
    print_graph(loss.grad_fn)

    assert w_1.grad is not None, (
        "w_1.grad is None -- graph is disconnected at alpha=0.5"
    )
    assert torch.isclose(w_1.grad, torch.tensor(TWO_LAYER_W1_GRAD), atol=TOL), (
        f"ANNEALING two-layer: w_1.grad={w_1.grad.item()}, expected {TWO_LAYER_W1_GRAD} "
        f"-- a value around -36.0 means a quantized branch has zero local gradient"
    )
    assert torch.isclose(b_1.grad, torch.tensor(TWO_LAYER_B1_GRAD), atol=TOL)
    assert torch.isclose(w_2.grad, torch.tensor(TWO_LAYER_W2_GRAD), atol=TOL), (
        f"ANNEALING two-layer: w_2.grad={w_2.grad.item()}, expected {TWO_LAYER_W2_GRAD}"
    )
    assert torch.isclose(b_2.grad, torch.tensor(TWO_LAYER_B2_GRAD), atol=TOL)


# ---------------------------------------------------------------------------
# NOTE: annealing blend path is NOT covered by any test above.
#
# base_quantizer.py line 111 (`result = quantized`) unconditionally overrides
# the blend computed on line 105, so test_ste_off_annealing_alpha_zero and
# test_ste_annealing_alpha_half both exercise the fully-quantized STE path.
# They pass because STE local slope == pass-through slope == 1.
#
# Once line 111 is removed, two graph-structure checks become meaningful:
#   • alpha=0  → graph must NOT contain FixedPointQuantFnTestingThingsBackward
#   • alpha=0.5 → graph must contain AddBackward (the blend sum has two inputs)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Real-model gradient tests: Brevitas WeightQuantProxy path
#
# The scalar tests above call FixedPointPerTensorQuantizer.forward() directly.
# In the training harness, Brevitas wraps the quantizer in a WeightQuantProxy.
# If that proxy inserts a .detach() or returns a tensor with no grad_fn,
# layer.weight.grad would be None even if the standalone quantizer tests pass.
# ---------------------------------------------------------------------------

def test_quant_linear_gradient_flow():
    """Single QuantLinear layer: gradient flows through Brevitas WeightQuantProxy.

    Code path:
        QuantLinear.forward → WeightQuantProxy → FixedPointPerTensorQuantizer
        → FixedPointQuantFnTestingThingsBackward → weight.grad

    A broken proxy would leave weight.grad as None even though the standalone
    quantizer tests pass.
    """
    layer = qnn.QuantLinear(4, 2, bias=False, weight_quant=FixedPointPerTensorWeightQuant)
    layer.train()

    x = torch.randn(2, 4)

    with torch.no_grad():
        _ = layer(x)  # calibration: sets search_done=True on the weight quantizer

    QuantizerManager().enable_quantization()

    loss = layer(x).sum()
    loss.backward()

    print("PRINT GRAPH (QuantLinear single layer, Brevitas proxy path)")
    print_graph(loss.grad_fn)

    assert layer.weight.grad is not None, (
        "weight.grad is None — WeightQuantProxy broke the gradient chain"
    )
    assert layer.weight.grad.abs().sum() > 0, "weight.grad is all zeros"


def test_quant_linear_two_layer_gradient_flow():
    """Two stacked QuantLinear layers: gradient flows from loss through layer2's
    proxy and quantizer all the way back to layer1.weight.

    Real-model equivalent of test_two_layer_ste_on.  If layer2's Brevitas proxy
    disconnects the graph, layer1.weight.grad will be None.
    """
    layer1 = qnn.QuantLinear(4, 4, bias=False, weight_quant=FixedPointPerTensorWeightQuant)
    layer2 = qnn.QuantLinear(4, 2, bias=False, weight_quant=FixedPointPerTensorWeightQuant)
    layer1.train()
    layer2.train()

    x = torch.randn(2, 4)

    with torch.no_grad():
        _ = layer2(layer1(x))  # calibrate both weight quantizers

    QuantizerManager().enable_quantization()

    h = layer1(x)
    loss = layer2(h).sum()
    loss.backward()

    print("PRINT GRAPH (two QuantLinear layers, Brevitas proxy path)")
    print_graph(loss.grad_fn)

    assert layer1.weight.grad is not None, (
        "layer1.weight.grad is None — layer2's proxy disconnected the graph"
    )
    assert layer1.weight.grad.abs().sum() > 0, "layer1.weight.grad is all zeros"
    assert layer2.weight.grad is not None, "layer2.weight.grad is None"
    assert layer2.weight.grad.abs().sum() > 0, "layer2.weight.grad is all zeros"
