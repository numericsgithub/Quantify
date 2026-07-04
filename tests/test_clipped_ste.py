"""
Clipped-STE gradient tests for FixedPointPerTensorQuantizer.

Plain STE uses local slope 1 EVERYWHERE (even where the forward clamp
saturated). Clipped STE — uses local slope
1 only for inputs INSIDE the representable range and slope 0 for inputs the
forward clamp saturated. The `clipped_ste` constructor toggle selects between
them so the two can be compared in an ablation.

These are NOT training runs. They use the same single-weight model as the
existing gradient tests:

    y_hat = quantizer(w) * x + b
    L     = (y_hat - y) ** 2                with b=1.0, x=3.0, y=10.0

Calibration grid (unsigned 3-bit, step=0.5):
    Warmup data [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] gives lsb=-1,
    step=0.5, representable range [0.0, 3.5] (codes 0..7). w=2.0 lands
    exactly on the grid; the top grid value is 3.5.

Hand-derived ground truth (do NOT recompute or alter these to make tests pass;
if a test fails, report the actual value):

  In-range  w=2.0  (on-grid):  q(w)=2.0, y_hat=7.0,  error=-3.0, dL/dy_hat=-6.0
      plain & clipped:  w.grad = -6.0 * 3.0 = -18.0 ,  b.grad = -6.0
      Clipping changes nothing inside the range.

  Out-of-range w=5.0 (above range max, clamped to 3.5):
      q(w)=3.5, y_hat=3.5*3+1=11.5, error=1.5, dL/dy_hat=3.0
      plain STE  : w.grad = 3.0 * 3.0 = 9.0   (slope 1 even though clamped)
      clipped STE: w.grad = 0.0              (slope 0, saturated -> masked)
      This is the distinguishing case.

  Boundary  w=3.5  (exactly the max grid value):
      In-range by the >=/<= convention, so clipped STE keeps slope 1:
      q(w)=3.5, y_hat=11.5, dL/dy_hat=3.0, w.grad = 9.0 (same as plain).
"""
import inspect

import pytest
import torch

from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer
from quantizers.manager import QuantizerManager

# ---- model constants ----
X = 3.0
B = 1.0
Y = 10.0
TOL = 1e-4

# ---- grid / weights ----
BIT_WIDTH = 3                     # unsigned 3-bit -> codes 0..7
CALIB_DATA = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]  # lsb=-1, step=0.5
GRID_MAX = 3.5                    # top representable value
IN_RANGE_W = 2.0                 # on-grid, inside range
OUT_RANGE_W = 5.0                # above range max, clamps to 3.5
BOUNDARY_W = GRID_MAX            # exactly the max grid value

# ---- hand-derived ground truth (do NOT alter to force passes) ----
EXPECTED_W_GRAD_IN_RANGE = -18.0
EXPECTED_B_GRAD_IN_RANGE = -6.0
EXPECTED_Y_HAT_IN_RANGE = 7.0
EXPECTED_W_GRAD_OUT_PLAIN = 9.0    # plain STE leaks gradient through the clamp
EXPECTED_W_GRAD_OUT_CLIPPED = 0.0  # clipped STE zeroes the saturated gradient
EXPECTED_W_GRAD_BOUNDARY = 9.0     # boundary counts as in-range -> nonzero


@pytest.fixture(autouse=True)
def reset_manager():
    """QuantizerManager is a singleton; wipe it before/after each test so
    quantizers from another test aren't hit by enable/disable_quantization()."""
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


def _make_calibrated_quantizer(clipped_ste: bool) -> FixedPointPerTensorQuantizer:
    """Build a fixed-point quantizer and run one warmup forward (train mode,
    no_grad) so search_done flips True before the gradient-measuring pass.

    Mirrors the existing tests/test_quantizer_gradients.py fixture: warmup on
    data spanning multiple unique values (required for search_done) that
    includes IN_RANGE_W exactly, so the grid represents w=2.0 exactly and the
    forward value is exact -- any failure is about gradients, not rounding.
    Uses BIT_WIDTH=3 so GRID_MAX is 3.5 and OUT_RANGE_W=5.0 actually saturates
    the forward clamp (bit_width=8 would put the max at 255 and nothing clamps).
    """
    q = FixedPointPerTensorQuantizer(
        bit_width=BIT_WIDTH, signed=True, clipped_ste=clipped_ste
    )
    q.train()
    with torch.no_grad():
        q(torch.tensor(CALIB_DATA))  # triggers _calibrate() -> search_done=True
    assert q.search_done.item(), "calibration did not set search_done=True"
    q.quantizer_manager.enable_quantization()  # alpha=1 -> STE backward is live
    assert q.annealing_alpha.item() == 1.0
    return q


def _run_single_weight(q, w_value):
    """y_hat = q(w) * x + b ; L = (y_hat - y)**2 ; returns (w, b, quantized_w)."""
    w = torch.tensor(w_value, requires_grad=True)
    b = torch.tensor(B, requires_grad=True)
    x = torch.tensor(X)
    y = torch.tensor(Y)

    quantized_w, _scale, _zp, _bw = q(w)
    y_hat = quantized_w * x + b
    loss = (y_hat - y) ** 2
    loss.backward()
    return w, b, quantized_w


# ---------------------------------------------------------------------------
# In-range: clipping must not disturb gradients inside the representable range
# ---------------------------------------------------------------------------

def test_in_range_plain_ste():
    """In-range w=2.0, plain STE -> w.grad == -18.0 (baseline STE behavior)."""
    q = _make_calibrated_quantizer(clipped_ste=False)
    w, b, qw = _run_single_weight(q, IN_RANGE_W)

    print(f"[in-range/plain]  w={IN_RANGE_W} q(w)={qw.item()} "
          f"w.grad={w.grad.item()} b.grad={b.grad.item()}")
    assert torch.isclose(qw, torch.tensor(IN_RANGE_W), atol=TOL), (
        f"w=2.0 must be on-grid: q(w)={qw.item()}"
    )
    assert torch.isclose(w.grad, torch.tensor(EXPECTED_W_GRAD_IN_RANGE), atol=TOL), (
        f"expected w.grad={EXPECTED_W_GRAD_IN_RANGE}, got {w.grad.item()}"
    )
    assert torch.isclose(b.grad, torch.tensor(EXPECTED_B_GRAD_IN_RANGE), atol=TOL)


def test_in_range_clipped_ste():
    """In-range w=2.0, clipped STE -> w.grad == -18.0.
    Proves clipping does NOT disturb in-range gradients."""
    q = _make_calibrated_quantizer(clipped_ste=True)
    w, b, qw = _run_single_weight(q, IN_RANGE_W)

    print(f"[in-range/clipped]  w={IN_RANGE_W} q(w)={qw.item()} "
          f"w.grad={w.grad.item()} b.grad={b.grad.item()}")
    assert torch.isclose(qw, torch.tensor(IN_RANGE_W), atol=TOL)
    assert torch.isclose(w.grad, torch.tensor(EXPECTED_W_GRAD_IN_RANGE), atol=TOL), (
        f"clipped STE must leave in-range gradient untouched: "
        f"expected {EXPECTED_W_GRAD_IN_RANGE}, got {w.grad.item()}"
    )
    assert torch.isclose(b.grad, torch.tensor(EXPECTED_B_GRAD_IN_RANGE), atol=TOL)


# ---------------------------------------------------------------------------
# Out-of-range: the distinguishing case
# ---------------------------------------------------------------------------

def test_out_of_range_plain_ste():
    """Out-of-range w=5.0, plain STE -> nonzero (~9.0). Documents the OLD
    behavior clipped STE improves on: gradient leaks through the saturated
    clamp with local slope 1."""
    q = _make_calibrated_quantizer(clipped_ste=False)
    w, b, qw = _run_single_weight(q, OUT_RANGE_W)

    print(f"[out-of-range/plain]  w={OUT_RANGE_W} q(w)={qw.item()} "
          f"w.grad={w.grad.item()} (expected ~{EXPECTED_W_GRAD_OUT_PLAIN})")
    assert torch.isclose(qw, torch.tensor(GRID_MAX), atol=TOL), (
        f"w=5.0 must clamp to grid max {GRID_MAX}: q(w)={qw.item()}"
    )
    assert w.grad.abs().item() > TOL, "plain STE must leak a nonzero gradient here"
    assert torch.isclose(w.grad, torch.tensor(EXPECTED_W_GRAD_OUT_PLAIN), atol=TOL), (
        f"expected plain-STE w.grad~{EXPECTED_W_GRAD_OUT_PLAIN}, got {w.grad.item()}"
    )


def test_out_of_range_clipped_ste():
    """Out-of-range w=5.0, clipped STE -> w.grad == 0.0. KEY new-behavior test:
    the forward clamp saturated, so the local slope is 0 and no gradient flows
    to the weight. If this is not exactly 0, report the actual number -- that
    is a real finding, not a tolerance issue."""
    q = _make_calibrated_quantizer(clipped_ste=True)
    w, b, qw = _run_single_weight(q, OUT_RANGE_W)

    print(f"[out-of-range/clipped]  w={OUT_RANGE_W} q(w)={qw.item()} "
          f"w.grad={w.grad.item()} (expected {EXPECTED_W_GRAD_OUT_CLIPPED})")
    assert torch.isclose(qw, torch.tensor(GRID_MAX), atol=TOL)
    assert w.grad.item() == EXPECTED_W_GRAD_OUT_CLIPPED, (
        f"clipped STE must zero the saturated gradient: "
        f"expected exactly {EXPECTED_W_GRAD_OUT_CLIPPED}, got {w.grad.item()}"
    )
    # b never touches the quantizer, so its gradient must still flow.
    assert b.grad.abs().item() > TOL, "b.grad must be nonzero (b bypasses the quantizer)"


# ---------------------------------------------------------------------------
# Boundary convention: exact max grid value counts as IN-range
# ---------------------------------------------------------------------------

def test_boundary_exactly_at_max_is_in_range():
    """w exactly equal to the max grid value (3.5). Documented convention is
    >=/<= (inclusive), so this counts as IN-range and clipped STE keeps slope
    1 -> nonzero gradient. Prints the actual value so the convention is
    verified, not assumed."""
    q = _make_calibrated_quantizer(clipped_ste=True)
    w, b, qw = _run_single_weight(q, BOUNDARY_W)

    print(f"[boundary/clipped]  w={BOUNDARY_W} (==grid max {GRID_MAX}) "
          f"q(w)={qw.item()} w.grad={w.grad.item()} "
          f"(convention: inclusive -> in-range -> nonzero)")
    assert torch.isclose(qw, torch.tensor(GRID_MAX), atol=TOL)
    assert w.grad.abs().item() > TOL, (
        f"boundary weight must be treated as in-range (inclusive convention), "
        f"so w.grad must be nonzero; got {w.grad.item()}"
    )
    assert torch.isclose(w.grad, torch.tensor(EXPECTED_W_GRAD_BOUNDARY), atol=TOL), (
        f"expected boundary w.grad={EXPECTED_W_GRAD_BOUNDARY}, got {w.grad.item()}"
    )


# ---------------------------------------------------------------------------
# Two-layer graph-connectivity: clipped STE must not sever the graph
# ---------------------------------------------------------------------------

def test_two_layer_graph_stays_connected():
    """Add a second linear op after the quantizer:
        inner = q(w) * x ;  y_hat = w2 * inner + b ;  L = (y_hat - y)**2
    With an IN-range w and clipped STE, gradients must reach BOTH w and w2 and
    neither may be None. This checks the graph stays connected (the
    graph-cutting failure mode), independent of the exact gradient values."""
    q = _make_calibrated_quantizer(clipped_ste=True)

    w = torch.tensor(IN_RANGE_W, requires_grad=True)
    w2 = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(B, requires_grad=True)
    x = torch.tensor(X)
    y = torch.tensor(Y)

    quantized_w, *_ = q(w)
    inner = quantized_w * x
    y_hat = w2 * inner + b
    loss = (y_hat - y) ** 2
    loss.backward()

    print(f"[two-layer/clipped]  q(w)={quantized_w.item()} inner={inner.item()} "
          f"y_hat={y_hat.item()} w.grad={w.grad} w2.grad={w2.grad}")

    assert w.grad is not None, "w.grad is None -- clipped STE severed the graph to w"
    assert w2.grad is not None, "w2.grad is None -- graph disconnected before w2"
    assert w.grad.abs().item() > TOL, "w.grad is zero for an in-range weight (should flow)"
    assert w2.grad.abs().item() > TOL, "w2.grad is zero (second op should receive gradient)"


# ---------------------------------------------------------------------------
# Other quantizers: clipped STE should live in shared logic. If the toggle
# isn't wired into a given quantizer yet, skip with a clear reason rather than
# faking a pass.
# ---------------------------------------------------------------------------

def _coefficient_quantizer_factory():
    import os
    import tempfile
    from quantizers.coefficient_per_tensor_weights import CoefficientPerTensorWeightQuantizer

    # A coefficient set that (scaled) can represent 2.0 but caps below 5.0.
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write("0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5\n")

    def factory(clipped_ste):
        # Will raise TypeError if clipped_ste isn't a constructor param -> skip.
        return CoefficientPerTensorWeightQuantizer(
            filepath=path, bit_width=BIT_WIDTH, clipped_ste=clipped_ste
        )

    return factory


def _silu_quantizer_factory():
    from quantizers.silu_quant import SiLUTensorQuant

    def factory(clipped_ste):
        return SiLUTensorQuant(bit_width=BIT_WIDTH, clipped_ste=clipped_ste)

    return factory


IN_RANGE_SMALL_W = 0.5  # on both grids (coeff set + silu grid) -> safely in-range


def _calibrate_and_measure(q, w_value):
    """Calibrate q on CALIB_DATA (train mode), enable quantization, then measure
    dL/dw for  L = (q(w)*x - y)**2 . Returns (w.grad.item(), q(w).item())."""
    q.train()
    with torch.no_grad():
        q(torch.tensor(CALIB_DATA))  # -> search_done=True
    assert q.search_done.item(), "calibration did not set search_done"
    q.quantizer_manager.enable_quantization()  # alpha=1 -> STE backward live

    w = torch.tensor(w_value, requires_grad=True)
    x = torch.tensor(X)
    qw, *_ = q(w)
    loss = (qw * x - torch.tensor(Y)) ** 2
    loss.backward()
    return w.grad.item(), qw.item()


@pytest.mark.parametrize(
    "name, factory",
    [
        ("coefficient", _coefficient_quantizer_factory()),
        ("silu", _silu_quantizer_factory()),
    ],
)
def test_other_quantizers_honor_clipped_ste(name, factory):
    """The clipped-STE toggle lives in shared logic, so the other quantizers
    must honor it too. If a quantizer doesn't accept/store the flag yet, skip
    with a clear reason rather than faking a pass; otherwise assert the real
    contract:
        out-of-range input, plain STE   -> nonzero gradient (leaks through clamp)
        out-of-range input, clipped STE -> exactly 0        (saturated -> masked)
        in-range input,     clipped STE -> nonzero gradient (still flows)
    """
    # Does this quantizer accept AND store the toggle?
    try:
        q_probe = factory(clipped_ste=True)
    except TypeError as e:
        pytest.skip(
            f"{name} quantizer does not accept clipped_ste yet "
            f"(toggle not wired into shared logic): {e}"
        )
    if not getattr(q_probe, "clipped_ste", False):
        pytest.skip(
            f"{name} quantizer accepts clipped_ste but does not store it; "
            f"skipping rather than asserting a fake pass."
        )

    # Each measurement uses a freshly-registered quantizer; reset the singleton
    # manager between them so enable_quantization() only touches the one under
    # test (the probe above also registered one).
    QuantizerManager().reset()
    plain_out_grad, plain_out_q = _calibrate_and_measure(factory(clipped_ste=False), OUT_RANGE_W)
    QuantizerManager().reset()
    clipped_out_grad, clipped_out_q = _calibrate_and_measure(factory(clipped_ste=True), OUT_RANGE_W)
    QuantizerManager().reset()
    clipped_in_grad, clipped_in_q = _calibrate_and_measure(factory(clipped_ste=True), IN_RANGE_SMALL_W)

    print(f"[other/{name}]  out-of-range(w={OUT_RANGE_W}): "
          f"plain w.grad={plain_out_grad} (q={plain_out_q}), "
          f"clipped w.grad={clipped_out_grad} (q={clipped_out_q}) ; "
          f"in-range(w={IN_RANGE_SMALL_W}): clipped w.grad={clipped_in_grad} (q={clipped_in_q})")

    assert abs(plain_out_grad) > TOL, (
        f"{name}: plain STE must leak a nonzero gradient on a saturated input, "
        f"got {plain_out_grad}"
    )
    assert clipped_out_grad == 0.0, (
        f"{name}: clipped STE must zero the saturated gradient exactly, "
        f"got {clipped_out_grad} -- report the actual number, not a tolerance"
    )
    assert abs(clipped_in_grad) > TOL, (
        f"{name}: clipped STE must keep in-range gradients flowing, "
        f"got {clipped_in_grad}"
    )
