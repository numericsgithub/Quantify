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

from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer
from quantizers.manager import QuantizerManager

EXPECTED_W_GRAD = -18.0
EXPECTED_B_GRAD = -6.0
EXPECTED_Y_HAT = 7.0
TOL = 1e-4


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

    actual_w_grad = w.grad.item()
    assert torch.isclose(w.grad, torch.tensor(EXPECTED_W_GRAD), atol=TOL), (
        f"ANNEALING state (alpha=0.5): expected w.grad={EXPECTED_W_GRAD}, "
        f"got {actual_w_grad} -- a value around -9.0 means the quantized "
        f"branch contributes zero local gradient (no STE), so the blend "
        f"only passes through half of the true straight-through slope."
    )
    assert torch.isclose(b.grad, torch.tensor(EXPECTED_B_GRAD), atol=TOL)
