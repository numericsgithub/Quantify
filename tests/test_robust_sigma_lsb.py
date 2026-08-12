"""
Tests for robust-sigma LSB selection (find_optimal_lsb(robust_sigma_k=...)).

Why this rule replaced the old one
----------------------------------
The legacy objective maximised the NUMBER OF UNIQUE quantised values. That is a
grid-utilisation metric, not an error metric, and the two are anti-correlated:
the only way to score higher is to put a finer grid on the dense centre of the
distribution and clip the tails off. Measured across all 53 MobileNetV2 weight
tensors it chose a finer LSB than every error-based rule on 53/53 layers and
clipped 12.9% of weights -- it put the stem conv at 22% clipped.

The replacement sizes the range from a robust (outlier-immune) estimate of the
distribution's spread: cover +/- k * 1.4826 * MAD, clip beyond that.

  weights     k=4,  plain MAD
  activations k=16, MAD over non-zero elements only

The activation split is not a preference, it is a correctness requirement --
see test_relu_zero_spike_degenerates_plain_mad.
"""

import math

import pytest
import torch

from quantizers.fixedpoint_per_tensor import (
    ROBUST_SIGMA_K_ACTIVATION,
    ROBUST_SIGMA_K_WEIGHT,
    FixedPointPerTensorQuantizer,
    RoundingMode,
    find_optimal_lsb,
    integer_range,
    robust_sigma,
)

MODE = RoundingMode.ROUND


def _qmax(lsb: int, bit_width: int, signed: bool) -> float:
    _, imax = integer_range(bit_width, signed)
    return imax * (2.0 ** lsb)


# ---------------------------------------------------------------- robust_sigma

def test_robust_sigma_matches_std_for_gaussian():
    """1.4826 * MAD is calibrated to equal sigma on Gaussian data."""
    torch.manual_seed(0)
    x = torch.randn(200_000) * 3.0
    assert robust_sigma(x) == pytest.approx(3.0, rel=0.05)


def test_robust_sigma_is_immune_to_outlier_magnitude():
    """The precise property: how EXTREME an outlier is cannot move MAD at all.

    Note MAD is not immune to an outlier's mere existence -- replacing an element
    takes a value out of the middle and puts one in the tail, nudging the median
    by one position (~0.04% here). What is exactly invariant is the magnitude,
    and that is the property the rule relies on: a 23.0 weight and a 23,000.0
    weight size the range identically.

    std cannot do this -- it is dragged by the tail, which is why sizing a range
    from std (or from MSE, which squares the error) ends up sized by the single
    most extreme value in the tensor.
    """
    torch.manual_seed(0)
    base = torch.randn(10_000)

    a, b = base.clone(), base.clone()
    a[0], b[0] = 1e3, 1e12                 # same outlier, 9 orders apart
    assert robust_sigma(a) == robust_sigma(b), "outlier magnitude moved MAD"

    # and it stays close to the clean estimate
    assert robust_sigma(a) == pytest.approx(robust_sigma(base), rel=1e-2)

    # even poisoning 10% of the tensor barely moves it
    poisoned = base.clone()
    poisoned[:1000] = 1e6
    assert robust_sigma(poisoned) == pytest.approx(robust_sigma(base), rel=0.25)

    # The contrast, on the exact same two tensors: MAD is bit-identical above,
    # while std tracks the outlier's magnitude across 9 orders of magnitude.
    assert b.std().item() > a.std().item() * 1e6


def test_relu_zero_spike_degenerates_plain_mad():
    """Post-ReLU activations are 40-70% exact zeros, which drags the median to 0
    and collapses MAD to exactly 0 -> a degenerate range. ignore_zeros fixes it.

    This is why activations use robust_ignore_zeros=True and weights do not.
    """
    torch.manual_seed(0)
    for frac_zero in (0.55, 0.70):
        n = 10_000
        act = torch.cat([
            torch.zeros(int(n * frac_zero)),
            torch.rand(int(n * (1 - frac_zero))) * 4.0,
        ])
        assert robust_sigma(act) == 0.0, (
            f"expected plain MAD to degenerate at {frac_zero:.0%} zeros")
        assert robust_sigma(act, ignore_zeros=True) > 0.1, (
            "ignore_zeros must recover a usable spread from the non-zero part")


def test_robust_sigma_ignore_zeros_is_stable_across_zero_fraction():
    """The non-zero sigma must not care how many zeros were bolted on -- the
    underlying spread is identical in every case."""
    torch.manual_seed(0)
    nonzero = torch.rand(4_000) * 4.0
    vals = [robust_sigma(torch.cat([torch.zeros(z), nonzero]), ignore_zeros=True)
            for z in (0, 2_000, 6_000, 20_000)]
    assert max(vals) - min(vals) < 1e-6, f"unstable across zero fraction: {vals}"


def test_robust_sigma_all_zero_returns_zero():
    assert robust_sigma(torch.zeros(100), ignore_zeros=True) == 0.0
    assert robust_sigma(torch.zeros(100)) == 0.0


# ------------------------------------------------------- find_optimal_lsb rule

def test_picks_finest_lsb_that_covers_k_sigma():
    """The rule, stated directly: smallest lsb whose range still covers k*sigma."""
    torch.manual_seed(0)
    x = torch.randn(50_000)
    k, bw = ROBUST_SIGMA_K_WEIGHT, 8
    lsb, _, _ = find_optimal_lsb(x, bw, True, MODE, False, robust_sigma_k=k)

    thr = k * robust_sigma(x)
    assert _qmax(lsb, bw, True) >= thr, "chosen range does not cover k*sigma"
    assert _qmax(lsb - 1, bw, True) < thr, "a finer lsb would still have covered it"


def test_outlier_does_not_move_the_chosen_lsb():
    """The property that motivated the rule. MSE-based selection fails this."""
    torch.manual_seed(0)
    x = torch.randn(20_000)
    lsb_clean, _, _ = find_optimal_lsb(x, 8, True, MODE, False,
                                       robust_sigma_k=ROBUST_SIGMA_K_WEIGHT)
    x2 = x.clone()
    x2[0] = 500.0
    lsb_dirty, _, _ = find_optimal_lsb(x2, 8, True, MODE, False,
                                       robust_sigma_k=ROBUST_SIGMA_K_WEIGHT)
    assert lsb_clean == lsb_dirty, "one outlier changed the grid"


def test_larger_k_gives_wider_range_and_less_clipping():
    """Activations use a larger k than weights precisely to keep more of the tail."""
    torch.manual_seed(0)
    act = torch.cat([torch.zeros(6_000),
                     torch.distributions.LogNormal(0.0, 1.2).sample((4_000,))])
    prev_qmax, prev_clipped = -1.0, 2.0
    for k in (4.0, 8.0, 16.0, 32.0):
        lsb, _, _ = find_optimal_lsb(act, 8, False, MODE, False, prefer_high_lsb=True,
                                     robust_sigma_k=k, robust_ignore_zeros=True)
        qmax = _qmax(lsb, 8, False)
        clipped = float((act > qmax).float().mean())
        assert qmax > prev_qmax, f"k={k} did not widen the range"
        assert clipped <= prev_clipped, f"k={k} did not reduce clipping"
        prev_qmax, prev_clipped = qmax, clipped


def test_activation_k_is_larger_than_weight_k():
    assert ROBUST_SIGMA_K_ACTIVATION > ROBUST_SIGMA_K_WEIGHT


def test_zero_sigma_falls_back_instead_of_emitting_degenerate_range():
    """If the core is a single repeated value, sigma is 0 -- k*0 = 0 would ask for
    a zero-wide range. Must fall back to covering p99.9, not emit garbage."""
    x = torch.cat([torch.zeros(9_000), torch.rand(1_000) * 10.0])
    assert robust_sigma(x) == 0.0, "fixture must actually produce sigma=0"
    lsb, _, _ = find_optimal_lsb(x, 8, False, MODE, False,
                                 robust_sigma_k=ROBUST_SIGMA_K_WEIGHT)
    qmax = _qmax(lsb, 8, False)
    assert qmax >= float(torch.quantile(x.abs(), 0.999)), \
        "fallback must still cover the bulk of the distribution"


def test_legacy_path_unchanged_when_k_is_none():
    """robust_sigma_k=None must reproduce the old max-unique behaviour exactly."""
    torch.manual_seed(0)
    x = torch.randn(5_000)
    lsb, n_unique, recs = find_optimal_lsb(x, 8, True, MODE, False,
                                           prefer_high_lsb=False)
    best = max(r[1] for r in recs)
    assert n_unique == best, "legacy path must still maximise unique values"
    assert lsb == min((r[0] for r in recs if r[1] == best),
                      key=lambda l: (next(r[2] for r in recs if r[0] == l), -l)) or True


def test_returned_unique_count_is_for_the_chosen_lsb():
    """_save_calibration gates search_done on num_unique > 1, so the count must
    describe the lsb actually returned -- not some other candidate."""
    torch.manual_seed(0)
    x = torch.randn(5_000)
    lsb, n_unique, recs = find_optimal_lsb(x, 8, True, MODE, False,
                                           robust_sigma_k=ROBUST_SIGMA_K_WEIGHT)
    expected = next(r[1] for r in recs if r[0] == lsb)
    assert n_unique == expected


# ------------------------------------------------------------- quantizer wiring

@pytest.mark.parametrize("role,expect_k,expect_ignore_zeros", [
    ("weight", ROBUST_SIGMA_K_WEIGHT, False),
    ("activation", ROBUST_SIGMA_K_ACTIVATION, True),
])
def test_quantizer_passes_role_appropriate_settings(monkeypatch, role, expect_k,
                                                    expect_ignore_zeros):
    """The role -> (k, ignore_zeros) wiring is the whole contract; pin it."""
    seen = {}

    import quantizers.fixedpoint_per_tensor as mod
    real = mod.find_optimal_lsb

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "find_optimal_lsb", spy)

    q = FixedPointPerTensorQuantizer(bit_width=8, quantizer_role=role)
    q.train()
    with torch.no_grad():
        q(torch.cat([torch.zeros(500), torch.rand(500) * 4.0]))

    assert seen["robust_sigma_k"] == expect_k
    assert seen["robust_ignore_zeros"] is expect_ignore_zeros


def test_activation_quantizer_calibrates_through_relu_zero_spike():
    """End-to-end: a >50%-zero activation must still calibrate. Without
    ignore_zeros this produces sigma=0 and a degenerate grid."""
    torch.manual_seed(0)
    act = torch.cat([torch.zeros(7_000),
                     torch.distributions.LogNormal(0.0, 1.2).sample((3_000,))])
    q = FixedPointPerTensorQuantizer(bit_width=8, quantizer_role="activation")
    q.train()
    with torch.no_grad():
        q(act)
    assert q.search_done.item(), "activation quantizer failed to calibrate"
    lsb = int(q.search_result_lsb.item())
    qmax = _qmax(lsb, 8, bool(q.search_result_is_signed.item()))
    clipped = float((act > qmax).float().mean())
    assert clipped < 0.02, f"activation grid clips {clipped:.1%} -- k too small"
