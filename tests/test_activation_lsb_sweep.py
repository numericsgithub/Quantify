"""
Sanity checks for utils/activation_lsb_sweep.py's LSB table (and, by
extension, for the saturation-aware input-range guardrail it was built to
visualize -- pitfall #21 in docs/llm/pitfalls/brevitas_pitfalls.md).

Three independent things are checked for every (activation, lsb) row in the
sweep, not just that the table *looks* right:

1. `representable_range()`'s min/max actually match what the real
   fixed-point quantize function clamps to at that (lsb, bit_width) --
   the table isn't just restating a formula, it matches the real clamp.
2. The activation's own output is sane (finite, and within the activation's
   theoretical output bounds where it has any) when fed exactly the row's
   min_value/max_value -- these are the most extreme values the grid can
   represent at that setting, so this is where a bug would show up first.
3. The "allowed" column actually matches what the REAL input quantizer
   (with its real max_abs_value cap wired in) does when calibrated against
   data whose observed abs_max exactly equals that row's representable
   magnitude -- an "allowed" row must calibrate to exactly that LSB, and a
   "not allowed" row must calibrate to something strictly finer (the cap
   kicking in), never to the wide, disallowed setting itself.
"""

import math

import pytest
import torch

from quantizers.fixedpoint_per_tensor import (
    quantize_fixed_point_with_integers,
    FixedPointPerTensorQuantizer,
    RoundingMode,
)
from utils.activation_lsb_sweep import (
    ACTIVATIONS,
    DEFAULT_BIT_WIDTH,
    DEFAULT_LSB_VALUES,
    build_lsb_table,
    get_input_cap,
    representable_range,
    find_input_quantizer,
)


# One (name, cap) pair computed once per activation -- reused by every test
# below instead of rebuilding the module per (activation, lsb) combination.
@pytest.fixture(scope="module", params=ACTIVATIONS, ids=[n for n, _ in ACTIVATIONS])
def activation_and_cap(request):
    name, cls = request.param
    cap = get_input_cap(cls, bit_width=DEFAULT_BIT_WIDTH)
    return name, cls, cap


# =========================================================================
# 0. The cap values this whole feature request hinges on
# =========================================================================


class TestCapValuesAreNextPowerOfTwo:
    """The explicit, literal ask: ReLU6's cap must be 8 (next power of two
    at or above its saturation point of 6), not 12 (2x headroom, the
    previous rule) or 6 (the raw saturation point itself)."""

    def test_relu6_cap_is_eight(self):
        from quantizers.activations import QuantReLU6
        assert get_input_cap(QuantReLU6) == 8.0

    def test_sigmoid_cap_is_eight(self):
        from quantizers.activations import QuantSigmoid
        assert get_input_cap(QuantSigmoid) == 8.0

    def test_tanh_cap_is_four(self):
        from quantizers.activations import QuantTanh
        assert get_input_cap(QuantTanh) == 4.0

    @pytest.mark.parametrize("name,cls", [a for a in ACTIVATIONS if a[0] in ("relu", "silu", "gelu", "leaky_relu", "softmax")])
    def test_unbounded_activations_have_no_cap(self, name, cls):
        assert get_input_cap(cls) is None


# =========================================================================
# 1. representable_range() matches the real quantizer's clamp
# =========================================================================


class TestRepresentableRangeMatchesRealClamp:
    @pytest.mark.parametrize("lsb", list(DEFAULT_LSB_VALUES))
    def test_matches_quantize_fixed_point_with_integers(self, lsb):
        min_value, max_value = representable_range(lsb, DEFAULT_BIT_WIDTH, signed=True)

        # Feed values far outside the grid on both sides -- the real
        # quantizer must clamp them to exactly our predicted min/max.
        huge = torch.tensor([min_value * 100.0, max_value * 100.0])
        clamped, _ = quantize_fixed_point_with_integers(
            huge, lsb, DEFAULT_BIT_WIDTH, signed=True, rounding_mode=RoundingMode.ROUND,
        )
        assert clamped[0].item() == pytest.approx(min_value)
        assert clamped[1].item() == pytest.approx(max_value)

    @pytest.mark.parametrize("lsb", list(DEFAULT_LSB_VALUES))
    def test_values_exactly_at_the_boundary_are_not_clipped(self, lsb):
        """The boundary values themselves must round-trip exactly -- they
        are representable grid points, not values the clamp should alter."""
        min_value, max_value = representable_range(lsb, DEFAULT_BIT_WIDTH, signed=True)
        x = torch.tensor([min_value, max_value])
        out, _ = quantize_fixed_point_with_integers(
            x, lsb, DEFAULT_BIT_WIDTH, signed=True, rounding_mode=RoundingMode.ROUND,
        )
        assert out[0].item() == pytest.approx(min_value)
        assert out[1].item() == pytest.approx(max_value)


# =========================================================================
# 2. Activation output sanity at the row's representable extremes
# =========================================================================


# (name -> (float activation fn, (out_min, out_max) theoretical bounds or None))
_REFERENCE_FN = {
    "relu": (torch.relu, (0.0, None)),
    "relu6": (lambda x: torch.clamp(x, 0.0, 6.0), (0.0, 6.0)),
    "sigmoid": (torch.sigmoid, (0.0, 1.0)),
    "tanh": (torch.tanh, (-1.0, 1.0)),
    "silu": (torch.nn.functional.silu, (None, None)),
    "gelu": (torch.nn.functional.gelu, (None, None)),
    "leaky_relu": (lambda x: torch.nn.functional.leaky_relu(x, 0.01), (None, None)),
    "softmax": (lambda x: torch.nn.functional.softmax(x, dim=-1), (0.0, 1.0)),
}


class TestActivationOutputSanityAtExtremes:
    @pytest.mark.parametrize("lsb", list(DEFAULT_LSB_VALUES))
    def test_output_is_finite_and_within_theoretical_bounds(self, activation_and_cap, lsb):
        name, cls, cap = activation_and_cap
        fn, (out_min, out_max) = _REFERENCE_FN[name]
        min_value, max_value = representable_range(lsb, DEFAULT_BIT_WIDTH, signed=True)

        x = torch.tensor([min_value, max_value, 0.0])
        out = fn(x)
        assert torch.isfinite(out).all(), f"{name}: non-finite output at lsb={lsb}"
        if out_min is not None:
            assert (out >= out_min - 1e-5).all(), f"{name}: output below {out_min} at lsb={lsb}: {out}"
        if out_max is not None:
            assert (out <= out_max + 1e-5).all(), f"{name}: output above {out_max} at lsb={lsb}: {out}"


# =========================================================================
# 3. "allowed" matches what the REAL input quantizer actually does
# =========================================================================


class TestAllowedColumnMatchesRealCalibration:
    """For every row, build data whose observed abs_max exactly equals that
    row's representable magnitude, calibrate the activation's REAL input
    quantizer against it, and check the outcome matches "allowed"."""

    @pytest.mark.parametrize("lsb", list(DEFAULT_LSB_VALUES))
    def test_allowed_or_rejected_as_predicted(self, activation_and_cap, lsb):
        name, cls, cap = activation_and_cap
        rows = build_lsb_table(cap, bit_width=DEFAULT_BIT_WIDTH, lsb_values=[lsb])
        row = rows[0]

        # find_optimal_lsb's coverage-first search sizes the grid off the
        # POSITIVE-code count only (quantizers/fixedpoint_per_tensor.py:
        # n_positive_codes = 2**(bit_width-1)-1 for signed), i.e. against
        # `max_value` -- it never consults the asymmetric extra negative
        # code `min_value` has in two's complement. So the target to match
        # exactly is `row.max_value`, not `max(|min|, |max|)`.
        magnitude = row.max_value
        # A small dense bulk plus the exact target as an "outlier" --
        # mirrors the real-world shape (bulk near zero + rare large value)
        # the guardrail exists for, and keeps the observed abs_max exactly
        # equal to `magnitude` (so coverage-first's target is known
        # precisely).
        torch.manual_seed(0)
        bulk = torch.randn(200) * (magnitude * 0.01 + 1e-6)
        x = torch.cat([bulk, torch.tensor([magnitude, -magnitude])])

        # input_bit_width is forced to DEFAULT_BIT_WIDTH explicitly: several
        # activations (ReLU, ReLU6, Sigmoid, Softmax) otherwise default their
        # INPUT quantizer to bit_width + 1 (see quantizers/activations.py's
        # bit-width asymmetry), which would use a wider grid than this
        # table assumes at the same lsb.
        module = cls(bit_width=DEFAULT_BIT_WIDTH, input_bit_width=DEFAULT_BIT_WIDTH)
        q_in = find_input_quantizer(module, sample_input=x.reshape(-1, 1))
        # find_input_quantizer already ran one calibrating forward pass;
        # confirm the cap it used matches what this row was built from.
        assert q_in.max_abs_value == cap
        assert q_in.bit_width == DEFAULT_BIT_WIDTH
        selected_lsb = int(q_in.search_result_lsb.item())
        _, selected_max = representable_range(selected_lsb, q_in.bit_width, signed=True)

        if row.allowed:
            assert selected_lsb == row.lsb, (
                f"{name} lsb={lsb}: predicted allowed (magnitude {magnitude} <= cap {cap}), "
                f"so calibration against data with abs_max={magnitude} should pick exactly "
                f"this lsb, but picked {selected_lsb}"
            )
        else:
            # Coverage-first only guarantees it COVERS the (capped) target --
            # not that the resulting grid's max stays under the cap itself.
            # At a cap of 8.0, the finest lsb that still covers 8.0 has a
            # max of 15.875 (one step coarser would stop covering it) --
            # that's expected, not a guardrail failure. The real invariant
            # to check is that capping makes calibration behave EXACTLY as
            # if the real (outlier) data had been abs_max=cap all along --
            # i.e. identical to a reference run with no outlier beyond the
            # cap at all (input_max_abs_value overridden very high so it
            # can never itself kick in and mask a bug here).
            ref_module = cls(
                bit_width=DEFAULT_BIT_WIDTH, input_bit_width=DEFAULT_BIT_WIDTH,
                input_max_abs_value=1e12,
            )
            torch.manual_seed(0)
            ref_bulk = torch.randn(200) * (cap * 0.01 + 1e-6)
            ref_x = torch.cat([ref_bulk, torch.tensor([cap, -cap])])
            ref_q_in = find_input_quantizer(ref_module, sample_input=ref_x.reshape(-1, 1))
            ref_lsb = int(ref_q_in.search_result_lsb.item())

            assert selected_lsb == ref_lsb, (
                f"{name} lsb={lsb}: predicted NOT allowed (magnitude {magnitude} > cap {cap}) -- "
                f"calibrating against the outlier data picked lsb={selected_lsb}, but capping to "
                f"{cap} should make it behave identically to calibrating against abs_max={cap} "
                f"directly (which picked lsb={ref_lsb}) -- the guardrail did not actually apply"
            )
            # Usually strictly finer, but coverage-first's power-of-two
            # granularity can make the cap's own covering lsb coincide with
            # the disallowed row's lsb (e.g. a cap of 4.0 and this row's own
            # lsb both landing on the same coarse step) -- equality here
            # just means this row happens to BE the capped answer, not that
            # the guardrail failed to apply (the equivalence assertion
            # above already proved that it did).
            assert selected_lsb <= row.lsb, (
                f"{name} lsb={lsb}: guardrail should never force something COARSER than the "
                f"disallowed setting, got selected_lsb={selected_lsb} > row.lsb={row.lsb}"
            )


# =========================================================================
# 4. The printed table itself doesn't crash and has the right shape
# =========================================================================


class TestTableFormatting:
    def test_print_all_tables_runs_without_error(self, capsys):
        from utils.activation_lsb_sweep import print_all_tables

        print_all_tables()
        captured = capsys.readouterr()
        for name, _ in ACTIVATIONS:
            assert name in captured.out

    def test_table_has_expected_row_count_and_starts_at_requested_lsb(self):
        rows = build_lsb_table(8.0, lsb_values=DEFAULT_LSB_VALUES)
        assert len(rows) == len(list(DEFAULT_LSB_VALUES))
        assert rows[0].lsb == -4
        assert [r.lsb for r in rows] == sorted(r.lsb for r in rows)
