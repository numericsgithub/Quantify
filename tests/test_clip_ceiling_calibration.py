"""
Tests for the per-role clipping ceiling enforced during fixed-point LSB
calibration (quantizers/fixedpoint_per_tensor.py::find_optimal_lsb /
FixedPointPerTensorQuantizer._calibrate):

  - weight quantizers: never choose an LSB that clips more than 15% of values
  - bias quantizers: never choose an LSB that clips ANY value (0%)
  - activation quantizers: unaffected (unchanged coverage-first rule)
  - an unconstrained ("unknown" role) quantizer keeps the old pre-existing
    "maximise unique count" behavior, which can clip outliers heavily --
    this is the baseline the weight/bias ceilings are meant to fix

"Clipped" means a value falls outside the representable
[q_min, q_max] grid range at the selected LSB -- e.g. 5 values
[0.1, 0.2, 0.3, 0.8, 1.1] against a range capped at 0.5 clips 2/5 = 40%.
"""

import torch
import pytest

from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorQuantizer,
    FixedPointPerTensorWeightQuant,
    FixedPointPerTensorBiasQuant,
    FixedPointPerTensorActivationQuant,
    find_optimal_lsb,
    quantize_fixed_point,
    RoundingMode,
)


def _clip_pct(x: torch.Tensor, lsb: int, bit_width: int, signed: bool, narrow_range: bool = False) -> float:
    step = 2.0 ** lsb
    if signed:
        imin = -(2 ** (bit_width - 1))
        if narrow_range:
            imin += 1
        imax = 2 ** (bit_width - 1) - 1
    else:
        imin, imax = 0, 2 ** bit_width - 1
    q_min, q_max = imin * step, imax * step
    n_clip = int(((x < q_min) | (x > q_max)).sum().item())
    return 100.0 * n_clip / x.numel()


def _outlier_tensor():
    """Dense bulk near zero plus a handful of large outliers -- the
    unconstrained "maximise unique count" objective clips these outliers
    heavily (it packs resolution into the dense bulk instead), which is
    exactly the behavior the weight/bias ceilings must prevent."""
    torch.manual_seed(0)
    bulk = torch.randn(970) * 0.05
    outliers = torch.tensor([3.0, -3.0, 2.5, -2.5, 4.0] * 6)
    return torch.cat([bulk, outliers])


# =========================================================================
# 1. The user's literal example
# =========================================================================


def test_literal_example_two_of_five_is_forty_percent():
    x = torch.tensor([0.1, 0.2, 0.3, 0.8, 1.1])
    # bit_width=1, unsigned -> integer_max=1; lsb=-1 -> step=0.5 -> q_max=0.5
    pct = _clip_pct(x, lsb=-1, bit_width=1, signed=False)
    assert pct == pytest.approx(40.0)


# =========================================================================
# 2. find_optimal_lsb's max_clip_pct parameter, directly
# =========================================================================


class TestFindOptimalLsbClipCeiling:
    def test_unconstrained_can_exceed_fifteen_percent(self):
        """Baseline: without a ceiling, the old 'maximise unique count'
        objective is free to clip heavily -- confirms the outlier tensor
        actually exercises the scenario the ceiling is meant to fix."""
        x = _outlier_tensor()
        lsb, _, _ = find_optimal_lsb(
            x, bit_width=8, signed=True, rounding_mode=RoundingMode.ROUND,
            prefer_high_lsb=False, max_clip_pct=None,
        )
        assert _clip_pct(x, lsb, 8, True) > 15.0

    def test_fifteen_percent_ceiling_is_respected(self):
        x = _outlier_tensor()
        lsb, _, _ = find_optimal_lsb(
            x, bit_width=8, signed=True, rounding_mode=RoundingMode.ROUND,
            prefer_high_lsb=False, max_clip_pct=15.0,
        )
        assert _clip_pct(x, lsb, 8, True) <= 15.0

    def test_zero_percent_ceiling_clips_nothing(self):
        x = _outlier_tensor()
        lsb, _, _ = find_optimal_lsb(
            x, bit_width=8, signed=True, rounding_mode=RoundingMode.ROUND,
            prefer_high_lsb=False, max_clip_pct=0.0,
        )
        assert _clip_pct(x, lsb, 8, True) == 0.0

    def test_zero_ceiling_is_coarser_or_equal_to_fifteen_ceiling(self):
        """A stricter (lower) ceiling can never select a finer-or-equal LSB
        than a looser one for the same data -- 0% clipping requires at
        least as coarse a grid as 15% clipping."""
        x = _outlier_tensor()
        lsb_15, _, _ = find_optimal_lsb(
            x, bit_width=8, signed=True, rounding_mode=RoundingMode.ROUND,
            prefer_high_lsb=False, max_clip_pct=15.0,
        )
        lsb_0, _, _ = find_optimal_lsb(
            x, bit_width=8, signed=True, rounding_mode=RoundingMode.ROUND,
            prefer_high_lsb=False, max_clip_pct=0.0,
        )
        assert lsb_0 >= lsb_15

    def test_ceiling_still_maximizes_unique_count_among_qualifying_candidates(self):
        """The ceiling only disqualifies candidates -- among the qualifying
        ones, the original weight objective (max unique, SAD tie-break)
        still applies. A plain, well-behaved (no outliers) tensor should
        pick the same LSB with or without a generous ceiling."""
        x = torch.randn(2000) * 0.5
        lsb_unconstrained, _, _ = find_optimal_lsb(
            x, bit_width=8, signed=True, rounding_mode=RoundingMode.ROUND,
            prefer_high_lsb=False, max_clip_pct=None,
        )
        lsb_generous, _, _ = find_optimal_lsb(
            x, bit_width=8, signed=True, rounding_mode=RoundingMode.ROUND,
            prefer_high_lsb=False, max_clip_pct=99.0,
        )
        assert lsb_unconstrained == lsb_generous

    def test_all_zero_tensor_short_circuits(self):
        x = torch.zeros(10)
        lsb, n_unique, records = find_optimal_lsb(
            x, bit_width=8, signed=True, rounding_mode=RoundingMode.ROUND,
            prefer_high_lsb=False, max_clip_pct=0.0,
        )
        assert lsb == 0
        assert n_unique == 1


# =========================================================================
# 3. FixedPointPerTensorQuantizer._calibrate, by role
# =========================================================================


class TestCalibrationByRole:
    def test_weight_role_respects_fifteen_percent(self):
        q = FixedPointPerTensorQuantizer(bit_width=8, quantizer_role="weight",
                                          rounding_mode=RoundingMode.ROUND)
        x = _outlier_tensor()
        q(x)
        lsb = int(q.search_result_lsb.item())
        signed = bool(q.search_result_is_signed.item())
        assert _clip_pct(x, lsb, 8, signed) <= 15.0

    def test_bias_role_clips_nothing(self):
        q = FixedPointPerTensorQuantizer(bit_width=8, quantizer_role="bias",
                                          rounding_mode=RoundingMode.ROUND)
        x = _outlier_tensor()
        q(x)
        lsb = int(q.search_result_lsb.item())
        signed = bool(q.search_result_is_signed.item())
        assert _clip_pct(x, lsb, 8, signed) == 0.0

    def test_unknown_role_is_unconstrained(self):
        q = FixedPointPerTensorQuantizer(bit_width=8, quantizer_role="unknown",
                                          rounding_mode=RoundingMode.ROUND)
        x = _outlier_tensor()
        q(x)
        lsb = int(q.search_result_lsb.item())
        signed = bool(q.search_result_is_signed.item())
        assert _clip_pct(x, lsb, 8, signed) > 15.0

    def test_activation_role_unaffected_still_zero_clip_via_coverage_first(self):
        """Activations keep their own, separate coverage-first rule
        (prefer_high_lsb=True) -- unchanged by this feature, verified here
        only to confirm nothing regressed."""
        q = FixedPointPerTensorQuantizer(bit_width=8, quantizer_role="activation",
                                          rounding_mode=RoundingMode.FLOOR)
        x = _outlier_tensor().abs()  # activations are typically unsigned/non-negative
        q(x)
        lsb = int(q.search_result_lsb.item())
        signed = bool(q.search_result_is_signed.item())
        assert _clip_pct(x, lsb, 8, signed) == 0.0

    def test_weight_calibration_more_conservative_than_unconstrained(self):
        """Direct before/after comparison on the same data: constraining
        weights to <=15% clipping must select a coarser-or-equal LSB than
        the old unconstrained objective would have."""
        x = _outlier_tensor()
        q_unconstrained = FixedPointPerTensorQuantizer(
            bit_width=8, quantizer_role="unknown", rounding_mode=RoundingMode.ROUND)
        q_unconstrained(x)

        q_weight = FixedPointPerTensorQuantizer(
            bit_width=8, quantizer_role="weight", rounding_mode=RoundingMode.ROUND)
        q_weight(x)

        assert q_weight.search_result_lsb.item() >= q_unconstrained.search_result_lsb.item()


# =========================================================================
# 4. Brevitas injector wiring -- the actual public API surface
# =========================================================================


class TestBrevitasInjectorWiring:
    def test_weight_quant_injector_respects_ceiling(self):
        import brevitas.nn as qnn

        layer = qnn.QuantConv2d(3, 8, 3, padding=1, bias=False,
                                 weight_quant=FixedPointPerTensorWeightQuant)
        # Inject outliers into the real conv weight so calibration sees them.
        with torch.no_grad():
            layer.weight.view(-1)[:5] = torch.tensor([3.0, -3.0, 2.5, -2.5, 4.0])
        layer.train()
        layer(torch.randn(1, 3, 8, 8))

        tq = layer.weight_quant.tensor_quant
        lsb = int(tq.search_result_lsb.item())
        signed = bool(tq.search_result_is_signed.item())
        assert _clip_pct(layer.weight.detach(), lsb, tq.bit_width, signed) <= 15.0

    def test_bias_quant_injector_clips_nothing(self):
        import brevitas.nn as qnn

        layer = qnn.QuantConv2d(
            3, 8, 3, padding=1, bias=True,
            weight_quant=FixedPointPerTensorWeightQuant,
            input_quant=FixedPointPerTensorActivationQuant,
            bias_quant=FixedPointPerTensorBiasQuant,
        )
        with torch.no_grad():
            layer.bias.view(-1)[:3] = torch.tensor([10.0, -10.0, 5.0])
        layer.train()
        layer(torch.randn(1, 3, 8, 8))

        tq = layer.bias_quant.tensor_quant
        lsb = int(tq.search_result_lsb.item())
        signed = bool(tq.search_result_is_signed.item())
        assert _clip_pct(layer.bias.detach(), lsb, tq.bit_width, signed) == 0.0
