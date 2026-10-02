"""
Regression tests for a real bug found by inspecting an exported model
(`/conv5_DP/act/input_quant/.../FixedPointQuant`): ReLU6's INPUT quantizer
calibrated to `lsb=-4` (step 0.0625) while its OUTPUT quantizer calibrated to
`lsb=-5` (step 0.03125) for the exact same real-valued region `[0, 6]` --
i.e. the input quantizer was COARSER than the output quantizer, so the
output's apparent extra precision was illusory: the achievable resolution of
the whole activation was bottlenecked by the input side.

Root cause: `_input_max_abs_value()` / `_next_pow2_at_least()` in
`quantizers/activations.py` round the saturation-aware cap UP to the next
power of two before handing it to `find_optimal_lsb`'s `max_abs_value`
(ReLU6/Sigmoid's true saturation point 6.0 -> cap 8.0; Tanh's 3.0 -> cap
4.0). This is unnecessary padding: coverage-first (`prefer_high_lsb=True`,
the correct and UNCHANGED calibration rule for activations) already finds
the finest LSB that covers whatever cap it's given, power-of-two or not --
rounding the cap itself up just forces that search to cover a bigger,
partly-wasted range. The fix capped at the exact per-activation saturation
value does the opposite -- it does not change which calibration algorithm
runs at all, only the cap VALUE fed into the existing activation-mode
(coverage-first) calibration.

This file is written FIRST, before the fix, and is expected to FAIL against
the current (buggy) power-of-two-padded cap. Every assumption behind the fix
is pinned down as its own test:

  A1. The input quantizer's calibrated range, when outliers are present,
      tracks the activation's EXACT saturation point (6.0 for ReLU6/Sigmoid,
      3.0 for Tanh) -- not a power-of-two-inflated version of it.
  A2. Equivalently: calibrating directly against the raw saturation value
      (bypassing the module entirely) yields the same LSB the module's
      input quantizer actually picks -- i.e. the module applies no hidden
      padding on top of the documented saturation constants.
  A3. The end-to-end achievable resolution (number of distinct values that
      survive input_quant -> activation -> output_quant) is NOT bottlenecked
      well below what the OUTPUT quantizer's own bit width supports, even
      when outliers are present in the pre-activation data.
  A4. The fix must not regress the "not too strict" guarantee: data that
      never approaches the saturation point must still calibrate without
      clipping (sanity check that the cap is a ceiling, not a new, tighter
      replacement for normal calibration).
"""

import torch
import pytest

from quantizers.activations import (
    QuantReLU6,
    QuantSigmoid,
    QuantTanh,
    _RELU6_SATURATION,
    _TANH_SATURATION,
    _SIGMOID_SATURATION,
)
from quantizers.fixedpoint_per_tensor import find_optimal_lsb, RoundingMode
from quantizers.base_quantizer import BaseQuantizer


def _quantizers_of(module):
    return [m for m in module.modules() if isinstance(m, BaseQuantizer)]


def _find_by_role_position(module, want_input: bool):
    quantizers = sorted(_quantizers_of(module), key=lambda q: q.inference_sequence_id)
    return quantizers[0] if want_input else quantizers[-1]


def _calibrated_q_max(quantizer) -> float:
    lsb = int(quantizer.search_result_lsb.item())
    signed = bool(quantizer.search_result_is_signed.item())
    step = 2.0 ** lsb
    if signed:
        integer_max = 2 ** (quantizer.bit_width - 1) - 1
    else:
        integer_max = 2 ** quantizer.bit_width - 1
    return integer_max * step


def _calibrated_lsb(quantizer) -> int:
    return int(quantizer.search_result_lsb.item())


def _outlier_data():
    """Dense bulk well inside the saturation region, plus a handful of rare
    extreme outliers well beyond it -- exactly the shape that exposed the
    bug in the real exported model (most data unremarkable, a few outliers
    driving calibration)."""
    torch.manual_seed(0)
    bulk = torch.randn(2000, 1) * 2.0 + 1.0
    outliers = torch.tensor([[50.0]] * 5 + [[-40.0]] * 5)
    return torch.cat([bulk, outliers])


# =========================================================================
# A1: calibrated input range tracks the EXACT saturation point, not a
# power-of-two-padded version of it.
# =========================================================================


class TestInputCapTracksExactSaturationPoint:
    """Before the fix: ReLU6/Sigmoid cap at 8.0 (pow2 of 6.0), Tanh at 4.0
    (pow2 of 3.0) -- all strictly looser than the activation's real
    saturation point, costing a full extra bit of resolution. After the fix:
    the cap equals the saturation point exactly, so the calibrated q_max
    sits just above it (the smallest coverage-first q_max >= saturation),
    not up near the padded power-of-two value.
    """

    def test_relu6_input_q_max_close_to_six_not_eight(self):
        module = QuantReLU6(bit_width=8)
        module.train()
        module(_outlier_data())
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        # Exact-cap (6.0) coverage-first lands at q_max == 7.96875.
        # Padded-cap (8.0) coverage-first lands at q_max == 15.9375.
        # Threshold splits the two cleanly.
        assert q_max < 10.0, (
            f"ReLU6 input quantizer calibrated q_max={q_max}, expected close to "
            f"the exact saturation point (6.0), not a power-of-two-padded (8.0) "
            f"version of it"
        )

    def test_sigmoid_input_q_max_close_to_six_not_eight(self):
        module = QuantSigmoid(bit_width=8)
        module.train()
        module(_outlier_data())
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        assert q_max < 10.0, (
            f"Sigmoid input quantizer calibrated q_max={q_max}, expected close to "
            f"the exact saturation point (6.0), not a power-of-two-padded (8.0) "
            f"version of it"
        )

    def test_tanh_input_q_max_close_to_three_not_four(self):
        module = QuantTanh(bit_width=8)
        module.train()
        module(_outlier_data())
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        # Exact-cap (3.0) coverage-first lands at q_max == 3.96875.
        # Padded-cap (4.0) coverage-first lands at q_max == 7.9375.
        assert q_max < 5.0, (
            f"Tanh input quantizer calibrated q_max={q_max}, expected close to "
            f"the exact saturation point (3.0), not a power-of-two-padded (4.0) "
            f"version of it"
        )


# =========================================================================
# A2: the module applies no hidden padding beyond the documented saturation
# constants -- calibrating directly against the raw constant reproduces the
# exact same LSB the module's input quantizer actually picks.
# =========================================================================


class TestModuleCapMatchesRawSaturationConstant:
    @pytest.mark.parametrize("cls,saturation", [
        (QuantReLU6, _RELU6_SATURATION),
        (QuantSigmoid, _SIGMOID_SATURATION),
        (QuantTanh, _TANH_SATURATION),
    ])
    def test_module_lsb_matches_direct_cap_at_exact_saturation(self, cls, saturation):
        x = _outlier_data()
        module = cls(bit_width=8)
        module.train()
        module(x)
        q_in = _find_by_role_position(module, want_input=True)
        module_lsb = _calibrated_lsb(q_in)

        expected_lsb, _, _ = find_optimal_lsb(
            x, q_in.bit_width, bool(q_in.search_result_is_signed.item()),
            RoundingMode.ROUND_TO_NEAREST_EVEN, prefer_high_lsb=True,
            max_abs_value=saturation,
        )
        assert module_lsb == expected_lsb, (
            f"{cls.__name__}'s input quantizer picked lsb={module_lsb}, but "
            f"calibrating directly against the raw saturation constant "
            f"({saturation}) gives lsb={expected_lsb} -- the module is padding "
            f"the cap beyond the documented saturation point"
        )


# =========================================================================
# A3: end-to-end achievable resolution must not be bottlenecked by the input
# side when outliers are present -- the test the user explicitly asked for:
# count unique values at the END of the activation (after the output
# quantizer), not just inspect the guardrail in isolation.
# =========================================================================


class TestEndToEndResolutionNotBottleneckedByInput:
    """Reproduces the real-model symptom directly: with outliers in the
    pre-activation data, the OLD (padded-cap) input quantizer is coarser
    than the output quantizer, so the output's own resolution can never be
    reached -- the number of distinct values seen at the very end of the
    pipeline is silently capped far below what the output bit_width
    supports. The fix (exact-saturation cap) removes that bottleneck.
    """

    def test_relu6_end_to_end_unique_value_count_not_bottlenecked(self):
        module = QuantReLU6(bit_width=8)
        module.train()
        y = module(_outlier_data())
        n_unique = int(torch.unique(y).numel())
        # Old buggy behavior (input lsb=-4, step 0.0625) caps achievable
        # unique values at roughly 6/0.0625 ~= 96 regardless of what the
        # output quantizer (lsb=-5, step 0.03125) could otherwise resolve.
        # Fixed behavior (input lsb=-5, matching the output) should get
        # much closer to the output's own ~193-value budget over [0, 6].
        assert n_unique >= 150, (
            f"only {n_unique} distinct output values reachable end-to-end -- "
            f"the input quantizer is bottlenecking resolution that the output "
            f"quantizer's own bit_width could otherwise deliver"
        )


# =========================================================================
# A4: the fix must not make the guardrail stricter than before for data that
# never approaches the saturation point -- still just a ceiling.
# =========================================================================


class TestExactCapStillJustACeiling:
    def test_relu6_normal_range_data_still_not_clipped(self):
        torch.manual_seed(1)
        x = (torch.randn(2000, 1) * 1.0).clamp(-4.0, 4.0)
        module = QuantReLU6(bit_width=8)
        module.train()
        module(x)
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        assert q_max >= float(x.abs().max()), (
            f"guardrail is too strict: clipped normal data (abs max "
            f"{float(x.abs().max())}) to q_max={q_max}"
        )

    def test_tanh_normal_range_data_still_not_clipped(self):
        torch.manual_seed(2)
        x = (torch.randn(2000, 1) * 0.5).clamp(-2.0, 2.0)
        module = QuantTanh(bit_width=8)
        module.train()
        module(x)
        q_in = _find_by_role_position(module, want_input=True)
        q_max = _calibrated_q_max(q_in)
        assert q_max >= float(x.abs().max())
