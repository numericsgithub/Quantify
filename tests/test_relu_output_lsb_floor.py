"""
Tests for the "no empty LSBs" rule on ReLU / ReLU6 output quantizers.

ReLU only zeroes values and ReLU6 only clamps them, so their output already
lies on the input quantizer's grid. An output quantizer calibrated to a
FINER LSB than the input's has codes that can never occur (and, via a
narrower range, can clip legitimate values). Observed in a real exported
model: input lsb=-3 (range +-32) followed by output lsb=-4 (range 0..15.9).

Rule under test: output LSB >= input LSB (never finer). See pitfall #25.
"""

import copy

import pytest
import torch

from quantizers.activations import QuantReLU, QuantReLU6, QuantSigmoid, QuantTanh, QuantLeakyReLU
from quantizers.base_quantizer import BaseQuantizer, reset_calibration_state
from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorQuantizer,
    LsbFloorLink,
    RoundingMode,
    find_optimal_lsb,
)
from quantizers.manager import QuantizerManager


def _in_out(module):
    """(input quantizer, output quantizer) -- input is the one whose forward
    runs first, which is also the one registered first."""
    qs = [m for m in module.modules() if isinstance(m, FixedPointPerTensorQuantizer)]
    assert len(qs) == 2
    return qs[0], qs[1]


def _negative_heavy_batch(seed=0):
    """Pre-activations whose |min| (-20) dominates the input quantizer's
    range while the positive max (12) -- all ReLU lets through -- is
    smaller: the input calibrates to lsb=-3, the unconstrained ReLU output
    to lsb=-4 (exactly the pair seen in the exported model)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(8, 32, 50, generator=g) * 10 - 5
    x.view(-1)[0] = -20.0
    x.view(-1)[1] = 12.0
    return x


def _fresh(cls, **kw):
    QuantizerManager().reset()
    return cls(**kw).train()


# ---------------------------------------------------------------------------
# find_optimal_lsb(min_lsb=...)
# ---------------------------------------------------------------------------

class TestFindOptimalLsbFloor:
    X = torch.linspace(0, 10, 500)

    @pytest.mark.parametrize("prefer_high_lsb", [True, False])
    def test_none_is_unchanged(self, prefer_high_lsb):
        a = find_optimal_lsb(self.X, 8, False, RoundingMode.FLOOR, prefer_high_lsb=prefer_high_lsb)
        b = find_optimal_lsb(self.X, 8, False, RoundingMode.FLOOR, prefer_high_lsb=prefer_high_lsb,
                             min_lsb=None)
        assert a[:2] == b[:2]

    @pytest.mark.parametrize("prefer_high_lsb", [True, False])
    def test_floor_above_natural_choice_is_applied(self, prefer_high_lsb):
        natural = find_optimal_lsb(self.X, 8, False, RoundingMode.FLOOR,
                                   prefer_high_lsb=prefer_high_lsb)[0]
        lsb = find_optimal_lsb(self.X, 8, False, RoundingMode.FLOOR,
                               prefer_high_lsb=prefer_high_lsb, min_lsb=natural + 2)[0]
        assert lsb == natural + 2

    @pytest.mark.parametrize("prefer_high_lsb", [True, False])
    def test_floor_below_natural_choice_changes_nothing(self, prefer_high_lsb):
        natural = find_optimal_lsb(self.X, 8, False, RoundingMode.FLOOR,
                                   prefer_high_lsb=prefer_high_lsb)[0]
        lsb = find_optimal_lsb(self.X, 8, False, RoundingMode.FLOOR,
                               prefer_high_lsb=prefer_high_lsb, min_lsb=natural - 3)[0]
        assert lsb == natural

    def test_floor_far_above_search_window_is_still_honoured(self):
        # window is only ceil(ideal)+12 wide; a floor beyond it must still win
        lsb = find_optimal_lsb(self.X, 8, False, RoundingMode.FLOOR,
                               prefer_high_lsb=True, min_lsb=40)[0]
        assert lsb == 40

    def test_all_zero_tensor_honours_floor(self):
        z = torch.zeros(16)
        assert find_optimal_lsb(z, 8, False, RoundingMode.FLOOR, min_lsb=3)[0] == 3
        assert find_optimal_lsb(z, 8, False, RoundingMode.FLOOR, min_lsb=-3)[0] == 0
        assert find_optimal_lsb(z, 8, False, RoundingMode.FLOOR)[0] == 0

    @pytest.mark.parametrize("min_lsb", [-9, -4, -1, 0, 2])
    @pytest.mark.parametrize("prefer_high_lsb", [True, False])
    def test_never_finer_than_floor_random(self, min_lsb, prefer_high_lsb):
        g = torch.Generator().manual_seed(0)
        for scale in (0.01, 0.3, 1, 7, 90):
            x = torch.randn(300, generator=g).abs() * scale
            lsb = find_optimal_lsb(x, 6, False, RoundingMode.FLOOR,
                                   prefer_high_lsb=prefer_high_lsb, min_lsb=min_lsb)[0]
            assert lsb >= min_lsb


# ---------------------------------------------------------------------------
# QuantReLU
# ---------------------------------------------------------------------------

class TestQuantReLUFloor:
    def test_output_grid_not_finer_than_input_grid(self):
        m = _fresh(QuantReLU)
        m(torch.randn(8, 32, 50) * 9)
        qin, qout = _in_out(m)
        assert qout._lsb_value >= qin._lsb_value

    def test_reported_case_input_coarser_than_natural_output(self):
        """Input data whose ReLU output would calibrate one bit finer than the
        input grid without the rule (input lsb=-3 -> unconstrained output -4)."""
        x = _negative_heavy_batch()
        off = _fresh(QuantReLU, match_input_lsb=False)
        off(x)
        qin, qout = _in_out(off)
        assert (qin._lsb_value, qout._lsb_value) == (-3, -4), "precondition: the bug reproduces without the rule"

        on = _fresh(QuantReLU)
        on(x)
        qin, qout = _in_out(on)
        assert qout._lsb_value == qin._lsb_value == -3

    def test_output_range_not_narrower_than_input_positive_range(self):
        m = _fresh(QuantReLU)
        m(_negative_heavy_batch())
        qin, qout = _in_out(m)
        _, in_max = qin._range_limits({"lsb": qin._lsb_value, "signed": qin._signed_value})
        _, out_max = qout._range_limits({"lsb": qout._lsb_value, "signed": qout._signed_value})
        assert out_max >= in_max

    def test_no_clipping_of_values_input_quantizer_let_through(self):
        m = _fresh(QuantReLU)
        x = torch.randn(8, 32, 50) * 9
        m(x)  # calibrate
        m.eval()
        with torch.no_grad():
            qin, _ = _in_out(m)
            expect = torch.relu(
                m.input_quant(x).detach() if not isinstance(m.input_quant(x), tuple) else m.input_quant(x)[0]
            )
            out = m(x)
            out = out.value if hasattr(out, "value") else out
        assert torch.allclose(out, expect)  # ReLU after quantization is lossless

    @pytest.mark.parametrize("bit_width", [2, 3, 4, 6, 8, 12])
    @pytest.mark.parametrize("scale", [0.02, 0.5, 3, 40, 900])
    @pytest.mark.parametrize("input_bit_width", [None, 4, 8])
    def test_invariant_over_widths_and_scales(self, bit_width, scale, input_bit_width):
        m = _fresh(QuantReLU, bit_width=bit_width, input_bit_width=input_bit_width)
        m(torch.randn(4, 16, 40) * scale + 0.3 * scale)
        qin, qout = _in_out(m)
        assert qout._lsb_value >= qin._lsb_value

    def test_every_output_code_is_reachable(self):
        """With the rule, output integers are the input integers (no
        always-empty odd codes): all output values are integer multiples of
        the output step AND already multiples of the input step."""
        m = _fresh(QuantReLU)
        x = torch.randn(16, 32, 50) * 9
        m(x)
        m.eval()
        with torch.no_grad():
            qin, qout = _in_out(m)
            out = m(x)
            out = out.value if hasattr(out, "value") else out
            step = 2.0 ** qout._lsb_value
            ints = torch.round(out / step)
            assert ints.unique().numel() > 2
            # equal grids -> every integer code between 0 and the max observed
            # that ReLU can produce is on the shared grid
            assert qout._lsb_value == qin._lsb_value

    def test_opt_out_flag_restores_independent_calibration(self):
        m = _fresh(QuantReLU, match_input_lsb=False)
        assert all(q.lsb_floor_link is None for q in m.modules() if isinstance(q, BaseQuantizer))

    def test_input_quantizer_never_gets_a_link(self):
        m = _fresh(QuantReLU)
        qin, qout = _in_out(m)
        assert qin.lsb_floor_link is None
        assert isinstance(qout.lsb_floor_link, LsbFloorLink)

    def test_no_new_state_dict_entries(self):
        a = _fresh(QuantReLU).state_dict().keys()
        b = _fresh(QuantReLU, match_input_lsb=False).state_dict().keys()
        assert list(a) == list(b)

    def test_fully_quantized_gated_input_means_no_floor_yet(self):
        m = _fresh(QuantReLU)
        qin, qout = _in_out(m)
        assert qout.lsb_floor_link.lsb_floor() is None  # input not calibrated
        m(torch.randn(4, 8, 20))
        assert qout.lsb_floor_link.lsb_floor() == qin._lsb_value

    def test_input_constant_never_calibrates_means_no_floor(self):
        m = _fresh(QuantReLU)
        qin, qout = _in_out(m)
        m(torch.zeros(4, 8, 20))  # num_unique == 1 -> search_done stays False
        assert qout.lsb_floor_link.lsb_floor() is None

    def test_recalibration_keeps_rule(self):
        m = _fresh(QuantReLU)
        m(torch.randn(8, 32, 50) * 9)
        for scale in (0.1, 30):
            reset_calibration_state(m)  # what the training harness does
            m(torch.randn(8, 32, 50) * scale)
            m(torch.randn(8, 32, 50) * scale)
            qin, qout = _in_out(m)
            assert qout._lsb_value >= qin._lsb_value

    def test_link_survives_load_state_dict_proxy_recreation(self):
        m = _fresh(QuantReLU)
        m(torch.randn(8, 32, 50) * 9)
        sd = copy.deepcopy(m.state_dict())
        m.load_state_dict(sd)  # Brevitas recreates tensor_quant objects here
        qin, qout = _in_out(m)
        assert isinstance(qout.lsb_floor_link, LsbFloorLink)
        assert qout.lsb_floor_link.lsb_floor() == qin._lsb_value
        # recalibrate from scratch on the new objects
        for q in (qin, qout):
            q.set_search_done(False)
        m(torch.randn(8, 32, 50) * 9)
        qin, qout = _in_out(m)
        assert qout._lsb_value >= qin._lsb_value

    def test_deepcopy_rebinds_link_to_the_copy(self):
        m = _fresh(QuantReLU)
        m(torch.randn(8, 32, 50) * 9)
        c = copy.deepcopy(m)
        _, qout_c = _in_out(c)
        assert qout_c.lsb_floor_link.owner is c
        assert qout_c.lsb_floor_link.owner is not m

    def test_custom_non_fixedpoint_input_quant_is_tolerated(self):
        """A link is a no-op when the input side isn't a FixedPoint quantizer."""
        link = LsbFloorLink().bind(torch.nn.Module())
        assert link.lsb_floor() is None
        assert LsbFloorLink().lsb_floor() is None  # unbound

    def test_works_inside_a_conv_network(self):
        net = torch.nn.Sequential(
            torch.nn.Conv1d(4, 8, 3, padding=1), QuantReLU(), torch.nn.Conv1d(8, 8, 3, padding=1), QuantReLU(),
        ).train()
        QuantizerManager().reset()
        net(torch.randn(4, 4, 64) * 5)
        for relu in (net[1], net[3]):
            qin, qout = _in_out(relu)
            assert qout._lsb_value >= qin._lsb_value


# ---------------------------------------------------------------------------
# QuantReLU6
# ---------------------------------------------------------------------------

class TestQuantReLU6Floor:
    def test_output_grid_not_finer_than_input_grid(self):
        m = _fresh(QuantReLU6)
        m(torch.randn(8, 32, 50) * 9)
        qin, qout = _in_out(m)
        assert qout._lsb_value >= qin._lsb_value or qout._lsb_value >= 1

    @pytest.mark.parametrize("scale", [0.05, 1, 5, 80])
    @pytest.mark.parametrize("bit_width", [3, 4, 6, 8])
    def test_never_finer_than_input_grid_up_to_the_six_cap(self, scale, bit_width):
        m = _fresh(QuantReLU6, bit_width=bit_width)
        m(torch.randn(8, 16, 40) * scale)
        qin, qout = _in_out(m)
        assert qout._lsb_value >= min(qin._lsb_value, 1)

    def test_floor_is_capped_so_six_stays_a_grid_point(self):
        """With a coarse input grid (lsb > 1) the output may still be as fine
        as lsb=1 (6 == 3 * 2**1), never forced coarser than that by the rule."""
        m = _fresh(QuantReLU6, bit_width=3, input_bit_width=3, input_max_abs_value=1000.0)
        m(torch.randn(8, 16, 40) * 500)
        qin, qout = _in_out(m)
        assert qin._lsb_value > 1, "precondition: coarse input grid"
        assert qout.lsb_floor_link.lsb_floor() == 1
        assert qout._lsb_value <= qin._lsb_value

    def test_opt_out(self):
        m = _fresh(QuantReLU6, match_input_lsb=False)
        _, qout = _in_out(m)
        assert qout.lsb_floor_link is None

    def test_rule_survives_deepcopy(self):
        m = _fresh(QuantReLU6)
        m(torch.randn(8, 32, 50) * 3)
        c = copy.deepcopy(m)
        _, qout = _in_out(c)
        assert qout.lsb_floor_link.owner is c


# ---------------------------------------------------------------------------
# Activations where LSBs are NOT empty must be untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [QuantSigmoid, QuantTanh, QuantLeakyReLU])
def test_other_activations_have_no_floor(cls):
    m = _fresh(cls)
    for q in m.modules():
        if isinstance(q, FixedPointPerTensorQuantizer):
            assert q.lsb_floor_link is None
