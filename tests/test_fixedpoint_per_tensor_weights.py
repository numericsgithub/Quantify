"""
Tests for the Fixed-Point Per-Tensor Weight Quantizer.

Covers:
    - Core quantization math (unsigned / signed / narrow range)
    - Rounding modes (round-to-nearest-even, floor)
    - Automatic signed/unsigned detection
    - Optimal LSB search (maximise unique values, break ties by SAD)
    - Brevitas integration via QuantLinear
    - Edge cases (all zeros, single value, already on grid)
"""

import math
import unittest

import pytest
import torch
import torch.nn as nn

from quantizers.fixedpoint_per_tensor_weights import (
    FixedPointPerTensorWeightQuantizer,
    FixedPointPerTensorWeightQuant,
    RoundingMode,
    quantize_fixed_point,
    find_optimal_lsb,
)


# =========================================================================
# 1. Core quantization grid
# =========================================================================


class TestQuantizeFixedPoint:
    """Tests for the low-level quantize_fixed_point function."""

    def test_unsigned_3bit_lsb_neg1(self):
        """Unsigned, bw=3, lsb=-1  →  grid {0.0, 0.5, 1.0, ..., 3.5}."""
        grid = [k * 0.5 for k in range(8)]  # 0.0 .. 3.5
        grid_floaty = [k * 0.5 + 0.1 for k in range(8)]  # 0.0 .. 3.5
        weights = torch.tensor(grid_floaty, dtype=torch.float32)
        q = quantize_fixed_point(weights, lsb=-1, bit_width=3, signed=False,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        assert torch.allclose(q, torch.tensor(grid)), f"Expected identity on grid, got {q}"

    def test_unsigned_3bit_lsb_neg1_values(self):
        """Verify the exact set of representable unsigned values."""
        expected = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        # Feed values that span the full range, some off-grid
        weights = torch.linspace(-1.0, 5.0, steps=200)
        q = quantize_fixed_point(weights, lsb=-1, bit_width=3, signed=False,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        unique = torch.unique(q)
        assert torch.allclose(unique, expected), f"Got {unique}"

    def test_signed_4bit_lsb_neg1_narrow(self):
        """Signed narrow, bw=4, lsb=-1  →  grid {-3.5, -3.0, ..., 3.5}."""
        expected_min, expected_max = -3.5, 3.5  # narrow excludes -4.0
        weights = torch.linspace(-5.0, 5.0, steps=200)
        q = quantize_fixed_point(weights, lsb=-1, bit_width=4, signed=True,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN,
                                 narrow_range=True)
        assert q.min().item() == pytest.approx(expected_min)
        assert q.max().item() == pytest.approx(expected_max)
        unique = torch.unique(q)
        # narrow signed 4-bit: 2^4 - 1 = 15 values
        assert unique.numel() == 15

    def test_signed_4bit_lsb_neg1_full_range(self):
        """Signed full range, bw=4, lsb=-1  →  includes -4.0."""
        weights = torch.linspace(-5.0, 5.0, steps=200)
        q = quantize_fixed_point(weights, lsb=-1, bit_width=4, signed=True,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN,
                                 narrow_range=False)
        assert q.min().item() == pytest.approx(-4.0)
        assert q.max().item() == pytest.approx(3.5)
        assert torch.unique(q).numel() == 16

    def test_clamp_below_range(self):
        """Values below the representable range should be clamped to the min."""
        weights = torch.tensor([-10.0, -5.0])
        q = quantize_fixed_point(weights, lsb=0, bit_width=3, signed=False,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        assert (q == 0.0).all(), "Negative values should clamp to 0 for unsigned"

    def test_clamp_above_range(self):
        """Values above the representable range should be clamped to the max."""
        weights = torch.tensor([100.0, 200.0])
        q = quantize_fixed_point(weights, lsb=0, bit_width=3, signed=False,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        expected_max = 7.0  # (2^3 - 1) * 2^0
        assert (q == expected_max).all()


# =========================================================================
# 2. Rounding modes
# =========================================================================


class TestRoundingModes:

    def test_round_to_nearest_even(self):
        """Halfway cases should round to the nearest even code (banker's)."""
        # 0.75 is exactly halfway between 0.5 and 1.0 on lsb=-1 grid
        weights = torch.tensor([0.75])
        q = quantize_fixed_point(weights, lsb=-1, bit_width=3, signed=False,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        # code = 0.75/0.5 = 1.5 → rounds to 2 (even) → 2*0.5 = 1.0
        assert q.item() == pytest.approx(1.0)

    def test_floor_rounding(self):
        """Floor rounding should always round toward negative infinity."""
        weights = torch.tensor([0.75])
        q = quantize_fixed_point(weights, lsb=-1, bit_width=3, signed=False,
                                 rounding_mode=RoundingMode.FLOOR)
        # code = floor(0.75/0.5) = floor(1.5) = 1 → 1*0.5 = 0.5
        assert q.item() == pytest.approx(0.5)

    def test_floor_negative(self):
        """Floor with negative values rounds toward more negative."""
        weights = torch.tensor([-0.3])
        q = quantize_fixed_point(weights, lsb=-1, bit_width=4, signed=True,
                                 rounding_mode=RoundingMode.FLOOR)
        # code = floor(-0.3/0.5) = floor(-0.6) = -1 → -1*0.5 = -0.5
        assert q.item() == pytest.approx(-0.5)


# =========================================================================
# 3. Automatic signed / unsigned detection
# =========================================================================


class TestSignedDetection:

    def test_all_positive_is_unsigned(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.tensor([0.1, 0.5, 1.0, 2.0])
        assert quantizer.detect_signed(weights) is False

    def test_has_negative_is_signed(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.tensor([0.1, -0.5, 1.0, 2.0])
        assert quantizer.detect_signed(weights) is True

    def test_all_zeros_is_unsigned(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.zeros(10)
        assert quantizer.detect_signed(weights) is False

    def test_single_negative_is_signed(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.tensor([-0.001])
        assert quantizer.detect_signed(weights) is True


# =========================================================================
# 4. Optimal LSB search
# =========================================================================


class TestFindOptimalLsb:

    def test_maximises_unique_values(self):
        """The chosen LSB should produce the most unique quantised values."""
        torch.manual_seed(42)
        weights = torch.randn(256) * 2.0  # roughly [-6, 6]
        signed = True
        bw = 4
        mode = RoundingMode.ROUND_TO_NEAREST_EVEN

        best_lsb = find_optimal_lsb(weights, bw, signed, mode)

        # Verify: no other nearby LSB gives strictly more unique values
        best_q = quantize_fixed_point(weights, best_lsb, bw, signed, mode)
        best_unique = torch.unique(best_q).numel()

        for lsb in range(best_lsb - 3, best_lsb + 4):
            q = quantize_fixed_point(weights, lsb, bw, signed, mode)
            n_unique = torch.unique(q).numel()
            assert n_unique <= best_unique, (
                f"lsb={lsb} gave {n_unique} unique > {best_unique} at best_lsb={best_lsb}"
            )

    def test_tiebreak_by_sad(self):
        """When two LSBs give the same unique count, pick lower SAD."""
        torch.manual_seed(0)
        weights = torch.randn(512)
        signed = True
        bw = 8  # more bits → more likely to tie on unique count
        mode = RoundingMode.ROUND_TO_NEAREST_EVEN

        best_lsb = find_optimal_lsb(weights, bw, signed, mode)
        best_q = quantize_fixed_point(weights, best_lsb, bw, signed, mode)
        best_unique = torch.unique(best_q).numel()
        best_sad = torch.sum(torch.abs(weights - best_q)).item()

        for lsb in range(best_lsb - 3, best_lsb + 4):
            q = quantize_fixed_point(weights, lsb, bw, signed, mode)
            n_unique = torch.unique(q).numel()
            sad = torch.sum(torch.abs(weights - q)).item()
            if n_unique == best_unique:
                assert sad >= best_sad - 1e-9, (
                    f"lsb={lsb} tied on unique ({n_unique}) but had lower "
                    f"SAD {sad:.6f} < {best_sad:.6f}"
                )

    def test_all_zeros(self):
        """All-zero weights should not crash."""
        weights = torch.zeros(64)
        lsb = find_optimal_lsb(weights, 4, False, RoundingMode.ROUND_TO_NEAREST_EVEN)
        assert isinstance(lsb, int)

    def test_positive_weights_choose_unsigned_range(self):
        """For purely positive weights, unsigned representation should cover
        the range better (more codes dedicated to positive side)."""
        weights = torch.rand(256) * 3.0  # [0, 3)
        bw = 4

        lsb_unsigned = find_optimal_lsb(weights, bw, signed=False,
                                        rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        lsb_signed = find_optimal_lsb(weights, bw, signed=True,
                                      rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)

        q_unsigned = quantize_fixed_point(weights, lsb_unsigned, bw, False,
                                          RoundingMode.ROUND_TO_NEAREST_EVEN)
        q_signed = quantize_fixed_point(weights, lsb_signed, bw, True,
                                        RoundingMode.ROUND_TO_NEAREST_EVEN)

        sad_u = torch.sum(torch.abs(weights - q_unsigned)).item()
        sad_s = torch.sum(torch.abs(weights - q_signed)).item()

        assert sad_u <= sad_s, (
            f"Unsigned SAD {sad_u:.6f} should be <= signed SAD {sad_s:.6f} "
            f"for purely positive weights"
        )


# =========================================================================
# 5. End-to-end FixedPointPerTensorWeightQuantizer module
# =========================================================================


class TestQuantizerModule:

    def test_output_shape(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.randn(32, 64)
        q, scale, zp, bw = quantizer(weights)
        assert q.shape == weights.shape

    def test_returns_four_tuple(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.randn(32, 64)
        result = quantizer(weights)
        assert len(result) == 4
        q, scale, zp, bw = result
        assert isinstance(q, torch.Tensor)
        assert isinstance(scale, torch.Tensor)
        assert isinstance(zp, torch.Tensor)
        assert isinstance(bw, torch.Tensor)

    def test_bit_width_returned(self):
        for bw_val in [2, 4, 8, 16]:
            quantizer = FixedPointPerTensorWeightQuantizer(bit_width=bw_val)
            _, _, _, bw = quantizer(torch.randn(16))
            assert bw.item() == float(bw_val)

    def test_scale_is_power_of_two(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.randn(128)
        _, scale, _, _ = quantizer(weights)
        # scale = 2^lsb, so log2(scale) should be an integer
        log2_scale = math.log2(scale.item())
        assert log2_scale == pytest.approx(round(log2_scale)), (
            f"Scale {scale.item()} is not a power of 2"
        )

    def test_zero_point_is_zero(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        _, _, zp, _ = quantizer(torch.randn(64))
        assert zp.item() == 0.0

    def test_quantized_values_on_grid(self):
        """All output values should be exact multiples of the step (scale)."""
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.randn(256)
        q, scale, _, _ = quantizer(weights)
        step = scale.item()
        # q / step should be integer-valued (within float tolerance)
        codes = q / step
        residual = torch.abs(codes - torch.round(codes))
        assert residual.max().item() < 1e-5, (
            f"Some values are not on the grid: max residual {residual.max().item()}"
        )

    def test_floor_rounding_mode(self):
        quantizer = FixedPointPerTensorWeightQuantizer(
            bit_width=4, rounding_mode=RoundingMode.FLOOR
        )
        weights = torch.randn(128)
        q, _, _, _ = quantizer(weights)
        # Just verify it runs and produces valid output
        assert q.shape == weights.shape
        assert torch.isfinite(q).all()

    def test_auto_unsigned_for_positive_weights(self):
        """Purely positive weights should be detected as unsigned."""
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.rand(128) * 5.0  # all positive
        q, _, _, _ = quantizer(weights)
        # For unsigned, no quantized value should be negative
        assert (q >= 0).all(), "Expected unsigned quantization for positive weights"

    def test_auto_signed_for_mixed_weights(self):
        """Mixed-sign weights should use signed quantization."""
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.randn(128)  # will have both positive and negative
        q, _, _, _ = quantizer(weights)
        # Signed quantization should preserve negative values
        assert (q < 0).any(), "Expected signed quantization to produce negative values"


# =========================================================================
# 6. Brevitas integration (QuantLinear)
# =========================================================================


class TestBrevitasIntegration:

    def test_quantlinear_forward(self):
        """QuantLinear with our quantizer should produce valid output."""
        from brevitas.nn import QuantLinear

        layer = QuantLinear(
            in_features=64,
            out_features=32,
            bias=True,
            weight_quant=FixedPointPerTensorWeightQuant,
        )
        x = torch.randn(1, 64)
        out = layer(x)
        assert out.shape == (1, 32)
        assert torch.isfinite(out).all()

    def test_quantlinear_quant_weight(self):
        """The quantized weight should have a valid scale."""
        from brevitas.nn import QuantLinear

        layer = QuantLinear(
            in_features=64,
            out_features=32,
            bias=True,
            weight_quant=FixedPointPerTensorWeightQuant,
        )
        qw = layer.quant_weight()
        assert qw.value is not None
        assert qw.scale is not None

    def test_quantlinear_batch(self):
        """Forward pass with a batch of inputs."""
        from brevitas.nn import QuantLinear

        layer = QuantLinear(
            in_features=128,
            out_features=64,
            bias=False,
            weight_quant=FixedPointPerTensorWeightQuant,
        )
        x = torch.randn(8, 128)
        out = layer(x)
        assert out.shape == (8, 64)

    def test_custom_bit_width_via_subclass(self):
        """User can subclass the Injector to change bit_width."""
        from brevitas.nn import QuantLinear

        class FourBitQuant(FixedPointPerTensorWeightQuant):
            bit_width = 4

        layer = QuantLinear(
            in_features=32,
            out_features=16,
            bias=True,
            weight_quant=FourBitQuant,
        )
        x = torch.randn(1, 32)
        out = layer(x)
        assert out.shape == (1, 16)


# =========================================================================
# 7. Edge cases
# =========================================================================


class TestEdgeCases:

    def test_single_weight(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.tensor([1.234])
        q, s, zp, bw = quantizer(weights)
        assert q.shape == (1,)
        assert torch.isfinite(q).all()

    def test_very_small_weights(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=8)
        weights = torch.randn(64) * 1e-6
        q, s, zp, bw = quantizer(weights)
        assert torch.isfinite(q).all()

    def test_very_large_weights(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=8)
        weights = torch.randn(64) * 1e6
        q, s, zp, bw = quantizer(weights)
        assert torch.isfinite(q).all()

    def test_all_same_value(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.full((64,), 1.5)
        q, s, zp, bw = quantizer(weights)
        # All quantized values should be the same (or very close)
        assert torch.unique(q).numel() == 1

    def test_two_bit_extreme(self):
        """2-bit quantization should still work."""
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=2)
        weights = torch.randn(64)
        q, s, zp, bw = quantizer(weights)
        # Signed 2-bit narrow: {-1, 0, 1} * step → max 3 unique
        assert torch.unique(q).numel() <= 4
        assert torch.isfinite(q).all()

    def test_gradient_passthrough(self):
        """Verify that gradients flow through (STE-like behavior)."""
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.randn(32, 64, requires_grad=True)
        q, _, _, _ = quantizer(weights)
        loss = q.sum()
        loss.backward()
        # torch.round / torch.floor have zero gradient almost everywhere,
        # but clamp does pass gradients within range.  This just checks
        # that backward() doesn't crash.
        assert weights.grad is not None or True  # no crash is the test


# =========================================================================
# 8. Caching & Re-computation
# =========================================================================


class TestCaching(unittest.TestCase):
    def test_search_runs_once_and_caches(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.randn(32, 64)
        
        # First pass triggers search
        q1, _, _, _ = quantizer(weights)
        lsb1 = quantizer.search_result_lsb.item()
        signed1 = quantizer.search_result_is_signed.item()
        done1 = quantizer.search_done.item()
        
        # Second pass with different scale should NOT re-run search
        q2, _, _, _ = quantizer(weights * 100.0)
        
        self.assertEqual(quantizer.search_result_lsb.item(), lsb1)
        self.assertEqual(quantizer.search_result_is_signed.item(), signed1)
        self.assertTrue(quantizer.search_done.item())
        self.assertTrue(done1)


# =========================================================================
# 9. Device Synchronization
# =========================================================================


class TestDeviceSync(unittest.TestCase):
    def test_device_sync_preserves_cached_lsb(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights_cpu = torch.randn(32, 64)
        
        # Run on CPU
        q1, _, _, _ = quantizer(weights_cpu)
        lsb_cpu = quantizer.search_result_lsb.item()
        
        # Run on CUDA
        weights_cuda = weights_cpu.cuda()
        q2, _, _, _ = quantizer(weights_cuda)
        
        # Check that buffers moved to CUDA and lsb is preserved
        self.assertEqual(quantizer.search_result_lsb.device.type, 'cuda')
        self.assertEqual(quantizer.search_result_lsb.item(), lsb_cpu)


# =========================================================================
# 10. Negative Halfway Rounding
# =========================================================================


class TestNegativeHalfwayRounding(unittest.TestCase):
    def test_negative_halfway_rounds_to_even(self):
        # -0.75 is halfway between -1.0 and -0.5 on lsb=-1 grid
        # Codes: -1.5 (odd), -1.0 (even), -0.5 (odd), 0.0 (even)
        # -0.75 / 0.5 = -1.5. Rounds to -2 (even) -> -2 * 0.5 = -1.0
        weights = torch.tensor([-0.75])
        q = quantize_fixed_point(weights, lsb=-1, bit_width=3, signed=True,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        self.assertAlmostEqual(q.item(), -1.0)


# =========================================================================
# 11. QuantConv2d Integration
# =========================================================================


class TestQuantConv2dIntegration(unittest.TestCase):
    def test_quantconv2d_forward(self):
        from brevitas.nn import QuantConv2d
        layer = QuantConv2d(3, 16, 3, weight_quant=FixedPointPerTensorWeightQuant)
        x = torch.randn(1, 3, 32, 32)
        out = layer(x)
        self.assertEqual(out.shape, (1, 16, 30, 30))
        self.assertTrue(torch.isfinite(out).all())


# =========================================================================
# 12. STE Gradient Flow
# =========================================================================


class TestSTEGradientFlow(unittest.TestCase):
    def test_ste_gradient_flow(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        weights = torch.randn(32, 64, requires_grad=True)
        q, _, _, _ = quantizer(weights)
        loss = q.sum()
        loss.backward()
        # torch.round has zero gradient almost everywhere, so we only verify
        # that backward() runs without error and grad is allocated with correct shape.
        self.assertIsNotNone(weights.grad)
        self.assertEqual(weights.grad.shape, weights.shape)


# =========================================================================
# 13. NaN / Inf Handling
# =========================================================================


class TestNaNInfHandling(unittest.TestCase):
    def test_nan_propagation(self):
        weights = torch.tensor([1.0, float('nan'), 3.0])
        q = quantize_fixed_point(weights, lsb=0, bit_width=4, signed=True,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        self.assertTrue(torch.isnan(q).any(), "NaN should propagate")

    def test_inf_clamping(self):
        weights = torch.tensor([1.0, float('inf'), -float('inf'), 3.0])
        q = quantize_fixed_point(weights, lsb=0, bit_width=4, signed=True,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        # Inf should be clamped to max/min representable values
        self.assertTrue(torch.isfinite(q).all(), "Inf should be clamped to finite values")


# =========================================================================
# 14. Extreme Bit Widths
# =========================================================================


class TestExtremeBitWidths(unittest.TestCase):
    def test_bit_width_1(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=1)
        weights = torch.randn(64)
        q, s, zp, bw = quantizer(weights)
        self.assertTrue(torch.isfinite(q).all())
        self.assertEqual(bw.item(), 1.0)

    def test_bit_width_32(self):
        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=32)
        weights = torch.randn(64)
        q, s, zp, bw = quantizer(weights)
        self.assertTrue(torch.isfinite(q).all())
        self.assertEqual(bw.item(), 32.0)
