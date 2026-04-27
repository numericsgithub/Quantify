"""
Tests for the Fixed-Point Per-Tensor Activation Quantizer.

Covers:
    - Core quantization math (unsigned / signed / narrow range)
    - Rounding modes (round-to-nearest-even, floor)
    - Automatic signed/unsigned detection
    - Optimal LSB search (maximise unique values, break ties by SAD)
    - Brevitas integration via QuantReLU
    - Edge cases (all zeros, single value, already on grid)
"""

import math
import unittest

import pytest
import torch
import torch.nn as nn

from quantizers.fixedpoint_per_tensor_activations import (
    FixedPointPerTensorActivationQuantizer,
    FixedPointPerTensorActivationQuant,
    RoundingMode,
    quantize_fixed_point,
    find_optimal_lsb,
)


# =========================================================================
# 1. Core quantization grid
# =========================================================================


class TestQuantizeFixedPoint(unittest.TestCase):
    """Tests for the low-level quantize_fixed_point function."""

    def test_unsigned_3bit_lsb_neg1(self):
        """Unsigned, bw=3, lsb=-1  →  grid {0.0, 0.5, 1.0, ..., 3.5}."""
        grid = [k * 0.5 for k in range(8)]  # 0.0 .. 3.5
        grid_floaty = [k * 0.5 + 0.1 for k in range(8)]  # 0.0 .. 3.5
        weights = torch.tensor(grid_floaty, dtype=torch.float32)
        q = quantize_fixed_point(weights, lsb=-1, bit_width=3, signed=False,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        self.assertTrue(torch.allclose(q, torch.tensor(grid)), f"Expected identity on grid, got {q}")

    def test_unsigned_3bit_lsb_neg1_values(self):
        """Verify the exact set of representable unsigned values."""
        expected = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        # Feed values that span the full range, some off-grid
        weights = torch.linspace(-1.0, 5.0, steps=200)
        q = quantize_fixed_point(weights, lsb=-1, bit_width=3, signed=False,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        unique = torch.unique(q)
        self.assertTrue(torch.allclose(unique, expected), f"Got {unique}")

    def test_signed_4bit_lsb_neg1_narrow(self):
        """Signed narrow, bw=4, lsb=-1  →  grid {-3.5, -3.0, ..., 3.5}."""
        expected_min, expected_max = -3.5, 3.5  # narrow excludes -4.0
        weights = torch.linspace(-5.0, 5.0, steps=200)
        q = quantize_fixed_point(weights, lsb=-1, bit_width=4, signed=True,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN,
                                 narrow_range=True)
        self.assertAlmostEqual(q.min().item(), expected_min)
        self.assertAlmostEqual(q.max().item(), expected_max)
        unique = torch.unique(q)
        # narrow signed 4-bit: 2^4 - 1 = 15 values
        self.assertEqual(unique.numel(), 15)

    def test_signed_4bit_lsb_neg1_full_range(self):
        """Signed full range, bw=4, lsb=-1  →  includes -4.0."""
        weights = torch.linspace(-5.0, 5.0, steps=200)
        q = quantize_fixed_point(weights, lsb=-1, bit_width=4, signed=True,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN,
                                 narrow_range=False)
        self.assertAlmostEqual(q.min().item(), -4.0)
        self.assertAlmostEqual(q.max().item(), 3.5)
        self.assertEqual(torch.unique(q).numel(), 16)

    def test_clamp_below_range(self):
        """Values below the representable range should be clamped to the min."""
        weights = torch.tensor([-10.0, -5.0])
        q = quantize_fixed_point(weights, lsb=0, bit_width=3, signed=False,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        self.assertTrue((q == 0.0).all(), "Negative values should clamp to 0 for unsigned")

    def test_clamp_above_range(self):
        """Values above the representable range should be clamped to the max."""
        weights = torch.tensor([100.0, 200.0])
        q = quantize_fixed_point(weights, lsb=0, bit_width=3, signed=False,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        expected_max = 7.0  # (2^3 - 1) * 2^0
        self.assertTrue((q == expected_max).all())


# =========================================================================
# 2. Rounding modes
# =========================================================================


class TestRoundingModes(unittest.TestCase):

    def test_round_to_nearest_even(self):
        """Halfway cases should round to the nearest even code (banker's)."""
        # 0.75 is exactly halfway between 0.5 and 1.0 on lsb=-1 grid
        weights = torch.tensor([0.75])
        q = quantize_fixed_point(weights, lsb=-1, bit_width=3, signed=False,
                                 rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        # code = 0.75/0.5 = 1.5 → rounds to 2 (even) → 2*0.5 = 1.0
        self.assertAlmostEqual(q.item(), 1.0)

    def test_floor_rounding(self):
        """Floor rounding should always round toward negative infinity."""
        weights = torch.tensor([0.75])
        q = quantize_fixed_point(weights, lsb=-1, bit_width=3, signed=False,
                                 rounding_mode=RoundingMode.FLOOR)
        # code = floor(0.75/0.5) = floor(1.5) = 1 → 1*0.5 = 0.5
        self.assertAlmostEqual(q.item(), 0.5)

    def test_floor_negative(self):
        """Floor with negative values rounds toward more negative."""
        weights = torch.tensor([-0.3])
        q = quantize_fixed_point(weights, lsb=-1, bit_width=4, signed=True,
                                 rounding_mode=RoundingMode.FLOOR)
        # code = floor(-0.3/0.5) = floor(-0.6) = -1 → -1*0.5 = -0.5
        self.assertAlmostEqual(q.item(), -0.5)


# =========================================================================
# 3. Automatic signed / unsigned detection
# =========================================================================


class TestSignedDetection(unittest.TestCase):

    def test_all_positive_is_unsigned(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.tensor([0.1, 0.5, 1.0, 2.0])
        self.assertFalse(quantizer.detect_signed(weights))

    def test_has_negative_is_signed(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.tensor([0.1, -0.5, 1.0, 2.0])
        self.assertTrue(quantizer.detect_signed(weights))

    def test_all_zeros_is_unsigned(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.zeros(10)
        self.assertFalse(quantizer.detect_signed(weights))

    def test_single_negative_is_signed(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.tensor([-0.001])
        self.assertTrue(quantizer.detect_signed(weights))


# =========================================================================
# 4. Optimal LSB search
# =========================================================================


class TestFindOptimalLsb(unittest.TestCase):

    def test_maximises_unique_values(self):
        """The chosen LSB should produce the most unique quantised values."""
        torch.manual_seed(42)
        weights = torch.randn(256) * 2.0  # roughly [-6, 6]
        signed = True
        bw = 4
        mode = RoundingMode.ROUND_TO_NEAREST_EVEN

        best_lsb, _ = find_optimal_lsb(weights, bw, signed, mode)

        # Verify: no other nearby LSB gives strictly more unique values
        best_q = quantize_fixed_point(weights, best_lsb, bw, signed, mode)
        best_unique = torch.unique(best_q).numel()

        for lsb in range(best_lsb - 3, best_lsb + 4):
            q = quantize_fixed_point(weights, lsb, bw, signed, mode)
            n_unique = torch.unique(q).numel()
            self.assertLessEqual(n_unique, best_unique,
                                 f"lsb={lsb} gave {n_unique} unique > {best_unique} at best_lsb={best_lsb}")

    def test_tiebreak_by_sad(self):
        """When two LSBs give the same unique count, pick lower SAD."""
        torch.manual_seed(0)
        weights = torch.randn(512)
        signed = True
        bw = 8  # more bits → more likely to tie on unique count
        mode = RoundingMode.ROUND_TO_NEAREST_EVEN

        best_lsb, _ = find_optimal_lsb(weights, bw, signed, mode)
        best_q = quantize_fixed_point(weights, best_lsb, bw, signed, mode)
        best_unique = torch.unique(best_q).numel()
        best_sad = torch.sum(torch.abs(weights - best_q)).item()

        for lsb in range(best_lsb - 3, best_lsb + 4):
            q = quantize_fixed_point(weights, lsb, bw, signed, mode)
            n_unique = torch.unique(q).numel()
            sad = torch.sum(torch.abs(weights - q)).item()
            if n_unique == best_unique:
                self.assertGreaterEqual(sad, best_sad - 1e-9,
                                        f"lsb={lsb} tied on unique ({n_unique}) but had lower "
                                        f"SAD {sad:.6f} < {best_sad:.6f}")

    def test_all_zeros(self):
        """All-zero weights should not crash."""
        weights = torch.zeros(64)
        lsb, _ = find_optimal_lsb(weights, 4, False, RoundingMode.ROUND_TO_NEAREST_EVEN)
        self.assertIsInstance(lsb, int)

    def test_positive_weights_choose_unsigned_range(self):
        """For purely positive weights, unsigned representation should cover
        the range better (more codes dedicated to positive side)."""
        weights = torch.rand(256) * 3.0  # [0, 3)
        bw = 4

        lsb_unsigned, _ = find_optimal_lsb(weights, bw, signed=False,
                                        rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        lsb_signed, _ = find_optimal_lsb(weights, bw, signed=True,
                                      rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)

        q_unsigned = quantize_fixed_point(weights, lsb_unsigned, bw, False,
                                          RoundingMode.ROUND_TO_NEAREST_EVEN)
        q_signed = quantize_fixed_point(weights, lsb_signed, bw, True,
                                        RoundingMode.ROUND_TO_NEAREST_EVEN)

        sad_u = torch.sum(torch.abs(weights - q_unsigned)).item()
        sad_s = torch.sum(torch.abs(weights - q_signed)).item()

        self.assertLessEqual(sad_u, sad_s,
                             f"Unsigned SAD {sad_u:.6f} should be <= signed SAD {sad_s:.6f} "
                             f"for purely positive weights")


# =========================================================================
# 5. End-to-end FixedPointPerTensorActivationQuantizer module
# =========================================================================


class TestQuantizerModule(unittest.TestCase):

    def test_output_shape(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.randn(32, 64)
        q, scale, zp, bw = quantizer(weights)
        self.assertEqual(q.shape, weights.shape)

    def test_returns_four_tuple(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.randn(32, 64)
        result = quantizer(weights)
        self.assertEqual(len(result), 4)
        q, scale, zp, bw = result
        self.assertIsInstance(q, torch.Tensor)
        self.assertIsInstance(scale, torch.Tensor)
        self.assertIsInstance(zp, torch.Tensor)
        self.assertIsInstance(bw, torch.Tensor)

    def test_bit_width_returned(self):
        for bw_val in [2, 4, 8, 16]:
            quantizer = FixedPointPerTensorActivationQuantizer(bit_width=bw_val)
            _, _, _, bw = quantizer(torch.randn(16))
            self.assertAlmostEqual(bw.item(), float(bw_val))

    def test_scale_is_power_of_two(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.randn(128)
        _, scale, _, _ = quantizer(weights)
        # scale = 2^lsb, so log2(scale) should be an integer
        log2_scale = math.log2(scale.item())
        self.assertAlmostEqual(log2_scale, round(log2_scale), places=5,
                               msg=f"Scale {scale.item()} is not a power of 2")

    def test_zero_point_is_zero(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        _, _, zp, _ = quantizer(torch.randn(64))
        self.assertAlmostEqual(zp.item(), 0.0)

    def test_quantized_values_on_grid(self):
        """All output values should be exact multiples of the step (scale)."""
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.randn(256)
        q, scale, _, _ = quantizer(weights)
        step = scale.item()
        # q / step should be integer-valued (within float tolerance)
        codes = q / step
        residual = torch.abs(codes - torch.round(codes))
        self.assertLess(residual.max().item(), 1e-5,
                        f"Some values are not on the grid: max residual {residual.max().item()}")

    def test_floor_rounding_mode(self):
        quantizer = FixedPointPerTensorActivationQuantizer(
            bit_width=4, rounding_mode=RoundingMode.FLOOR
        )
        weights = torch.randn(128)
        q, _, _, _ = quantizer(weights)
        # Just verify it runs and produces valid output
        self.assertEqual(q.shape, weights.shape)
        self.assertTrue(torch.isfinite(q).all())

    def test_auto_unsigned_for_positive_weights(self):
        """Purely positive weights should be detected as unsigned."""
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.rand(128) * 5.0  # all positive
        q, _, _, _ = quantizer(weights)
        # For unsigned, no quantized value should be negative
        self.assertTrue((q >= 0).all(), "Expected unsigned quantization for positive weights")

    def test_auto_signed_for_mixed_weights(self):
        """Mixed-sign weights should use signed quantization."""
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.randn(128)  # will have both positive and negative
        q, _, _, _ = quantizer(weights)
        # Signed quantization should preserve negative values
        self.assertTrue((q < 0).any(), "Expected signed quantization to produce negative values")


# =========================================================================
# 6. Brevitas integration (QuantReLU)
# =========================================================================


class TestBrevitasIntegration(unittest.TestCase):

    def test_quantrelu_forward(self):
        """QuantReLU with our quantizer should produce valid output."""
        from brevitas.nn import QuantReLU

        act = QuantReLU(act_quant=FixedPointPerTensorActivationQuant)
        x = torch.randn(2, 3, 4, 4)
        out = act(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_quantrelu_quant_act(self):
        """The quantized activation should have a valid scale."""
        from brevitas.nn import QuantReLU

        act = QuantReLU(act_quant=FixedPointPerTensorActivationQuant)
        x = torch.randn(2, 3, 4, 4)
        _ = act(x)
        # Verify quantizer proxy is attached
        self.assertTrue(hasattr(act, 'act_quant'))
        # Brevitas proxies expose scale as 'quant_scale' or 'scale' depending on version
        self.assertTrue(hasattr(act.act_quant, 'quant_scale') or hasattr(act.act_quant, 'scale'))

    def test_quantrelu_batch(self):
        """Forward pass with a batch of inputs."""
        from brevitas.nn import QuantReLU

        act = QuantReLU(act_quant=FixedPointPerTensorActivationQuant)
        x = torch.randn(8, 16, 4, 4)
        out = act(x)
        self.assertEqual(out.shape, x.shape)

    def test_custom_bit_width_via_subclass(self):
        """User can subclass the Injector to change bit_width."""
        from brevitas.nn import QuantReLU

        class FourBitActQuant(FixedPointPerTensorActivationQuant):
            bit_width = 4

        act = QuantReLU(act_quant=FourBitActQuant)
        x = torch.randn(2, 3, 4, 4)
        out = act(x)
        self.assertEqual(out.shape, x.shape)


# =========================================================================
# 7. Edge cases
# =========================================================================


class TestEdgeCases(unittest.TestCase):

    def test_single_weight(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.tensor([1.234])
        q, s, zp, bw = quantizer(weights)
        self.assertEqual(q.shape, (1,))
        self.assertTrue(torch.isfinite(q).all())

    def test_very_small_weights(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=8)
        weights = torch.randn(64) * 1e-6
        q, s, zp, bw = quantizer(weights)
        self.assertTrue(torch.isfinite(q).all())

    def test_very_large_weights(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=8)
        weights = torch.randn(64) * 1e6
        q, s, zp, bw = quantizer(weights)
        self.assertTrue(torch.isfinite(q).all())

    def test_all_same_value(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.full((64,), 1.5)
        q, s, zp, bw = quantizer(weights)
        # All quantized values should be the same (or very close)
        self.assertEqual(torch.unique(q).numel(), 1)

    def test_two_bit_extreme(self):
        """2-bit quantization should still work."""
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=2)
        weights = torch.randn(64)
        q, s, zp, bw = quantizer(weights)
        # Signed 2-bit narrow: {-1, 0, 1} * step → max 3 unique
        self.assertLessEqual(torch.unique(q).numel(), 4)
        self.assertTrue(torch.isfinite(q).all())

    def test_gradient_passthrough(self):
        """Verify that gradients flow through (STE-like behavior)."""
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.randn(32, 64, requires_grad=True)
        q, _, _, _ = quantizer(weights)
        loss = q.sum()
        loss.backward()
        # torch.round / torch.floor have zero gradient almost everywhere,
        # but clamp does pass gradients within range.  This just checks
        # that backward() doesn't crash.
        self.assertIsNotNone(weights.grad)


# =========================================================================
# 8. Caching & Re-computation
# =========================================================================


class TestCaching(unittest.TestCase):
    def test_search_runs_once_and_caches(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
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
            
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
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
# 11. STE Gradient Flow
# =========================================================================


class TestSTEGradientFlow(unittest.TestCase):
    def test_ste_gradient_flow(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        weights = torch.randn(32, 64, requires_grad=True)
        q, _, _, _ = quantizer(weights)
        loss = q.sum()
        loss.backward()
        self.assertIsNotNone(weights.grad)
        self.assertEqual(weights.grad.shape, weights.shape)


# =========================================================================
# 12. NaN / Inf Handling
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
# 13. Extreme Bit Widths
# =========================================================================


class TestExtremeBitWidths(unittest.TestCase):
    def test_bit_width_1(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=1)
        weights = torch.randn(64)
        q, s, zp, bw = quantizer(weights)
        self.assertTrue(torch.isfinite(q).all())
        self.assertAlmostEqual(bw.item(), 1.0)

    def test_bit_width_32(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=32)
        weights = torch.randn(64)
        q, s, zp, bw = quantizer(weights)
        self.assertTrue(torch.isfinite(q).all())
        self.assertAlmostEqual(bw.item(), 32.0)
