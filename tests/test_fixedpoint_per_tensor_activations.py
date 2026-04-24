import unittest
import torch
import torch.nn as nn
from quantizers.fixedpoint_per_tensor_activations import (
    quantize_fixed_point,
    find_optimal_lsb,
    FixedPointPerTensorActivationQuantizer,
    FixedPointPerTensorActivationQuant,
    RoundingMode,
)

class TestQuantizeFixedPoint(unittest.TestCase):
    def test_unsigned_3bit_lsb_neg1(self):
        tensor = torch.tensor([0.1, 0.5, 1.0, 2.0])
        q = quantize_fixed_point(tensor, lsb=-1, bit_width=3, signed=False, rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        self.assertEqual(q.shape, tensor.shape)

    def test_signed_4bit_lsb_neg1_narrow(self):
        tensor = torch.tensor([-1.5, 0.0, 1.5, 2.0])
        q = quantize_fixed_point(tensor, lsb=-1, bit_width=4, signed=True, rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN, narrow_range=True)
        self.assertEqual(q.shape, tensor.shape)

    def test_clamp_below_range(self):
        tensor = torch.tensor([-10.0])
        q = quantize_fixed_point(tensor, lsb=0, bit_width=4, signed=True, rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        self.assertEqual(q.item(), -7.0)  # narrow range min for 4-bit signed is -7

    def test_clamp_above_range(self):
        tensor = torch.tensor([10.0])
        q = quantize_fixed_point(tensor, lsb=0, bit_width=4, signed=True, rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
        self.assertEqual(q.item(), 7.0)  # narrow range max for 4-bit signed is 7

class TestFindOptimalLsb(unittest.TestCase):
    def test_maximises_unique_values(self):
        torch.manual_seed(42)
        tensor = torch.randn(256) * 2.0
        signed = True
        bw = 4
        mode = RoundingMode.ROUND_TO_NEAREST_EVEN
        best_lsb = find_optimal_lsb(tensor, bw, signed, mode)
        self.assertIsInstance(best_lsb, int)

    def test_all_zeros(self):
        tensor = torch.zeros(10)
        lsb = find_optimal_lsb(tensor, 8, True, RoundingMode.ROUND_TO_NEAREST_EVEN)
        self.assertEqual(lsb, 0)

    def test_positive_weights_choose_unsigned_range(self):
        tensor = torch.tensor([0.1, 0.5, 1.0, 2.0])
        lsb = find_optimal_lsb(tensor, 4, False, RoundingMode.ROUND_TO_NEAREST_EVEN)
        self.assertIsInstance(lsb, int)

class TestQuantizerModule(unittest.TestCase):
    def test_output_shape(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        tensor = torch.randn(32, 64)
        q, scale, zp, bw = quantizer(tensor)
        self.assertEqual(q.shape, tensor.shape)

    def test_returns_four_tuple(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        tensor = torch.randn(10)
        out = quantizer(tensor)
        self.assertEqual(len(out), 4)

    def test_bit_width_returned(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        tensor = torch.randn(10)
        _, _, _, bw = quantizer(tensor)
        self.assertEqual(bw.item(), 4.0)

    def test_scale_is_power_of_two(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        tensor = torch.randn(10)
        _, scale, _, _ = quantizer(tensor)
        self.assertTrue(torch.isclose(scale, torch.tensor(2.0 ** round(torch.log2(scale).item()))))

    def test_zero_point_is_zero(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        tensor = torch.randn(10)
        _, _, zp, _ = quantizer(tensor)
        self.assertEqual(zp.item(), 0.0)

    def test_quantized_values_on_grid(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        tensor = torch.tensor([0.75, 1.25, -0.5])
        q, _, _, _ = quantizer(tensor)
        step = 2.0 ** quantizer.search_result_lsb.item()
        self.assertTrue(torch.allclose(q, torch.round(q / step) * step))

    def test_auto_unsigned_for_positive_weights(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        tensor = torch.tensor([0.1, 0.5, 1.0, 2.0])
        quantizer(tensor)
        self.assertFalse(quantizer.search_result_is_signed.item())

    def test_auto_signed_for_mixed_weights(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        tensor = torch.tensor([0.1, -0.5, 1.0, 2.0])
        quantizer(tensor)
        self.assertTrue(quantizer.search_result_is_signed.item())

class TestBrevitasIntegration(unittest.TestCase):
    def test_quantrelu_forward(self):
        from brevitas.nn import QuantReLU
        act = QuantReLU(act_quant=FixedPointPerTensorActivationQuant)
        x = torch.randn(2, 3, 4, 4)
        out = act(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_quantrelu_quant_act(self):
        from brevitas.nn import QuantReLU
        act = QuantReLU(act_quant=FixedPointPerTensorActivationQuant)
        x = torch.randn(2, 3, 4, 4)
        _ = act(x)
        # Verify quantizer proxy is attached and exposes scale attributes
        self.assertTrue(hasattr(act, 'act_quant'))
        self.assertTrue(hasattr(act.act_quant, 'quant_scale'))

    def test_custom_bit_width_via_subclass(self):
        from brevitas.nn import QuantReLU
        class My4bitActQuant(FixedPointPerTensorActivationQuant):
            bit_width = 4
        act = QuantReLU(act_quant=My4bitActQuant)
        x = torch.randn(2, 3, 4, 4)
        out = act(x)
        self.assertEqual(out.shape, x.shape)

class TestEdgeCases(unittest.TestCase):
    def test_single_value(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        tensor = torch.tensor([1.234])
        q, s, zp, bw = quantizer(tensor)
        self.assertEqual(q.shape, (1,))

    def test_very_small_values(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        tensor = torch.tensor([1e-6, 2e-6])
        q, _, _, _ = quantizer(tensor)
        self.assertEqual(q.shape, tensor.shape)

    def test_very_large_values(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        tensor = torch.tensor([1e6, 2e6])
        q, _, _, _ = quantizer(tensor)
        self.assertEqual(q.shape, tensor.shape)

    def test_gradient_passthrough(self):
        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        tensor = torch.randn(10, requires_grad=True)
        q, _, _, _ = quantizer(tensor)
        loss = q.sum()
        loss.backward()
        self.assertIsNotNone(tensor.grad)

if __name__ == "__main__":
    unittest.main()
