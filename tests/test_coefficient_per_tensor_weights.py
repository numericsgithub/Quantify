import unittest
import torch
import os
from typing import Tuple

from quantizers.coefficient_per_tensor_weights import CoefficientPerTensorWeightQuantizer, CoefficientPerTensorWeightQuant
from brevitas.nn import QuantLinear

class TestCoefficientQuantizer(unittest.TestCase):
    def setUp(self):
        self.coeff_file = "tests/dummy_coeffs.txt"
        # Verify the dummy file exists before proceeding
        if not os.path.exists(self.coeff_file):
            self.fail(f"Required coefficient file {self.coeff_file} not found. Please ensure it is created.")

    def test_init_and_loading(self):
        """Test that coefficients are correctly loaded from the text file."""
        quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        self.assertEqual(len(quantizer.coefficient_sets), 2)
        # Set 0: [-1.0, 0.0, 1.0]
        self.assertTrue(torch.allclose(quantizer.coefficient_sets[0], torch.tensor([-1.0, 0.0, 1.0])))
        # Set 1: [-0.5, -0.25, 0.0, 0.25, 0.5]
        self.assertTrue(torch.allclose(quantizer.coefficient_sets[1], torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5])))

    def test_optimal_search_simple(self):
        """Test search for the best set and scale when weights clearly fit a specific set."""
        quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        # Weights clearly fitting set 0 with scale 1.0 (bit_shift_scale=0)
        weights = torch.tensor([-1.1, 0.1, 0.9])
        q, scale, zp, bw = quantizer(weights)
        
        self.assertEqual(quantizer.best_set_idx.item(), 0)
        self.assertEqual(quantizer.best_bit_shift_scale.item(), 0)
        self.assertTrue(torch.allclose(q, torch.tensor([-1.0, 0.0, 1.0])))
        self.assertEqual(scale.item(), 1.0)
        self.assertEqual(bw.item(), 3.0)

    def test_optimal_search_scaling(self):
        """Test that the quantizer finds the optimal power-of-two scale."""
        quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        # Weights fitting set 0 with scale 2.0 (bit_shift_scale=1) -> [-2.0, 0.0, 2.0]
        weights = torch.tensor([-2.1, 0.1, 1.9])
        q, scale, zp, bw = quantizer(weights)
        
        self.assertEqual(quantizer.best_bit_shift_scale.item(), 1)
        self.assertTrue(torch.allclose(q, torch.tensor([-2.0, 0.0, 2.0])))
        self.assertEqual(scale.item(), 2.0)

    def test_optimal_search_set_selection(self):
        """Test that the quantizer selects the set that minimizes SAD."""
        quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        # Weights fitting set 1 with scale 1.0 (bit_shift_scale=0) -> [-0.5, -0.25, 0.0, 0.25, 0.5]
        weights = torch.tensor([-0.48, -0.26, 0.01, 0.24, 0.51])
        q, scale, zp, bw = quantizer(weights)
        
        self.assertEqual(quantizer.best_set_idx.item(), 1)
        self.assertEqual(bw.item(), 5.0)

    def test_serialization(self):
        """Test that search results are preserved in the state_dict."""
        quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        weights = torch.tensor([-1.1, 0.1, 0.9])
        quantizer(weights) # Trigger search
        
        # Simulate saving and loading
        state = quantizer.state_dict()
        new_quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        new_quantizer.load_state_dict(state)
        
        self.assertTrue(new_quantizer.search_done.item())
        self.assertEqual(new_quantizer.best_set_idx.item(), 0)
        self.assertEqual(new_quantizer.best_bit_shift_scale.item(), 0)

    def test_brevitas_integration(self):
        """Test integration with Brevitas QuantLinear via the Injector."""
        # Create a subclass to specify the filepath for the test
        class TestCoeffQuant(CoefficientPerTensorWeightQuant):
            filepath = "tests/dummy_coeffs.txt"

        layer = QuantLinear(
            in_features=10,
            out_features=5,
            weight_quant=TestCoeffQuant
        )
        
        # Forward pass should trigger the search in the internal quantizer
        _ = layer(torch.randn(1, 10))
        
        # In Brevitas, the weight quantizer is instantiated as a proxy.
        # The actual quantizer module is stored in the tensor_quant attribute.
        self.assertTrue(layer.weight_quant.tensor_quant.search_done.item())

    def test_all_zeros(self):
        """Test behavior with all-zero weights."""
        quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        weights = torch.zeros(10)
        q, scale, zp, bw = quantizer(weights)
        self.assertTrue(torch.all(q == 0))

    def test_missing_file(self):
        """Test that a missing file raises the appropriate error."""
        with self.assertRaises(FileNotFoundError):
            CoefficientPerTensorWeightQuantizer("non_existent_file_12345.txt")

if __name__ == '__main__':
    unittest.main()
