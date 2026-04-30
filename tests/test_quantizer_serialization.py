import unittest
import torch
from quantizers import FixedPointPerTensorQuantizer

class TestQuantizerSerialization(unittest.TestCase):
    """
    Tests that the FixedPointPerTensorQuantizer correctly serializes
    its search results (LSB and signedness) so that they are not re-calculated
    after loading a model.
    """

    def test_serialization_preserves_search_results(self):
        # 1. Setup
        bit_width = 4
        quantizer = FixedPointPerTensorQuantizer(bit_width=bit_width)
        
        # Use weights that will trigger a specific LSB search result
        # Weights in range [-1, 1] with bw=4 should result in a specific LSB
        weights = torch.randn(10, 10)
        
        # 2. Use the quantizer once to trigger LSB search
        quantizer(weights)
        
        # Capture the results of the search from the buffers
        original_lsb = quantizer.search_result_lsb.item()
        original_signed = quantizer.search_result_is_signed.item()
        original_done = quantizer.search_done.item()
        
        self.assertTrue(original_done, "Search should be marked as done after first forward pass")
        
        # 3. Save the state_dict
        state_dict = quantizer.state_dict()
        
        # 4. Create a new quantizer and load the state_dict
        new_quantizer = FixedPointPerTensorQuantizer(bit_width=bit_width)
        new_quantizer.load_state_dict(state_dict)
        
        # 5. Verify that the search results are preserved in the new instance
        self.assertEqual(new_quantizer.search_done.item(), original_done)
        self.assertEqual(new_quantizer.search_result_lsb.item(), original_lsb)
        self.assertEqual(new_quantizer.search_result_is_signed.item(), original_signed)
        
        # 6. Verify that it uses the loaded LSB even if provided with weights 
        # that would normally trigger a different LSB search result.
        
        # Use weights that are much larger, which would normally result in a much higher LSB
        different_weights = torch.randn(10, 10) * 1000.0 
        
        q_new, scale_new, _, _ = new_quantizer(different_weights)
        
        # The scale should be based on the loaded LSB (2^original_lsb), 
        # NOT the LSB that would be optimal for different_weights.
        expected_scale = 2.0 ** original_lsb
        self.assertAlmostEqual(scale_new.item(), expected_scale, places=5, 
                               msg="Quantizer re-searched LSB instead of using serialized value")

if __name__ == '__main__':
    unittest.main()
