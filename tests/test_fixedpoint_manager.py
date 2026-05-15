import torch
import unittest
from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorQuantizer, 
    RoundingMode
)
from quantizers.manager import QuantizerManager


class TestFixedPointManager(unittest.TestCase):
    def setUp(self):
        # Create a fresh manager for each test to avoid state leakage
        self.manager = QuantizerManager()

    def test_quantizer_registration(self):
        """Verify that quantizer instances are automatically registered with the manager and given IDs."""
        q1 = FixedPointPerTensorQuantizer(bit_width=8, quantizer_manager=self.manager)
        q2 = FixedPointPerTensorQuantizer(bit_width=4, quantizer_manager=self.manager)
        
        self.assertIn(q1, self.manager.quantizers.values())
        self.assertIn(q2, self.manager.quantizers.values())
        self.assertEqual(len(self.manager.quantizers), 2)
        
        # Verify unique IDs were assigned
        self.assertTrue(hasattr(q1, 'quant_id'))
        self.assertTrue(hasattr(q2, 'quant_id'))
        self.assertNotEqual(q1.quant_id, q2.quant_id)

    def test_global_recalibration(self):
        """Verify that trigger_global_recalibration forces a re-run of LSB search."""
        q = FixedPointPerTensorQuantizer(bit_width=8, quantizer_manager=self.manager)
        
        # 1. Initial forward pass to calibrate
        # Use a tensor with a specific range to lock in an LSB
        data1 = torch.randn(100) * 1.0 
        q(data1)
        
        initial_lsb = q.search_result_lsb.item()
        self.assertTrue(q.search_done.item())

        # 2. Change data to something that would require a different LSB
        # (e.g., much larger values)
        data2 = torch.randn(100) * 100.0
        
        # Run forward without global recalibration flag
        q(data2)
        
        # LSB should NOT have changed because search_done is True
        self.assertEqual(q.search_result_lsb.item(), initial_lsb, 
                         "LSB changed without global recalibration flag being set")

        # 3. Trigger global recalibration
        self.manager.trigger_global_recalibration()
        
        # Run forward again
        q(data2)
        
        new_lsb = q.search_result_lsb.item()
        self.assertNotEqual(new_lsb, initial_lsb, 
                            "LSB did not change after global recalibration was triggered")
        
        # 4. Reset flag and verify it stops recalibrating
        self.manager.reset_global_flag()
        
        # Change data again
        data3 = torch.randn(100) * 0.001
        q(data3)
        
        self.assertEqual(q.search_result_lsb.item(), new_lsb, 
                         "LSB changed after global recalibration flag was reset")


if __name__ == "__main__":
    unittest.main()
