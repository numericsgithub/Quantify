import torch
import unittest
from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorQuantizer, 
    RoundingMode
)
from quantizers.manager import QuantizerManager


class TestFixedPointManager(unittest.TestCase):
    def setUp(self):
        # Reset the singleton manager to avoid state leakage from previous tests
        QuantizerManager().reset()
        self.manager = QuantizerManager()

    def tearDown(self):
        # Reset again so state set by this test (e.g. quantization_start_gap)
        # doesn't leak into later test files via the shared singleton.
        QuantizerManager().reset()

    def test_quantizer_registration(self):
        """Verify that quantizer instances are automatically registered with the manager and given IDs."""
        # Instantiate quantizers normally (they create their own local managers by default)
        q1 = FixedPointPerTensorQuantizer(bit_width=8)
        q2 = FixedPointPerTensorQuantizer(bit_width=4)
        
        # Create a shared manager to test cross-instance ID uniqueness
        shared_manager = QuantizerManager()
        
        # Swap the manager to the shared one for this test
        q1.quantizer_manager = shared_manager
        q2.quantizer_manager = shared_manager
        
        # Register them in the shared manager
        shared_manager.register_quantizer(q1)
        shared_manager.register_quantizer(q2)
        
        self.assertIn(q1, shared_manager.quantizers.values())
        self.assertIn(q2, shared_manager.quantizers.values())
        self.assertEqual(len(shared_manager.quantizers), 2)
        
        # Verify unique IDs were assigned
        self.assertTrue(hasattr(q1, 'quant_id'))
        self.assertTrue(hasattr(q2, 'quant_id'))
        self.assertNotEqual(q1.quant_id, q2.quant_id)

    def test_global_recalibration(self):
        """Verify that trigger_global_recalibration forces a re-run of LSB search."""
        q = FixedPointPerTensorQuantizer(bit_width=8)
        manager = q.quantizer_manager
        
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
        manager.trigger_global_recalibration()
        
        # Run forward again
        q(data2)
        
        new_lsb = q.search_result_lsb.item()
        self.assertNotEqual(new_lsb, initial_lsb, 
                            "LSB did not change after global recalibration was triggered")
        
        # 4. Reset flag and verify it stops recalibrating
        manager.reset_global_flag()
        
        # Change data again
        data3 = torch.randn(100) * 0.001
        q(data3)
        
        self.assertEqual(q.search_result_lsb.item(), new_lsb,
                         "LSB changed after global recalibration flag was reset")

    def test_quantizers_in_execution_order_basic(self):
        """Sequence id ordering reflects forward call order, not registration order."""
        q1 = FixedPointPerTensorQuantizer(bit_width=8)
        q2 = FixedPointPerTensorQuantizer(bit_width=8)
        q3 = FixedPointPerTensorQuantizer(bit_width=8)
        # Forward in a deliberately different order than registration (q1, q2, q3).
        q3(torch.randn(10))
        q1(torch.randn(10))
        q2(torch.randn(10))

        ordered = self.manager.quantizers_in_execution_order()
        self.assertEqual([q.quant_id for q in ordered], [q3.quant_id, q1.quant_id, q2.quant_id])

    def test_quantizers_in_execution_order_raises_before_forward(self):
        """Calling before any forward pass must raise, not return a misleading order."""
        FixedPointPerTensorQuantizer(bit_width=8)
        FixedPointPerTensorQuantizer(bit_width=8)
        with self.assertRaises(RuntimeError):
            self.manager.quantizers_in_execution_order()

    def test_quantizers_in_execution_order_empty_registry_returns_empty_list(self):
        """Empty registry is a normal state (e.g. right after reset()), not a misuse."""
        self.assertEqual(self.manager.quantizers_in_execution_order(), [])

    def test_quantizers_in_execution_order_excludes_ghosts_by_default(self):
        """A registered-but-never-forwarded quantizer (inference_sequence_id == -1)
        is dropped by default, matching the Brevitas injector-ghost scenario."""
        q1 = FixedPointPerTensorQuantizer(bit_width=8)
        q2 = FixedPointPerTensorQuantizer(bit_width=8)  # never forwarded — simulates a ghost
        q1(torch.randn(10))

        ordered = self.manager.quantizers_in_execution_order()
        self.assertEqual([q.quant_id for q in ordered], [q1.quant_id])

        ordered_incl = self.manager.quantizers_in_execution_order(include_unreached=True)
        self.assertIn(q2.quant_id, [q.quant_id for q in ordered_incl])

    def test_skip_gating_bypasses_calibrated_quantizer(self):
        """A preserved (search_done=True) quantizer must skip the staggered
        gating wait, not just have its annealing alpha forced to 1.0."""
        q = FixedPointPerTensorQuantizer(bit_width=8)
        q(torch.randn(10))  # assigns inference_sequence_id, calibrates (search_done=True)
        q.inference_sequence_id = 5  # pretend this is a deep-in-the-network quantizer
        self.manager.quantization_start_gap = 100

        self.manager.skip_gating_for_calibrated_quantizers()
        self.assertEqual(q.inference_counter, 5 * 100)

        q.train()
        _, scale, _, _ = q(torch.randn(10))
        self.assertNotEqual(scale.item(), 1.0, "quantizer was gated instead of bypassed")

    def test_skip_gating_leaves_uncalibrated_quantizer_untouched(self):
        """A never-calibrated quantizer (search_done=False) must still go
        through the normal staggered gating wait."""
        q = FixedPointPerTensorQuantizer(bit_width=8)
        q(torch.randn(10))
        q.search_done.fill_(False)
        q.inference_sequence_id = 5
        self.manager.quantization_start_gap = 100

        self.manager.skip_gating_for_calibrated_quantizers()
        self.assertEqual(q.inference_counter, 0)


if __name__ == "__main__":
    unittest.main()
