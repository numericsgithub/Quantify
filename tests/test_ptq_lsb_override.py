"""
Tests that verify search_result_lsb.fill_(candidate_lsb) actually changes
the quantization output on the next forward pass.

This is the mechanism used by find_perfect_lsbs_imagenet_ptq.py to sweep
LSB candidates: after calibration, the script sets search_result_lsb to
each candidate value and evaluates the model.  These tests confirm that
the buffer write propagates through _load_calibration -> _quantize correctly.
"""

import unittest

import torch

from quantizers import (
    FixedPointPerTensorQuantizer,
    RoundingMode,
    quantize_fixed_point,
)
from quantizers.manager import QuantizerManager


class TestLSBOverride(unittest.TestCase):

    def setUp(self):
        QuantizerManager().reset()
        self.q = FixedPointPerTensorQuantizer(bit_width=8)

    def _calibrate(self, x: torch.Tensor) -> int:
        """Run one training-mode forward to calibrate; return the chosen LSB."""
        self.q.train()
        self.q(x)
        self.assertTrue(self.q.search_done.item(), "calibration did not complete")
        return int(self.q.search_result_lsb.item())

    # ------------------------------------------------------------------

    def test_fill_changes_quantization_output(self):
        """After calibration, fill_(new_lsb) must change the forward output."""
        x = torch.randn(64, 64) * 10.0
        calib_lsb = self._calibrate(x)
        new_lsb   = calib_lsb + 2   # deliberately different

        self.q.eval()
        with torch.no_grad():
            out_calib, _, _, _ = self.q(x)

        # Override the LSB
        self.q.search_result_lsb.fill_(new_lsb)

        self.q.eval()
        with torch.no_grad():
            out_override, _, _, _ = self.q(x)

        # Outputs must differ (coarser grid → visible differences for random data)
        self.assertFalse(
            torch.equal(out_calib, out_override),
            f"LSB override from {calib_lsb} to {new_lsb} had no effect on output"
        )

    def test_fill_output_matches_expected_quantization(self):
        """The output after fill_(target_lsb) must equal quantize_fixed_point(x, target_lsb)."""
        x = torch.randn(128) * 5.0
        self._calibrate(x)

        signed = bool(self.q.search_result_is_signed.item())

        for target_lsb in (-8, -5, -3, -1, 0, 2):
            self.q.search_result_lsb.fill_(target_lsb)
            self.q.search_done.fill_(True)

            self.q.eval()
            with torch.no_grad():
                out, _, _, _ = self.q(x)

            expected = quantize_fixed_point(x, target_lsb, self.q.bit_width,
                                            signed, RoundingMode.ROUND)
            self.assertTrue(
                torch.allclose(out, expected),
                f"LSB={target_lsb}: output does not match quantize_fixed_point. "
                f"max diff={( out - expected).abs().max().item():.6f}"
            )

    def test_fill_with_negative_lsb(self):
        """Negative LSBs (sub-integer precision) must round-trip correctly."""
        x = torch.randn(64) * 3.0
        self._calibrate(x)

        signed = bool(self.q.search_result_is_signed.item())
        target_lsb = -6

        self.q.search_result_lsb.fill_(target_lsb)
        self.q.search_done.fill_(True)

        self.q.eval()
        with torch.no_grad():
            out, _, _, _ = self.q(x)

        expected = quantize_fixed_point(x, target_lsb, self.q.bit_width,
                                        signed, RoundingMode.ROUND)
        self.assertTrue(torch.allclose(out, expected))

    def test_lsb_override_does_not_trigger_recalibration(self):
        """Calling fill_ on the buffer must not cause _calibrate to run again."""
        x = torch.randn(64, 64)
        self._calibrate(x)

        # Count calibrations by watching search_done toggle (it goes False→True during calib)
        # We verify: after the override, search_done stays True
        self.q.search_result_lsb.fill_(-3)
        self.q.search_done.fill_(True)

        self.q.eval()
        with torch.no_grad():
            self.q(x)

        self.assertTrue(
            self.q.search_done.item(),
            "search_done was reset — recalibration triggered unexpectedly after lsb override"
        )

    def test_scale_metadata_reflects_overridden_lsb(self):
        """The scale tensor returned in the 4-tuple must be 2**lsb for the overridden value."""
        x = torch.randn(64, 64)
        self._calibrate(x)

        target_lsb = -4
        self.q.search_result_lsb.fill_(target_lsb)
        self.q.search_done.fill_(True)

        self.q.eval()
        with torch.no_grad():
            _, scale, _, _ = self.q(x)

        expected_scale = 2.0 ** target_lsb
        self.assertAlmostEqual(
            scale.item(), expected_scale, places=6,
            msg=f"scale={scale.item()} does not match 2**{target_lsb}={expected_scale}"
        )

    def test_sequential_overrides_are_independent(self):
        """Multiple fill_ calls in a loop must each produce distinct, correct outputs."""
        x = torch.randn(128) * 4.0
        self._calibrate(x)
        signed = bool(self.q.search_result_is_signed.item())

        lsb_candidates = [-7, -6, -5, -4, -3]
        outputs = {}

        for lsb in lsb_candidates:
            self.q.search_result_lsb.fill_(lsb)
            self.q.search_done.fill_(True)
            self.q.eval()
            with torch.no_grad():
                out, _, _, _ = self.q(x)
            outputs[lsb] = out.clone()

        # Every output must match the direct quantize call
        for lsb, out in outputs.items():
            expected = quantize_fixed_point(x, lsb, self.q.bit_width,
                                            signed, RoundingMode.ROUND)
            self.assertTrue(
                torch.allclose(out, expected),
                f"LSB={lsb}: output differs from quantize_fixed_point"
            )

        # Adjacent LSBs produce different outputs (coarser grid has strictly larger step)
        for i in range(len(lsb_candidates) - 1):
            a = lsb_candidates[i]
            b = lsb_candidates[i + 1]
            self.assertFalse(
                torch.equal(outputs[a], outputs[b]),
                f"LSB={a} and LSB={b} produced identical outputs — override had no effect"
            )


if __name__ == "__main__":
    unittest.main()
