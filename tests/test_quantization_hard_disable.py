"""Regression tests for QuantizerManager.disable_quantization() genuinely
skipping calibration/quantize compute, not just discarding the result via
annealing_alpha=0.

Real-world bug report: a tiny (~1,700 param) model with 18 quantizers was
2.5-6x slower per epoch on GPU than CPU. Root cause: disable_quantization()
only zeroed annealing_alpha; BaseQuantizer.forward() still ran the full
_calibrate()/_quantize() pipeline every forward call (since
quantization_start_gap defaults to 0, the gate never blocks anything), and
the result was thrown away by AnnealingBlendFn.apply(x, quantized, alpha=0.0)
blending 100% back to the float input. Measured breakdown (RTX 4090,
batch=256): plain PyTorch 0.841s/epoch -> Brevitas wrapping only 0.949s ->
+ permanently gated (never quantizes) 1.332s -> + disable_quantization() as
it worked before this fix 2.316s. The last step (+0.984s) was pure wasted
compute with no effect on the output.
"""
import torch

from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer
from quantizers.manager import QuantizerManager


def _fresh_quantizer(bit_width: int = 8) -> FixedPointPerTensorQuantizer:
    return FixedPointPerTensorQuantizer(bit_width=bit_width)


class TestDisableQuantizationSkipsCompute:
    def setup_method(self):
        QuantizerManager().reset()

    def teardown_method(self):
        QuantizerManager().reset()

    def test_disabled_quantizer_never_calibrates(self):
        """search_done must stay False -- _calibrate() must never run --
        while quantization is globally disabled."""
        q = _fresh_quantizer()
        QuantizerManager().disable_quantization()
        q.train()
        for _ in range(10):
            q(torch.randn(16))
        assert q.search_done_value is False

    def test_disabled_quantizer_output_is_the_exact_input(self):
        """Not just numerically close via an alpha=0 blend -- the literal
        same tensor should come back untouched (true passthrough, no
        quantize math computed and discarded)."""
        q = _fresh_quantizer()
        QuantizerManager().disable_quantization()
        q.train()
        x = torch.randn(16)
        out, scale, zero_point, bit_width = q(x)
        assert out is x
        assert scale.item() == 1.0
        assert zero_point.item() == 0.0
        assert bit_width.item() == 8.0

    def test_disabled_then_reenabled_via_manager_flag(self):
        q = _fresh_quantizer()
        mgr = QuantizerManager()
        mgr.disable_quantization()
        q.train()
        q(torch.randn(16))
        assert q.search_done_value is False

        mgr.enable_quantization()
        q(torch.randn(16))
        assert q.search_done_value is True  # calibration now actually ran

    def test_disabled_then_reactivated_via_set_annealing_for_n_inferences(self):
        """The real production path (training_harness/trainer_v2.py
        _activate_qat()): float warmup calls disable_quantization(), QAT
        activation calls set_annealing_for_n_inferences() directly -- it
        never calls enable_quantization(). Regression guard: this specific
        sequence must actually turn quantization back on, not leave it
        silently hard-disabled forever (that was a bug this fix's own first
        draft introduced and this test caught)."""
        q = _fresh_quantizer()
        mgr = QuantizerManager()

        mgr.disable_quantization()  # float warmup
        q.train()
        for _ in range(5):
            q(torch.randn(16))
        assert q.search_done_value is False

        mgr.set_annealing_for_n_inferences(5)  # QAT activation, no enable_quantization() call
        mgr.quantization_start_gap = 0
        q(torch.randn(16))
        assert q.search_done_value is True
        assert q.annealing_alpha.item() > 0.0

    def test_passthrough_metadata_tensors_are_cached_not_reallocated(self):
        """The gated-off/disabled return path used to build 3 fresh
        torch.tensor(...) objects every call; now cached per (dtype, device)."""
        q = _fresh_quantizer()
        QuantizerManager().disable_quantization()
        q.train()
        x1 = torch.randn(16)
        x2 = torch.randn(16)
        _, scale1, zp1, bw1 = q(x1)
        _, scale2, zp2, bw2 = q(x2)
        assert scale1 is scale2
        assert zp1 is zp2
        assert bw1 is bw2

    def test_reset_clears_the_disabled_flag(self):
        mgr = QuantizerManager()
        mgr.disable_quantization()
        assert mgr.quantization_globally_disabled is True
        mgr.reset()
        assert mgr.quantization_globally_disabled is False
