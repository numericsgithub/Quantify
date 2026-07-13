"""
Regression tests for QuantizerManager.skip_gating_for_calibrated_quantizers().

Bug (fixed): the old implementation set inference_counter = inference_sequence_id
* gap, guarded by `inference_sequence_id != -1`. When called from
_activate_qat BEFORE the first forward pass — which is the normal case for a
model initialised from a PTQ/QAT checkpoint — every inference_sequence_id is
still -1, so the call was a silent no-op. The staggered-activation gate then
re-engaged once real sequence ids were assigned on the first forward, so a
model that was supposed to be fully quantized from step 0 instead quantized
only gradually over ~sequence_id * gap steps (a spurious accuracy dip that
looked like QAT "getting worse").

These tests pin the intended behaviour: after skip_gating_for_calibrated_
quantizers(), an already-calibrated quantizer quantizes on its very first
forward pass regardless of its sequence id or the gap size.
"""

import torch

from quantizers.manager import QuantizerManager
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer


def _make_calibrated_quantizer() -> FixedPointPerTensorQuantizer:
    """Build a fixed-point quantizer and calibrate it via one training forward
    (with the gap temporarily at 0 so calibration itself isn't gated), then
    reset its runtime gating state to mimic a fresh checkpoint load (search_done
    buffer restored = True, but inference_sequence_id back to -1 and
    inference_counter back to 0)."""
    mgr = QuantizerManager()
    mgr.quantization_start_gap = 0  # don't gate the calibration forward
    q = FixedPointPerTensorQuantizer(bit_width=8, signed=True, quantizer_role="weight")
    q.train()
    torch.manual_seed(0)
    q(torch.randn(4096) * 2.0)
    assert q.search_done.item(), "setup failed: quantizer did not calibrate"
    # Simulate a fresh process that loaded this quantizer's buffers from a
    # checkpoint: buffers persist, runtime counters do not.
    q.inference_sequence_id = -1
    q.inference_counter = 0
    q.gating_enabled = True
    return q


def test_gate_blocks_calibrated_quantizer_without_skip():
    """Control: with a large gap and a positive sequence id, a calibrated
    quantizer that has NOT had skip_gating applied is gated off (passthrough)
    on its first forward — proving the gate is actually engaged in this setup."""
    QuantizerManager().reset()
    mgr = QuantizerManager()
    q = _make_calibrated_quantizer()
    # Now impose a large gap; the calibration forward already advanced the
    # manager's sequence counter, so q's next forward gets a positive id.
    mgr.quantization_start_gap = 100

    q.eval()
    x = torch.randn(4096) * 2.0
    out, _, _, _ = q(x)

    assert q.inference_sequence_id >= 1, "test needs a positive sequence id"
    assert torch.equal(out, x), (
        "expected passthrough (gated) but the quantizer altered its input"
    )


def test_skip_gating_activates_calibrated_quantizer_on_first_forward():
    """After skip_gating_for_calibrated_quantizers() — called BEFORE the first
    forward, when inference_sequence_id is still -1 — the calibrated quantizer
    must quantize immediately, not wait out sequence_id * gap steps."""
    QuantizerManager().reset()
    mgr = QuantizerManager()
    q = _make_calibrated_quantizer()
    mgr.quantization_start_gap = 100
    assert q.inference_sequence_id == -1  # not forwarded yet, like a fresh load

    mgr.skip_gating_for_calibrated_quantizers()
    assert q.gating_enabled is False, "skip_gating did not disable the gate"

    q.eval()
    x = torch.randn(4096) * 2.0
    out, _, _, _ = q(x)

    assert q.inference_sequence_id >= 1, "test needs a positive sequence id"
    assert not torch.allclose(out, x), (
        "quantizer was still gated off after skip_gating — the gate was not "
        "bypassed (the original no-op bug)"
    )
    # And the output must actually lie on the fixed-point grid.
    step = 2.0 ** int(q.search_result_lsb.item())
    residual = out / step - torch.round(out / step)
    assert residual.abs().max().item() < 1e-4, "output is not on the quant grid"


def test_fresh_uncalibrated_quantizer_still_cascades():
    """skip_gating must only affect calibrated quantizers; a not-yet-calibrated
    one (search_done=False) keeps gating_enabled=True so the staggered QAT
    cascade still works for freshly-calibrated quantizers."""
    QuantizerManager().reset()
    mgr = QuantizerManager()
    q = FixedPointPerTensorQuantizer(bit_width=8, signed=True, quantizer_role="weight")
    assert q.search_done.item() is False
    mgr.skip_gating_for_calibrated_quantizers()
    assert q.gating_enabled is True, (
        "skip_gating should not touch uncalibrated quantizers"
    )
