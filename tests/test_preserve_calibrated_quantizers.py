"""
Tests for QATTrainerV2._activate_qat() with preserve_calibrated_quantizers.

Verifies that quantizers already calibrated (search_done=True) before QAT
activates — the situation after loading a PTQ checkpoint produced by
examples/find_perfect_lsbs_imagenet_ptq.py — keep their search_result_lsb
and jump straight to annealing_alpha=1.0 instead of being wiped back to
search_done=False / annealing_alpha=0.0 like a fresh, never-calibrated
quantizer. Quantizers that are NOT yet calibrated must still go through
the normal reset + gradual annealing ramp regardless of the flag.
"""

import pytest
import torch
import torch.nn as nn
import brevitas.nn as qnn

from quantizers.manager import QuantizerManager
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorWeightQuant
from training_harness.trainer_v2 import QATTrainerV2, _reset_and_register
from training_harness.config_v2 import TrainerConfigV2, QATScheduleConfigV2


class _WQ(FixedPointPerTensorWeightQuant):
    bit_width = 8


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(3, 4, 3, bias=False, weight_quant=_WQ)
        self.conv2 = qnn.QuantConv2d(4, 4, 3, bias=False, weight_quant=_WQ)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


@pytest.fixture(autouse=True)
def reset_manager():
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


def _make_trainer(tmp_path, preserve: bool) -> QATTrainerV2:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 2, (4,))
    loader = [(x, y)]

    config = TrainerConfigV2(
        experiment_name="test_preserve_calibrated",
        output_dir=str(tmp_path),
        epochs=1,
        qat=QATScheduleConfigV2(
            float_warmup_epochs=0,
            annealing_steps=10,
            quantization_start_gap=5,
            preserve_calibrated_quantizers=preserve,
        ),
        early_stopping_patience=None,
    )
    return QATTrainerV2(
        config=config, model=model, optimizer=optimizer,
        train_loader=loader, val_loader=None, loss_fn=nn.CrossEntropyLoss(),
    )


def _register_and_calibrate(model: nn.Module, lsb: int = -9) -> None:
    """Simulate a PTQ checkpoint already having been loaded: every quantizer
    is calibrated, fully annealed, and holding a specific found LSB."""
    _reset_and_register(model)
    for q in QuantizerManager().quantizers.values():
        q.search_result_lsb.fill_(lsb)
        q.search_done.fill_(True)
        q.annealing_alpha.data.fill_(1.0)


class TestPreserveCalibratedQuantizersGating:
    """Gating (inference_counter vs. sequence_id * gap) is independent from
    annealing (alpha). preserve_calibrated_quantizers must bypass BOTH for an
    already-calibrated quantizer, not just annealing — otherwise a preserved
    quantizer still silently runs as float passthrough for its own
    sequence_id * quantization_start_gap forward calls despite alpha=1.0."""

    def test_preserve_true_bypasses_gating_immediately(self, tmp_path):
        trainer = _make_trainer(tmp_path, preserve=True)
        _register_and_calibrate(trainer.model)

        # Simulate the real production flow: a forward pass (e.g. a
        # pre-training evaluation) happens before _activate_qat(), assigning
        # real inference_sequence_id values to every quantizer.
        trainer.model.eval()
        with torch.no_grad():
            trainer.model(torch.randn(4, 3, 8, 8))

        trainer._activate_qat()

        # conv2's weight quantizer is second in execution order
        # (inference_sequence_id == 1) — under the old (buggy) behavior it
        # would still need inference_counter >= 1 * gap (5) before
        # quantizing at all, despite being "preserved".
        q1 = next(
            q for q in QuantizerManager().quantizers.values()
            if q.inference_sequence_id == 1
        )
        assert q1.inference_counter == 1 * trainer.config.qat.quantization_start_gap

        trainer.model.train()
        _, scale, _, _ = q1(torch.randn(4, 4, 6, 6))
        assert scale.item() != 1.0, (
            "preserved quantizer was gated (float passthrough) instead of "
            "bypassing the staggered activation delay"
        )

    def test_preserve_false_still_uses_normal_gating(self, tmp_path):
        """Regression: without preserve, fresh/reset quantizers must still
        go through the real staggered gating wait — unchanged behavior for
        a from-scratch QAT run."""
        trainer = _make_trainer(tmp_path, preserve=False)
        _register_and_calibrate(trainer.model)

        trainer.model.eval()
        with torch.no_grad():
            trainer.model(torch.randn(4, 3, 8, 8))

        trainer._activate_qat()

        q1 = next(
            q for q in QuantizerManager().quantizers.values()
            if q.inference_sequence_id == 1
        )
        assert q1.inference_counter == 0, "gating bypass must not run when preserve=False"

        trainer.model.train()
        _, scale, _, _ = q1(torch.randn(4, 4, 6, 6))
        assert scale.item() == 1.0, (
            "quantizer should still be gated (float passthrough) since "
            "preserve_calibrated_quantizers=False"
        )


class TestPreserveCalibratedQuantizers:

    def test_default_preserve_false_resets_calibrated_quantizer(self, tmp_path):
        """Backward-compat baseline: without the flag, _activate_qat must
        still wipe search_done and reset alpha to 0, even when a quantizer
        was already calibrated."""
        trainer = _make_trainer(tmp_path, preserve=False)
        _register_and_calibrate(trainer.model)

        trainer._activate_qat()

        for q in QuantizerManager().quantizers.values():
            assert q.search_done.item() is False
            assert q.annealing_alpha.item() == pytest.approx(0.0)

    def test_preserve_true_keeps_calibrated_quantizer_active(self, tmp_path):
        trainer = _make_trainer(tmp_path, preserve=True)
        _register_and_calibrate(trainer.model)

        trainer._activate_qat()

        for q in QuantizerManager().quantizers.values():
            assert q.search_done.item() is True
            assert q.annealing_alpha.item() == pytest.approx(1.0)
            assert q.annealing_alpha_step == pytest.approx(0.0)

    def test_preserve_true_keeps_calibrated_lsb_value(self, tmp_path):
        """The actual found LSB must survive _activate_qat unchanged."""
        trainer = _make_trainer(tmp_path, preserve=True)
        _register_and_calibrate(trainer.model, lsb=-11)

        trainer._activate_qat()

        for q in QuantizerManager().quantizers.values():
            assert int(q.search_result_lsb.item()) == -11

    def test_preserve_true_still_resets_uncalibrated_quantizer(self, tmp_path):
        """A quantizer that was NOT calibrated must still go through the
        normal reset + ramp, even when preserve_calibrated_quantizers=True."""
        trainer = _make_trainer(tmp_path, preserve=True)
        _reset_and_register(trainer.model)
        for q in QuantizerManager().quantizers.values():
            q.search_done.fill_(False)
            q.annealing_alpha.data.fill_(1.0)  # leftover state, must still be reset

        trainer._activate_qat()

        for q in QuantizerManager().quantizers.values():
            assert q.search_done.item() is False
            assert q.annealing_alpha.item() == pytest.approx(0.0)
            assert q.annealing_alpha_step == pytest.approx(1.0 / 10)  # annealing_steps=10

    def test_preserve_true_mixed_calibrated_and_uncalibrated(self, tmp_path):
        """One quantizer pre-calibrated, one not — each must be handled
        according to its own state, not a blanket all-or-nothing reset."""
        trainer = _make_trainer(tmp_path, preserve=True)
        _reset_and_register(trainer.model)

        qs = list(QuantizerManager().quantizers.values())
        assert len(qs) == 2
        calibrated, uncalibrated = qs[0], qs[1]
        calibrated.search_result_lsb.fill_(-7)
        calibrated.search_done.fill_(True)
        calibrated.annealing_alpha.data.fill_(1.0)
        uncalibrated.search_done.fill_(False)

        trainer._activate_qat()

        assert calibrated.search_done.item() is True
        assert int(calibrated.search_result_lsb.item()) == -7
        assert calibrated.annealing_alpha.item() == pytest.approx(1.0)

        assert uncalibrated.search_done.item() is False
        assert uncalibrated.annealing_alpha.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# QuantizerManager.set_annealing_for_n_inferences(skip_calibrated=...)
# ---------------------------------------------------------------------------

class TestManagerSkipCalibrated:

    def test_skip_calibrated_false_is_default_behavior(self):
        from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer
        q = FixedPointPerTensorQuantizer(bit_width=8)
        q.search_done.fill_(True)
        q.annealing_alpha.data.fill_(1.0)

        QuantizerManager().set_annealing_for_n_inferences(10)

        assert q.annealing_alpha.item() == pytest.approx(0.0)
        assert q.annealing_alpha_step == pytest.approx(0.1)

    def test_skip_calibrated_true_preserves_calibrated_quantizer(self):
        from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer
        q = FixedPointPerTensorQuantizer(bit_width=8)
        q.search_result_lsb.fill_(-3)
        q.search_done.fill_(True)
        q.annealing_alpha.data.fill_(1.0)

        QuantizerManager().set_annealing_for_n_inferences(10, skip_calibrated=True)

        assert q.search_done.item() is True
        assert int(q.search_result_lsb.item()) == -3
        assert q.annealing_alpha.item() == pytest.approx(1.0)
        assert q.annealing_alpha_step == pytest.approx(0.0)

    def test_skip_calibrated_true_still_resets_uncalibrated_quantizer(self):
        from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer
        q = FixedPointPerTensorQuantizer(bit_width=8)
        q.search_done.fill_(False)
        q.annealing_alpha.data.fill_(1.0)  # leftover state

        QuantizerManager().set_annealing_for_n_inferences(10, skip_calibrated=True)

        assert q.annealing_alpha.item() == pytest.approx(0.0)
        assert q.annealing_alpha_step == pytest.approx(0.1)
