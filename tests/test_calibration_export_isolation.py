"""
Regression tests for pitfall #19 (docs/llm/pitfalls/brevitas_pitfalls.md):
two independent ways a quantizer used to get calibrated, gated open, and/or
have its one-shot lifecycle log flags tripped WITHOUT any real, gated
training forward pass ever happening -- silently breaking
`quantization_start_gap` staggering.

Bug 1 -- `training_harness/schedulers.py::collect_scale_factors` (called once
per epoch as soon as QAT is active, `track_scale_factors=True` by default):
reading a Brevitas `WeightQuantProxyFromInjector`'s `.scale()` on an
uncalibrated quantizer runs a REAL forward pass on the live weight tensor as
a side effect (`retrieve_attribute` -> `self.__call__(weight)`), in eval
mode, outside the normal training loop. Any quantizer whose staggered gate
threshold is 0 (the first one reached in forward order always has
threshold `0 * gap = 0`) calibrated, opened its gate, and started annealing
the instant QAT activated -- before any real training batch -- regardless of
`quantization_start_gap`.

Bug 2 -- `utils/onnx_export.py::export_onnx_with_io`: both the
`torch.onnx.export()` trace and its post-trace `model(dummy_input)`
reference-output pass are real forward calls. An uncalibrated quantizer
reached during either one calibrated against meaningless dummy data (e.g.
the hardcoded `torch.randn(1, 3, 32, 32)` fallback in
`training_harness/checkpointing.py::CheckpointManager._export_onnx`, called
once per epoch as part of the normal `save()` path). Separately,
`_freeze_annealing`'s temporary `annealing_alpha=1.0` (needed so the export
shows a clean quantized value, not a float/quantized blend) could trip the
one-shot "annealing complete" log flag for an already-calibrated quantizer
that is for real still mid-anneal, permanently suppressing the real event.
"""

import tempfile

import pytest
import torch
import torch.nn as nn
import brevitas.nn as qnn

from quantizers import FixedPointPerTensorWeightQuant
from quantizers.base_quantizer import BaseQuantizer
from quantizers.manager import QuantizerManager
from quantizers.naming import assign_descriptive_quant_ids
from training_harness.schedulers import collect_scale_factors
from utils.onnx_export import export_onnx_with_io


@pytest.fixture(autouse=True)
def _isolated_manager():
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


def _quantized_conv():
    return qnn.QuantConv2d(3, 4, 3, padding=1, weight_quant=FixedPointPerTensorWeightQuant)


def _quantizer_of(module) -> BaseQuantizer:
    return next(m for m in module.modules() if isinstance(m, BaseQuantizer))


# =========================================================================
# Bug 1: collect_scale_factors() must not calibrate an uncalibrated quantizer
# =========================================================================


class TestCollectScaleFactorsDoesNotCalibrate:
    def test_uncalibrated_quantizer_is_skipped_not_calibrated(self, caplog):
        """A freshly-built (uncalibrated) quantized layer must stay
        uncalibrated after collect_scale_factors(); its scale must not
        appear in the returned dict either (nothing to report yet)."""
        model = _quantized_conv()
        q = _quantizer_of(model)
        assert q.search_done_value is False

        with caplog.at_level("INFO", logger="quantizers"):
            scales = collect_scale_factors(model)

        assert q.search_done_value is False, (
            "collect_scale_factors() must not trigger calibration as a side effect"
        )
        assert not any("0.weight_quant.scale" in k for k in scales), scales
        assert not any(r.name == "quantizers" for r in caplog.records), (
            "no lifecycle event should have fired from a scale-factor read"
        )

    def test_calibrated_quantizer_scale_is_still_reported(self):
        """The fix must not break the legitimate case: an already-calibrated
        quantizer's scale is still collected normally."""
        model = _quantized_conv()
        model.train()
        model(torch.randn(2, 3, 8, 8))  # calibrates
        q = _quantizer_of(model)
        assert q.search_done_value is True

        scales = collect_scale_factors(model)
        assert any(k.endswith("weight_quant.scale") for k in scales), scales

    def test_does_not_open_zero_threshold_gate_early(self, caplog):
        """Reproduces the exact real-world symptom: a quantizer with
        inference_sequence_id=0 (threshold 0*gap=0, always the first one
        reached) must NOT calibrate/gate-open merely because
        collect_scale_factors() was called -- only a real forward pass
        should do that."""
        model = _quantized_conv()
        mgr = QuantizerManager()
        mgr.quantization_start_gap = 50
        assign_descriptive_quant_ids(model, mgr)
        q = _quantizer_of(model)

        # Simulate "QAT just activated, epoch loop calls collect_scale_factors
        # before any real training batch of the new phase has run" -- model
        # is in eval mode here, exactly like the real call site.
        model.eval()
        with caplog.at_level("INFO", logger="quantizers"):
            collect_scale_factors(model)

        assert q.search_done_value is False
        assert q.inference_sequence_id == -1, (
            "a pure scale-factor read must not even assign a sequence id "
            "-- it must never reach BaseQuantizer.forward() at all for an "
            "uncalibrated quantizer"
        )
        gate_msgs = [r.message for r in caplog.records if "gate opened" in r.message]
        assert not gate_msgs, gate_msgs


# =========================================================================
# Bug 2: export_onnx_with_io() must not calibrate / log from its own
# internal forward passes
# =========================================================================


class TestExportDoesNotCalibrateOrLog:
    def test_uncalibrated_quantizer_exported_as_passthrough_stays_uncalibrated(self, caplog, tmp_path):
        model = _quantized_conv()
        q = _quantizer_of(model)
        assert q.search_done_value is False

        with caplog.at_level("INFO", logger="quantizers"):
            export_onnx_with_io(
                model, torch.randn(1, 3, 8, 8), str(tmp_path / "m.onnx"),
                opset_version=13, custom_opsets={"Quantify": 1}, dynamo=False,
            )

        assert q.search_done_value is False, (
            "export must not calibrate an uncalibrated quantizer against dummy_input"
        )
        assert not any(r.name == "quantizers" for r in caplog.records), (
            "no lifecycle event should fire from an export's internal forward passes"
        )
        # Flags fully reset -- a later REAL calibration must still log normally.
        assert q._log_gate_opened is False
        assert q._log_calibration_count == 0

    def test_exported_graph_has_no_quantize_node_for_uncalibrated_quantizer(self, tmp_path):
        """The exported graph should show a plain passthrough (no
        `Quantify::FixedPointQuant` node) for a quantizer that hasn't
        started quantizing in the live model -- embedding a quantize node
        with a bogus, uncalibrated LSB would misrepresent the real model."""
        import onnx

        model = _quantized_conv()
        onnx_path = tmp_path / "m.onnx"
        export_onnx_with_io(
            model, torch.randn(1, 3, 8, 8), str(onnx_path),
            opset_version=13, custom_opsets={"Quantify": 1}, dynamo=False,
        )
        onnx_model = onnx.load(str(onnx_path))
        quant_nodes = [
            n for n in onnx_model.graph.node
            if n.op_type == "FixedPointQuant" and n.domain == "Quantify"
        ]
        assert quant_nodes == []

    def test_real_calibration_after_export_still_logs_normally(self, caplog, tmp_path):
        """After an export has (harmlessly) touched an uncalibrated
        quantizer, a genuine subsequent training forward pass must still
        calibrate it and log the real event -- the export must not have
        permanently tripped the one-shot flags."""
        model = _quantized_conv()
        export_onnx_with_io(
            model, torch.randn(1, 3, 8, 8), str(tmp_path / "m.onnx"),
            opset_version=13, custom_opsets={"Quantify": 1}, dynamo=False,
        )
        q = _quantizer_of(model)
        assert q.search_done_value is False

        model.train()
        with caplog.at_level("INFO", logger="quantizers"):
            model(torch.randn(2, 3, 8, 8))

        assert q.search_done_value is True
        messages = [r.message for r in caplog.records if r.name == "quantizers"]
        assert any("gate opened" in m for m in messages), messages
        assert any("calibration completed" in m for m in messages), messages

    def test_mid_anneal_export_does_not_suppress_real_annealing_complete_log(self, caplog, tmp_path):
        """An already-calibrated, mid-anneal quantizer's forced
        annealing_alpha=1.0 during export must not permanently trip the
        one-shot 'annealing complete' flag -- the real completion later in
        training must still log."""
        model = _quantized_conv()
        model.train()
        model(torch.randn(2, 3, 8, 8))  # calibrate
        q = _quantizer_of(model)
        # Simulate the real QAT-activation state: annealing_alpha freshly
        # reset to ramp 0->1 (BaseQuantizer defaults annealing_alpha=1.0 at
        # construction, so the calibration call just above already logged an
        # immediate "annealing complete" -- reset the flag to mimic a
        # genuinely-still-annealing quantizer).
        q.annealing_alpha.fill_(0.5)
        q.annealing_alpha_step = 0.5
        q._log_annealing_complete = False
        assert q._log_annealing_complete is False

        with caplog.at_level("INFO", logger="quantizers"):
            export_onnx_with_io(
                model, torch.randn(1, 3, 8, 8), str(tmp_path / "export1.onnx"),
                opset_version=13, custom_opsets={"Quantify": 1}, dynamo=False,
            )

        assert q._log_annealing_complete is False, (
            "export's temporarily-forced alpha=1.0 must not trip the real flag"
        )
        assert q.annealing_alpha.item() == pytest.approx(0.5), (
            "annealing_alpha must be restored to its real, mid-anneal value"
        )
        caplog.clear()

        # Now genuinely finish annealing via a real training forward pass.
        with caplog.at_level("INFO", logger="quantizers"):
            model(torch.randn(2, 3, 8, 8))  # alpha: 0.5 -> 1.0

        assert q.annealing_alpha.item() == pytest.approx(1.0)
        messages = [r.message for r in caplog.records if r.name == "quantizers"]
        assert any("annealing complete" in m for m in messages), messages


# =========================================================================
# End-to-end: staggered gap must be honored across a realistic sequence of
# training batches interleaved with scale-factor collection and export
# =========================================================================


class TestStaggeredGapEndToEnd:
    def test_gap_is_honored_despite_scale_collection_and_export_between_batches(self, tmp_path, caplog):
        """Two quantizers with inference_sequence_id 0 and 1 and
        quantization_start_gap=3: the second one must not calibrate until it
        has genuinely seen 3 gated-off real training forward passes, even
        with collect_scale_factors()/export_onnx_with_io() calls spliced in
        between (simulating per-epoch checkpoint/scale-tracking)."""

        class TwoConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = qnn.QuantConv2d(3, 4, 3, padding=1, bias=False, weight_quant=FixedPointPerTensorWeightQuant)
                self.conv2 = qnn.QuantConv2d(4, 4, 3, padding=1, bias=False, weight_quant=FixedPointPerTensorWeightQuant)

            def forward(self, x):
                return self.conv2(self.conv1(x))

        model = TwoConv()
        mgr = QuantizerManager()
        mgr.quantization_start_gap = 3
        assign_descriptive_quant_ids(model, mgr)

        q1 = model.conv1.weight_quant.tensor_quant
        q2 = model.conv2.weight_quant.tensor_quant

        model.train()
        x = torch.randn(1, 3, 8, 8)

        # Batch 0: q1 (threshold 0) calibrates immediately; q2 (threshold 3)
        # stays gated off.
        model(x)
        assert q1.search_done_value is True
        assert q2.search_done_value is False

        # Interleave scale collection + export, as a per-epoch checkpoint
        # would -- these must be complete no-ops w.r.t. gating/calibration.
        collect_scale_factors(model)
        export_onnx_with_io(
            model, x, str(tmp_path / "e0.onnx"),
            opset_version=13, custom_opsets={"Quantify": 1}, dynamo=False,
        )
        assert q2.search_done_value is False
        assert q2.inference_counter == 1, (
            "export_onnx_with_io must restore training mode afterward -- "
            "otherwise this (real, train-mode) batch's gating increment "
            "would be silently lost"
        )

        # Two more real batches (counter -> 3): still below threshold 3 --
        # the THIRD gated-off batch, so the gate hasn't opened YET, just
        # reached the threshold.
        model(x)
        model(x)
        collect_scale_factors(model)
        export_onnx_with_io(
            model, x, str(tmp_path / "e1.onnx"),
            opset_version=13, custom_opsets={"Quantify": 1}, dynamo=False,
        )
        assert q2.search_done_value is False
        assert q2.inference_counter == 3

        # Third real batch: counter reaches 3 == threshold -> gate opens.
        with caplog.at_level("INFO", logger="quantizers"):
            model(x)
        assert q2.search_done_value is True
        messages = [r.message for r in caplog.records if r.name == "quantizers"]
        assert any("gate opened" in m and "waited 3 gated-off" in m for m in messages), messages
