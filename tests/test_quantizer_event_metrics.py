"""
Tests for the rich metrics dict attached to "calibration_completed"/
"calibration_rerun" and "annealing_complete" lifecycle log events
(`BaseQuantizer._compute_event_metrics`, `utils/quantizer_diagnostics.py`).

Covers:
    - compute_metrics() correctness: histogram sums to n_elements, exact
      quantized-value counts sum to n_elements, clip counts, grid/range info
    - record.metrics is populated on calibration_completed/calibration_rerun
      and annealing_complete, and is None on gate_opened/annealing_started
    - record.metrics is None (and nothing is computed) while
      suppress_lifecycle_logging is set (export isolation)
    - plot_quantizer_metrics() / plot_quantizer_metrics_grid() build a
      figure from a metrics dict without raising
    - the post-annealing metrics genuinely differ from the post-calibration
      ones when more/different data has flowed through by then (the user's
      "see how much things changed over training" ask)
"""

import logging

import pytest
import torch

from quantizers import FixedPointPerTensorQuantizer
from quantizers.manager import QuantizerManager
from utils.quantizer_diagnostics import (
    compute_metrics,
    plot_quantizer_metrics,
    plot_quantizer_metrics_grid,
)


@pytest.fixture(autouse=True)
def _isolated_manager():
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


def _capture_quantizer_logs():
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("quantizers")
    logger.setLevel(logging.INFO)
    handler = _Capture()
    logger.addHandler(handler)
    return records, handler


# =========================================================================
# 1. compute_metrics() correctness
# =========================================================================


class TestComputeMetrics:
    def test_histogram_sums_to_total_elements(self):
        x = torch.randn(10_000) * 5
        q = torch.round(x)  # stand-in "quantized" tensor
        m = compute_metrics(x, q, lsb=0, bit_width=8, signed=True)
        assert m["hist_counts"].sum() == pytest.approx(x.numel())

    def test_quant_counts_sum_to_total_elements(self):
        x = torch.randn(10_000) * 5
        q = torch.round(x)
        m = compute_metrics(x, q, lsb=0, bit_width=8, signed=True)
        assert int(m["quant_counts"].sum()) == x.numel()

    def test_clip_counts_match_manual_computation(self):
        x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        q = torch.clamp(x, -8.0, 7.0)
        m = compute_metrics(x, q, lsb=0, bit_width=4, signed=True)
        # q_min=-8, q_max=7 for 4-bit signed, lsb=0
        assert m["q_min"] == -8.0
        assert m["q_max"] == 7.0
        assert m["n_clipped"] == 2  # -100 and 100
        assert m["n_unclipped"] == 3
        assert m["n_clipped"] + m["n_unclipped"] == x.numel()

    def test_no_clipping_when_input_within_range(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        q = x.clone()
        m = compute_metrics(x, q, lsb=0, bit_width=8, signed=True)
        assert m["n_clipped"] == 0
        assert m["n_unclipped"] == x.numel()

    def test_unique_and_coverage(self):
        x = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        q = x.clone()
        m = compute_metrics(x, q, lsb=0, bit_width=2, signed=False)
        assert m["n_unique"] == 3
        assert m["n_representable"] == 4
        assert m["coverage_pct"] == pytest.approx(75.0)

    def test_grid_and_input_range_reported(self):
        x = torch.tensor([-3.0, 5.0])
        q = torch.clamp(x, -8.0, 7.0)
        m = compute_metrics(x, q, lsb=0, bit_width=4, signed=True)
        assert m["input_min"] == -3.0
        assert m["input_max"] == 5.0
        assert m["q_min"] == -8.0
        assert m["q_max"] == 7.0

    def test_works_on_gpu_tensor_if_available(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        x = (torch.randn(1000) * 5).cuda()
        q = torch.clamp(x, -8.0, 7.0)
        m = compute_metrics(x, q, lsb=0, bit_width=4, signed=True)
        assert m["hist_counts"].sum() == pytest.approx(1000)
        # Returned arrays must already be plain CPU numpy, not CUDA tensors.
        import numpy as np
        assert isinstance(m["hist_counts"], np.ndarray)
        assert isinstance(m["quant_values"], np.ndarray)


# =========================================================================
# 2. Lifecycle log events carry (or don't carry) metrics
# =========================================================================


class TestEventMetricsAttachment:
    def test_calibration_completed_has_metrics(self):
        records, _ = _capture_quantizer_logs()
        q = FixedPointPerTensorQuantizer(bit_width=4, quantizer_role="activation")
        q(torch.randn(256))

        cal = next(r for r in records if r.event == "calibration_completed")
        assert cal.metrics is not None
        assert cal.metrics["bit_width"] == 4
        assert "hist_counts" in cal.metrics
        assert "quant_values" in cal.metrics

    def test_gate_opened_has_no_metrics(self):
        records, _ = _capture_quantizer_logs()
        q = FixedPointPerTensorQuantizer(bit_width=4, quantizer_role="activation")
        q(torch.randn(256))

        gate = next(r for r in records if r.event == "gate_opened")
        assert gate.metrics is None

    def test_annealing_started_has_no_metrics(self):
        records, _ = _capture_quantizer_logs()
        q = FixedPointPerTensorQuantizer(bit_width=4, quantizer_role="activation")
        q.annealing_alpha.fill_(0.0)
        q.annealing_alpha_step = 0.5
        q(torch.randn(256))

        started = next(r for r in records if r.event == "annealing_started")
        assert started.metrics is None

    def test_annealing_complete_has_metrics(self):
        records, _ = _capture_quantizer_logs()
        q = FixedPointPerTensorQuantizer(bit_width=4, quantizer_role="activation")
        q.annealing_alpha.fill_(0.0)
        q.annealing_alpha_step = 1.0  # completes on this very call
        q(torch.randn(256))

        complete = next(r for r in records if r.event == "annealing_complete")
        assert complete.metrics is not None
        assert "n_clipped" in complete.metrics

    def test_calibration_rerun_has_metrics(self):
        records, _ = _capture_quantizer_logs()
        q = FixedPointPerTensorQuantizer(bit_width=4, quantizer_role="activation")
        q(torch.randn(256))
        records.clear()

        QuantizerManager().trigger_global_recalibration()
        q(torch.randn(256) * 10)

        rerun = next(r for r in records if r.event == "calibration_rerun")
        assert rerun.metrics is not None

    def test_suppressed_logging_computes_no_metrics(self, monkeypatch):
        """During export isolation (suppress_lifecycle_logging=True), no
        event should fire at all -- and therefore compute_metrics() should
        never even be called (not just discarded)."""
        import utils.quantizer_diagnostics as diag_mod

        calls = []
        orig = diag_mod.compute_metrics

        def _spy(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)

        monkeypatch.setattr(diag_mod, "compute_metrics", _spy)

        q = FixedPointPerTensorQuantizer(bit_width=4, quantizer_role="activation")
        q.suppress_lifecycle_logging = True
        q(torch.randn(256))

        assert calls == []
        # suppress_lifecycle_logging doesn't stop real calibration from
        # happening -- only the logging/metrics side effect.
        assert q.search_done_value is True

    def test_base_quantizer_without_diagnostics_support_has_none_metrics(self):
        """A subclass with the base `_get_diagnostics_params` (returns None)
        must not crash and must simply carry no metrics."""
        from quantizers.base_quantizer import BaseQuantizer

        class _NoDiagQuant(BaseQuantizer):
            def _calibrate(self, x):
                return {}

            def _save_calibration(self, params):
                self.set_search_done(True)

            def _load_calibration(self):
                return {}

            def _quantize(self, x, params):
                return x

            def _get_metadata(self, params, x):
                return (
                    torch.tensor(1.0), torch.tensor(0.0), torch.tensor(float(self.bit_width)),
                )

        q = _NoDiagQuant(bit_width=8)
        records, _ = _capture_quantizer_logs()
        q(torch.randn(32))

        cal = next(r for r in records if r.event == "calibration_completed")
        assert cal.metrics is None


# =========================================================================
# 3. Plotting helper
# =========================================================================


class TestPlotHelper:
    def test_plot_quantizer_metrics_builds_figure(self):
        x = torch.randn(2000) * 3
        q = torch.clamp(torch.round(x), -8, 7)
        m = compute_metrics(x, q, lsb=0, bit_width=4, signed=True, quantizer_role="weight")
        fig = plot_quantizer_metrics(m, quant_id="demo", trigger="calibration_completed")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_quantizer_metrics_log_scale(self):
        x = torch.randn(2000) * 3
        q = torch.clamp(torch.round(x), -8, 7)
        m = compute_metrics(x, q, lsb=0, bit_width=4, signed=True)
        fig = plot_quantizer_metrics(m, log_scale=True)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_quantizer_metrics_grid_builds_figure(self):
        x = torch.randn(2000) * 3
        q = torch.clamp(torch.round(x), -8, 7)
        m = compute_metrics(x, q, lsb=0, bit_width=4, signed=True)
        fig = plot_quantizer_metrics_grid(m, quant_id="demo", trigger="annealing_complete")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_from_real_log_record(self):
        """End-to-end: a real log record's `.metrics` plots directly."""
        records, _ = _capture_quantizer_logs()
        q = FixedPointPerTensorQuantizer(bit_width=4, quantizer_role="activation")
        q(torch.randn(500))

        cal = next(r for r in records if r.event == "calibration_completed")
        fig = plot_quantizer_metrics(cal.metrics, quant_id=cal.quant_id, trigger=cal.event)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_into_existing_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = torch.randn(500) * 2
        q = torch.clamp(torch.round(x), -8, 7)
        m = compute_metrics(x, q, lsb=0, bit_width=4, signed=True)

        fig, ax = plt.subplots()
        returned_fig = plot_quantizer_metrics(m, ax=ax)
        assert returned_fig is fig
        plt.close(fig)


# =========================================================================
# 4. "See how much things changed" -- calibration vs. post-annealing metrics
# =========================================================================


class TestCalibrationVsAnnealingComplete:
    def test_metrics_reflect_the_data_seen_at_each_event(self):
        """Calibrate on a narrow distribution, then feed much wider data
        during the annealing ramp -- the annealing_complete event's metrics
        must reflect the LATEST forward call's data (e.g. a different clip
        count), not a frozen copy of the calibration-time metrics. This is
        exactly the comparison the feature exists for: seeing how the
        quantizer's behavior evolved from calibration to the end of
        annealing."""
        q = FixedPointPerTensorQuantizer(bit_width=4, quantizer_role="activation")
        q.annealing_alpha.fill_(0.0)
        q.annealing_alpha_step = 0.5

        records, _ = _capture_quantizer_logs()
        q(torch.randn(500) * 0.1)  # calibrates on a narrow range -> small step
        records.clear()

        # Second call: alpha reaches 1.0 (step=0.5, two calls), with data
        # wide enough to clip against the narrow calibrated range.
        q(torch.randn(500) * 100.0)

        complete = next(r for r in records if r.event == "annealing_complete")
        assert complete.metrics is not None
        # The narrow calibration range all but guarantees heavy clipping
        # against this much wider second batch.
        assert complete.metrics["n_clipped"] > 0
        assert complete.metrics["input_max"] > complete.metrics["q_max"] or (
            complete.metrics["input_min"] < complete.metrics["q_min"]
        )
