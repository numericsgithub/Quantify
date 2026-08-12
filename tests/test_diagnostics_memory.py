"""
Regression tests for quantizer-diagnostics GPU memory.

Background: during the activation-introduction QAT run (batch=1024) the trainer
OOM'd inside utils/quantizer_diagnostics._compute_metrics. The first post-stem
activation tensor is 1024×32×112×112 = 411M elements; torch.unique() over it
SORTS the whole tensor and tried to allocate 9.19 GiB in one op — on top of an
already-full training step. Weight quantizers never hit this because their
tensors are tiny.

The fix reduces the scalar metrics over at most MAX_METRIC_SAMPLES elements (a
uniform random subsample for larger tensors). These tests pin two properties:

  1. Peak GPU memory used by _compute_metrics is BOUNDED and, crucially,
     INDEPENDENT of batch size — the exact thing that blew up. (CUDA only.)
  2. The subsampled metrics still match the full-tensor metrics closely, and
     tensors at or below the cap are still measured exactly.
"""

from __future__ import annotations

import math

import pytest
import torch

import utils.quantizer_diagnostics as qd
from utils.quantizer_diagnostics import _compute_metrics, MAX_METRIC_SAMPLES


def _fake_quant(x: torch.Tensor, lsb: int) -> torch.Tensor:
    step = 2.0 ** lsb
    return torch.round(x / step) * step


# ---------------------------------------------------------------------------
# GPU memory bound (the actual OOM)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compute_metrics_peak_memory_is_independent_of_batch():
    """Peak extra memory must NOT scale with the input tensor size.

    Before the fix every reduction (esp. torch.unique) ran on the full tensor,
    so peak memory scaled with batch. After the fix it is pinned near the
    sample cap. We compare an 8x-larger tensor and assert its peak extra memory
    is not proportionally larger.
    """
    dev = torch.device("cuda")

    def peak_extra_for(batch: int) -> int:
        x = torch.randn(batch, 32, 112, 112, device=dev)   # C×H×W = 401,408
        q = _fake_quant(x, lsb=-4)
        assert x.numel() > MAX_METRIC_SAMPLES
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev)
        _compute_metrics(x, q, lsb=-4, bit_width=8, signed=True,
                         input_shape=tuple(x.shape), quantizer_role="activation")
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated(dev) - base
        del x, q
        torch.cuda.empty_cache()
        return extra

    small = peak_extra_for(32)    # 12.8M elements
    big   = peak_extra_for(256)   # 102.8M elements  (8x the data)

    # If memory still scaled with the tensor, big would be ~8x small. Bounded,
    # it must stay well under 2x and comfortably below a fixed ceiling. The
    # sample cap is 4M elements; the working set is a small multiple of that.
    ceiling = MAX_METRIC_SAMPLES * 4 * 20  # ~320 MB, vs multi-GB for full unique
    assert big < ceiling, f"big peak {big/1e6:.0f} MB exceeds {ceiling/1e6:.0f} MB ceiling"
    assert big < small * 2.0, (
        f"peak scaled with batch: small={small/1e6:.0f} MB big={big/1e6:.0f} MB "
        f"(fix should decouple memory from batch size)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_run_diagnostics_does_not_oom_on_real_activation_size(tmp_path):
    """End-to-end run_diagnostics on the exact tensor size that OOM'd."""
    free, _ = torch.cuda.mem_get_info()
    # x and q are ~1.6 GiB each at this size; need headroom to build them.
    if free < 6 * 1024**3:
        pytest.skip("need >6 GiB free to build the 411M-element repro tensors")

    import matplotlib
    matplotlib.use("Agg")

    dev = torch.device("cuda")
    x = torch.randn(1024, 32, 112, 112, device=dev)  # 411M elements — the OOM case
    q = _fake_quant(x, lsb=-4)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(dev)
    base = torch.cuda.memory_allocated(dev)
    qd.run_diagnostics(
        quant_id="quant_test", x=x, quantized=q, lsb=-4, bit_width=8, signed=True,
        quantizer_role="activation", trigger="calibration_1", out_dir=tmp_path,
    )
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated(dev) - base

    # The whole diagnostics pass must stay far under the 9.19 GiB it used to try.
    assert extra < 1 * 1024**3, f"diagnostics used {extra/1e9:.2f} GiB extra"
    assert (tmp_path / "quantizer_quant_test.txt").exists()


# ---------------------------------------------------------------------------
# Correctness of the subsampled metrics (CPU — no CUDA required)
# ---------------------------------------------------------------------------

def test_subsampled_metrics_close_to_full_tensor(monkeypatch):
    monkeypatch.setattr(qd, "MAX_METRIC_SAMPLES", 20_000)
    torch.manual_seed(0)

    x = torch.randn(500_000) * 0.7
    q = _fake_quant(x, lsb=-4)

    m = _compute_metrics(x, q, lsb=-4, bit_width=8, signed=True,
                         input_shape=(500_000,), quantizer_role="activation")
    assert m["n_metric_samples"] == 20_000
    assert m["n_elements"] == 500_000

    err = (x - q)
    full_mse = float((err ** 2).mean())
    full_mae = float(err.abs().mean())

    assert math.isclose(m["mse"], full_mse, rel_tol=0.15)
    assert math.isclose(m["mae"], full_mae, rel_tol=0.15)


def test_small_tensor_measured_exactly():
    torch.manual_seed(1)
    x = torch.randn(1000) * 0.5
    q = _fake_quant(x, lsb=-4)

    m = _compute_metrics(x, q, lsb=-4, bit_width=8, signed=True,
                         input_shape=(1000,), quantizer_role="weight")
    # Below the cap → no subsampling, exact.
    assert m["n_metric_samples"] == 1000
    err = (x - q)
    assert math.isclose(m["mse"], float((err ** 2).mean()), rel_tol=1e-5)
    assert math.isclose(m["mae"], float(err.abs().mean()), rel_tol=1e-5)


def test_subsample_still_captures_all_grid_codes(monkeypatch):
    """Every code used by a non-trivial fraction of a large tensor must appear
    in the subsample — n_unique should equal the true grid usage."""
    monkeypatch.setattr(qd, "MAX_METRIC_SAMPLES", 50_000)
    torch.manual_seed(2)

    # 8 distinct quantized levels, each used by ~1/8 of a 1M-element tensor.
    step = 2.0 ** -4
    codes = torch.randint(0, 8, (1_000_000,))
    q = codes.float() * step
    x = q + torch.randn(1_000_000) * (step / 8)  # small noise, same rounding

    m = _compute_metrics(x, _fake_quant(x, lsb=-4), lsb=-4, bit_width=8,
                         signed=True, input_shape=(1_000_000,),
                         quantizer_role="activation")
    assert m["n_metric_samples"] == 50_000
    # All 8 frequent codes must survive the subsample.
    assert m["n_unique"] == 8
