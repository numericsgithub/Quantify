"""Engine robustness tests: vmap-fallback equivalence, OOM chunk-halving,
uint8 quantization error bound.
"""
import logging

import numpy as np
import pytest
import torch

from importance import analyze
from importance.storage import _quantize_uint8
import importance.engine as engine_mod

from tests.importance_test_models import Small2DCNN, make_image_loader


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_forced_fallback_matches_vmap_path():
    """allow_vmap=False should give (statistically) the same abs-metric
    values as the vmap path, since both compute the exact same quantity --
    just proving the fallback loop is not silently wrong."""
    torch.manual_seed(42)
    model = Small2DCNN(n_classes=3).eval()
    data = make_image_loader(n_batches=4, batch_size=4, hw=8)

    r_vmap = analyze(model, data, max_samples=16, store_samples=0, device="cpu", allow_vmap=True)
    assert r_vmap.manifest["settings"]["path_used"] == "vmap"

    r_fallback = analyze(model, data, max_samples=16, store_samples=0, device="cpu", allow_vmap=False)
    assert r_fallback.manifest["settings"]["path_used"] == "fallback"

    np.testing.assert_allclose(
        r_vmap.filter("conv1", metric="mean_abs_s"),
        r_fallback.filter("conv1", metric="mean_abs_s"),
        rtol=1e-3, atol=1e-6,
    )
    np.testing.assert_allclose(
        r_vmap.weight("fc", metric="mean_abs_s"),
        r_fallback.weight("fc", metric="mean_abs_s"),
        rtol=1e-3, atol=1e-6,
    )
    # signed metric is always computed the same (eager) way regardless of path
    np.testing.assert_allclose(
        r_vmap.filter("conv1", metric="mean_s"),
        r_fallback.filter("conv1", metric="mean_s"),
        rtol=1e-6, atol=1e-8,
    )


def test_vmap_failure_is_logged_and_falls_back(caplog):
    from tests.importance_test_models import QuantifyFixedPointModel

    model = QuantifyFixedPointModel().eval()
    model.calibrate(torch.randn(4, 3, 8, 8))
    data = make_image_loader(n_batches=2)
    with caplog.at_level(logging.WARNING, logger="importance"):
        result = analyze(model, data, max_samples=8, store_samples=0, device="cpu")
    assert result.manifest["settings"]["path_used"] == "fallback"
    assert any("vmap path failed" in rec.message for rec in caplog.records)


def test_oom_triggers_chunk_halving(monkeypatch, caplog):
    """Simulate a CUDA-OOM on the first (larger) chunk size and verify the
    engine halves the chunk and retries instead of crashing."""
    model = Small2DCNN(n_classes=3).eval()
    data = make_image_loader(n_batches=1, batch_size=4, hw=8)

    calls = {"n": 0}
    real_fn = engine_mod._try_vmap_abs_pass

    def flaky(model, x, layers, accumulators, output_spec, out_slice, n_out_total, chunk_outputs, use_quantized_weight):
        calls["n"] += 1
        if chunk_outputs > 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        return real_fn(model, x, layers, accumulators, output_spec, out_slice, n_out_total,
                        chunk_outputs, use_quantized_weight)

    monkeypatch.setattr(engine_mod, "_try_vmap_abs_pass", flaky)

    with caplog.at_level(logging.WARNING, logger="importance"):
        result = analyze(model, data, max_samples=4, store_samples=0, device="cpu",
                          chunk_outputs=4, chunk_samples=4)

    assert result.manifest["settings"]["path_used"] == "vmap"
    assert calls["n"] > 1, "expected at least one retry after the simulated OOM"
    assert any("halving chunk" in rec.message for rec in caplog.records)
    assert np.all(np.isfinite(result.filter("conv1")))


def test_uint8_quantization_error_bound():
    rng = np.random.default_rng(0)
    arr = rng.normal(scale=3.0, size=(6, 4, 3, 3)).astype(np.float32)
    q, scale = _quantize_uint8(arr)
    dequant = q.astype(np.float32) * scale
    max_err = np.abs(dequant - arr).max()
    # rounding to the nearest int on a step of `scale` bounds the error to
    # half a step (plus clamp effects for outliers beyond the 127-level range)
    assert max_err <= scale / 2 + 1e-4
    assert q.dtype == np.int8
    assert np.abs(q).max() <= 127


def test_uint8_quantization_all_zero_array_is_safe():
    arr = np.zeros((2, 3), dtype=np.float32)
    q, scale = _quantize_uint8(arr)
    assert scale == 1.0
    assert np.all(q == 0)


def test_weight_dtype_uint8_roundtrip_close(tmp_path):
    model = Small2DCNN(n_classes=3).eval()
    data = make_image_loader(n_batches=2)
    result = analyze(model, data, max_samples=8, store_samples=0, device="cpu", weight_dtype="uint8")
    out_dir = tmp_path / "res"
    result.save(str(out_dir))
    from importance import load
    reloaded = load(str(out_dir))
    dense = result.weight("conv1", metric="mean_abs_s")
    quantized_back = reloaded.weight("conv1", metric="mean_abs_s")
    scale = float(dense.max()) / 127.0 if dense.max() > 0 else 1.0
    np.testing.assert_allclose(quantized_back, dense, atol=scale + 1e-6)
