"""Public API: analyze(), view(). See importance/__init__.py for exports."""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from importance.discovery import discover_layers, detect_batch, build_output_spec, warn_if_probability_head
from importance.engine import AnalyzeConfig, run_analysis
from importance.storage import Result, DEFAULT_SIZE_WARNING_BYTES


def analyze(
    model: nn.Module,
    dataloader,
    loss_fn: Optional[Callable] = None,
    device: Optional[str] = None,
    max_samples: int = 2000,
    output_fn: Optional[Callable] = None,
    batch_fn: Optional[Callable] = None,
    output_reduce: Any = "auto",
    levels: Tuple[str, ...] = ("filter", "kernel", "weight"),
    store_samples: int = 16,
    use_quantized_weight: bool = False,
    weight_dtype: str = "uint8",
    max_outputs: int = 256,
    chunk_samples: Optional[int] = None,
    chunk_outputs: Optional[int] = None,
    seed: int = 0,
    allow_vmap: bool = True,
    size_warning_bytes: int = DEFAULT_SIZE_WARNING_BYTES,
    progress: bool = True,
) -> Result:
    """Run Taylor-attribution importance analysis of `model` over `dataloader`.

    See docs/llm/importance_analysis.md for the metrics computed and
    new_feature.md for the full spec this implements.
    """
    if weight_dtype not in ("float32", "float16", "uint8"):
        raise ValueError(f"weight_dtype must be one of float32/float16/uint8, got {weight_dtype!r}")

    resolved_device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(resolved_device)

    was_training = model.training
    model.eval()

    try:
        it = iter(dataloader)
        first_batch = next(it)
    except StopIteration:
        raise ValueError("dataloader is empty")

    x0, extra0 = detect_batch(first_batch, batch_fn)
    x0 = x0.to(resolved_device)
    dummy = x0[:1]

    layers = discover_layers(model, dummy)
    if not layers:
        raise ValueError(
            "No Conv1d/2d/3d or Linear layers found in this model (checked "
            "isinstance against torch.nn / Brevitas subclasses). Nothing to analyze."
        )

    warn_if_probability_head(model)

    with torch.no_grad():
        raw_out0 = model(x0)
    output_spec = build_output_spec(raw_out0, max_outputs=max_outputs, output_reduce=output_reduce,
                                     output_fn=output_fn)

    samples = _collect_samples(model, dataloader, batch_fn, output_spec, store_samples, resolved_device)

    config = AnalyzeConfig(
        loss_fn=loss_fn, device=str(resolved_device), max_samples=max_samples, output_fn=output_fn,
        batch_fn=batch_fn, output_reduce=output_reduce, levels=tuple(levels), store_samples=store_samples,
        max_outputs=max_outputs, use_quantized_weight=use_quantized_weight,
        chunk_samples=chunk_samples, chunk_outputs=chunk_outputs, seed=seed, allow_vmap=allow_vmap,
        progress=progress,
    )

    engine_result = run_analysis(model, dataloader, layers, output_spec, config, resolved_device)

    model.train(was_training)

    model_summary = {
        "class_name": type(model).__name__,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_analyzed_layers": len(layers),
        "device": str(resolved_device),
    }
    dataset_info = {"input_shape": list(x0.shape[1:])}

    result = Result.from_engine_result(
        engine_result, layers, config, output_spec, weight_dtype, model_summary, dataset_info,
        samples, size_warning_bytes=size_warning_bytes,
    )
    return result


def _collect_samples(model, dataloader, batch_fn, output_spec, store_samples, device):
    if store_samples <= 0:
        return {"arrays": {}, "meta": {"count": 0}}
    inputs = []
    outputs = []
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            x, _extra = detect_batch(batch, batch_fn)
            x = x.to(device)
            raw_out = model(x)
            out = output_spec.transform(raw_out)
            out_all = out.mean(dim=1, keepdim=True)
            out_cat = torch.cat([out_all, out], dim=1)
            take = min(store_samples - count, x.shape[0])
            inputs.append(x[:take].cpu().numpy())
            outputs.append(out_cat[:take].cpu().numpy())
            count += take
            if count >= store_samples:
                break
    if not inputs:
        return {"arrays": {}, "meta": {"count": 0}}
    input_arr = np.concatenate(inputs, axis=0)
    output_arr = np.concatenate(outputs, axis=0)
    return {
        "arrays": {"inputs": input_arr, "outputs": output_arr},
        "meta": {"count": int(input_arr.shape[0]), "input_shape": list(input_arr.shape[1:])},
    }


def view(result_dir: str, port: int = 8000, open_browser: bool = True, host: str = "127.0.0.1",
         block: bool = True):
    """Start the local importance viewer web app for a saved result directory.

    Pass host="0.0.0.0" to expose it on the LAN instead of localhost-only.
    """
    from importance.serve import serve
    return serve(result_dir, port=port, open_browser=open_browser, host=host, block=block)


def load(path: str) -> Result:
    return Result.load(path)
