"""Taylor-attribution importance engine.

For output feature j, sample n, parameter w:  s = w * d out_j / d w

Two independent computations happen per layer/output:

- **Signed weight/bias importance** (`mean_s`) and **activation-based filter
  importance** (`act_filter`) are both computed with a single "eager" forward
  pass per data batch plus one plain ``backward()`` call per output feature
  (see ``_eager_pass``). This never needs ``vmap`` -- it works for *any*
  autograd-compatible model, including Quantify's custom
  ``torch.autograd.Function`` quantizers, because summing a batched output
  column over the batch dimension before calling ``backward()`` gives
  *exactly* the per-sample gradient in each sample's slot of the resulting
  ``.grad`` tensor (true as long as samples don't interact -- i.e. the model
  is in eval mode). See docs/llm/importance_analysis.md.

- **Absolute weight/bias importance** (`mean_abs_s`) genuinely needs
  *per-sample* gradients of a value that is shared across the batch (the
  weight), which the trick above cannot give (only their sum). This is
  computed either with a vectorized ``torch.func`` ``vmap(jacrev(...))`` path
  (fast, primary) or, if that raises (e.g. an unsupported custom autograd
  Function), with a per-sample loop reusing ``_eager_pass`` at batch size 1
  (slow but always correct, fallback). See pitfall in
  docs/llm/pitfalls/brevitas_pitfalls.md.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from importance.discovery import LayerInfo, OutputSpec

logger = logging.getLogger("importance")

LEVEL_PRIORITY = ["weight", "kernel", "filter"]


@dataclass
class AnalyzeConfig:
    loss_fn: Optional[Callable] = None
    device: Optional[str] = None
    max_samples: int = 5000
    output_fn: Optional[Callable] = None
    batch_fn: Optional[Callable] = None
    output_reduce: Any = "auto"
    levels: Tuple[str, ...] = ("filter", "kernel", "weight")
    store_samples: int = 16
    max_outputs: int = 256
    use_quantized_weight: bool = False
    chunk_samples: Optional[int] = None
    chunk_outputs: Optional[int] = None
    seed: int = 0
    allow_vmap: bool = True
    progress: bool = True


def finest_level(levels: Tuple[str, ...]) -> str:
    for lvl in LEVEL_PRIORITY:
        if lvl in levels:
            return lvl
    return "filter"


def _reduce_grad(grad: torch.Tensor, level: str, ndim_kernel: int) -> torch.Tensor:
    """grad: [..., F, C, *kdims] (kdims empty for Linear; "..." is zero or more
    leading batch/chunk dims) -> reduced level shape, dims counted from the end
    so this works whether or not there's a leading vmap batch dim."""
    if level == "weight":
        return grad
    if level == "kernel":
        if ndim_kernel == 0:
            return grad
        dims = tuple(range(grad.dim() - ndim_kernel, grad.dim()))
        return grad.sum(dim=dims) if dims else grad
    if level == "filter":
        n = 1 + ndim_kernel  # C plus kernel dims
        dims = tuple(range(grad.dim() - n, grad.dim()))
        return grad.sum(dim=dims) if dims else grad
    raise ValueError(level)


def _slice_like(obj: Any, i) -> Any:
    """Slice a (possibly nested) batch-like object by index `i` (int, giving a
    length-1 slice) or a `slice` object."""
    if obj is None:
        return None
    sel = i if isinstance(i, slice) else slice(i, i + 1)
    if isinstance(obj, torch.Tensor):
        if sel.stop is not None and sel.stop > obj.shape[0]:
            sel = slice(sel.start, obj.shape[0])
        return obj[sel]
    if isinstance(obj, (list, tuple)):
        return type(obj)(_slice_like(x, i) for x in obj)
    if isinstance(obj, dict):
        return {k: _slice_like(v, i) for k, v in obj.items()}
    return obj


def _to_device(obj: Any, device) -> Any:
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _default_chunk_sizes(n_params_total: int, device: torch.device) -> Tuple[int, int]:
    if device.type == "cuda" and n_params_total > 0:
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except Exception:
            free_bytes = 2 * 1024 ** 3
        budget = int(free_bytes * 0.25)
        bytes_per_elem = 4
        max_cells = max(1, budget // (bytes_per_elem * max(n_params_total, 1)))
        chunk_out = max(1, min(8, max_cells))
        chunk_b = max(1, min(16, max_cells // max(chunk_out, 1)))
        return chunk_b, chunk_out
    return 4, 4


class _Accumulator:
    """Running sums for one layer, at the finest requested level."""

    def __init__(self, n_out_total: int, level: str, weight_shape: Tuple[int, ...],
                 has_bias: bool, ndim_kernel: int, out_channels: int):
        self.level = level
        self.ndim_kernel = ndim_kernel
        reduced_shape = self._level_shape(weight_shape, level, ndim_kernel)
        self.abs_sum = np.zeros((n_out_total,) + reduced_shape, dtype=np.float64)
        self.signed_sum = np.zeros((n_out_total,) + reduced_shape, dtype=np.float64)
        self.has_bias = has_bias
        if has_bias:
            self.bias_abs_sum = np.zeros((n_out_total, out_channels), dtype=np.float64)
            self.bias_signed_sum = np.zeros((n_out_total, out_channels), dtype=np.float64)
        self.act_abs_sum = np.zeros((n_out_total, out_channels), dtype=np.float64)
        self.act_signed_sum = np.zeros((n_out_total, out_channels), dtype=np.float64)
        self.act_count = np.zeros((n_out_total,), dtype=np.float64)

    @staticmethod
    def _level_shape(weight_shape, level, ndim_kernel):
        F = weight_shape[0]
        if len(weight_shape) > 1:
            C = weight_shape[1]
        else:
            C = 1
        k = weight_shape[2:]
        if level == "weight":
            return (F, C) + tuple(k)
        if level == "kernel":
            return (F, C)
        if level == "filter":
            return (F,)
        raise ValueError(level)


def _quantized_weight(module: nn.Module) -> Optional[torch.Tensor]:
    qw = getattr(module, "quant_weight", None)
    if callable(qw):
        try:
            out = qw()
            return getattr(out, "value", out)
        except Exception:
            return None
    return None


def _eager_pass(
    model: nn.Module,
    x: torch.Tensor,
    extra: Any,
    layers: List[LayerInfo],
    accumulators: Dict[str, _Accumulator],
    output_spec: OutputSpec,
    out_slice: List[int],
    n_out_total: int,
    loss_fn: Optional[Callable],
    chunk_outputs: int,
    use_quantized_weight: bool,
    collect_abs: bool,
):
    """One forward + one backward-per-output-feature pass.

    Always updates the signed metric and act_filter (abs+signed).
    If collect_abs is True (only meaningful when x has batch size 1), also
    updates the abs weight/bias metric -- this is the fallback path.
    """
    B = x.shape[0]
    acts: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(layer_id):
        def hook(mod, inp, out):
            t = out
            if not isinstance(t, torch.Tensor):
                t = getattr(t, "value", None)  # unwrap Brevitas QuantTensor
            if isinstance(t, torch.Tensor) and t.requires_grad:
                t.retain_grad()
                acts[layer_id] = t
        return hook

    for layer in layers:
        handles.append(layer.module.register_forward_hook(make_hook(layer.id)))

    try:
        raw_out = model(x)
    finally:
        for h in handles:
            h.remove()

    out_full = output_spec.transform(raw_out)  # [B, K]
    out_all = out_full.mean(dim=1, keepdim=True)
    out_cat = torch.cat([out_all, out_full], dim=1)  # [B, 1+K]
    # loss_fn is unused here -- the loss row is handled separately by
    # _loss_pass since it needs per-sample targets and an arbitrary
    # loss_fn(output, target) signature that can't go through this batched path.
    del loss_fn

    real_out_slice = [j for j in out_slice if j < out_cat.shape[1]]

    for start in range(0, len(real_out_slice), chunk_outputs):
        chunk = real_out_slice[start:start + chunk_outputs]
        for j in chunk:
            model.zero_grad(set_to_none=True)
            for t in acts.values():
                t.grad = None
            retain = True
            out_cat[:, j].sum().backward(retain_graph=retain)

            for layer in layers:
                acc = accumulators[layer.id]
                w = layer.module.weight
                w_eff = w
                if use_quantized_weight:
                    qw = _quantized_weight(layer.module)
                    if qw is not None:
                        w_eff = qw
                if w.grad is not None:
                    contrib = (w_eff.detach() * w.grad.detach())
                    reduced = _reduce_grad(contrib.unsqueeze(0), acc.level, acc.ndim_kernel)[0]
                    reduced_np = reduced.cpu().numpy()
                    if collect_abs:
                        acc.abs_sum[j] += np.abs(reduced_np)
                    else:
                        acc.signed_sum[j] += reduced_np
                if acc.has_bias and layer.module.bias is not None and layer.module.bias.grad is not None:
                    b = layer.module.bias
                    contrib_b = (b.detach() * b.grad.detach()).cpu().numpy()
                    if collect_abs:
                        acc.bias_abs_sum[j] += np.abs(contrib_b)
                    else:
                        acc.bias_signed_sum[j] += contrib_b

                if not collect_abs and layer.id in acts:
                    a = acts[layer.id]
                    g = a.grad
                    if g is not None:
                        contrib_a = (a.detach() * g.detach())
                        # [B, F, *spatial] -> per-channel sum over batch+spatial
                        if contrib_a.dim() > 2:
                            flat = contrib_a.flatten(2)
                            n_positions = flat.shape[0] * flat.shape[2]
                            abs_sum = flat.abs().sum(dim=(0, 2)).cpu().numpy()
                            signed_sum = flat.sum(dim=(0, 2)).cpu().numpy()
                        else:
                            n_positions = contrib_a.shape[0]
                            abs_sum = contrib_a.abs().sum(dim=0).cpu().numpy()
                            signed_sum = contrib_a.sum(dim=0).cpu().numpy()
                        acc.act_abs_sum[j] += abs_sum
                        acc.act_signed_sum[j] += signed_sum
                        acc.act_count[j] += n_positions

    model.zero_grad(set_to_none=True)


def _try_vmap_abs_pass(
    model: nn.Module,
    x: torch.Tensor,
    layers: List[LayerInfo],
    accumulators: Dict[str, _Accumulator],
    output_spec: OutputSpec,
    out_slice: List[int],
    n_out_total: int,
    chunk_outputs: int,
    use_quantized_weight: bool,
) -> None:
    from torch.func import functional_call, vmap, jacrev

    all_params = dict(model.named_parameters())
    all_buffers = dict(model.named_buffers())
    target_names = []
    for layer in layers:
        target_names.append(layer.weight_param_name)
        if layer.bias_param_name is not None:
            target_names.append(layer.bias_param_name)
    target_params = {n: all_params[n] for n in target_names if n in all_params}
    context_params = {n: p for n, p in all_params.items() if n not in target_params}

    real_out_slice = [j for j in out_slice if j <= output_spec.k]

    def make_f(out_chunk):
        def f(tparams, x_single):
            merged = {**context_params, **tparams}
            x_b = x_single.unsqueeze(0)
            raw_out = functional_call(model, (merged, all_buffers), (x_b,))
            out_full = output_spec.transform(raw_out)
            out_all = out_full.mean(dim=1, keepdim=True)
            out_cat = torch.cat([out_all, out_full], dim=1)
            idx = torch.tensor(out_chunk, device=out_cat.device)
            return out_cat[0].index_select(0, idx)
        return f

    for start in range(0, len(real_out_slice), chunk_outputs):
        chunk = real_out_slice[start:start + chunk_outputs]
        f = make_f(chunk)
        jac_fn = jacrev(f, argnums=0)
        per_sample = vmap(jac_fn, in_dims=(None, 0))(target_params, x)
        for layer in layers:
            acc = accumulators[layer.id]
            wname = layer.weight_param_name
            if wname in per_sample:
                grad = per_sample[wname]  # [B, chunk_out, *weight_shape]
                if use_quantized_weight:
                    qw = _quantized_weight(layer.module)
                    w_eff = qw if qw is not None else layer.module.weight
                else:
                    w_eff = layer.module.weight
                contrib = grad * w_eff.detach()
                reduced = _reduce_grad(contrib, acc.level, acc.ndim_kernel)  # [B, chunk_out, *level_shape]
                reduced_np = reduced.detach().cpu().numpy()
                for ci, j in enumerate(chunk):
                    acc.abs_sum[j] += np.abs(reduced_np[:, ci]).sum(axis=0)
            bname = layer.bias_param_name
            if bname is not None and bname in per_sample and acc.has_bias:
                gradb = per_sample[bname]  # [B, chunk_out, F]
                contribb = gradb * layer.module.bias.detach()
                contribb_np = contribb.detach().cpu().numpy()
                for ci, j in enumerate(chunk):
                    acc.bias_abs_sum[j] += np.abs(contribb_np[:, ci]).sum(axis=0)


def _loss_pass(
    model: nn.Module,
    x: torch.Tensor,
    extra: Any,
    layers: List[LayerInfo],
    accumulators: Dict[str, _Accumulator],
    loss_fn: Callable,
    loss_col: int,
    collect_abs: bool,
    use_quantized_weight: bool,
):
    """Per-sample loop computing the loss row (always per-sample; loss_fn's
    signature is arbitrary so this cannot generically go through vmap)."""
    B = x.shape[0]
    for i in range(B):
        xi = x[i:i + 1]
        extra_i = _slice_like(extra, i)
        model.zero_grad(set_to_none=True)
        out_i = model(xi)
        try:
            loss = loss_fn(out_i, extra_i)
        except TypeError:
            loss = loss_fn(out_i, extra)
        loss.backward()
        for layer in layers:
            acc = accumulators[layer.id]
            w = layer.module.weight
            if w.grad is not None:
                w_eff = w
                if use_quantized_weight:
                    qw = _quantized_weight(layer.module)
                    if qw is not None:
                        w_eff = qw
                contrib = (w_eff.detach() * w.grad.detach())
                reduced = _reduce_grad(contrib.unsqueeze(0), acc.level, acc.ndim_kernel)[0].cpu().numpy()
                if collect_abs:
                    acc.abs_sum[loss_col] += np.abs(reduced)
                else:
                    acc.signed_sum[loss_col] += reduced
            if acc.has_bias and layer.module.bias is not None and layer.module.bias.grad is not None:
                b = layer.module.bias
                contrib_b = (b.detach() * b.grad.detach()).cpu().numpy()
                if collect_abs:
                    acc.bias_abs_sum[loss_col] += np.abs(contrib_b)
                else:
                    acc.bias_signed_sum[loss_col] += contrib_b
    model.zero_grad(set_to_none=True)


@dataclass
class EngineResult:
    accumulators: Dict[str, _Accumulator]
    n_samples: int
    path_used: str
    n_out_total: int
    has_loss: bool
    output_names: List[str]
    level: str


def run_analysis(
    model: nn.Module,
    dataloader,
    layers: List[LayerInfo],
    output_spec: OutputSpec,
    config: AnalyzeConfig,
    device: torch.device,
) -> EngineResult:
    from importance.discovery import detect_batch

    torch.manual_seed(config.seed)
    level = finest_level(config.levels)
    n_out_total = 1 + output_spec.k + (1 if config.loss_fn is not None else 0)
    loss_col = n_out_total - 1 if config.loss_fn is not None else None

    accumulators: Dict[str, _Accumulator] = {}
    for layer in layers:
        weight_shape = layer.weight_shape
        accumulators[layer.id] = _Accumulator(
            n_out_total=n_out_total,
            level=level,
            weight_shape=weight_shape,
            has_bias=layer.has_bias,
            ndim_kernel=len(weight_shape) - 2 if len(weight_shape) > 2 else 0,
            out_channels=layer.out_channels or weight_shape[0],
        )

    n_params_total = sum(int(np.prod(l.weight_shape)) for l in layers if l.weight_shape)
    chunk_b, chunk_out = config.chunk_samples, config.chunk_outputs
    if chunk_b is None or chunk_out is None:
        auto_b, auto_out = _default_chunk_sizes(n_params_total, device)
        chunk_b = chunk_b or auto_b
        chunk_out = chunk_out or auto_out

    out_slice_all = list(range(1 + output_spec.k))  # __all__ + real outputs (loss handled separately)

    path_used = "vmap" if config.allow_vmap else "fallback"
    vmap_failed = not config.allow_vmap
    n_samples = 0
    n_batches_seen = 0

    for batch in dataloader:
        if n_samples >= config.max_samples:
            break
        x_raw, extra = detect_batch(batch, config.batch_fn)
        x = x_raw.to(device)
        extra = _to_device(extra, device)
        B = x.shape[0]
        if n_samples + B > config.max_samples:
            keep = config.max_samples - n_samples
            x = x[:keep]
            extra = _slice_like(extra, slice(0, keep))
            B = keep

        # 1) signed metric + act_filter (always eager, robust)
        for start in range(0, B, chunk_b):
            xb = x[start:start + chunk_b]
            extra_b = _slice_like(extra, slice(start, start + chunk_b)) if extra is not None else None
            _eager_pass(
                model, xb, extra_b, layers, accumulators, output_spec, out_slice_all,
                n_out_total, None, chunk_out, config.use_quantized_weight, collect_abs=False,
            )

        # 2) abs metric: vmap primary, per-sample fallback
        if not vmap_failed:
            try:
                for start in range(0, B, chunk_b):
                    xb = x[start:start + chunk_b]
                    cb, co = chunk_b, chunk_out
                    while True:
                        try:
                            _try_vmap_abs_pass(
                                model, xb, layers, accumulators, output_spec, out_slice_all,
                                n_out_total, co, config.use_quantized_weight,
                            )
                            break
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower() and (cb > 1 or co > 1):
                                torch.cuda.empty_cache()
                                if co > 1:
                                    co = max(1, co // 2)
                                else:
                                    cb = max(1, cb // 2)
                                    xb = xb[:cb]
                                logger.warning("importance: OOM, halving chunk to (b=%d, out=%d)", cb, co)
                                continue
                            raise
                path_used = "vmap"
            except Exception as e:  # noqa: BLE001 - vmap can fail in many ways for custom autograd Fns
                logger.warning(
                    "importance: vmap path failed (%s: %s); falling back to the "
                    "per-sample loop for the abs metric.", type(e).__name__, e,
                )
                vmap_failed = True
                path_used = "fallback"

        if vmap_failed:
            for i in range(B):
                _eager_pass(
                    model, x[i:i + 1], None, layers, accumulators, output_spec, out_slice_all,
                    n_out_total, None, chunk_out, config.use_quantized_weight, collect_abs=True,
                )

        # 3) loss row
        if config.loss_fn is not None:
            for i in range(B):
                _loss_pass(
                    model, x[i:i + 1], _slice_like(extra, i), layers, accumulators,
                    config.loss_fn, loss_col, collect_abs=False, use_quantized_weight=config.use_quantized_weight,
                )
                _loss_pass(
                    model, x[i:i + 1], _slice_like(extra, i), layers, accumulators,
                    config.loss_fn, loss_col, collect_abs=True, use_quantized_weight=config.use_quantized_weight,
                )

        n_samples += B
        n_batches_seen += 1
        if config.progress and n_batches_seen % 10 == 0:
            logger.info("importance: processed %d samples", n_samples)

    if n_samples == 0:
        raise ValueError("Dataloader produced no samples (or max_samples=0).")

    for acc in accumulators.values():
        acc.abs_sum /= n_samples
        acc.signed_sum /= n_samples
        if acc.has_bias:
            acc.bias_abs_sum /= n_samples
            acc.bias_signed_sum /= n_samples
        safe_count = np.where(acc.act_count > 0, acc.act_count, 1.0)
        acc.act_abs_sum = acc.act_abs_sum / safe_count[:, None]
        acc.act_signed_sum = acc.act_signed_sum / safe_count[:, None]

    output_names = ["__all__"] + output_spec.names
    if config.loss_fn is not None:
        output_names = output_names + ["loss"]

    return EngineResult(
        accumulators=accumulators,
        n_samples=n_samples,
        path_used=path_used,
        n_out_total=n_out_total,
        has_loss=config.loss_fn is not None,
        output_names=output_names,
        level=level,
    )
