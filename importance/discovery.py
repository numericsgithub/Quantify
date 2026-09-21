"""Model introspection: layer discovery, execution order, batch/output auto-detection.

See docs/llm/importance_analysis.md for the metrics this feeds into.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
LINEAR_TYPES = (nn.Linear,)
NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)
PARAM_LEVEL_TYPES = CONV_TYPES + LINEAR_TYPES

_BATCH_DICT_KEYS = ("img", "image", "input", "x", "inputs", "pixel_values")


class BatchDetectionError(ValueError):
    pass


@dataclass
class LayerInfo:
    id: str
    type: str
    kind: str  # "conv" | "linear" | "other"
    module: Any = field(repr=False)
    execution_index: int = -1
    weight_shape: Optional[Tuple[int, ...]] = None
    has_bias: bool = False
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    kernel_size: Optional[Tuple[int, ...]] = None
    groups: int = 1
    has_bn_after: bool = False
    ndim: int = 0  # 1/2/3 for conv spatial dims, 0 for linear

    @property
    def weight_param_name(self) -> str:
        return f"{self.id}.weight"

    @property
    def bias_param_name(self) -> Optional[str]:
        return f"{self.id}.bias" if self.has_bias else None

    def to_manifest_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "kind": self.kind,
            "execution_index": self.execution_index,
            "weight_shape": list(self.weight_shape) if self.weight_shape else None,
            "has_bias": self.has_bias,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": list(self.kernel_size) if self.kernel_size else None,
            "groups": self.groups,
            "has_bn_after": self.has_bn_after,
            "ndim": self.ndim,
        }


def _classify(module: nn.Module) -> str:
    if isinstance(module, CONV_TYPES):
        return "conv"
    if isinstance(module, LINEAR_TYPES):
        return "linear"
    return "other"


def _spatial_ndim(module: nn.Module) -> int:
    if isinstance(module, nn.Conv1d):
        return 1
    if isinstance(module, nn.Conv2d):
        return 2
    if isinstance(module, nn.Conv3d):
        return 3
    return 0


def discover_layers(model: nn.Module, dummy_input: torch.Tensor) -> List[LayerInfo]:
    """Discover Conv*/Linear layers in forward-execution order via a dry-run hook pass.

    Falls back to declaration order (named_modules()) only for the has_bn_after
    heuristic when a module never fires during the dry run (dead branch).
    """
    order: List[str] = []
    module_by_name = dict(model.named_modules())
    name_by_module = {m: n for n, m in module_by_name.items() if n}

    handles = []

    def _hook(mod, inp, out):
        name = name_by_module.get(mod)
        if name is not None:
            order.append(name)

    for name, mod in module_by_name.items():
        if not name:
            continue
        if isinstance(mod, PARAM_LEVEL_TYPES + NORM_TYPES):
            handles.append(mod.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    # de-duplicate while preserving first-seen order (a module can fire more
    # than once, e.g. weight sharing / recurrent use)
    seen = set()
    exec_order = []
    for name in order:
        if name not in seen:
            seen.add(name)
            exec_order.append(name)

    # append any parameterized module that never fired (dead branch) at the end,
    # in declaration order, so it's still analyzable.
    for name, mod in module_by_name.items():
        if name and isinstance(mod, PARAM_LEVEL_TYPES) and name not in seen:
            seen.add(name)
            exec_order.append(name)

    layers: List[LayerInfo] = []
    for idx, name in enumerate(exec_order):
        mod = module_by_name[name]
        if not isinstance(mod, PARAM_LEVEL_TYPES):
            continue
        kind = _classify(mod)
        weight = getattr(mod, "weight", None)
        has_bias = getattr(mod, "bias", None) is not None
        info = LayerInfo(
            id=name,
            type=type(mod).__name__,
            kind=kind,
            module=mod,
            execution_index=idx,
            weight_shape=tuple(weight.shape) if weight is not None else None,
            has_bias=has_bias,
        )
        if kind == "conv":
            info.in_channels = mod.in_channels
            info.out_channels = mod.out_channels
            info.kernel_size = tuple(mod.kernel_size) if isinstance(mod.kernel_size, (tuple, list)) else (mod.kernel_size,)
            info.groups = mod.groups
            info.ndim = _spatial_ndim(mod)
        elif kind == "linear":
            info.in_channels = mod.in_features
            info.out_channels = mod.out_features
            info.kernel_size = ()
            info.groups = 1
            info.ndim = 0
        layers.append(info)

    # has_bn_after: the next thing to *execute* after this layer's id is a norm module.
    exec_index_of = {name: i for i, name in enumerate(exec_order)}
    for info in layers:
        pos = exec_index_of[info.id]
        if pos + 1 < len(exec_order):
            next_name = exec_order[pos + 1]
            next_mod = module_by_name[next_name]
            info.has_bn_after = isinstance(next_mod, NORM_TYPES)

    return layers


def detect_batch(batch: Any, batch_fn: Optional[Callable] = None) -> Tuple[Any, Any]:
    """Return (model_input, extra) from a dataloader batch.

    extra carries whatever else the batch held (targets, full batch dict, ...),
    made available to loss_fn as needed.
    """
    if batch_fn is not None:
        return batch_fn(batch)

    if isinstance(batch, torch.Tensor):
        return batch, None

    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise BatchDetectionError(
                "Got an empty list/tuple batch from the dataloader; pass batch_fn=lambda "
                "batch: (model_input, extra) to tell importance.analyze how to unpack it."
            )
        if len(batch) == 1:
            return batch[0], None
        if len(batch) == 2:
            # common (input, target) convention -- unwrap so loss_fn(out, target) works directly
            return batch[0], batch[1]
        return batch[0], batch[1:]

    if isinstance(batch, dict):
        for key in _BATCH_DICT_KEYS:
            if key in batch:
                return batch[key], batch
        raise BatchDetectionError(
            f"Could not find an input tensor in batch dict with keys {list(batch.keys())}. "
            f"Expected one of {_BATCH_DICT_KEYS}. Pass batch_fn=lambda batch: "
            f"(model_input, extra) to importance.analyze() to handle this dataloader."
        )

    raise BatchDetectionError(
        f"Don't know how to extract a model input from a batch of type {type(batch)}. "
        f"Pass batch_fn=lambda batch: (model_input, extra) to importance.analyze()."
    )


def detect_output(output: Any, output_fn: Optional[Callable] = None) -> torch.Tensor:
    """Return a plain float tensor [B, ...] from a raw model output."""
    if output_fn is not None:
        return output_fn(output)

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                warnings.warn(
                    "Model output was a tuple/list; using the first tensor found "
                    "for importance analysis. Pass output_fn=... to override.",
                    stacklevel=2,
                )
                return item
        raise ValueError(
            "Model output tuple/list contained no tensors; pass output_fn=... to "
            "importance.analyze() to extract the tensor to analyze."
        )

    if isinstance(output, dict):
        for item in output.values():
            if isinstance(item, torch.Tensor):
                warnings.warn(
                    "Model output was a dict; using the first tensor value found "
                    "for importance analysis. Pass output_fn=... to override.",
                    stacklevel=2,
                )
                return item
        raise ValueError(
            "Model output dict contained no tensor values; pass output_fn=... to "
            "importance.analyze() to extract the tensor to analyze."
        )

    raise ValueError(
        f"Don't know how to extract a tensor from model output of type {type(output)}. "
        f"Pass output_fn=... to importance.analyze()."
    )


@dataclass
class OutputSpec:
    """Describes how to turn a raw model output into a fixed-size [B, K] tensor
    of "output features" that importance is computed with respect to.
    """
    k: int
    names: List[str]
    source_shape: Tuple[int, ...]
    reduce_kind: str
    channel_reduce: bool  # True if source was [B, C, *spatial] -> summed over spatial
    topk_indices: Optional[torch.Tensor] = None  # only for flatten_topk
    output_fn: Optional[Callable] = None

    def transform(self, raw_output: Any) -> torch.Tensor:
        out = detect_output(raw_output, self.output_fn)
        out = out.float()
        if self.channel_reduce:
            # [B, C, *spatial] -> [B, C]
            out = out.flatten(2).sum(dim=2) if out.dim() > 2 else out
        else:
            out = out.flatten(1)
        if self.topk_indices is not None:
            out = out.index_select(1, self.topk_indices.to(out.device))
        return out


def build_output_spec(
    sample_output: Any,
    max_outputs: int = 256,
    output_reduce: Any = "auto",
    output_fn: Optional[Callable] = None,
) -> OutputSpec:
    out = detect_output(sample_output, output_fn)
    shape = tuple(out.shape)

    if callable(output_reduce) and output_reduce not in ("auto", "channel", "flatten_topk"):
        reduced = output_reduce(sample_output)
        k = reduced.shape[1]
        return OutputSpec(
            k=k, names=[f"out_{i}" for i in range(k)], source_shape=shape,
            reduce_kind="callable", channel_reduce=False, output_fn=lambda o, _fn=output_reduce: _fn(o),
        )

    if out.dim() <= 2:
        channel_reduce = False
        n_features = out.shape[1] if out.dim() == 2 else 1
    else:
        channel_reduce = True
        n_features = out.shape[1]

    kind = output_reduce
    if kind == "auto":
        kind = "channel" if channel_reduce else "flatten_topk"

    topk_indices = None
    if n_features > max_outputs:
        if kind == "channel" and channel_reduce:
            # still too many channels: fall back to a deterministic top-k by
            # activation magnitude on this sample batch (documented approximation)
            flat = out.flatten(2).sum(dim=2) if out.dim() > 2 else out
            scores = flat.abs().mean(dim=0)
            topk_indices = torch.topk(scores, k=max_outputs).indices.sort().values
            n_features = max_outputs
        elif kind == "flatten_topk":
            flat = out.flatten(1)
            scores = flat.abs().mean(dim=0)
            topk_indices = torch.topk(scores, k=max_outputs).indices.sort().values
            n_features = max_outputs
            channel_reduce = False
        else:
            raise ValueError(
                f"Output has {n_features} features (> max_outputs={max_outputs}) and "
                f"output_reduce={output_reduce!r} does not reduce it. Pass "
                f"output_reduce='flatten_topk', a smaller output_fn, or raise max_outputs."
            )

    names = [f"out_{i}" for i in range(n_features)]
    return OutputSpec(
        k=n_features, names=names, source_shape=shape, reduce_kind=kind,
        channel_reduce=channel_reduce, topk_indices=topk_indices, output_fn=output_fn,
    )


def warn_if_probability_head(model: nn.Module) -> None:
    """Best-effort warning if the model's last module looks like a Softmax/Sigmoid."""
    last = None
    for m in model.modules():
        last = m
    if isinstance(last, (nn.Softmax, nn.LogSoftmax, nn.Sigmoid)):
        warnings.warn(
            f"Model's last module is {type(last).__name__}; importance analysis uses "
            f"raw (pre-activation) outputs by convention -- consider exposing logits "
            f"via output_fn if this module is part of the traced output.",
            stacklevel=2,
        )
