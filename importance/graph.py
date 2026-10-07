"""Dataflow-graph extraction for the viewer's Netron-style graph.

`trace_graph(model, example_input)` runs one forward pass and records which
module / functional op produced every tensor and which ones consumed it, so
residual adds, concatenations, branches and multi-output heads show up as
real graph edges instead of the flat execution-order list the viewer used
before.

How it works (no torch.fx, which can't trace data-dependent control flow
and chokes on Brevitas' QuantTensor plumbing):

- Global module pre/forward hooks bracket every call of an *atomic* module
  (a module with no children, any Conv/Linear/Norm, or anything from the
  `brevitas`/`quantizers` packages -- e.g. a QuantConv2d with its injected
  quantizer submodules is one node, not twenty). Everything that happens
  inside an atomic module is hidden behind that one node.
- A `TorchFunctionMode` sees every torch function / Tensor method executed
  *outside* atomic modules (``x + y``, ``torch.cat``, ``x.flatten(1)``,
  ``F.relu`` ...). Each one whose inputs include a traced tensor becomes an
  "op" node; ops that only touch parameters/constants are skipped.
- Tensors are matched by Python identity. All traced tensors are kept
  alive until the trace is done, so an id can't be recycled mid-trace.

The result is plain JSON (nodes + edges) stored as ``graph.json`` next to
``manifest.json``. If tracing fails for some exotic model, `analyze()` logs a
warning and the viewer falls back to a sequential chain of the analyzed
layers.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

from importance.discovery import NORM_TYPES, PARAM_LEVEL_TYPES

logger = logging.getLogger("importance")

# Same values, possibly a new tensor object (dtype/device/layout moves): no
# node of their own, but the result inherits the input's producer so the
# edge to whatever consumes it isn't lost.
_PASSTHROUGH_OPS = {"contiguous", "to", "float", "half", "double", "cpu", "cuda", "detach", "clone", "type_as"}

# Tensor methods / functions that are pure bookkeeping and never worth a node.
_IGNORED_OPS = _PASSTHROUGH_OPS | {
    "size", "dim", "numel", "__len__", "__get__", "__getattribute__", "__format__", "__repr__",
    "__bool__", "__index__", "__int__", "__float__", "item", "tolist", "is_floating_point",
    "requires_grad_", "__setstate__", "__reduce_ex__", "__deepcopy__", "data_ptr", "element_size",
    "stride", "storage_offset", "is_contiguous", "get_device", "__iter__", "__eq__", "__ne__",
}

# Ops that don't change values, only shape, collapse into a "reshape" family
# for colouring; everything else keeps its own name.
_OP_CATEGORY = {
    "add": "merge", "__add__": "merge", "__radd__": "merge", "__iadd__": "merge", "sub": "merge",
    "__sub__": "merge", "__rsub__": "merge", "mul": "elementwise", "__mul__": "elementwise",
    "__rmul__": "elementwise", "__imul__": "elementwise", "div": "elementwise",
    "__truediv__": "elementwise", "cat": "merge", "concat": "merge", "stack": "merge",
    "flatten": "shape", "view": "shape", "reshape": "shape", "permute": "shape",
    "transpose": "shape", "squeeze": "shape", "unsqueeze": "shape", "chunk": "shape",
    "split": "shape", "__getitem__": "shape", "mean": "pool", "sum": "pool",
    "adaptive_avg_pool2d": "pool", "adaptive_avg_pool1d": "pool", "max_pool2d": "pool",
    "avg_pool2d": "pool", "relu": "activation", "relu_": "activation", "sigmoid": "activation",
    "silu": "activation", "gelu": "activation", "tanh": "activation", "softmax": "activation",
    "interpolate": "shape",
}

_OP_DISPLAY = {
    "__add__": "add", "__radd__": "add", "__iadd__": "add", "__sub__": "sub", "__rsub__": "sub",
    "__mul__": "mul", "__rmul__": "mul", "__imul__": "mul", "__truediv__": "div",
    "__getitem__": "slice", "__matmul__": "matmul",
}


def _module_category(mod: nn.Module) -> str:
    name = type(mod).__name__.lower()
    if isinstance(mod, PARAM_LEVEL_TYPES):
        return "conv" if "conv" in name else "linear"
    if isinstance(mod, NORM_TYPES) or "norm" in name:
        return "norm"
    if any(k in name for k in ("relu", "sigmoid", "tanh", "silu", "gelu", "softmax", "hardswish", "mish", "elu")):
        return "activation"
    if "pool" in name:
        return "pool"
    if any(k in name for k in ("flatten", "reshape", "upsample", "identity", "dropout")):
        return "shape"
    if "quant" in name:
        return "quant"
    return "other"


def _is_atomic(mod: nn.Module) -> bool:
    if isinstance(mod, PARAM_LEVEL_TYPES + NORM_TYPES):
        return True
    if next(mod.children(), None) is None:
        return True
    pkg = type(mod).__module__ or ""
    return pkg.startswith("brevitas") or pkg.startswith("quantizers")


def _tensors_in(obj: Any, out: List[torch.Tensor], depth: int = 0) -> List[torch.Tensor]:
    if depth > 4:
        return out
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        # Brevitas QuantTensor is a tuple subclass whose first field is the value
        for x in obj:
            _tensors_in(x, out, depth + 1)
    elif isinstance(obj, dict):
        for x in obj.values():
            _tensors_in(x, out, depth + 1)
    else:
        val = getattr(obj, "value", None)
        if isinstance(val, torch.Tensor):
            out.append(val)
    return out


def _value_repr(v: Any) -> Any:
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (tuple, list)) and all(isinstance(x, (bool, int, float)) for x in v):
        return list(v)
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return float(v.detach().cpu())
    return str(v)[:80]


def module_attributes(mod: nn.Module) -> Dict[str, Any]:
    """Netron-style attribute list: hyper-parameters, not tensors."""
    attrs: Dict[str, Any] = {}
    for key in ("in_channels", "out_channels", "in_features", "out_features", "kernel_size", "stride",
                "padding", "dilation", "groups", "padding_mode", "num_features", "eps", "momentum",
                "affine", "track_running_stats", "normalized_shape", "num_groups", "negative_slope",
                "inplace", "output_size", "start_dim", "end_dim", "p", "dim", "scale_factor", "mode",
                "min_val", "max_val", "ceil_mode"):
        if hasattr(mod, key):
            try:
                attrs[key] = _value_repr(getattr(mod, key))
            except Exception:  # noqa: BLE001 - attribute access on exotic modules
                pass
    return attrs


def _bit_width_of(q: Any) -> Optional[float]:
    for attr in ("bit_width",):
        bw = getattr(q, attr, None)
        try:
            if callable(bw):
                bw = bw()
            if isinstance(bw, torch.Tensor):
                return float(bw.detach().cpu())
            if isinstance(bw, (int, float)):
                return float(bw)
        except Exception:  # noqa: BLE001
            return None
    return None


def quantization_info(mod: nn.Module) -> Dict[str, Any]:
    """Best-effort summary of the quantizers attached to a module (Brevitas
    `*_quant` proxies and Quantify quantizers), for the properties panel."""
    info: Dict[str, Any] = {}
    for proxy_name in ("weight_quant", "bias_quant", "input_quant", "output_quant", "act_quant"):
        proxy = getattr(mod, proxy_name, None)
        if proxy is None:
            continue
        entry: Dict[str, Any] = {}
        enabled = getattr(proxy, "is_quant_enabled", None)
        if enabled is not None:
            entry["enabled"] = bool(enabled)
            if not enabled:
                info[proxy_name] = entry
                continue
        tensor_quant = getattr(proxy, "tensor_quant", None)
        if tensor_quant is not None:
            entry["quantizer"] = type(tensor_quant).__name__
        else:
            entry["quantizer"] = type(proxy).__name__
        bw = _bit_width_of(proxy)
        if bw is None and tensor_quant is not None:
            bw = _bit_width_of(tensor_quant)
        if bw is not None:
            entry["bit_width"] = bw
        # Quantify quantizers keep a few cheap, interesting buffers
        for sub in (tensor_quant, proxy):
            if sub is None:
                continue
            for attr in ("signed", "narrow_range", "lsb", "search_done", "annealing_alpha"):
                if attr in entry or not hasattr(sub, attr):
                    continue
                try:
                    entry[attr] = _value_repr(getattr(sub, attr))
                except Exception:  # noqa: BLE001
                    pass
        info[proxy_name] = entry
    return info


class _Tracer(TorchFunctionMode):
    def __init__(self, graph: "_GraphBuilder"):
        super().__init__()
        self.graph = graph

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        g = self.graph
        if g.atomic_depth == 0 and not g.suspended:
            name = getattr(func, "__name__", None) or str(func)
            if name in _PASSTHROUGH_OPS:
                g.alias(args[0] if args else None, out)
            elif name not in _IGNORED_OPS:
                g.suspended = True  # don't trace our own bookkeeping
                try:
                    g.record_op(name, args, kwargs, out)
                finally:
                    g.suspended = False
        return out


class _GraphBuilder:
    def __init__(self, model: nn.Module, analyzed_ids: set):
        self.model = model
        self.analyzed_ids = analyzed_ids
        self.name_of = {m: n for n, m in model.named_modules() if n}
        self.nodes: List[dict] = []
        self.edges: List[dict] = []
        self.producer: Dict[int, tuple] = {}  # id(tensor) -> (node_id, output_slot)
        self.keepalive: List[torch.Tensor] = []
        self.atomic_depth = 0
        self.suspended = False
        self.scope: List[str] = []
        self.call_count: Dict[str, int] = {}
        self._pending: List[tuple] = []  # stack of (module, node_id) for atomic calls

    # -- helpers ---------------------------------------------------------
    def _new_node(self, **kw) -> str:
        node_id = f"n{len(self.nodes)}"
        kw["id"] = node_id
        self.nodes.append(kw)
        return node_id

    def _connect_inputs(self, node_id: str, tensors: List[torch.Tensor]) -> int:
        n = 0
        seen = set()
        for t in tensors:
            src = self.producer.get(id(t))
            if src is None or (src[0], id(t)) in seen:
                continue
            seen.add((src[0], id(t)))
            self.edges.append({"source": src[0], "target": node_id, "shape": list(t.shape)})
            n += 1
        return n

    def _register_outputs(self, node_id: str, out: Any) -> List[List[int]]:
        shapes = []
        for t in _tensors_in(out, []):
            self.producer[id(t)] = (node_id, len(shapes))
            self.keepalive.append(t)
            shapes.append(list(t.shape))
        return shapes

    def alias(self, src: Any, out: Any) -> None:
        if isinstance(src, torch.Tensor) and isinstance(out, torch.Tensor) and out is not src:
            producer = self.producer.get(id(src))
            if producer is not None:
                self.producer[id(out)] = producer
                self.keepalive.append(out)

    # -- entry points ----------------------------------------------------
    def add_input(self, x: Any) -> None:
        tensors = _tensors_in(x, [])
        for i, t in enumerate(tensors):
            nid = self._new_node(kind="input", name="input" if len(tensors) == 1 else f"input_{i}",
                                 type="Input", category="io", output_shapes=[list(t.shape)])
            self.producer[id(t)] = (nid, 0)
            self.keepalive.append(t)

    def add_output(self, out: Any) -> None:
        tensors = _tensors_in(out, [])
        for i, t in enumerate(tensors):
            nid = self._new_node(kind="output", name="output" if len(tensors) == 1 else f"output_{i}",
                                 type="Output", category="io", input_shapes=[list(t.shape)])
            self._connect_inputs(nid, [t])

    def pre_hook(self, mod, args):
        name = self.name_of.get(mod)
        if name is not None:
            self.scope.append(name)
        if name is None or self.suspended:
            return
        if self.atomic_depth == 0 and _is_atomic(mod):
            count = self.call_count.get(name, 0)
            self.call_count[name] = count + 1
            in_tensors = _tensors_in(args, [])
            node_id = self._new_node(
                kind="module", name=name if count == 0 else f"{name}#{count + 1}", module=name,
                type=type(mod).__name__, category=_module_category(mod),
                analyzed=name in self.analyzed_ids and count == 0,
                input_shapes=[list(t.shape) for t in in_tensors],
            )
            self._connect_inputs(node_id, in_tensors)
            self._pending.append((mod, node_id))
            self.atomic_depth += 1
        elif self.atomic_depth > 0:
            self.atomic_depth += 1

    def post_hook(self, mod, args, out):
        name = self.name_of.get(mod)
        if name is not None and self.scope and self.scope[-1] == name:
            self.scope.pop()
        if name is None or self.suspended or self.atomic_depth == 0:
            return
        self.atomic_depth -= 1
        if self.atomic_depth == 0 and self._pending and self._pending[-1][0] is mod:
            _, node_id = self._pending.pop()
            node = self.nodes[int(node_id[1:])]
            node["output_shapes"] = self._register_outputs(node_id, out)

    def record_op(self, name: str, args, kwargs, out) -> None:
        in_tensors = _tensors_in(list(args) + list(kwargs.values()), [])
        traced = [t for t in in_tensors if id(t) in self.producer]
        out_tensors = _tensors_in(out, [])
        if not traced or not out_tensors:
            return
        # in-place op on a traced tensor with no other traced inputs (e.g.
        # relu_ after a module) still deserves its own node -- fall through.
        display = _OP_DISPLAY.get(name, name)
        scope = self.scope[-1] if self.scope else ""
        count = self.call_count.get(f"{scope}::{display}", 0)
        self.call_count[f"{scope}::{display}"] = count + 1
        label = f"{scope}.{display}" if scope else display
        if count:
            label = f"{label}#{count + 1}"
        attrs = {}
        for k, v in kwargs.items():
            if not isinstance(v, torch.Tensor):
                attrs[k] = _value_repr(v)
        for i, a in enumerate(args):
            if not isinstance(a, (torch.Tensor, list, tuple)) or (
                    isinstance(a, (list, tuple)) and not _tensors_in(a, [])):
                attrs[f"arg{i}"] = _value_repr(a)
        node_id = self._new_node(kind="op", name=label, type=display, scope=scope,
                                 category=_OP_CATEGORY.get(name, "other"), attributes=attrs,
                                 input_shapes=[list(t.shape) for t in traced])
        self._connect_inputs(node_id, traced)
        self.nodes[-1]["output_shapes"] = self._register_outputs(node_id, out)


def _prune_dead_ends(nodes: List[dict], edges: List[dict]) -> tuple:
    """Drop op nodes whose result never reaches an output or a module (e.g.
    a ``x.shape``-adjacent helper computation, or an assert's comparison)."""
    succ: Dict[str, List[str]] = {}
    for e in edges:
        succ.setdefault(e["source"], []).append(e["target"])
    by_id = {n["id"]: n for n in nodes}
    alive = {n["id"] for n in nodes if n["kind"] == "output"}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n["id"] in alive:
                continue
            if any(t in alive for t in succ.get(n["id"], [])):
                alive.add(n["id"])
                changed = True
    keep = {nid for nid in by_id if nid in alive or by_id[nid]["kind"] in ("input", "module")}
    nodes = [n for n in nodes if n["id"] in keep]
    edges = [e for e in edges if e["source"] in keep and e["target"] in keep]
    return nodes, edges


def trace_graph(model: nn.Module, example_input: Any, analyzed_ids: Optional[set] = None) -> dict:
    """Trace one forward pass of `model` and return {"nodes": [...], "edges": [...]}.

    Node fields: id, kind ("input"|"module"|"op"|"output"), name, type,
    category (colouring), module (for module nodes: the named_modules() key),
    analyzed (True for the layers that have importance arrays),
    input_shapes/output_shapes, attributes, params, quantization.
    """
    builder = _GraphBuilder(model, set(analyzed_ids or ()))
    pre = torch.nn.modules.module.register_module_forward_pre_hook(builder.pre_hook)
    post = torch.nn.modules.module.register_module_forward_hook(builder.post_hook)
    was_training = model.training
    model.eval()
    try:
        builder.add_input(example_input)
        with torch.no_grad(), _Tracer(builder), warnings.catch_warnings():
            # Brevitas renames tensors internally; under a TorchFunctionMode
            # that surfaces PyTorch's "named tensors are experimental" warning
            warnings.filterwarnings("ignore", message="Named tensors")
            out = model(example_input)
        builder.suspended = True
        builder.add_output(out)
    finally:
        pre.remove()
        post.remove()
        model.train(was_training)

    modules = dict(model.named_modules())
    for node in builder.nodes:
        if node["kind"] != "module":
            continue
        mod = modules[node["module"]]
        node["attributes"] = module_attributes(mod)
        node["params"] = [
            {"name": pname, "shape": list(p.shape), "dtype": str(p.dtype).replace("torch.", "")}
            for pname, p in mod.named_parameters(recurse=False)
        ]
        q = quantization_info(mod)
        if q:
            node["quantization"] = q

    nodes, edges = _prune_dead_ends(builder.nodes, builder.edges)
    builder.keepalive.clear()
    return {"nodes": nodes, "edges": edges}


def sequential_fallback_graph(manifest: dict) -> dict:
    """Linear chain of the analyzed layers, for results saved before graph
    tracing existed or where tracing failed."""
    nodes = [{"id": "in", "kind": "input", "name": "input", "type": "Input", "category": "io",
              "output_shapes": [[None] + list(manifest.get("dataset_info", {}).get("input_shape", []))]}]
    edges = []
    prev = "in"
    for i, layer in enumerate(manifest["layers"]):
        nid = f"L{i}"
        attrs = {k: layer.get(k) for k in ("in_channels", "out_channels", "kernel_size", "groups")
                 if layer.get(k) not in (None, [], ())}
        nodes.append({"id": nid, "kind": "module", "name": layer["id"], "module": layer["id"],
                      "type": layer["type"], "category": layer["kind"], "analyzed": True,
                      "attributes": attrs,
                      "params": [{"name": "weight", "shape": layer.get("weight_shape"), "dtype": "float32"}]})
        edges.append({"source": prev, "target": nid})
        prev = nid
    nodes.append({"id": "out", "kind": "output", "name": "output", "type": "Output", "category": "io"})
    edges.append({"source": prev, "target": "out"})
    return {"nodes": nodes, "edges": edges, "fallback": True}
