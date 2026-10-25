"""
Canonical-graph rules for `utils/onnx_export.py::export_onnx_with_io`.

Downstream tooling consumes Quantify's ONNX exports, so beyond "runs in
onnxruntime" every export must be *canonical*:

1. **Dense** -- every `nn.Linear` (incl. Brevitas `QuantLinear`) is exactly
   one `Quantify::Dense(x, W, b)` node, whatever its input rank and whether
   or not it has a bias: `W` is an initializer (optionally behind its
   quantize node) in PyTorch's `[out, in]` layout -- no runtime `Transpose`
   -- and `b` is always present (a zero initializer for a bias-less Linear).
   `dense_export` routes Linears through `quantizers.layers.DenseFn` for the
   duration of the export only; nothing is left on the live model.
2. **Padding** -- every Conv / ConvTranspose / MaxPool / AveragePool is
   VALID (all pads 0) or exactly TF-style SAME (see
   `quantizers.layers.same_pads`); no `ceil_mode=1`, no `auto_pad` other
   than NOTSET/VALID/SAME_UPPER. Use the `padding="same"` layers in
   `quantizers/layers.py` to express SAME at stride > 1.
3. **Graph input** -- each graph input, after skipping only
   `Quantify::FixedPointQuant`, must feed straight into a compute layer
   (Conv / ConvTranspose / Quantify::Dense / Gemm / MatMul): no
   Unsqueeze/Transpose/Reshape/Gather/mask glue. Feed the input already in
   the shape the first layer expects.
4. **Activations** -- a plain activation with a Quantify equivalent
   (`nn.ReLU`, `nn.Softmax`, Brevitas' own `qnn.QuantReLU`, ...) is rejected,
   both from the module tree (names the module and the Quantify class to use)
   and in the exported graph (catches functional `F.relu` etc.). Exception:
   an activation whose output IS a graph output may stay plain. LayerNorm and
   activations without a Quantify equivalent only warn.

`check_onnx_graph` returns `(violations, warnings)`; `export_onnx_with_io`
raises `NonCanonicalONNXError` listing every violation at once (or warns,
per its `canonical_checks` argument).
"""

from __future__ import annotations

import contextlib
import copy
import types
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import onnx
import torch
import torch.nn as nn

QUANTIFY = "Quantify"


class NonCanonicalONNXError(RuntimeError):
    """The exported graph violates one or more canonical-graph rules."""


# ---------------------------------------------------------------------------
# 1. Dense: route every Linear through Quantify::Dense during export
# ---------------------------------------------------------------------------

def _zero_bias(weight: torch.Tensor) -> torch.Tensor:
    return torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)


@contextlib.contextmanager
def dense_export(model: nn.Module):
    """Patch every `nn.Linear` in `model` (instance attributes only) so it
    lowers to one `Quantify::Dense` node while `torch.onnx.is_in_onnx_export()`;
    eager behaviour is unchanged. Restored on exit, even on error."""
    from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer

    from quantizers.layers import DenseFn, unwrap_quant_tensor

    patched = []
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, QuantWeightBiasInputOutputLayer):
            original = module.inner_forward_impl

            def inner(self, x, quant_weight, quant_bias, _orig=original):
                if not torch.onnx.is_in_onnx_export():
                    return _orig(x, quant_weight, quant_bias)
                w = unwrap_quant_tensor(quant_weight)
                b = _zero_bias(w) if quant_bias is None else unwrap_quant_tensor(quant_bias)
                return DenseFn.apply(unwrap_quant_tensor(x), w, b)

            name = "inner_forward_impl"
            fn = inner
        else:
            original = module.forward

            def forward(self, x, _orig=original):
                if not torch.onnx.is_in_onnx_export():
                    return _orig(x)
                b = _zero_bias(self.weight) if self.bias is None else self.bias
                return DenseFn.apply(x, self.weight, b)

            name = "forward"
            fn = forward
        had_own = name in module.__dict__
        patched.append((module, name, had_own, module.__dict__.get(name)))
        module.__dict__[name] = types.MethodType(fn, module)
    try:
        yield
    finally:
        for module, name, had_own, prev in patched:
            if had_own:
                module.__dict__[name] = prev
            else:
                del module.__dict__[name]


def dense_constants_to_initializers(onnx_model: onnx.ModelProto) -> None:
    """The zero bias `dense_export` supplies for a bias-less Linear is traced
    as a `Constant` node; turn every `Constant` feeding a `Quantify::Dense`
    weight/bias into a plain initializer, so every Dense's W and b look the
    same (initializer, optionally behind its quantize node)."""
    g = onnx_model.graph
    dense_params = {i for n in g.node if (n.domain, n.op_type) == (QUANTIFY, "Dense")
                    for i in list(n.input)[1:]}
    keep = []
    for n in g.node:
        if not n.domain and n.op_type == "Constant" and n.output[0] in dense_params:
            value = next((a.t for a in n.attribute if a.name == "value"), None)
            if value is not None:
                init = copy.deepcopy(value)
                init.name = n.output[0]
                g.initializer.append(init)
                continue
        keep.append(n)
    if len(keep) != len(g.node):
        del g.node[:]
        g.node.extend(keep)


# ---------------------------------------------------------------------------
# 4a. Activations: module-tree walk (before export)
# ---------------------------------------------------------------------------

def _quantify_activation_types():
    from quantizers import activations as A
    return (A.QuantReLU, A.QuantReLU6, A.QuantSigmoid, A.QuantTanh, A.QuantSiLU,
            A.QuantGELU, A.QuantLeakyReLU, A.QuantSoftmax, A.QuantHardSwish,
            A.QuantHardSigmoid)


def _plain_activation_map():
    """(plain type, Quantify replacement) -- order matters: nn.ReLU6 is a
    subclass of nn.Hardtanh."""
    import brevitas.nn as qnn
    return [
        (nn.ReLU6, "QuantReLU6"), (nn.ReLU, "QuantReLU"), (nn.Sigmoid, "QuantSigmoid"),
        (nn.Tanh, "QuantTanh"), (nn.SiLU, "QuantSiLU"), (nn.GELU, "QuantGELU"),
        (nn.LeakyReLU, "QuantLeakyReLU"), (nn.Softmax, "QuantSoftmax"),
        (nn.Hardswish, "QuantHardSwish"), (nn.Hardsigmoid, "QuantHardSigmoid"),
        (qnn.QuantReLU, "QuantReLU"), (qnn.QuantSigmoid, "QuantSigmoid"),
        (qnn.QuantTanh, "QuantTanh"),
    ]


def _no_equivalent_types():
    import brevitas.nn as qnn
    return (nn.LayerNorm, nn.Hardtanh, nn.ELU, nn.SELU, nn.CELU, nn.Mish, nn.Softplus,
            nn.Softsign, nn.LogSoftmax, nn.LogSigmoid, nn.PReLU, nn.RReLU, nn.GLU,
            nn.Softmin, nn.Softmax2d, nn.Threshold, nn.Hardshrink, nn.Softshrink,
            nn.Tanhshrink, qnn.QuantHardTanh)


def _is_quantizer_internal(module: nn.Module) -> bool:
    from quantizers.base_quantizer import BaseQuantizer
    mod = type(module).__module__
    return isinstance(module, BaseQuantizer) or mod.startswith(("brevitas.proxy", "brevitas.core"))


def find_activation_modules(model: nn.Module):
    """Walk the module tree; return `(rejected, warn_only)`, each a list of
    `(qualified_name, module, message)`. Quantify activations and quantizer
    internals are not descended into (their own act_impl is fine)."""
    quantify_types = _quantify_activation_types()
    plain_map = _plain_activation_map()
    no_equiv = _no_equivalent_types()
    rejected, warn_only = [], []

    def visit(prefix: str, module: nn.Module):
        for name, child in module.named_children():
            qual = f"{prefix}.{name}" if prefix else name
            if isinstance(child, quantify_types) or _is_quantizer_internal(child):
                continue
            replacement = next((r for t, r in plain_map if isinstance(child, t)), None)
            if replacement is not None:
                rejected.append((qual, child,
                                 f"'{qual}' is a plain {type(child).__module__}.{type(child).__name__} -- "
                                 f"use quantizers.{replacement} instead (it exports as a quantized "
                                 f"Quantify::* node)"))
                continue
            if isinstance(child, no_equiv):
                warn_only.append((qual, child,
                                  f"'{qual}' ({type(child).__name__}) has no Quantify equivalent yet -- "
                                  f"it will export as a plain, unquantized ONNX op"))
                continue
            visit(qual, child)

    visit("", model)
    return rejected, warn_only


def _flatten_tensors(obj) -> List[torch.Tensor]:
    obj = getattr(obj, "value", obj)
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for o in obj for t in _flatten_tensors(o)]
    if isinstance(obj, dict):
        return [t for o in obj.values() for t in _flatten_tensors(o)]
    return []


@contextlib.contextmanager
def record_forward(model: nn.Module, watched: Sequence[nn.Module]):
    """Hooks for one reference forward pass: records the output tensor ids of
    each `watched` module, and how many times any `nn.Linear` was called.
    Yields a dict filled as the forward runs: `{"outputs": {module: [ids]},
    "linear_calls": int}`."""
    record = {"outputs": defaultdict(list), "linear_calls": 0}
    handles = []
    for m in watched:
        handles.append(m.register_forward_hook(
            lambda mod, inp, out: record["outputs"][mod].extend(id(t) for t in _flatten_tensors(out))))
    for m in model.modules():
        if isinstance(m, nn.Linear):
            def count(mod, inp, out):
                record["linear_calls"] += 1
            handles.append(m.register_forward_hook(count))
    try:
        yield record
    finally:
        for h in handles:
            h.remove()


def activation_module_findings(rejected, warn_only, record, model_output):
    """Drop findings whose module output is a model output (allowed to stay
    plain); return `(violations, warnings)` message lists."""
    out_ids = {id(t) for t in _flatten_tensors(model_output)}

    def at_output(m):
        ids = record["outputs"].get(m, [])
        return bool(ids) and all(i in out_ids for i in ids)

    violations = [msg for _, m, msg in rejected if not at_output(m)]
    warnings_ = [msg for _, m, msg in warn_only if not at_output(m)]
    return violations, warnings_


# ---------------------------------------------------------------------------
# Graph checks (after export)
# ---------------------------------------------------------------------------

_COMPUTE_OPS = {("", "Conv"), ("", "ConvTranspose"), (QUANTIFY, "Dense"), ("", "Gemm"), ("", "MatMul")}
_QUANT_OPS = {(QUANTIFY, "FixedPointQuant"), (QUANTIFY, "CoefficientQuant")}
_PLAIN_ACT_OPS = {"Relu", "Sigmoid", "Tanh", "LeakyRelu", "Softmax", "LogSoftmax", "HardSwish",
                  "HardSigmoid", "Clip", "Elu", "Selu", "Celu", "Gelu", "Mish", "Softplus",
                  "Softsign", "PRelu", "ThresholdedRelu"}
_PADDED_OPS = {"Conv", "ConvTranspose", "MaxPool", "AveragePool"}


def _key(node):
    return (node.domain or "", node.op_type)


def _label(node):
    op = f"{node.domain}::{node.op_type}" if node.domain else node.op_type
    return f"{op} '{node.name}'" if node.name else op


def _attrs(node) -> dict:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


class _Graph:
    def __init__(self, model: onnx.ModelProto):
        g = model.graph
        self.g = g
        self.inits = {i.name: i for i in g.initializer}
        self.producer = {o: n for n in g.node for o in n.output}
        self.consumers = defaultdict(list)
        for n in g.node:
            for i in n.input:
                if i:
                    self.consumers[i].append(n)
        self.outputs = {o.name for o in g.output}
        self.inputs = [i.name for i in g.input if i.name not in self.inits]
        self.shapes: Dict[str, Optional[List[int]]] = {}
        for vi in list(g.input) + list(g.value_info) + list(g.output):
            t = vi.type.tensor_type
            if t.HasField("shape"):
                self.shapes[vi.name] = [d.dim_value if d.HasField("dim_value") else None
                                        for d in t.shape.dim]

    def const_dims(self, name: str, through_quant: bool = True) -> Optional[List[int]]:
        """Dims of `name` if it's a constant (initializer / Constant node),
        optionally behind one quantize node; else None."""
        if name in self.inits:
            return list(self.inits[name].dims)
        p = self.producer.get(name)
        if p is None:
            return None
        if p.op_type == "Constant" and not p.domain:
            v = _attrs(p).get("value")
            return list(v.dims) if v is not None else []
        if through_quant and _key(p) in _QUANT_OPS:
            return self.const_dims(p.input[0], through_quant=False)
        return None


def _check_dense(G: _Graph, expected_calls: Optional[int]) -> List[str]:
    v = []
    dense = [n for n in G.g.node if _key(n) == (QUANTIFY, "Dense")]
    if expected_calls is not None and len(dense) != expected_calls:
        v.append(f"[dense] {expected_calls} nn.Linear call(s) in the forward pass but "
                 f"{len(dense)} Quantify::Dense node(s) in the graph -- a Linear was lowered "
                 f"some other way (e.g. inside nn.MultiheadAttention, or a functional F.linear)")
    for n in dense:
        if len(n.input) != 3 or not all(n.input):
            v.append(f"[dense] {_label(n)} must have exactly 3 inputs (x, W, b), got {list(n.input)}")
            continue
        w, b = G.const_dims(n.input[1]), G.const_dims(n.input[2])
        if w is None or len(w) != 2:
            v.append(f"[dense] {_label(n)}: weight must be a constant [out, in] tensor "
                     f"(optionally behind its quantize node), got {w}")
        if b is None or len(b) != 1 or (w is not None and len(w) == 2 and b[0] != w[0]):
            v.append(f"[dense] {_label(n)}: bias must be a constant [out] tensor, got {b}")
    for n in G.g.node:
        if _key(n) == ("", "Gemm"):
            v.append(f"[dense] {_label(n)}: Gemm in the graph -- every Linear must export as "
                     f"Quantify::Dense")
        elif _key(n) == ("", "MatMul") and any(G.const_dims(i) is not None for i in n.input):
            v.append(f"[dense] {_label(n)}: MatMul with a constant weight -- a Linear-like op "
                     f"not exported as Quantify::Dense")
        elif _key(n) == ("", "Transpose") and G.const_dims(n.input[0]) is not None:
            v.append(f"[dense] {_label(n)}: runtime Transpose on a constant (weight) tensor")
    return v


def _expected_same(op: str, in_sizes, kernel, strides, dilations, output_padding):
    pads = []
    for ax, (i, k, s, d) in enumerate(zip(in_sizes, kernel, strides, dilations)):
        if op == "ConvTranspose":
            out = i * s
            op_ax = output_padding[ax] if output_padding else 0
            total = s * (i - 1) + op_ax + (k - 1) * d + 1 - out
            if total < 0:
                return None
        else:
            out = -(-i // s)
            total = max((out - 1) * s + (k - 1) * d + 1 - i, 0)
        pads.append((total // 2, total - total // 2))
    return pads


def _check_padding(G: _Graph) -> List[str]:
    v = []
    for n in G.g.node:
        if n.domain or n.op_type not in _PADDED_OPS:
            continue
        a = _attrs(n)
        auto_pad = a.get("auto_pad", b"NOTSET")
        auto_pad = auto_pad.decode() if isinstance(auto_pad, bytes) else auto_pad
        if a.get("ceil_mode", 0):
            v.append(f"[padding] {_label(n)}: ceil_mode=1 is not allowed (use padding='same' "
                     f"layers from quantizers/layers.py instead)")
        if auto_pad not in ("NOTSET", "VALID", "SAME_UPPER"):
            v.append(f"[padding] {_label(n)}: auto_pad={auto_pad} is not allowed "
                     f"(only NOTSET / VALID / SAME_UPPER)")
            continue
        if auto_pad != "NOTSET":
            continue
        shape = G.shapes.get(n.input[0])
        if "kernel_shape" in a:
            kernel = list(a["kernel_shape"])
        else:
            wd = G.const_dims(n.input[1]) if len(n.input) > 1 else None
            kernel = wd[2:] if wd else None
        nsp = len(kernel) if kernel else (len(shape) - 2 if shape else 0)
        pads = list(a.get("pads", [0] * (2 * nsp)))
        if all(p == 0 for p in pads):
            continue  # VALID
        if shape is None or kernel is None or any(s is None for s in shape[2:]):
            v.append(f"[padding] {_label(n)}: non-zero pads {pads} but the input shape is unknown, "
                     f"so they can't be verified as TF-SAME")
            continue
        in_sizes = shape[2:]
        strides = list(a.get("strides", [1] * nsp))
        dilations = list(a.get("dilations", [1] * nsp))
        expected = _expected_same(n.op_type, in_sizes, kernel, strides, dilations,
                                  list(a.get("output_padding", [])))
        got = list(zip(pads[:nsp], pads[nsp:]))
        if expected != got:
            exp = _flat(expected) if expected is not None else "not representable"
            v.append(f"[padding] {_label(n)}: input shape {shape}, kernel {kernel}, stride {strides}, "
                     f"dilation {dilations} has pads {pads}, which is neither VALID (all 0) nor "
                     f"TF-SAME ({exp}). Use padding='same' (quantizers/layers.py) or padding=0.")
    return v


def _flat(pairs):
    return [b for b, _ in pairs] + [e for _, e in pairs]


def _check_graph_inputs(G: _Graph, skip: frozenset = frozenset()) -> List[str]:
    v = []
    for inp in G.inputs:
        if inp in skip:  # recurrent memory input, not external data
            continue
        frontier = [(inp, [f"input '{inp}'"])]
        while frontier:
            name, chain = frontier.pop()
            for n in G.consumers.get(name, []):
                if _key(n) == (QUANTIFY, "FixedPointQuant"):
                    frontier.append((n.output[0], chain + [_label(n)]))
                elif _key(n) not in _COMPUTE_OPS:
                    tail = _describe_tail(G, n)
                    v.append(f"[input] {' -> '.join(chain + tail)}: the graph input must feed "
                             f"straight into the first compute layer (Conv / Dense / MatMul), only a "
                             f"FixedPointQuant may sit in between. Feed the input already in the shape "
                             f"the first layer expects instead of reshaping it inside the model.")
    return v


def _describe_tail(G: _Graph, node, limit: int = 4) -> List[str]:
    """`node` plus the first-consumer chain after it, up to a compute layer."""
    out = []
    while node is not None and len(out) < limit:
        out.append(_label(node))
        if _key(node) in _COMPUTE_OPS:
            break
        nxt = [c for o in node.output for c in G.consumers.get(o, [])]
        node = nxt[0] if nxt else None
    if node is not None and _key(node) not in _COMPUTE_OPS:
        out.append("...")
    return out


def _check_activations(G: _Graph) -> Tuple[List[str], List[str]]:
    v, w = [], []
    for n in G.g.node:
        if n.domain:
            continue
        at_output = all(o in G.outputs for o in n.output)
        if n.op_type in _PLAIN_ACT_OPS and not at_output:
            v.append(f"[activation] {_label(n)}: plain (unquantized) activation inside the graph -- "
                     f"use the Quantify activation module (quantizers/activations.py), which exports "
                     f"as a Quantify::* node. Only an activation producing a graph output may stay plain.")
        elif n.op_type == "LayerNormalization":
            w.append(f"[activation] {_label(n)}: LayerNormalization has no Quantify equivalent yet "
                     f"(exported as a plain float op)")
    return v, w


def _graph_for_checks(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """A copy with the Quantify FunctionProtos embedded (so shape inference
    can see through custom nodes) and shapes inferred."""
    from utils.onnx_self_contained import embed_self_contained_functions
    m = copy.deepcopy(onnx_model)
    embed_self_contained_functions(m)
    try:
        return onnx.shape_inference.infer_shapes(m)
    except Exception:
        return m


def check_onnx_graph(onnx_model: onnx.ModelProto,
                     expected_dense_calls: Optional[int] = None,
                     skip_inputs: frozenset = frozenset()) -> Tuple[List[str], List[str]]:
    """Run every graph-level canonical check; return `(violations, warnings)`."""
    G = _Graph(_graph_for_checks(onnx_model))
    violations = []
    violations += _check_dense(G, expected_dense_calls)
    violations += _check_padding(G)
    violations += _check_graph_inputs(G, frozenset(skip_inputs))
    act_v, act_w = _check_activations(G)
    violations += act_v
    return violations, act_w
