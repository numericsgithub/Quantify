"""
Make exported Quantify ONNX models self-contained and executable with
vanilla ONNX Runtime -- no custom op registration required.

The custom `Quantify::*` nodes this project emits (`FixedPointQuant`,
`CoefficientQuant`, `Dense`, and the activations `Relu`, `Relu6`,
`Sigmoid`, `Tanh`, `SiLU`, `GELU`, `LeakyRelu`, `Softmax`, `HardSwish`,
`HardSigmoid`) exist purely to
preserve exact quantization semantics in the exported graph for inspection
(see pitfall #8 in docs/llm/pitfalls/brevitas_pitfalls.md -- "custom nodes
are for graph inspection, not ORT inference"). This module closes that gap
using a standard, spec-compliant ONNX mechanism: a **local Function**
(`onnx.FunctionProto`) embedded directly in the model, implementing each
custom op's math using only standard ONNX operators (Clip, Round, Floor,
Sigmoid, Erf, ArgMin, Gather, ...). Any ONNX-Runtime-conformant executor
transparently expands and runs a node whose `(domain, op_type)` matches an
embedded function -- this is part of the ONNX spec itself, not a Quantify
extension, so it works in plain `onnxruntime.InferenceSession` with zero
custom kernels.

Each node's own attributes parameterize its function body via ONNX's
attribute-reference mechanism (`AttributeProto.ref_attr_name`): a `Constant`
node inside the function body declares `ref_attr_name="scale"`, and at each
call site that constant takes the VALUE of that specific node's own `scale`
attribute. This is why several symbolic() methods
(quantizers/fixedpoint_per_tensor.py, quantizers/activations.py,
quantizers/coefficient_per_tensor_weights.py)
emit a few redundant, precomputed attributes (`integer_min_f`,
`integer_max_f`, `rounding_mode_code_i`, `approximate_code_i`, `scale_f` on
CoefficientQuant) alongside their existing human-readable ones -- the
function bodies below consume those, computed once in Python at export time
rather than reconstructed via in-graph integer arithmetic or (impossible,
since ONNX has no string-equality op) string comparison.

IMPORTANT: PyTorch's ONNX exporter strips the type-suffix from a `g.op()`
kwarg when it writes the node's actual attribute (e.g. `scale_f=...` in
Python becomes an attribute literally named `scale` on the exported node --
see pitfall #22 in docs/llm/pitfalls/brevitas_pitfalls.md). Every
`ref_attr_name` below is therefore the STRIPPED name (`"scale"`, not
`"scale_f"`), and two of the new attributes above are named `..._code_i`
rather than the more obvious `rounding_mode_i`/`approximate_i` specifically
to avoid colliding, post-stripping, with the pre-existing
`rounding_mode_s`/`approximate_s` string attributes (both of which strip to
the same `rounding_mode`/`approximate` name).

`embed_self_contained_functions(onnx_model)` scans the graph for which
`Quantify::*` op_types actually appear and attaches only the matching
FunctionProtos (idempotent -- safe to call more than once, and a no-op for
an op_type the model doesn't use). `utils/onnx_export.py::export_onnx_with_io`
calls this automatically (`self_contained=True` by default).
"""

from __future__ import annotations

from typing import Dict, List, Set

import onnx
from onnx import TensorProto, helper

DOMAIN = "Quantify"
_FUNCTION_OPSET = 17  # internal to each FunctionProto; independent of the calling model's opset


def _ref_attr(output_name: str, ref_attr_name: str, kind: str, node_name: str = None):
    """A `Constant` node whose value is substituted, per call site, from the
    calling node's own `ref_attr_name` attribute (ONNX's function-attribute
    reference mechanism). `kind` is one of "float"/"int"/"string" -- the
    placeholder value below is never actually used (it's overwritten by the
    reference at call time), only its TYPE matters for the attribute proto.
    """
    placeholder = {"float": "value_float", "int": "value_int", "string": "value_string"}[kind]
    default = {"float": 0.0, "int": 0, "string": ""}[kind]
    node = helper.make_node("Constant", [], [output_name], name=node_name or f"{output_name}_const")
    attr = helper.make_attribute(placeholder, default)
    attr.ref_attr_name = ref_attr_name
    del node.attribute[:]
    node.attribute.append(attr)
    return node


def _ref_float(name: str, ref_attr_name: str) -> List[onnx.NodeProto]:
    """Like `_ref_attr(..., "float")`, but the value is cast to the dtype of the
    function input `x`, so the body runs unchanged in float32 AND float64
    (the exporter emits float64 graphs; a bare float32 Constant would be a type
    mismatch against a double `x`)."""
    return [
        _ref_attr(f"{name}_f32", ref_attr_name, "float"),
        helper.make_node("CastLike", [f"{name}_f32", "x"], [name]),
    ]


def _ref_tensor(name: str, ref_attr_name: str) -> List[onnx.NodeProto]:
    """A `Constant` node whose TENSOR value is substituted, per call site, from
    the calling node's own `ref_attr_name` attribute, then cast to the dtype of
    the function input `x` (same reason as `_ref_float`). Used to keep data
    that belongs to a node (e.g. a coefficient set) as an attribute of it
    instead of a separate graph input."""
    node = helper.make_node("Constant", [], [f"{name}_raw"], name=f"{name}_raw_const")
    attr = helper.make_attribute("value", helper.make_tensor("placeholder", TensorProto.FLOAT, [1], [0.0]))
    attr.ref_attr_name = ref_attr_name
    del node.attribute[:]
    node.attribute.append(attr)
    return [node, helper.make_node("CastLike", [f"{name}_raw", "x"], [name])]


def _fconst(name: str, value: float) -> List[onnx.NodeProto]:
    """A float constant held in double precision and cast to `x`'s dtype, so a
    float64 graph gets the full-precision value (not a float32-rounded one)."""
    return [
        helper.make_node("Constant", [], [f"{name}_f64"],
                         value=helper.make_tensor(f"{name}_t", TensorProto.DOUBLE, [], [value])),
        helper.make_node("CastLike", [f"{name}_f64", "x"], [name]),
    ]


def _opset():
    return [helper.make_opsetid("", _FUNCTION_OPSET)]


# ---------------------------------------------------------------------------
# Quantify::FixedPointQuant
# ---------------------------------------------------------------------------

def _build_fixed_point_quant_function(op_type: str = "FixedPointQuant") -> onnx.FunctionProto:
    """y = Clip(round_or_floor(x / scale), integer_min, integer_max) * scale

    Rounding mode (`rounding_mode_i`: 0=floor, 1=round-half-up,
    2=round-half-to-even) is chosen via `Where` rather than branching, since
    one function body serves every node of this op_type in the model
    (which may calibrate different quantizers with different modes) and
    ONNX functions can't dispatch on a graph structure per call. All three
    candidate roundings are cheap elementwise ops, computed unconditionally
    then selected. `Round` (ONNX) is round-half-to-even, matching
    RoundingMode.ROUND_TO_NEAREST_EVEN / torch.round exactly.
    """
    nodes = [
        # NOTE: ref_attr_name values below are the attribute names as they
        # actually appear on the exported node -- PyTorch's ONNX exporter
        # strips the type-suffix from g.op() kwargs (e.g. "scale_f" ->
        # "scale") when writing the node, so these must match the STRIPPED
        # form, not the g.op() kwarg spelling.
        *_ref_float("scale", "scale"),
        *_ref_float("imin", "integer_min"),
        *_ref_float("imax", "integer_max"),
        _ref_attr("rm", "rounding_mode_code", "int"),
        *_fconst("half", 0.5),
        helper.make_node("Constant", [], ["zero_i"], value_int=0),
        helper.make_node("Constant", [], ["one_i"], value_int=1),

        helper.make_node("Div", ["x", "scale"], ["ratio"]),

        helper.make_node("Floor", ["ratio"], ["floor_val"]),
        helper.make_node("Add", ["ratio", "half"], ["ratio_plus_half"]),
        helper.make_node("Floor", ["ratio_plus_half"], ["round_val"]),
        helper.make_node("Round", ["ratio"], ["roundeven_val"]),

        helper.make_node("Equal", ["rm", "zero_i"], ["is_floor"]),
        helper.make_node("Equal", ["rm", "one_i"], ["is_round"]),
        helper.make_node("Where", ["is_round", "round_val", "roundeven_val"], ["tmp"]),
        helper.make_node("Where", ["is_floor", "floor_val", "tmp"], ["rounded"]),

        helper.make_node("Clip", ["rounded", "imin", "imax"], ["clipped"]),
        helper.make_node("Mul", ["clipped", "scale"], ["y"]),
    ]
    return helper.make_function(
        domain=DOMAIN,
        fname=op_type,
        inputs=["x"],
        outputs=["y"],
        nodes=nodes,
        opset_imports=_opset(),
        attributes=["scale", "integer_min", "integer_max", "rounding_mode_code"],
    )


# ---------------------------------------------------------------------------
# Quantify::Relu6
# ---------------------------------------------------------------------------

def _build_relu6_function() -> onnx.FunctionProto:
    nodes = [
        *_ref_float("lo", "min"),
        *_ref_float("hi", "max"),
        helper.make_node("Clip", ["x", "lo", "hi"], ["y"]),
    ]
    return helper.make_function(
        domain=DOMAIN, fname="Relu6", inputs=["x"], outputs=["y"],
        nodes=nodes, opset_imports=_opset(), attributes=["min", "max"],
    )


# ---------------------------------------------------------------------------
# Quantify::SiLU  (plain activation, no fused quantize)
# ---------------------------------------------------------------------------

def _build_silu_function() -> onnx.FunctionProto:
    nodes = [
        helper.make_node("Sigmoid", ["x"], ["sig"]),
        helper.make_node("Mul", ["x", "sig"], ["y"]),
    ]
    return helper.make_function(
        domain=DOMAIN, fname="SiLU", inputs=["x"], outputs=["y"],
        nodes=nodes, opset_imports=_opset(), attributes=[],
    )


# ---------------------------------------------------------------------------
# Quantify::GELU
# ---------------------------------------------------------------------------

def _build_gelu_function() -> onnx.FunctionProto:
    """approximate_i: 0 -> exact (erf-based), 1 -> tanh approximation.
    Both candidates are computed unconditionally and selected via `Where`,
    same reasoning as the rounding-mode dispatch above."""
    sqrt_half = 0.7071067811865476   # 1/sqrt(2)
    c0 = 0.7978845608028654          # sqrt(2/pi)
    c1 = 0.044715
    nodes = [
        _ref_attr("approx", "approximate_code", "int"),
        helper.make_node("Constant", [], ["zero_i"], value_int=0),
        *_fconst("half", 0.5),
        *_fconst("one", 1.0),
        *_fconst("sqrt_half", sqrt_half),
        *_fconst("c0", c0),
        *_fconst("c1", c1),
        *_fconst("three", 3.0),

        # exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        helper.make_node("Mul", ["x", "sqrt_half"], ["_exact_arg"]),
        helper.make_node("Erf", ["_exact_arg"], ["_exact_erf"]),
        helper.make_node("Add", ["_exact_erf", "one"], ["_exact_1p"]),
        helper.make_node("Mul", ["x", "_exact_1p"], ["_exact_xp"]),
        helper.make_node("Mul", ["_exact_xp", "half"], ["exact_val"]),

        # tanh approx: 0.5 * x * (1 + tanh(c0 * (x + c1 * x^3)))
        helper.make_node("Pow", ["x", "three"], ["_x3"]),
        helper.make_node("Mul", ["_x3", "c1"], ["_c1x3"]),
        helper.make_node("Add", ["x", "_c1x3"], ["_inner"]),
        helper.make_node("Mul", ["_inner", "c0"], ["_tanh_arg"]),
        helper.make_node("Tanh", ["_tanh_arg"], ["_tanh_val"]),
        helper.make_node("Add", ["_tanh_val", "one"], ["_tanh_1p"]),
        helper.make_node("Mul", ["x", "_tanh_1p"], ["_tanh_xp"]),
        helper.make_node("Mul", ["_tanh_xp", "half"], ["tanh_val"]),

        helper.make_node("Equal", ["approx", "zero_i"], ["is_exact"]),
        helper.make_node("Where", ["is_exact", "exact_val", "tanh_val"], ["y"]),
    ]
    return helper.make_function(
        domain=DOMAIN, fname="GELU", inputs=["x"], outputs=["y"],
        nodes=nodes, opset_imports=_opset(), attributes=["approximate_code"],
    )


# ---------------------------------------------------------------------------
# Quantify::CoefficientQuant
# ---------------------------------------------------------------------------

def _build_coefficient_quant_function() -> onnx.FunctionProto:
    """Nearest-neighbor quantization to a nonuniform codebook: for each
    element of `x`, pick whichever of `coefficients * scale` it's closest
    to. Both `scale` and the `coefficients` set are attributes of the calling
    node (the set is part of the quantizer, not a graph input). `Gather` on a 1-D codebook with an arbitrary-shape index tensor is
    exactly numpy/torch "fancy indexing" (`codebook[indices]`), giving an
    output the same shape as `indices` (== the shape of `x`, since ArgMin
    with keepdims=0 removes the codebook axis introduced by Unsqueeze).
    """
    nodes = [
        *_ref_float("scale", "scale"),
        *_ref_tensor("coefficients", "coefficients"),
        helper.make_node("Constant", [], ["neg1"], value_ints=[-1]),

        helper.make_node("Mul", ["coefficients", "scale"], ["scaled_coeffs"]),
        helper.make_node("Unsqueeze", ["x", "neg1"], ["x_unsq"]),
        helper.make_node("Sub", ["x_unsq", "scaled_coeffs"], ["diffs"]),
        helper.make_node("Abs", ["diffs"], ["abs_diffs"]),
        helper.make_node("ArgMin", ["abs_diffs"], ["indices"], axis=-1, keepdims=0),
        helper.make_node("Gather", ["scaled_coeffs", "indices"], ["y"], axis=0),
    ]
    return helper.make_function(
        domain=DOMAIN, fname="CoefficientQuant", inputs=["x"], outputs=["y"],
        nodes=nodes, opset_imports=_opset(), attributes=["scale", "coefficients"],
    )


# ---------------------------------------------------------------------------
# Quantify::Relu / Sigmoid / Tanh / HardSwish / LeakyRelu / Softmax /
# HardSigmoid -- each a single standard op; the custom domain only exists so
# a Quantify activation stays distinguishable from a plain one in the graph.
# ---------------------------------------------------------------------------

def _ref_node(op_type: str, inputs: List[str], outputs: List[str], refs: Dict[str, tuple]):
    """A standard-op node whose attributes are all taken, per call site, from
    the calling node's attributes: `refs` maps attribute name ->
    (ref_attr_name, placeholder value whose TYPE matters)."""
    node = helper.make_node(op_type, inputs, outputs)
    for name, (ref_name, placeholder) in refs.items():
        attr = helper.make_attribute(name, placeholder)
        attr.ref_attr_name = ref_name
        node.attribute.append(attr)
    return node


def _build_unary_function(op_type: str, refs: Dict[str, tuple] = None) -> onnx.FunctionProto:
    refs = refs or {}
    return helper.make_function(
        domain=DOMAIN, fname=op_type, inputs=["x"], outputs=["y"],
        nodes=[_ref_node(op_type, ["x"], ["y"], refs)],
        opset_imports=_opset(), attributes=[r[0] for r in refs.values()],
    )


# ---------------------------------------------------------------------------
# Quantify::Dense
# ---------------------------------------------------------------------------

def _build_dense_function() -> onnx.FunctionProto:
    """y = x @ W^T + b for x of any rank >= 1, W in PyTorch's `[out, in]`
    layout, b of shape `[out]` (always present -- zeros for a bias-less
    Linear). The Transpose is an implementation detail of this function body
    only; the graph itself never carries a runtime Transpose on the weight."""
    nodes = [
        helper.make_node("Transpose", ["W"], ["Wt"], perm=[1, 0]),
        helper.make_node("MatMul", ["x", "Wt"], ["xw"]),
        helper.make_node("Add", ["xw", "b"], ["y"]),
    ]
    return helper.make_function(
        domain=DOMAIN, fname="Dense", inputs=["x", "W", "b"], outputs=["y"],
        nodes=nodes, opset_imports=_opset(), attributes=[],
    )


def _build_hard_sigmoid_function() -> onnx.FunctionProto:
    """PyTorch's hardsigmoid, `Clip(x + 3, 0, 6) / 6`, evaluated exactly like
    eager. The node's `alpha`/`beta` attributes (1/6, 0.5 -- kept for graph
    readability) are deliberately NOT used: ONNX float attributes are float32,
    and `float32(1/6) * x` in a float64 graph rounds differently from `x / 6`,
    which can flip a downstream quantizer's rounding at a tie."""
    nodes = [
        *_fconst("three", 3.0),
        *_fconst("zero", 0.0),
        *_fconst("six", 6.0),
        helper.make_node("Add", ["x", "three"], ["x3"]),
        helper.make_node("Clip", ["x3", "zero", "six"], ["c"]),
        helper.make_node("Div", ["c", "six"], ["y"]),
    ]
    return helper.make_function(
        domain=DOMAIN, fname="HardSigmoid", inputs=["x"], outputs=["y"],
        nodes=nodes, opset_imports=_opset(), attributes=[],
        attribute_protos=[helper.make_attribute("alpha", 0.0), helper.make_attribute("beta", 0.0)],
    )


_BUILDERS = {
    "FixedPointQuant": lambda: _build_fixed_point_quant_function(),
    "Relu6": _build_relu6_function,
    "SiLU": _build_silu_function,
    "GELU": _build_gelu_function,
    "CoefficientQuant": _build_coefficient_quant_function,
    "Relu": lambda: _build_unary_function("Relu"),
    "Sigmoid": lambda: _build_unary_function("Sigmoid"),
    "Tanh": lambda: _build_unary_function("Tanh"),
    "HardSwish": lambda: _build_unary_function("HardSwish"),
    "LeakyRelu": lambda: _build_unary_function("LeakyRelu", {"alpha": ("alpha", 0.0)}),
    "Softmax": lambda: _build_unary_function("Softmax", {"axis": ("axis", 0)}),
    "HardSigmoid": _build_hard_sigmoid_function,
    "Dense": _build_dense_function,
}


def _used_quantify_op_types(graph: onnx.GraphProto) -> Set[str]:
    used = set()
    for node in graph.node:
        if node.domain == DOMAIN and node.op_type in _BUILDERS:
            used.add(node.op_type)
        # Subgraphs (If/Loop/Scan bodies) -- none of our exports currently
        # produce these, but walk them for robustness if that ever changes.
        for attr in node.attribute:
            if attr.HasField("g"):
                used |= _used_quantify_op_types(attr.g)
            for g in attr.graphs:
                used |= _used_quantify_op_types(g)
    return used


def embed_self_contained_functions(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Attach a `FunctionProto` for every `Quantify::*` op_type actually
    present in `onnx_model`'s graph, so it can be loaded and run with plain
    `onnxruntime.InferenceSession` -- no custom op registration. Idempotent:
    safe to call more than once (skips op_types already embedded), and a
    no-op for any op_type this module doesn't know how to implement (the
    model still exports/validates fine either way; it just won't be
    self-contained for that node until this module gains support for it).
    """
    already = {f.name for f in onnx_model.functions if f.domain == DOMAIN}
    needed = _used_quantify_op_types(onnx_model.graph) - already

    for op_type in sorted(needed):
        onnx_model.functions.append(_BUILDERS[op_type]())

    # Functions resolve via the model's own opset_import for their domain --
    # make sure "Quantify" is declared even if the caller's export_kwargs
    # didn't already set custom_opsets (export_onnx_with_io's default does).
    if needed and not any(o.domain == DOMAIN for o in onnx_model.opset_import):
        onnx_model.opset_import.append(helper.make_opsetid(DOMAIN, 1))

    return onnx_model
