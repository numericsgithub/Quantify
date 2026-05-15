# Exporting Custom PyTorch Quantizers to ONNX as Custom Nodes

A complete developer guide covering all learnings, pitfalls, and best practices
for exporting `torch.autograd.Function` subclasses as custom ONNX nodes.

---

## Table of Contents

1. [Overview](#overview)
2. [The `autograd.Function` Contract](#the-autogradfunction-contract)
3. [The `symbolic` Method](#the-symbolic-method)
4. [Argument Types in `symbolic`: Values vs. Python Primitives](#argument-types-in-symbolic-values-vs-python-primitives)
5. [ONNX Inputs vs. Attributes](#onnx-inputs-vs-attributes)
6. [Attribute Type Suffixes](#attribute-type-suffixes)
7. [Tensor Attributes: Storing Complex Data](#tensor-attributes-storing-complex-data)
8. [Propagating Type Information with `.setType()`](#propagating-type-information-with-settype)
9. [Returning a 4-Tuple (Brevitas Contract)](#returning-a-4-tuple-brevitas-contract)
10. [Injecting Pre-Computed Data via Class Variables](#injecting-pre-computed-data-via-class-variables)
11. [Naming Inputs and Why It Is Dangerous](#naming-inputs-and-why-it-is-dangerous)
12. [Avoiding Traced Ops Before the Custom Node](#avoiding-traced-ops-before-the-custom-node)
13. [Signature Matching Between `forward` and `symbolic`](#signature-matching-between-forward-and-symbolic)
14. [`register_custom_op_symbolic` Does Not Work for `autograd.Function`](#register_custom_op_symbolic-does-not-work-for-autogradfunction)
15. [Complete Reference Implementation](#complete-reference-implementation)
16. [Quick Checklist](#quick-checklist)

---

## Overview

When building custom quantizers with Brevitas (or any framework that wraps
`torch.autograd.Function`), you often need the ONNX graph to contain a single
clean custom node — not a cloud of `Abs`, `ArgMin`, `Cast`, `Constant` nodes
that represent internal implementation details.

PyTorch supports this through the `symbolic` static method on your `Function`
subclass. The method speaks the TorchScript/JIT graph API and lets you emit
whatever ONNX node structure you want, regardless of what `forward` actually
does at runtime.

---

## The `autograd.Function` Contract

A class that inherits from `torch.autograd.Function` must implement:

| Method | Purpose |
|---|---|
| `forward(ctx, ...)` | Runs at training/inference time. Receives real tensors. |
| `backward(ctx, ...)` | Computes gradients (Straight-Through Estimator for quantizers). |
| `symbolic(g, ...)` | Called by the ONNX exporter. Receives graph `Value` objects, not tensors. |

All three must be `@staticmethod`.

---

## The `symbolic` Method

During `torch.onnx.export`, PyTorch intercepts every call to
`YourFunction.apply(...)` and instead calls `YourFunction.symbolic(g, ...)`,
where `g` is a `GraphContext` object. The method must build and return ONNX
graph nodes using `g.op(...)`.

### Basic structure

```python
@staticmethod
def symbolic(g, x, some_tensor_input, some_scalar):
    output = g.op(
        "MyDomain::MyOp",       # "domain::OpName"
        x,                      # positional tensor args → ONNX inputs
        some_tensor_input,      # positional tensor args → ONNX inputs
        my_scalar_i=some_scalar # keyword args with type suffix → ONNX attributes
    ).setType(x.type())         # always propagate type!
    return output
```

---

## Argument Types in `symbolic`: Values vs. Python Primitives

This is the most important concept to understand before anything else.

When `symbolic` is called by the ONNX exporter, **its arguments are not real
tensors**. They are `torch._C.Value` objects — opaque handles that represent
nodes in the TorchScript graph. This means:

- You **cannot** call `.flatten()`, `.tolist()`, `.item()`, `.shape`, or any
  tensor method on them.
- You **can** pass them directly to `g.op(...)` as positional arguments.
- You **can** call `.setDebugName()` on them (with caveats — see below).
- You **can** call `.setType(...)` on them.

Scalar arguments (e.g. `bit_shift_scale: int`) that were passed as plain Python
primitives to `apply()` **are** available as real Python values inside
`symbolic`, because the JIT treats them as compile-time constants.

### The consequence for dynamic data

If you need to include data that depends on the actual weight tensor values
(e.g. quantization indices, pre-computed quantized weights), you cannot compute
them inside `symbolic` from the `Value` arguments. You must compute them
**before** calling `apply()` and pass them in through a side channel. See
[Injecting Pre-Computed Data via Class Variables](#injecting-pre-computed-data-via-class-variables).

---

## ONNX Inputs vs. Attributes

`g.op(opname, ...)` maps arguments to ONNX concepts as follows:

| How you pass it | What it becomes in ONNX |
|---|---|
| Positional `torch._C.Value` argument | **Input** (an edge from another node) |
| Keyword argument with type suffix (e.g. `dim_i=3`) | **Attribute** (compile-time constant embedded in the node) |

### Making something an Input

Pass it as a positional argument to `g.op()`:

```python
output = g.op("MyDomain::MyOp", x, coefficients)
#                                ^  ^--- both become ONNX inputs
```

To turn pre-computed data (a tensor you own) into an input, first create a
`Constant` node for it, then pass the resulting `Value`:

```python
indices_node = g.op("Constant", value_t=my_indices_tensor)
output = g.op("MyDomain::MyOp", x, indices_node)
```

### Making something an Attribute

Pass it as a keyword argument with the correct type suffix:

```python
output = g.op("MyDomain::MyOp", x, bit_shift_scale_i=int(bit_shift_scale))
```

---

## Attribute Type Suffixes

PyTorch's `_add_attribute` recognises exactly these suffixes:

| Suffix | Python value type | ONNX attribute type |
|---|---|---|
| `_i` | `int` | INT |
| `_i` | `list[int]` | INTS |
| `_f` | `float` | FLOAT |
| `_f` | `list[float]` | FLOATS |
| `_s` | `str` | STRING |
| `_s` | `list[str]` | STRINGS |
| `_t` | `torch.Tensor` | TENSOR |
| `_g` | graph | GRAPH |

**There is no `_is`, `_fs`, or similar plural suffix.** The scalar vs. list
distinction is determined entirely by whether you pass an `int` or a
`list[int]`. Passing `_is` will raise:

```
ValueError: Invalid attribute specifier 'my_attr_is' names must be suffixed
with type, e.g. 'dim_i' or 'dims_i'
```

---

## Tensor Attributes: Storing Complex Data

If you need to store a tensor of arbitrary shape as an attribute (rather than
as an input), use the `_t` suffix:

```python
output = g.op(
    "MyDomain::MyOp",
    x,
    bit_shift_scale_i=int(bit_shift_scale),
    chosen_indices_t=my_indices_tensor,       # full tensor, any shape
    quantized_values_t=my_quantized_tensor,   # full tensor, any shape
).setType(x.type())
```

This is the **safest approach** for injecting pre-computed weight-derived data:

- The tensors appear in Netron under **ATTRIBUTES** with their full shape.
- You never touch any existing graph `Value` nodes, so nothing upstream is
  affected.
- No `Constant` nodes are created, so the graph stays clean.
- No risk of accidentally renaming shared graph values (see below).

The tensors must be on CPU and must be concrete (`torch.Tensor`), not
`torch._C.Value` objects. Compute them before the `apply()` call and stash them
in a class variable.

---

## Propagating Type Information with `.setType()`

Always call `.setType(x.type())` on the output of your `g.op(...)` call:

```python
output = g.op("MyDomain::MyOp", x, ...).setType(x.type())
```

Without this, the ONNX node's output has **unknown shape and dtype**. Any
downstream node (e.g. a convolution that consumes quantized weights) will fail
with an error like:

```
torch.onnx.errors.SymbolicValueError: Unsupported: ONNX export of convolution
for kernel of unknown shape.
```

This is one of the most common and confusing export failures because the error
points at the convolution, not at your custom node.

---

## Returning a 4-Tuple (Brevitas Contract)

Brevitas's quantization proxy expects every `tensor_quant` module to return a
4-tuple of `(quantized, scale, zero_point, bit_width)`. Both `forward` and
`symbolic` must return exactly this:

```python
# In forward:
return (
    quantized,
    torch.tensor(scale, dtype=x.dtype, device=x.device),
    torch.tensor(0.0,   dtype=x.dtype, device=x.device),
    torch.tensor(float(bit_width), dtype=x.dtype, device=x.device),
)

# In symbolic:
scale      = g.op("Constant", value_t=torch.tensor(2.0 ** bit_shift_scale))
zero_point = g.op("Constant", value_t=torch.tensor(0.0))
bw         = g.op("Constant", value_t=torch.tensor(float(bit_width)))
return quantized, scale, zero_point, bw
```

If `symbolic` returns fewer values than `forward`, or vice versa, the exporter
will fail with a `NotImplementedError: You must implement the forward function`
error, even though `forward` is clearly defined. This is because the exporter
uses return-count matching to verify it has found the right symbolic.

---

## Injecting Pre-Computed Data via Class Variables

When `symbolic` is called, its tensor arguments are `torch._C.Value` graph
nodes — you cannot call `.tolist()` or `.flatten()` on them. For data that
must be computed from the actual weight values (indices, quantized outputs),
use a class-level variable as a side channel:

```python
class MyQuantFn(Function):

    _captured_indices:   torch.Tensor = None
    _captured_quantized: torch.Tensor = None

    @staticmethod
    def symbolic(g, x, coefficients, bit_shift_scale, bit_width):
        # Read from class variable — real Python/tensor data, not a Value
        output = g.op(
            "MyDomain::MyOp",
            x,
            coefficients,
            bit_shift_scale_i=int(bit_shift_scale),
            chosen_indices_t=MyQuantFn._captured_indices,
            quantized_values_t=MyQuantFn._captured_quantized,
        ).setType(x.type())
        ...

    @staticmethod
    def forward(ctx, x, coefficients, bit_shift_scale, bit_width):
        # Normal forward — does not use the class variables
        ...
```

In `_quantize` (or wherever you call `apply`), populate the class variables
**before** the `apply()` call, inside the `is_in_onnx_export()` guard:

```python
if torch.onnx.is_in_onnx_export():
    with torch.no_grad():
        quantized_vals, _, indices = compute_quantization(x, coefficients, bit_shift_scale)
        MyQuantFn._captured_indices   = indices.flatten().cpu().to(torch.long)
        MyQuantFn._captured_quantized = quantized_vals.flatten().cpu()

    quantized, _, _, _ = MyQuantFn.apply(x, coefficients, bit_shift_scale, bit_width)
    return quantized
```

### Why `with torch.no_grad()`?

The pre-computation of indices and quantized values must not create autograd
graph nodes. Wrapping in `torch.no_grad()` ensures these ops are completely
invisible to the tracer.

### Why `.cpu()`?

ONNX tensor attributes must be on CPU. GPU tensors will cause a serialisation
error.

---

## Naming Inputs and Why It Is Dangerous

You might want to give human-readable names to your node's inputs for Netron
display. There are two approaches, one safe and one destructive.

### ❌ Dangerous: `node.inputsAt(i).setDebugName()`

```python
node = quantized.node()
node.inputsAt(0).setDebugName("weights")      # DO NOT DO THIS
node.inputsAt(1).setDebugName("coefficients") # DO NOT DO THIS
```

`inputsAt(0)` returns the **same** `torch._C.Value` object that the rest of
the graph uses for that tensor. Renaming it renames it **everywhere in the
graph** — including in the convolution that consumes the weights, which will
then fail because its kernel has an unrecognised name.

This can manifest as the same `convolution for kernel of unknown shape` error
as a missing `.setType()`, making it very hard to diagnose.

### ❌ Also Dangerous: `x.setDebugName()` before `g.op()`

```python
x.setDebugName("weights")  # renames the shared Value — DO NOT DO THIS
```

Same problem. `x` is a reference into the shared graph.

### ✓ Safe: `.setDebugName()` only on `Constant` nodes you created

```python
indices_const   = g.op("Constant", value_t=MyQuantFn._captured_indices)
quantized_const = g.op("Constant", value_t=MyQuantFn._captured_quantized)

# These are fresh nodes we own — safe to name
indices_const.setDebugName("chosen_indices")
quantized_const.setDebugName("quantized_values")
```

### ✓ Safest: Use tensor attributes instead

As described in [Tensor Attributes](#tensor-attributes-storing-complex-data),
putting pre-computed data in `_t` attributes avoids all naming concerns and
produces the cleanest graph.

---

## Avoiding Traced Ops Before the Custom Node

A common mistake is passing a tensor that was **computed inside the traced
region** as an argument to `apply()`. Even if the computation happens in
`_quantize` before the `apply()` call, if it is not inside
`torch.no_grad()` or otherwise isolated from the tracer, PyTorch will trace
every op that produced it and insert them into the graph before your custom
node.

For example, passing `indices` naively:

```python
# WRONG — tracer sees Abs, ArgMin, Cast before CoefficientQuant
_, _, indices = apply_non_uniform_quantization(x, coeffs, bit_shift)
quantized, _, _, _ = MyQuantFn.apply(x, coeffs, bit_shift, bw, indices)
```

In Netron this shows up as a chain of `Abs → ArgMin → Cast → CoefficientQuant`
instead of a single clean node.

The fix is to either:

1. Compute the data **outside** the trace via `torch.no_grad()` and stash it in
   a class variable (preferred), or
2. Create a `Constant` ONNX node inside `symbolic` from the class variable.

---

## Signature Matching Between `forward` and `symbolic`

The signatures of `forward` and `symbolic` **must match exactly** in number and
order of tensor arguments. The ONNX exporter uses argument count to verify it
has found the correct symbolic for a given `forward`.

| `forward` | `symbolic` |
|---|---|
| `forward(ctx, x, coefficients, bit_shift_scale, bit_width)` | `symbolic(g, x, coefficients, bit_shift_scale, bit_width)` |

Note that `ctx` in `forward` corresponds to `g` in `symbolic` — both are the
first argument and are not counted as tensor inputs.

If counts differ, you will see:

```
NotImplementedError: You must implement the forward function for custom autograd.Function.
```

Even though `forward` is clearly present. This error message is misleading —
the real cause is the signature mismatch.

---

## `register_custom_op_symbolic` Does Not Work for `autograd.Function`

You might be tempted to register the symbolic outside the class:

```python
torch.onnx.register_custom_op_symbolic(
    "prim::PythonOp.MyQuantFn",
    my_symbolic_fn,
    opset_version=11,
)
```

This will raise:

```
torch.onnx.OnnxExporterError: Failed to register operator
prim::PythonOp.MyQuantFn. The symbolic name must match the format
domain::name, and should start with a letter and contain only
alphanumerical characters
```

`register_custom_op_symbolic` is for custom C++ ops registered via
`torch.ops`, not for Python `autograd.Function` subclasses. For
`autograd.Function`, the symbolic **must** be a `@staticmethod` named
`symbolic` directly on the class. There is no alternative.

---

## Complete Reference Implementation

```python
import torch
import torch.nn as nn
from torch.autograd import Function


def apply_quantization(x: torch.Tensor, coefficients: torch.Tensor, bit_shift: int):
    """Your actual quantization logic."""
    scale = 2.0 ** bit_shift
    scaled = coefficients * scale
    diffs = torch.abs(x.unsqueeze(-1) - scaled)
    indices = torch.argmin(diffs, dim=-1)
    quantized = scaled[indices]
    return quantized, scale, indices


class MyQuantFn(Function):
    """
    Custom autograd.Function that exports as a single clean ONNX node.
    Pre-computed weight-derived data is injected via class variables
    to avoid tracing internal ops into the graph.
    """

    # Side-channel storage for data that must come from real tensors,
    # not from the graph Value objects that symbolic() receives.
    _captured_indices:   torch.Tensor = None
    _captured_quantized: torch.Tensor = None

    @staticmethod
    def symbolic(g, x, coefficients, bit_shift_scale, bit_width):
        # bit_shift_scale and bit_width are plain Python ints here —
        # they were passed as scalars to apply() so the JIT treats them
        # as compile-time constants.

        output = g.op(
            "MyDomain::MyOp",
            x,                       # ONNX input 0
            coefficients,            # ONNX input 1
            bit_shift_scale_i=int(bit_shift_scale),               # attribute
            chosen_indices_t=MyQuantFn._captured_indices,         # tensor attribute
            quantized_values_t=MyQuantFn._captured_quantized,     # tensor attribute
        ).setType(x.type())          # ← critical: propagates shape/dtype downstream

        # Brevitas requires a 4-tuple return
        scale      = g.op("Constant", value_t=torch.tensor(2.0 ** bit_shift_scale))
        zero_point = g.op("Constant", value_t=torch.tensor(0.0))
        bw         = g.op("Constant", value_t=torch.tensor(float(bit_width)))
        return output, scale, zero_point, bw

    @staticmethod
    def forward(ctx, x, coefficients, bit_shift_scale, bit_width):
        # Must have same tensor-arg signature as symbolic (excluding ctx/g)
        ctx.save_for_backward(x)
        quantized, scale, _ = apply_quantization(x, coefficients, bit_shift_scale)
        bw = torch.tensor(float(bit_width), dtype=x.dtype, device=x.device)
        return (
            quantized,
            torch.tensor(scale,  dtype=x.dtype, device=x.device),
            torch.tensor(0.0,    dtype=x.dtype, device=x.device),
            bw,
        )

    @staticmethod
    def backward(ctx, grad_out, grad_scale, grad_zp, grad_bw):
        # Straight-Through Estimator — one None per forward tensor input
        return grad_out, None, None, None


class MyQuantizer(nn.Module):

    def __init__(self, coefficients: torch.Tensor, bit_shift: int):
        super().__init__()
        self.register_buffer("coefficients", coefficients)
        self.bit_shift = bit_shift
        self.bit_width = len(coefficients)

    def forward(self, x: torch.Tensor):
        if torch.onnx.is_in_onnx_export():
            # Compute weight-derived data NOW, before apply() is traced.
            # torch.no_grad() ensures these ops are invisible to the tracer.
            with torch.no_grad():
                q, _, idx = apply_quantization(x, self.coefficients, self.bit_shift)
                MyQuantFn._captured_indices   = idx.flatten().cpu().to(torch.long)
                MyQuantFn._captured_quantized = q.flatten().cpu()

        quantized, scale, zero_point, bw = MyQuantFn.apply(
            x,
            self.coefficients,
            self.bit_shift,
            self.bit_width,
        )
        return quantized, scale, zero_point, bw
```

---

## Quick Checklist

Use this before every ONNX export attempt:

- [ ] `symbolic` and `forward` have **identical tensor-argument signatures**
      (same count, same order; `ctx` ↔ `g` don't count).
- [ ] `g.op(...).setType(x.type())` is called on every custom node output.
- [ ] `symbolic` returns the same number of values as `forward`.
- [ ] Any data derived from real tensors is pre-computed under `torch.no_grad()`
      and stored in a class variable — never computed inside `symbolic` from
      `Value` arguments.
- [ ] Pre-computed tensors stashed in class variables are on **CPU** before
      being used as `_t` attributes.
- [ ] You are **not** using `register_custom_op_symbolic` for an
      `autograd.Function`.
- [ ] You are **not** calling `.setDebugName()` on `inputsAt(i)` or on `x`/
      `coefficients` (shared graph values).
- [ ] Scalar arguments passed to `apply()` as plain `int`/`float` are used
      directly as Python values inside `symbolic` — no `.item()` needed.
- [ ] `backward` returns exactly one value per `forward` tensor input
      (use `None` for inputs that don't need gradients).
