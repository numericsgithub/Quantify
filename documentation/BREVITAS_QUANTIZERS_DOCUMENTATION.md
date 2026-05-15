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
7. [Extracting Constants from Graph Values with `_maybe_get_const`](#extracting-constants-from-graph-values-with-_maybe_get_const)
8. [Tensor Attributes: Storing Complex Data](#tensor-attributes-storing-complex-data)
9. [Enum Attributes](#enum-attributes)
10. [Propagating Type Information with `.setType()`](#propagating-type-information-with-settype)
11. [Returning a 4-Tuple (Brevitas Contract)](#returning-a-4-tuple-brevitas-contract)
12. [Side Channels for Pre-Computed Data](#side-channels-for-pre-computed-data)
    - [Pattern A: Single Class Variable](#pattern-a-single-class-variable)
    - [Pattern B: FIFO Deque (Recommended for Shared Function Classes)](#pattern-b-fifo-deque-recommended-for-shared-function-classes)
13. [Execution Order During ONNX Export](#execution-order-during-onnx-export)
14. [Where to Place the `is_in_onnx_export()` Guard](#where-to-place-the-is_in_onnx_export-guard)
15. [Naming Inputs and Why It Is Dangerous](#naming-inputs-and-why-it-is-dangerous)
16. [Avoiding Traced Ops Before the Custom Node](#avoiding-traced-ops-before-the-custom-node)
17. [Signature Matching Between `forward` and `symbolic`](#signature-matching-between-forward-and-symbolic)
18. [`register_custom_op_symbolic` Does Not Work for `autograd.Function`](#register_custom_op_symbolic-does-not-work-for-autogradfunction)
19. [Complete Reference Implementation](#complete-reference-implementation)
20. [Quick Checklist](#quick-checklist)

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
- You **can** unwrap them to real Python/tensor values *if and only if* they
  were produced by a `Constant` node in the traced graph — see
  [`_maybe_get_const`](#extracting-constants-from-graph-values-with-_maybe_get_const).

Scalar arguments (e.g. `bit_shift_scale: int`) that were passed as plain Python
primitives to `apply()` **are** available as real Python values inside
`symbolic`, because the JIT treats them as compile-time constants.

### The consequence for dynamic data

If you need to include data that depends on the actual weight tensor values
(e.g. quantization indices, pre-computed quantized weights), you cannot compute
them inside `symbolic` from the `Value` arguments. You must compute them
**before** the symbolic conversion pass runs and pass them in through a side
channel. See [Side Channels for Pre-Computed Data](#side-channels-for-pre-computed-data).

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

### Undocumented but useful: 0-dim tensors coerce to scalars

The official PyTorch docs state that `_f` "accepts either a single float, or a
list of floats." In practice, **a 0-dim `torch.Tensor` also works** because the
underlying C++ binding (`node.f_(name, value)`) goes through Python's argument
coercion, which calls `__float__()` on the tensor automatically. The same
applies to `_i` with `__int__()`.

This means the following pattern is valid:

```python
# scale_val is a 0-dim torch.Tensor (e.g. from _maybe_get_const)
quantized = g.op("MyDomain::MyOp", x, scale_f=scale_val)
```

This is undocumented but stable behaviour and is what `_maybe_get_const(v, "t")`
returns for scalar Constant nodes. **Pitfall:** It only works if the value is
actually a 0-dim concrete tensor, not a `torch._C.Value`. If `_maybe_get_const`
returns the original `Value` (because the upstream node was not a Constant),
passing it to `_f` will fail. Make sure your scale/zero_point inputs were
created from `torch.tensor(...)` literals so they trace as Constants.

---

## Extracting Constants from Graph Values with `_maybe_get_const`

`torch.onnx.symbolic_helper._maybe_get_const(value, desc)` is the canonical
helper for unwrapping a `torch._C.Value` back into its underlying Python or
tensor value — but only when that value originated from an `onnx::Constant`
node in the trace.

```python
from torch.onnx import symbolic_helper

scale_val = symbolic_helper._maybe_get_const(scale, "t")
```

The `desc` argument controls the return type:

| `desc` | Returns when value is a Constant |
|---|---|
| `"i"` | Python `int` |
| `"f"` | Python `float` |
| `"t"` | `torch.Tensor` (the raw tensor, any shape) |
| `"is"` | `list[int]` |
| `"fs"` | `list[float]` |
| `"v"` | the `Value` itself (no unwrapping) |

**Behaviour when the value is NOT a Constant:** `_maybe_get_const` returns
the original `Value` object unchanged. This is why the function is named
`_maybe_get_const`: it tries to unwrap, but gracefully falls back. You must
either guarantee the upstream is a Constant (typically by constructing the
input with `torch.tensor(...)` in the caller) or handle both cases.

### Why this matters

Scale and zero-point tensors created via `torch.tensor(2.0 ** lsb, ...)` in
`_quantize` get traced as `onnx::Constant` nodes. `_maybe_get_const(scale, "t")`
then returns a real 0-dim tensor, which can be passed directly as a `_f`
attribute (via the coercion described above) or as a `_t` attribute. This lets
you embed the scalar value into your custom node rather than threading it
through as a graph input.

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
`torch._C.Value` objects. Compute them before they are needed in `symbolic`
and stash them in a side channel (class variable or deque — see
[Side Channels](#side-channels-for-pre-computed-data)).

### Choosing between `_f`/`_i` and `_t` for scalars

For a scalar value, you have three options:

| Approach | When to use |
|---|---|
| `scale_f=float(val)` | You have a plain Python float in scope. |
| `scale_f=val` where `val` is a 0-dim tensor | `_maybe_get_const(..., "t")` returned a 0-dim tensor; let coercion handle it. |
| `scale_t=val` where `val` is a 0-dim tensor | You want it visible in Netron as a tensor attribute, or you want to be explicit. |

All three produce a valid ONNX node. `_f` shows up in Netron as `FLOAT` and
`_t` shows up as `TENSOR` — the runtime that consumes your custom node
determines which is more convenient.

---

## Enum Attributes

If your operator has an enum-typed parameter (e.g. a rounding mode), encode it
as a string attribute using its `.value`:

```python
class RoundingMode(Enum):
    ROUND_TO_NEAREST_EVEN = "round_to_nearest_even"
    FLOOR = "floor"

output = g.op(
    "MyDomain::MyOp",
    x,
    rounding_mode_s=str(rounding_mode.value),   # → STRING attribute
)
```

This keeps the value human-readable in Netron and easy to dispatch on inside
the runtime kernel. Don't try to pass the `Enum` member itself — `_s` expects
a `str`, not an `Enum`.

Alternatively, pick a stable integer encoding and use `_i`:

```python
ROUNDING_MODE_CODES = {
    RoundingMode.ROUND_TO_NEAREST_EVEN: 0,
    RoundingMode.FLOOR: 1,
}
output = g.op("MyDomain::MyOp", x, rounding_mode_i=ROUNDING_MODE_CODES[mode])
```

Strings are friendlier for debugging; ints are faster to dispatch on. Pick one
and document it.

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

# In symbolic — two valid options for the trailing values:

# Option A: create fresh Constant nodes
scale      = g.op("Constant", value_t=torch.tensor(2.0 ** bit_shift_scale))
zero_point = g.op("Constant", value_t=torch.tensor(0.0))
bw         = g.op("Constant", value_t=torch.tensor(float(bit_width)))
return quantized, scale, zero_point, bw

# Option B: pass through the incoming Values
# (scale and zero_point arrive as Constants traced from torch.tensor(...) calls
#  in _quantize; the JIT will fold them naturally)
bw = g.op("Constant", value_t=torch.tensor(float(bit_width)))
return quantized, scale, zero_point, bw
```

Both options produce valid graphs. Option B is fine and is what you get
naturally when you pass `torch.tensor(...)` values into `apply()` from the
caller; Option A is slightly more explicit.

If `symbolic` returns fewer values than `forward`, or vice versa, the exporter
will fail with a `NotImplementedError: You must implement the forward function`
error, even though `forward` is clearly defined. This is because the exporter
uses return-count matching to verify it has found the right symbolic.

---

## Side Channels for Pre-Computed Data

When `symbolic` is called, its tensor arguments are `torch._C.Value` graph
nodes — you cannot call `.tolist()` or `.flatten()` on them. For data that
must be computed from the actual weight values (indices, quantized outputs),
you need a side channel that bypasses the graph.

There are two patterns. **Pick the one that matches your usage**: a single
class variable works for a one-shot, single-instance case; a FIFO deque is
required when the same `Function` class is shared across multiple quantizer
instances (which is the common case for Brevitas injectors).

### Pattern A: Single Class Variable

Works when exactly **one** `apply()` call happens per export.

```python
class MyQuantFn(Function):

    _captured_indices:   torch.Tensor = None
    _captured_quantized: torch.Tensor = None

    @staticmethod
    def symbolic(g, x, coefficients, bit_shift_scale, bit_width):
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
        ...
```

In `_quantize`, populate the class variables **before** the `apply()` call,
inside the `is_in_onnx_export()` guard:

```python
if torch.onnx.is_in_onnx_export():
    with torch.no_grad():
        quantized_vals, _, indices = compute_quantization(x, coefficients, bit_shift_scale)
        MyQuantFn._captured_indices   = indices.flatten().cpu().to(torch.long)
        MyQuantFn._captured_quantized = quantized_vals.flatten().cpu()

    quantized, _, _, _ = MyQuantFn.apply(x, coefficients, bit_shift_scale, bit_width)
    return quantized
```

#### Why this breaks with multiple instances

If your `Function` class is shared across more than one quantizer module
(common in Brevitas: the same `tensor_quant = FixedPointPerTensorQuantizer`
is used for weights, activations, *and* bias), every call to `_quantize`
overwrites the class variable. By the time `symbolic` runs (which happens
after all `forward` calls — see [Execution Order](#execution-order-during-onnx-export)),
only the data from the *last* `_quantize` call is still in the variable.
Every emitted ONNX node will get the same wrong attributes.

This failure mode is **silent** — the graph exports successfully, but all
custom nodes reference the same captured data. Catch it by exporting a model
with two distinct quantized layers and verifying their tensor attributes
differ in Netron.

### Pattern B: FIFO Deque (Recommended for Shared Function Classes)

A `collections.deque` solves the multi-instance problem cleanly. Each
`forward()` call enqueues its captured data; each `symbolic()` call
dequeues from the front.

```python
from collections import deque

class FixedPointQuantFn(Function):

    _integer_queue: deque = deque()

    @staticmethod
    def symbolic(g, x, scale, zero_point, lsb, bit_width, signed, narrow_range, rounding_mode):
        scale_val      = symbolic_helper._maybe_get_const(scale, "t")
        zero_point_val = symbolic_helper._maybe_get_const(zero_point, "t")

        # Pop the integers the corresponding forward() enqueued
        captured = FixedPointQuantFn._integer_queue.popleft()

        quantized = g.op(
            "Quantify::FixedPointQuant",
            x,
            scale_f=scale_val,
            zero_point_f=zero_point_val,
            lsb_i=int(lsb),
            bit_width_i=int(bit_width),
            signed_i=int(signed),
            narrow_range_i=int(narrow_range),
            rounding_mode_s=str(rounding_mode.value),
            quantized_ints_t=captured,
        ).setType(x.type())

        bw = g.op("Constant", value_t=torch.tensor(float(bit_width)))
        return quantized, scale, zero_point, bw

    @staticmethod
    def forward(ctx, x, scale, zero_point, lsb, bit_width, signed, narrow_range, rounding_mode):
        ctx.save_for_backward(x)
        quantized, integers = quantize_fixed_point_with_integers(
            x, int(lsb), int(bit_width), signed, rounding_mode, narrow_range
        )

        if torch.onnx.is_in_onnx_export():
            with torch.no_grad():
                FixedPointQuantFn._integer_queue.append(integers.cpu().to(torch.long))

        bw = torch.tensor(float(bit_width), dtype=x.dtype, device=x.device)
        return quantized, scale, zero_point, bw

    @staticmethod
    def backward(ctx, grad_q, grad_s, grad_zp, grad_bw):
        return grad_q, None, None, None, None, None, None, None

    @classmethod
    def reset_capture_state(cls):
        cls._integer_queue.clear()
```

#### Why the deque ordering is correct

PyTorch tracing runs the model's forward pass to completion before the
symbolic conversion pass walks the resulting graph. So:

1. Every `forward()` call enqueues, in execution order.
2. The graph is built in that same execution order.
3. Symbolic conversion walks the graph top-to-bottom, calling `symbolic()`
   for each `prim::PythonOp`, which dequeues in the matching order.

FIFO order is preserved end-to-end. See
[Execution Order](#execution-order-during-onnx-export) for details.

#### Cleanup

Always call `reset_capture_state()` before each export, especially in test
suites or notebooks where multiple exports run in the same Python process:

```python
FixedPointQuantFn.reset_capture_state()
torch.onnx.export(model, dummy_input, "model.onnx", ...)
```

If a previous export aborted with an exception partway through, leftover
entries in the deque will silently corrupt the next export.

---

## Execution Order During ONNX Export

Understanding this order is essential for any side-channel design.

`torch.onnx.export` does **not** call `symbolic` inline as the model runs.
The export happens in two distinct phases:

```
Phase 1: Tracing
  - Model's forward() is executed end-to-end with the dummy input.
  - Every call to YourFunction.apply(...) runs YourFunction.forward(...).
  - Each apply() call is recorded as a prim::PythonOp node in the JIT graph.
  - At this point your Function's forward() can populate side channels
    (class variables, deques, files on disk, etc.) freely.

Phase 2: Symbolic conversion
  - PyTorch walks the JIT graph top-to-bottom.
  - For each prim::PythonOp, it looks up the matching `symbolic` staticmethod.
  - YourFunction.symbolic(g, ...) is called, receiving torch._C.Value args.
  - The returned Value(s) replace the prim::PythonOp in the graph.
```

Key consequences:

- All `forward()` calls happen before any `symbolic()` call. You can never
  read inside `symbolic` and expect a fresh result from `forward` — by then
  every `forward` has already completed.
- The two phases visit operations in the same order, because phase 2 walks
  the graph that phase 1 produced. A FIFO queue populated in phase 1 and
  consumed in phase 2 stays in sync.
- A single class variable populated repeatedly in phase 1 will only retain
  the *last* assignment by the time phase 2 starts.

---

## Where to Place the `is_in_onnx_export()` Guard

Two valid options:

### Option 1: Guard in the caller (`_quantize`)

```python
def _quantize(self, x, params):
    if torch.onnx.is_in_onnx_export():
        with torch.no_grad():
            _, _, indices = compute_quantization(x, ...)
            MyQuantFn._captured_indices = indices.cpu().to(torch.long)
        quantized, _, _, _ = MyQuantFn.apply(x, ...)
        return quantized
    return quantize_direct(x, ...)
```

Pros: Avoids running `apply()` at all when not exporting (no autograd
overhead in the hot training path). Cons: The capture logic and the
`apply()` call are split across two control flows.

### Option 2: Guard inside `forward()`

```python
@staticmethod
def forward(ctx, x, ...):
    quantized, integers = quantize_with_integers(x, ...)
    if torch.onnx.is_in_onnx_export():
        with torch.no_grad():
            MyQuantFn._integer_queue.append(integers.cpu().to(torch.long))
    return quantized, ...
```

Pros: All capture logic lives next to the forward math; the caller stays
clean. Cons: `apply()` runs unconditionally even during training, which
adds a small autograd overhead.

Both patterns work. Option 2 pairs naturally with the deque pattern because
the capture happens at the same point the integers are computed, with no
duplicated work.

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

1. Compute the data **outside** the trace via `torch.no_grad()` and stash it
   in a side channel (class variable or deque) — preferred, or
2. Create a `Constant` ONNX node inside `symbolic` from the side channel.

Note that data captured inside `forward()` itself is *not* traced as separate
ops, because the entire `forward()` body is recorded as a single
`prim::PythonOp` node. This is what makes the deque-inside-forward pattern
safe — the `integers.cpu().to(torch.long)` ops never enter the graph.

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

This is the deque-based pattern, which is the recommended default. It
correctly handles a `Function` class shared by multiple quantizer instances
(weights, activations, bias).

```python
import torch
from collections import deque
from enum import Enum
from torch.autograd import Function
from torch.onnx import symbolic_helper


class RoundingMode(Enum):
    ROUND_TO_NEAREST_EVEN = "round_to_nearest_even"
    FLOOR = "floor"


def quantize_with_integers(x, lsb, bit_width, signed, rounding_mode, narrow_range):
    """Your actual quantization logic. Returns (quantized, integers)."""
    ...


class FixedPointQuantFn(Function):
    """
    Custom autograd.Function that exports as a single clean ONNX node.

    Side-channel storage is a FIFO deque so the same class can be shared
    across multiple quantizer instances without state collision.
    """

    _integer_queue: deque = deque()

    @staticmethod
    def symbolic(g, x, scale, zero_point, lsb, bit_width, signed,
                 narrow_range, rounding_mode):
        # scale and zero_point arrive as torch._C.Value graph nodes.
        # _maybe_get_const unwraps them to 0-dim tensors because the caller
        # built them with torch.tensor(...) — so they trace as Constants.
        scale_val      = symbolic_helper._maybe_get_const(scale, "t")
        zero_point_val = symbolic_helper._maybe_get_const(zero_point, "t")

        # Dequeue the integers that the corresponding forward() enqueued.
        # FIFO order matches the graph walk, so each symbolic call gets
        # the data from its matching forward call.
        captured_integers = FixedPointQuantFn._integer_queue.popleft()

        quantized = g.op(
            "Quantify::FixedPointQuant",
            x,                                                  # ONNX input 0
            scale_f=scale_val,                                  # 0-dim tensor → FLOAT
            zero_point_f=zero_point_val,                        # 0-dim tensor → FLOAT
            lsb_i=int(lsb),                                     # INT
            bit_width_i=int(bit_width),                         # INT
            signed_i=int(signed),                               # INT (0 or 1)
            narrow_range_i=int(narrow_range),                   # INT (0 or 1)
            rounding_mode_s=str(rounding_mode.value),           # STRING (enum value)
            quantized_ints_t=captured_integers,                 # TENSOR (full shape)
        ).setType(x.type())                                     # propagate shape/dtype

        # Brevitas requires a 4-tuple return.
        bw = g.op("Constant", value_t=torch.tensor(float(bit_width)))
        return quantized, scale, zero_point, bw

    @staticmethod
    def forward(ctx, x, scale, zero_point, lsb, bit_width, signed,
                narrow_range, rounding_mode):
        ctx.save_for_backward(x)
        quantized, integers = quantize_with_integers(
            x, int(lsb), int(bit_width), signed, rounding_mode, narrow_range
        )

        # Capture happens here, inside the prim::PythonOp boundary, so the
        # .cpu()/.to() ops are NOT traced as separate graph nodes.
        if torch.onnx.is_in_onnx_export():
            with torch.no_grad():
                FixedPointQuantFn._integer_queue.append(
                    integers.cpu().to(torch.long)
                )

        bw = torch.tensor(float(bit_width), dtype=x.dtype, device=x.device)
        return quantized, scale, zero_point, bw

    @staticmethod
    def backward(ctx, grad_q, grad_s, grad_zp, grad_bw):
        # Straight-Through Estimator: one None per non-tensor forward input.
        return grad_q, None, None, None, None, None, None, None

    @classmethod
    def reset_capture_state(cls):
        """Call before each torch.onnx.export to clear any leftover state."""
        cls._integer_queue.clear()
```

The caller (a `nn.Module` or Brevitas injector) builds the scale and
zero-point as fresh `torch.tensor(...)` values so they trace as Constants:

```python
def _quantize(self, x, params):
    if torch.onnx.is_in_onnx_export():
        quantized, _, _, _ = FixedPointQuantFn.apply(
            x,
            torch.tensor(2.0 ** params['lsb'], dtype=x.dtype, device=x.device),
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
            params['lsb'],
            self.bit_width,
            params['signed'],
            self.narrow_range,
            self.rounding_mode,
        )
        return quantized
    return quantize_direct(x, int(params['lsb']), self.bit_width,
                           params['signed'], self.rounding_mode,
                           self.narrow_range)
```

And before exporting:

```python
FixedPointQuantFn.reset_capture_state()
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=13, ...)
```

---

## Quick Checklist

Use this before every ONNX export attempt:

- [ ] `symbolic` and `forward` have **identical tensor-argument signatures**
      (same count, same order; `ctx` ↔ `g` don't count).
- [ ] `g.op(...).setType(x.type())` is called on every custom node output.
- [ ] `symbolic` returns the same number of values as `forward`.
- [ ] Side-channel data is computed from real tensors (in `forward()` or
      under `torch.no_grad()` before `apply()`) — never from `Value` args
      inside `symbolic`.
- [ ] If your `Function` class is shared across multiple quantizer instances,
      you are using a **deque** (or equivalent FIFO), not a single class
      variable.
- [ ] `reset_capture_state()` (or equivalent) is called immediately before
      every `torch.onnx.export(...)` call.
- [ ] Pre-computed tensors are on **CPU** before being used as `_t` attributes.
- [ ] Scale/zero-point Values intended for `_f` attributes were built with
      `torch.tensor(...)` in the caller, so `_maybe_get_const(v, "t")` returns
      a 0-dim tensor and not the raw Value.
- [ ] Enums are passed as `enum_member.value` with the `_s` suffix (or mapped
      to ints with `_i`), never as the Enum member itself.
- [ ] You are **not** using `register_custom_op_symbolic` for an
      `autograd.Function`.
- [ ] You are **not** calling `.setDebugName()` on `inputsAt(i)` or on `x`/
      `coefficients` (shared graph values).
- [ ] Scalar arguments passed to `apply()` as plain `int`/`float` are used
      directly as Python values inside `symbolic` — no `.item()` needed.
- [ ] `backward` returns exactly one value per `forward` tensor input
      (use `None` for inputs that don't need gradients).
