"""
Autograd helpers shared by every quantizer: annealing blend and the STE variants.

Re-exported from `quantizers/base_quantizer.py`, which uses them in
`BaseQuantizer.forward`. See docs/llm/skills/ste-clipping-modes.md.
"""

import torch


class AnnealingBlendFn(torch.autograd.Function):
    """Compute (1-alpha)*x + alpha*quantized in the forward pass, but present
    a single straight-through node in the backward graph instead of the
    Add/Mul chain that tensor arithmetic would produce.

    alpha is passed as a plain Python float so it never appears as a graph
    input.  Gradient goes entirely through the `x` input (slope = 1); `None`
    is returned for `quantized` to avoid double-counting — both x and
    quantized are derived from the same upstream leaf.
    """

    @staticmethod
    def forward(ctx, x, quantized, alpha):
        return (1.0 - alpha) * x + alpha * quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class ClippedSTEFn(torch.autograd.Function):
    """Clipped Straight-Through Estimator, shared by all quantizers.

    Forward is a value-preserving identity on `quantized` (so the numeric
    output of the quantizer is completely unchanged). Backward multiplies the
    incoming gradient by a precomputed in-range mask, so the local slope is 1
    for inputs that landed INSIDE the quantizer's representable range and 0 for
    inputs the forward clamp saturated. This is the state-of-the-art refinement
    over plain STE, which uses slope 1 everywhere.

    The mask is a 0/1 tensor (in the quantized tensor's dtype) computed once in
    BaseQuantizer.forward from the ORIGINAL float input and the quantizer's
    range bounds; `None` is returned for it in backward since it is not a
    differentiable input.
    """

    @staticmethod
    def forward(ctx, quantized, in_range_mask):
        ctx.save_for_backward(in_range_mask)
        # Return a fresh tensor (not the input alias) so autograd treats this as
        # a distinct node; the value is identical to `quantized`.
        return quantized.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (in_range_mask,) = ctx.saved_tensors
        return grad_output * in_range_mask, None


class SingleDirectionClippedSTEFn(torch.autograd.Function):
    """Single-direction clipped STE, shared by all quantizers.

    Like ClippedSTEFn, forward is a value-preserving identity on `quantized`.
    Backward passes the gradient (slope 1) for in-range inputs, and for
    out-of-range inputs ONLY when it points back into the range -- i.e. when a
    gradient-descent step (`w -= lr * grad`) would move the value inward:

        side == 0  (in range)     -> pass
        side == +1 (above range)  -> pass iff grad > 0 (descent decreases it)
        side == -1 (below range)  -> pass iff grad < 0 (descent increases it)

    Gradients pushing a clipped value further out are zeroed, exactly like
    ClippedSTEFn, so a clipped value cannot drift away (the plain-STE
    problem). But unlike ClippedSTEFn, a clipped value is never permanently
    stuck: once the loss wants it back inside, it gets the gradient again.
    For a persistent parameter (a weight) with a fixed quantizer range, plain
    clipped STE would otherwise freeze it forever -- see
    examples/ste_clipping_demo.py (old_examples branch).

    `side` is a tensor of -1/0/+1 (in the quantized tensor's dtype) computed
    in BaseQuantizer.forward from the ORIGINAL float input via `_clip_side()`.
    """

    @staticmethod
    def forward(ctx, quantized, side):
        ctx.save_for_backward(side)
        return quantized.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (side,) = ctx.saved_tensors
        passes = (side == 0) | ((side > 0) & (grad_output > 0)) | ((side < 0) & (grad_output < 0))
        return grad_output * passes.to(grad_output.dtype), None


# Accepted values of a quantizer's `clipped_ste` setting. Booleans are still
# accepted for backward compatibility: False -> "not_clip", True -> "clip".
STE_MODES = ("not_clip", "clip", "single_direction_clip")


def normalize_ste_mode(value) -> str:
    """Map a `clipped_ste` setting (bool or one of STE_MODES) to its mode string."""
    if isinstance(value, bool):
        return "clip" if value else "not_clip"
    if isinstance(value, str) and value in STE_MODES:
        return value
    raise ValueError(f"clipped_ste must be a bool or one of {STE_MODES}, got {value!r}")
