"""
Quantized Activation Functions for Quantify.

Provides quantized wrappers for the standard PyTorch activation functions
(ReLU, ReLU6, Sigmoid, Tanh, SiLU, GELU, LeakyReLU, Softmax). Every wrapper
quantizes BOTH its input and its output, fixed-point by default (via
`FixedPointPerTensorActivationQuant`): the output quantizer quantizes the
activation's result; the input quantizer quantizes the pre-activation value
the nonlinearity itself consumes.

Two things are handled specially for the input quantizer -- see pitfall #21
in `docs/llm/pitfalls/brevitas_pitfalls.md` for the full writeup:

1. **Bit-width asymmetry.** An activation whose output is structurally
   forced non-negative (ReLU, ReLU6, Sigmoid, Softmax) never needs a sign
   bit on its OUTPUT quantizer -- all `bit_width` bits go to magnitude. Its
   INPUT quantizer, though, usually *does* need a sign bit (pre-activation
   values are typically signed), so by default it gets one extra bit
   (`input_bit_width = bit_width + 1`) to keep the same magnitude
   resolution on both sides of the activation. Activations whose output can
   be negative (Tanh, SiLU, GELU, LeakyReLU) get `input_bit_width ==
   bit_width` (no asymmetry -- both sides need a sign bit anyway).
2. **Saturation-aware range capping.** Activations that converge to a fixed
   value (ReLU6's hard clip to [0, 6]; Sigmoid/Tanh's asymptotic
   saturation) must not let a rare, huge pre-activation outlier blow up the
   input quantizer's calibrated range -- the activation clips/saturates
   that outlier away immediately afterward regardless, so resolution spent
   representing it is pure waste. These three get a `max_abs_value` cap
   derived from the activation's own saturation point (`find_optimal_lsb`'s
   `max_abs_value` parameter): real inputs beyond the cap get clipped by
   the quantizer, matching what the activation itself does to them anyway.
   Activations with no natural saturation point (ReLU -- unbounded above;
   SiLU/GELU -- grow ~linearly for large positive input; LeakyReLU --
   unbounded both directions; Softmax -- depends on the relative spread of
   logits, not an absolute magnitude) are never capped this way.

ONNX export contract: each activation emits exactly one ONNX node for the
nonlinearity itself, with its output feeding into a separate quantizer node
(`Quantify::FixedPointQuant` by default). For ReLU, Sigmoid, Tanh, LeakyReLU
and Softmax, the plain PyTorch op already lowers to a single native ONNX node
(`Relu`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Softmax`) under the legacy
TorchScript exporter, so no extra shimming is needed. ReLU6, SiLU and GELU do
NOT lower to a single node natively (PyTorch decomposes them into
Clip/Constant, Sigmoid/Mul, and Erf-based chains respectively) -- these three
use a `torch.autograd.Function` with a custom `symbolic()` (same pattern as
`quantizers/fixedpoint_per_tensor.py`/`quantizers/silu_quant.py`) to force a
single `Quantify::<Name>` node.

Note: this is a different, decoupled pattern from `quantizers/silu_quant.py`'s
`QuantSiLUActivationQuant`, which fuses the SiLU nonlinearity and the
fixed-point quantize step into a single `Quantify::QuantSiLU` node for use as
a Brevitas `act_quant` injector directly on `QuantConv2d`/`QuantLinear`. The
`QuantSiLU` module here is a standalone activation module (like `nn.SiLU`)
with its own separate quantizer, matching the other seven activations.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from torch.autograd import Function

from quantizers.fixedpoint_per_tensor import FixedPointPerTensorActivationQuant


# ---------------------------------------------------------------------------
# Input-quantizer defaults: bit-width asymmetry + saturation-aware range cap
# ---------------------------------------------------------------------------

# "Practical saturation" magnitude per activation -- the point beyond which
# the activation's output has converged (exactly, for ReLU6's hard clip;
# asymptotically, for Sigmoid/Tanh) and extra input resolution beyond it is
# wasted on a range the activation immediately clips/saturates away. `None`
# means the activation has no such point (unbounded, or -- Softmax -- its
# useful range depends on relative logit spread, not an absolute magnitude).
#
# The actual cap used is the NEXT POWER OF TWO at or above this value (see
# `_next_pow2_at_least`), not the raw saturation point itself -- fixed-point
# quantizer ranges are themselves powers of two (an 8-bit signed grid at
# lsb=L spans +/- 2**(lsb+7)), so rounding the cap down to a non-power-of-two
# number would still forbid the single LSB setting that actually matches the
# saturation point tightest, for no benefit. E.g. ReLU6's hard clip at 6 ->
# cap 8 (not 6 itself, and not some arbitrary multiple). Sigmoid and Tanh are
# exactly related (sigmoid(x) = (1 + tanh(x/2)) / 2), so Sigmoid's natural
# x-scale is double Tanh's; both numbers below follow from that
# relationship, not independent guesses.
_RELU6_SATURATION = 6.0
_TANH_SATURATION = 3.0
_SIGMOID_SATURATION = 2.0 * _TANH_SATURATION  # == 6.0


def _next_pow2_at_least(value: float) -> float:
    """Smallest power of two that is >= `value` (`value > 0`). A small
    epsilon guards against a value that's already an exact power of two
    landing a notch too high purely from floating-point log2 noise (e.g.
    `log2(8.0)` evaluating to `2.9999999999999996` instead of exactly `3.0`).
    """
    import math
    return 2.0 ** math.ceil(math.log2(value) - 1e-9)


def _input_bit_width(bit_width: int, unsigned_output: bool, override: Optional[int]) -> int:
    """Default input-quantizer bit-width: `bit_width + 1` when the
    activation's output is structurally non-negative (no sign bit spent
    there, so give it to the input instead -- see the module docstring),
    else plain `bit_width`. An explicit `override` always wins.
    """
    if override is not None:
        return override
    return bit_width + 1 if unsigned_output else bit_width


def _input_max_abs_value(saturation: Optional[float], override: Optional[float]) -> Optional[float]:
    """Default input-quantizer range cap: the next power of two at or above
    `saturation` (see `_next_pow2_at_least`), or `None` (uncapped) when
    `saturation` is `None`. An explicit `override` always wins -- including
    an explicit `None` passed deliberately to disable the cap (same
    semantics as every other "`None` means use the default" knob here, so
    `override` is only consulted when it is NOT None; pass the class's own
    saturation-derived value if you want to opt back out of an override at
    the call site).
    """
    if override is not None:
        return override
    if saturation is None:
        return None
    return _next_pow2_at_least(saturation)


# ---------------------------------------------------------------------------
# Custom single-node ONNX shims for activations PyTorch does not lower to a
# single native op (ReLU6, SiLU, GELU).
# ---------------------------------------------------------------------------

def _grad_via_recompute(fwd_fn, x, grad_output):
    """Compute the real gradient of `fwd_fn` at `x` by recomputing it under
    autograd, instead of hand-deriving a closed-form derivative for every
    activation. This is the true activation gradient (not a straight-through
    estimator) -- these Functions represent the actual nonlinearity, with
    quantization handled entirely by the separate quantizer module."""
    with torch.enable_grad():
        x_ = x.detach().requires_grad_(True)
        y = fwd_fn(x_)
    (grad_x,) = torch.autograd.grad(y, x_, grad_output)
    return grad_x


class Relu6Fn(Function):
    """Symbolic shim: emits a single `Quantify::Relu6` ONNX node."""

    @staticmethod
    def symbolic(g, x):
        return g.op("Quantify::Relu6", x, min_f=0.0, max_f=6.0).setType(x.type())

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0.0, max=6.0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Differentiate hardtanh (what nn.ReLU6 runs), not torch.clamp: they
        # agree in value but clamp's gradient is 1 at exactly x == 0 and
        # x == 6 while nn.ReLU6's is 0 there. Quantized pre-activations hit
        # those points exactly whenever they round to code 0 (or to 6).
        return _grad_via_recompute(lambda t: F.hardtanh(t, 0.0, 6.0), x, grad_output)


class SiLUFn(Function):
    """Symbolic shim: emits a single `Quantify::SiLU` ONNX node."""

    @staticmethod
    def symbolic(g, x):
        return g.op("Quantify::SiLU", x).setType(x.type())

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.silu(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return _grad_via_recompute(F.silu, x, grad_output)


class GELUFn(Function):
    """Symbolic shim: emits a single `Quantify::GELU` ONNX node."""

    @staticmethod
    def symbolic(g, x, approximate):
        approximate_val = torch.onnx.symbolic_helper._maybe_get_const(approximate, "s")
        # approximate_code_i: redundant int encoding (0="none"/erf-based,
        # 1="tanh") of approximate_s, for the embedded ONNX FunctionProto
        # body (utils/onnx_self_contained.py) to select a branch with --
        # there's no standard ONNX op for string comparison, so the function
        # body picks via Where on this instead. Named "..._code_i" (not
        # "approximate_i") because PyTorch's ONNX exporter strips the
        # trailing type-suffix from attribute names when writing the node
        # (e.g. "approximate_s" -> "approximate"), so "approximate_i" would
        # otherwise collide with "approximate_s" on the exported node.
        approximate_code_i = 1 if str(approximate_val) == "tanh" else 0
        return g.op(
            "Quantify::GELU", x,
            approximate_s=str(approximate_val),
            approximate_code_i=approximate_code_i,
        ).setType(x.type())

    @staticmethod
    def forward(ctx, x, approximate):
        ctx.save_for_backward(x)
        ctx.approximate = approximate
        return F.gelu(x, approximate=approximate)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        approximate = ctx.approximate
        return _grad_via_recompute(lambda t: F.gelu(t, approximate=approximate), x, grad_output), None


# ---------------------------------------------------------------------------
# Quantized activation modules
# ---------------------------------------------------------------------------


class QuantReLU(qnn.QuantReLU):
    """ReLU with fixed-point (by default) input AND output quantizers.

    Exports as the input quantizer's node, a single native `Relu` ONNX node,
    then the output quantizer's node (`Quantify::FixedPointQuant` by default
    for both). ReLU's output is structurally non-negative, so by default the
    saved output sign bit goes to the input instead
    (`input_bit_width = bit_width + 1`) -- see the module docstring. ReLU has
    no upper saturation point, so its input is never range-capped.
    """

    def __init__(
        self,
        act_quant=FixedPointPerTensorActivationQuant,
        input_quant=FixedPointPerTensorActivationQuant,
        bit_width: int = 8,
        input_bit_width: Optional[int] = None,
        **kwargs,
    ):
        ibw = _input_bit_width(bit_width, unsigned_output=True, override=input_bit_width)
        super().__init__(act_quant=act_quant, input_quant=input_quant,
                          bit_width=bit_width, input_bit_width=ibw, **kwargs)


class QuantSigmoid(qnn.QuantSigmoid):
    """Sigmoid with fixed-point (by default) input AND output quantizers.

    Exports as the input quantizer's node, a single native `Sigmoid` ONNX
    node, then the output quantizer's node. Sigmoid's output is structurally
    non-negative, so by default the saved output sign bit goes to the input
    instead (`input_bit_width = bit_width + 1`). Sigmoid saturates
    asymptotically, so its input is range-capped by default (see the module
    docstring) to avoid a rare huge pre-activation outlier ballooning the
    calibrated range for no benefit.
    """

    def __init__(
        self,
        act_quant=FixedPointPerTensorActivationQuant,
        input_quant=FixedPointPerTensorActivationQuant,
        bit_width: int = 8,
        input_bit_width: Optional[int] = None,
        input_max_abs_value: Optional[float] = None,
        **kwargs,
    ):
        ibw = _input_bit_width(bit_width, unsigned_output=True, override=input_bit_width)
        imax = _input_max_abs_value(_SIGMOID_SATURATION, input_max_abs_value)
        super().__init__(act_quant=act_quant, input_quant=input_quant,
                          bit_width=bit_width, input_bit_width=ibw,
                          input_max_abs_value=imax, **kwargs)


class QuantTanh(qnn.QuantTanh):
    """Tanh with fixed-point (by default) input AND output quantizers.

    Exports as the input quantizer's node, a single native `Tanh` ONNX node,
    then the output quantizer's node. Tanh's output can be negative, so
    `input_bit_width == bit_width` (no asymmetry -- both sides need a sign
    bit). Tanh saturates asymptotically, so its input is range-capped by
    default (see the module docstring).
    """

    def __init__(
        self,
        act_quant=FixedPointPerTensorActivationQuant,
        input_quant=FixedPointPerTensorActivationQuant,
        bit_width: int = 8,
        input_bit_width: Optional[int] = None,
        input_max_abs_value: Optional[float] = None,
        **kwargs,
    ):
        ibw = _input_bit_width(bit_width, unsigned_output=False, override=input_bit_width)
        imax = _input_max_abs_value(_TANH_SATURATION, input_max_abs_value)
        super().__init__(act_quant=act_quant, input_quant=input_quant,
                          bit_width=bit_width, input_bit_width=ibw,
                          input_max_abs_value=imax, **kwargs)


class QuantLeakyReLU(nn.Module):
    """LeakyReLU with fixed-point (by default) input AND output quantizers.

    Exports as the input quantizer's node, a single native `LeakyRelu` ONNX
    node, then the output quantizer's node. Brevitas has no built-in
    `QuantLeakyReLU`, so this applies the nonlinearity directly between two
    `qnn.QuantIdentity` instances. LeakyReLU's output can be negative, so
    `input_bit_width == bit_width` (no asymmetry), and it has no saturation
    point in either direction, so its input is never range-capped.
    """

    def __init__(
        self,
        negative_slope: float = 0.01,
        act_quant=FixedPointPerTensorActivationQuant,
        input_quant=FixedPointPerTensorActivationQuant,
        bit_width: int = 8,
        input_bit_width: Optional[int] = None,
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.negative_slope = negative_slope
        ibw = _input_bit_width(bit_width, unsigned_output=False, override=input_bit_width)
        self.input_quant = qnn.QuantIdentity(
            act_quant=input_quant, bit_width=ibw, return_quant_tensor=False,
        )
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, bit_width=bit_width,
            return_quant_tensor=return_quant_tensor, **kwargs
        )

    def forward(self, x):
        x = self.input_quant(x)
        return self.output_quant(F.leaky_relu(x, negative_slope=self.negative_slope))


class QuantSoftmax(nn.Module):
    """Softmax with fixed-point (by default) input AND output quantizers.

    Exports as the input quantizer's node, a single native `Softmax` ONNX
    node, then the output quantizer's node. Brevitas has no built-in
    `QuantSoftmax`, so this applies the nonlinearity directly between two
    `qnn.QuantIdentity` instances. Softmax's output (probabilities) is
    structurally non-negative, so by default the saved output sign bit goes
    to the input instead (`input_bit_width = bit_width + 1`) -- logits are
    typically signed. Softmax's useful input range depends on the relative
    spread of the logits, not an absolute magnitude, so it has no
    saturation-point range cap (unlike ReLU6/Sigmoid/Tanh).
    """

    def __init__(
        self,
        dim: int = -1,
        act_quant=FixedPointPerTensorActivationQuant,
        input_quant=FixedPointPerTensorActivationQuant,
        bit_width: int = 8,
        input_bit_width: Optional[int] = None,
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        ibw = _input_bit_width(bit_width, unsigned_output=True, override=input_bit_width)
        self.input_quant = qnn.QuantIdentity(
            act_quant=input_quant, bit_width=ibw, return_quant_tensor=False,
        )
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, bit_width=bit_width,
            return_quant_tensor=return_quant_tensor, **kwargs
        )

    def forward(self, x):
        x = self.input_quant(x)
        return self.output_quant(F.softmax(x, dim=self.dim))


class QuantReLU6(nn.Module):
    """ReLU6 with fixed-point (by default) input AND output quantizers.

    PyTorch's own `nn.ReLU6` decomposes into `Constant, Constant, Clip` on
    export, so this uses `Relu6Fn` (a custom `torch.autograd.Function`) to
    force a single `Quantify::Relu6` ONNX node, with the input quantizer's
    node before it and the output quantizer's node after. ReLU6's output is
    structurally non-negative, so by default the saved output sign bit goes
    to the input instead (`input_bit_width = bit_width + 1`). ReLU6 hard-
    clips to `[0, 6]`, so its input is range-capped by default (see the
    module docstring) to avoid a rare huge pre-activation outlier
    ballooning the calibrated range for no benefit -- everything beyond the
    clip point is thrown away by the activation regardless.
    """

    def __init__(
        self,
        act_quant=FixedPointPerTensorActivationQuant,
        input_quant=FixedPointPerTensorActivationQuant,
        bit_width: int = 8,
        input_bit_width: Optional[int] = None,
        input_max_abs_value: Optional[float] = None,
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        super().__init__()
        ibw = _input_bit_width(bit_width, unsigned_output=True, override=input_bit_width)
        imax = _input_max_abs_value(_RELU6_SATURATION, input_max_abs_value)
        self.input_quant = qnn.QuantIdentity(
            act_quant=input_quant, bit_width=ibw, max_abs_value=imax,
            return_quant_tensor=False,
        )
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, bit_width=bit_width,
            return_quant_tensor=return_quant_tensor, **kwargs
        )

    def forward(self, x):
        x = self.input_quant(x)
        return self.output_quant(Relu6Fn.apply(x))


class QuantSiLU(nn.Module):
    """SiLU (Swish) with fixed-point (by default) input AND output
    quantizers.

    `nn.SiLU`/`F.silu` decomposes into `Sigmoid, Mul` on export, so this uses
    `SiLUFn` (a custom `torch.autograd.Function`) to force a single
    `Quantify::SiLU` ONNX node, with the input quantizer's node before it
    and the output quantizer's node after. SiLU's output can be negative
    (down to about -0.278), so `input_bit_width == bit_width` (no
    asymmetry). SiLU grows ~linearly for large positive input (no upper
    saturation point), so its input is never range-capped.

    See the module docstring for how this differs from
    `quantizers.silu_quant.QuantSiLUActivationQuant`.
    """

    def __init__(
        self,
        act_quant=FixedPointPerTensorActivationQuant,
        input_quant=FixedPointPerTensorActivationQuant,
        bit_width: int = 8,
        input_bit_width: Optional[int] = None,
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        super().__init__()
        ibw = _input_bit_width(bit_width, unsigned_output=False, override=input_bit_width)
        self.input_quant = qnn.QuantIdentity(
            act_quant=input_quant, bit_width=ibw, return_quant_tensor=False,
        )
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, bit_width=bit_width,
            return_quant_tensor=return_quant_tensor, **kwargs
        )

    def forward(self, x):
        x = self.input_quant(x)
        return self.output_quant(SiLUFn.apply(x))


class QuantGELU(nn.Module):
    """GELU with fixed-point (by default) input AND output quantizers.

    `nn.GELU`/`F.gelu` decomposes into a long Erf- (or Tanh-)based chain on
    export, so this uses `GELUFn` (a custom `torch.autograd.Function`) to
    force a single `Quantify::GELU` ONNX node, with the input quantizer's
    node before it and the output quantizer's node after. GELU's output can
    be negative (down to about -0.17), so `input_bit_width == bit_width`
    (no asymmetry). GELU grows ~linearly for large positive input (no upper
    saturation point), so its input is never range-capped.
    """

    def __init__(
        self,
        approximate: str = "none",
        act_quant=FixedPointPerTensorActivationQuant,
        input_quant=FixedPointPerTensorActivationQuant,
        bit_width: int = 8,
        input_bit_width: Optional[int] = None,
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.approximate = approximate
        ibw = _input_bit_width(bit_width, unsigned_output=False, override=input_bit_width)
        self.input_quant = qnn.QuantIdentity(
            act_quant=input_quant, bit_width=ibw, return_quant_tensor=False,
        )
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, bit_width=bit_width,
            return_quant_tensor=return_quant_tensor, **kwargs
        )

    def forward(self, x):
        x = self.input_quant(x)
        return self.output_quant(GELUFn.apply(x, self.approximate))
