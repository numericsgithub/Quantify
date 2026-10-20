"""
Quantized Activation Functions for Quantify.

Provides quantized wrappers for the standard PyTorch activation functions
(ReLU, ReLU6, Sigmoid, Tanh, SiLU, GELU, LeakyReLU, Softmax, HardSwish,
HardSigmoid). Every wrapper
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

ONNX export contract: each activation emits exactly one `Quantify::<Name>`
ONNX node for the nonlinearity itself (`Relu`, `Relu6`, `Sigmoid`, `Tanh`,
`SiLU`, `GELU`, `LeakyRelu`, `Softmax`, `HardSwish`, `HardSigmoid`), between
its input and output quantizer nodes (`Quantify::FixedPointQuant` by
default). A custom node -- never the native ONNX op -- so a Quantify
activation can always be told apart in the graph from a plain `nn.ReLU` etc.
that slipped through unquantized; `utils/onnx_export.py::export_onnx_with_io`
rejects the latter (see `utils/onnx_canonical.py`). Every one of these nodes
also has an embedded `FunctionProto` (`utils/onnx_self_contained.py`), so the
graph still runs in plain onnxruntime.

ReLU6, SiLU and GELU use a `torch.autograd.Function` with a custom
`symbolic()` (same pattern as `quantizers/fixedpoint_per_tensor.py`) in
every forward pass. The others only route through their `*Fn` shim while
`torch.onnx.is_in_onnx_export()` (`_export_as`), running the plain torch op
otherwise -- identical numerics and gradients, with no extra autograd node
on the training hot path.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.nn import quant_activation as _brevitas_act
from brevitas.nn.quant_layer import QuantNonLinearActLayer
from torch.autograd import Function

from quantizers.fixedpoint_per_tensor import FixedPointPerTensorActivationQuant, LsbFloorLink


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
# The cap used is this value EXACTLY, not padded up to a power of two --
# coverage-first (the calibration rule this cap feeds) already finds the
# finest LSB that covers whatever cap it's given, so padding the cap up only
# forces a wider, partly-wasted range for no benefit (see pitfall #23 in
# `docs/llm/pitfalls/brevitas_pitfalls.md`). Sigmoid and Tanh are exactly
# related (sigmoid(x) = (1 + tanh(x/2)) / 2), so Sigmoid's natural x-scale is
# double Tanh's; both numbers below follow from that relationship, not
# independent guesses.
_RELU6_SATURATION = 6.0
_TANH_SATURATION = 3.0
_SIGMOID_SATURATION = 2.0 * _TANH_SATURATION  # == 6.0
# HardSigmoid is exactly relu6(x + 3) / 6: hard-clipped to 0 below -3 and to
# 1 above +3.
_HARDSIGMOID_SATURATION = 3.0
# 6 == 3 * 2**1, so ReLU6's clamp point is only on grids with lsb <= 1.
_RELU6_MAX_FLOOR_LSB = 1


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
    """Default input-quantizer range cap: the activation's own saturation
    point exactly, or `None` (uncapped) when `saturation` is `None`. An
    explicit `override` always wins -- including an explicit `None` passed
    deliberately to disable the cap (same semantics as every other "`None`
    means use the default" knob here, so `override` is only consulted when
    it is NOT None; pass the class's own saturation-derived value if you
    want to opt back out of an override at the call site).

    The cap is the RAW saturation value, not padded up to a power of two.
    Coverage-first (`prefer_high_lsb=True`, the calibration rule this cap
    feeds -- see `find_optimal_lsb`) already finds the finest LSB that
    covers whatever cap it's given, power-of-two or not; padding the cap up
    only forces that search to cover a wider, partly-wasted range, costing
    real resolution for no benefit. E.g. ReLU6 hard-clips to [0, 6]: capping
    at exactly 6.0 calibrates to lsb=-5 (q_max=7.96875); capping at a
    padded-up 8.0 calibrates to lsb=-4 (q_max=15.9375) -- a full extra bit
    wasted on headroom the activation throws away regardless. See pitfall
    #23 in `docs/llm/pitfalls/brevitas_pitfalls.md`.
    """
    if override is not None:
        return override
    return saturation


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
# Export-only single-node ONNX shims for activations PyTorch WOULD lower to a
# native op (Relu, Sigmoid, ...). Emitting `Quantify::<Name>` instead keeps a
# Quantify activation distinguishable from an unquantized `nn.ReLU` in the
# graph. Only applied while exporting (`_export_as`), so they need no
# backward().
# ---------------------------------------------------------------------------

class ReluFn(Function):
    """Symbolic shim: emits a single `Quantify::Relu` ONNX node."""

    @staticmethod
    def symbolic(g, x):
        return g.op("Quantify::Relu", x).setType(x.type())

    @staticmethod
    def forward(ctx, x):
        return F.relu(x)


class SigmoidFn(Function):
    """Symbolic shim: emits a single `Quantify::Sigmoid` ONNX node."""

    @staticmethod
    def symbolic(g, x):
        return g.op("Quantify::Sigmoid", x).setType(x.type())

    @staticmethod
    def forward(ctx, x):
        return torch.sigmoid(x)


class TanhFn(Function):
    """Symbolic shim: emits a single `Quantify::Tanh` ONNX node."""

    @staticmethod
    def symbolic(g, x):
        return g.op("Quantify::Tanh", x).setType(x.type())

    @staticmethod
    def forward(ctx, x):
        return torch.tanh(x)


class LeakyReluFn(Function):
    """Symbolic shim: emits a single `Quantify::LeakyRelu` ONNX node
    (`alpha` = negative slope, same attribute name as ONNX's LeakyRelu)."""

    @staticmethod
    def symbolic(g, x, negative_slope):
        return g.op("Quantify::LeakyRelu", x, alpha_f=float(negative_slope)).setType(x.type())

    @staticmethod
    def forward(ctx, x, negative_slope):
        return F.leaky_relu(x, negative_slope=negative_slope)


class SoftmaxFn(Function):
    """Symbolic shim: emits a single `Quantify::Softmax` ONNX node (`axis`,
    same attribute name/semantics as ONNX's opset-13+ Softmax)."""

    @staticmethod
    def symbolic(g, x, dim):
        return g.op("Quantify::Softmax", x, axis_i=int(dim)).setType(x.type())

    @staticmethod
    def forward(ctx, x, dim):
        return F.softmax(x, dim=dim)


class HardSwishFn(Function):
    """Symbolic shim: emits a single `Quantify::HardSwish` ONNX node."""

    @staticmethod
    def symbolic(g, x):
        return g.op("Quantify::HardSwish", x).setType(x.type())

    @staticmethod
    def forward(ctx, x):
        return F.hardswish(x)


class HardSigmoidFn(Function):
    """Symbolic shim: emits a single `Quantify::HardSigmoid` ONNX node.
    `alpha`/`beta` follow ONNX's HardSigmoid (`clip(alpha*x + beta, 0, 1)`);
    PyTorch's hardsigmoid is exactly alpha=1/6, beta=0.5."""

    @staticmethod
    def symbolic(g, x):
        return g.op("Quantify::HardSigmoid", x, alpha_f=1.0 / 6.0, beta_f=0.5).setType(x.type())

    @staticmethod
    def forward(ctx, x):
        return F.hardsigmoid(x)


def _export_as(fn_cls, eager_fn, x, *args):
    """`fn_cls.apply(x, *args)` while exporting to ONNX (single
    `Quantify::<Name>` node), the plain `eager_fn(x, *args)` otherwise."""
    if torch.onnx.is_in_onnx_export():
        return fn_cls.apply(x, *args)
    return eager_fn(x, *args)


# `act_impl` classes for the Brevitas-based QuantReLU/QuantSigmoid/QuantTanh.
# Subclassing the original impl keeps every `isinstance` check inside
# Brevitas (and the parameter-free state_dict) unchanged; only the export
# lowering differs. Explicit no-arg __init__ for the same dependency-injection
# reason as brevitas.nn.quant_activation.Sigmoid/Tanh.

class _ReLUImpl(nn.ReLU):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _export_as(ReluFn, F.relu, x)


class _SigmoidImpl(_brevitas_act.Sigmoid):
    def forward(self, x):
        return _export_as(SigmoidFn, torch.sigmoid, x)


class _TanhImpl(_brevitas_act.Tanh):
    def forward(self, x):
        return _export_as(TanhFn, torch.tanh, x)


# ---------------------------------------------------------------------------
# Quantized activation modules
# ---------------------------------------------------------------------------


class QuantReLU(qnn.QuantReLU):
    """ReLU with fixed-point (by default) input AND output quantizers.

    Exports as the input quantizer's node, a single `Quantify::Relu` ONNX
    node, then the output quantizer's node (`Quantify::FixedPointQuant` by default
    for both). ReLU's output is structurally non-negative, so by default the
    saved output sign bit goes to the input instead
    (`input_bit_width = bit_width + 1`) -- see the module docstring. ReLU has
    no upper saturation point, so its input is never range-capped.

    ReLU only zeroes values, so its output already lies on the input
    quantizer's grid. By default the output quantizer is therefore never
    calibrated finer than the input quantizer's LSB (`match_input_lsb=True`):
    a finer grid would only add codes that can never occur, spending bits
    that should have gone to range. Pass `match_input_lsb=False` to disable.
    See pitfall #25.
    """

    def __init__(
        self,
        act_quant=FixedPointPerTensorActivationQuant,
        input_quant=FixedPointPerTensorActivationQuant,
        bit_width: int = 8,
        input_bit_width: Optional[int] = None,
        match_input_lsb: bool = True,
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        ibw = _input_bit_width(bit_width, unsigned_output=True, override=input_bit_width)
        link = LsbFloorLink() if (match_input_lsb and input_quant is not None) else None
        extra = {} if link is None else {"lsb_floor_link": link}
        # Same as qnn.QuantReLU.__init__, but with the export-aware act_impl.
        QuantNonLinearActLayer.__init__(
            self, act_impl=_ReLUImpl, passthrough_act=True,
            input_quant=input_quant, act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            bit_width=bit_width, input_bit_width=ibw, **extra, **kwargs)
        if link is not None:
            link.bind(self)


class QuantSigmoid(qnn.QuantSigmoid):
    """Sigmoid with fixed-point (by default) input AND output quantizers.

    Exports as the input quantizer's node, a single `Quantify::Sigmoid`
    ONNX node, then the output quantizer's node. Sigmoid's output is structurally
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
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        ibw = _input_bit_width(bit_width, unsigned_output=True, override=input_bit_width)
        imax = _input_max_abs_value(_SIGMOID_SATURATION, input_max_abs_value)
        QuantNonLinearActLayer.__init__(
            self, act_impl=_SigmoidImpl, passthrough_act=False,
            input_quant=input_quant, act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            bit_width=bit_width, input_bit_width=ibw,
            input_max_abs_value=imax, **kwargs)


class QuantTanh(qnn.QuantTanh):
    """Tanh with fixed-point (by default) input AND output quantizers.

    Exports as the input quantizer's node, a single `Quantify::Tanh` ONNX
    node, then the output quantizer's node. Tanh's output can be negative, so
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
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        ibw = _input_bit_width(bit_width, unsigned_output=False, override=input_bit_width)
        imax = _input_max_abs_value(_TANH_SATURATION, input_max_abs_value)
        QuantNonLinearActLayer.__init__(
            self, act_impl=_TanhImpl, passthrough_act=False,
            input_quant=input_quant, act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            bit_width=bit_width, input_bit_width=ibw,
            input_max_abs_value=imax, **kwargs)


class QuantLeakyReLU(nn.Module):
    """LeakyReLU with fixed-point (by default) input AND output quantizers.

    Exports as the input quantizer's node, a single `Quantify::LeakyRelu`
    ONNX node, then the output quantizer's node. Brevitas has no built-in
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
        return self.output_quant(_export_as(LeakyReluFn, F.leaky_relu, x, self.negative_slope))


class QuantSoftmax(nn.Module):
    """Softmax with fixed-point (by default) input AND output quantizers.

    Exports as the input quantizer's node, a single `Quantify::Softmax`
    ONNX node, then the output quantizer's node. Brevitas has no built-in
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
        return self.output_quant(_export_as(SoftmaxFn, F.softmax, x, self.dim))


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
        match_input_lsb: bool = True,
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
        # The clamp value 6 is a multiple of 2**1 but of no coarser grid, so
        # the floor is capped at lsb 1 (output must stay able to hit 6).
        extra = ({"lsb_floor_link": LsbFloorLink(max_floor=_RELU6_MAX_FLOOR_LSB)}
                 if match_input_lsb and input_quant is not None else {})
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, bit_width=bit_width,
            return_quant_tensor=return_quant_tensor, **extra, **kwargs
        )
        if extra:
            extra["lsb_floor_link"].bind(self)

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


class QuantHardSwish(nn.Module):
    """HardSwish with fixed-point (by default) input AND output quantizers.

    Exports as the input quantizer's node, a single `Quantify::HardSwish`
    ONNX node, then the output quantizer's node. HardSwish's output can be
    negative (down to -0.375), so `input_bit_width == bit_width` (no
    asymmetry). It grows linearly for large positive input (no upper
    saturation point), so its input is never range-capped.
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
        return self.output_quant(_export_as(HardSwishFn, F.hardswish, x))


class QuantHardSigmoid(nn.Module):
    """HardSigmoid with fixed-point (by default) input AND output quantizers.

    Exports as the input quantizer's node, a single `Quantify::HardSigmoid`
    ONNX node, then the output quantizer's node. HardSigmoid's output is in
    `[0, 1]`, so by default the saved output sign bit goes to the input
    instead (`input_bit_width = bit_width + 1`). It hard-clips for
    `|x| >= 3`, so its input is range-capped there by default (see the
    module docstring).
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
        imax = _input_max_abs_value(_HARDSIGMOID_SATURATION, input_max_abs_value)
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
        return self.output_quant(_export_as(HardSigmoidFn, F.hardsigmoid, x))
