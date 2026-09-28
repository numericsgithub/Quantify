"""
Quantized Activation Functions for Quantify.

Provides quantized wrappers for the standard PyTorch activation functions
(ReLU, ReLU6, Sigmoid, Tanh, SiLU, GELU, LeakyReLU, Softmax). Every wrapper
applies the activation function, then quantizes the output with an
`act_quant` injector (fixed-point by default, via
`FixedPointPerTensorActivationQuant`).

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from torch.autograd import Function

from quantizers.fixedpoint_per_tensor import FixedPointPerTensorActivationQuant


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
        return _grad_via_recompute(lambda t: torch.clamp(t, min=0.0, max=6.0), x, grad_output)


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
        return g.op("Quantify::GELU", x, approximate_s=str(approximate_val)).setType(x.type())

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
    """ReLU with a fixed-point (by default) output quantizer.

    Exports as a single native `Relu` ONNX node followed by the quantizer's
    node (`Quantify::FixedPointQuant` by default).
    """

    def __init__(self, act_quant=FixedPointPerTensorActivationQuant, **kwargs):
        super().__init__(act_quant=act_quant, **kwargs)


class QuantSigmoid(qnn.QuantSigmoid):
    """Sigmoid with a fixed-point (by default) output quantizer.

    Exports as a single native `Sigmoid` ONNX node followed by the
    quantizer's node.
    """

    def __init__(self, act_quant=FixedPointPerTensorActivationQuant, **kwargs):
        super().__init__(act_quant=act_quant, **kwargs)


class QuantTanh(qnn.QuantTanh):
    """Tanh with a fixed-point (by default) output quantizer.

    Exports as a single native `Tanh` ONNX node followed by the quantizer's
    node.
    """

    def __init__(self, act_quant=FixedPointPerTensorActivationQuant, **kwargs):
        super().__init__(act_quant=act_quant, **kwargs)


class QuantLeakyReLU(nn.Module):
    """LeakyReLU with a fixed-point (by default) output quantizer.

    Exports as a single native `LeakyRelu` ONNX node followed by the
    quantizer's node. Brevitas has no built-in `QuantLeakyReLU`, so this
    applies the nonlinearity directly and quantizes the result with
    `qnn.QuantIdentity`.
    """

    def __init__(
        self,
        negative_slope: float = 0.01,
        act_quant=FixedPointPerTensorActivationQuant,
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.negative_slope = negative_slope
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=return_quant_tensor, **kwargs
        )

    def forward(self, x):
        return self.output_quant(F.leaky_relu(x, negative_slope=self.negative_slope))


class QuantSoftmax(nn.Module):
    """Softmax with a fixed-point (by default) output quantizer.

    Exports as a single native `Softmax` ONNX node followed by the
    quantizer's node. Brevitas has no built-in `QuantSoftmax`, so this
    applies the nonlinearity directly and quantizes the result with
    `qnn.QuantIdentity`.
    """

    def __init__(
        self,
        dim: int = -1,
        act_quant=FixedPointPerTensorActivationQuant,
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=return_quant_tensor, **kwargs
        )

    def forward(self, x):
        return self.output_quant(F.softmax(x, dim=self.dim))


class QuantReLU6(nn.Module):
    """ReLU6 with a fixed-point (by default) output quantizer.

    PyTorch's own `nn.ReLU6` decomposes into `Constant, Constant, Clip` on
    export, so this uses `Relu6Fn` (a custom `torch.autograd.Function`) to
    force a single `Quantify::Relu6` ONNX node, followed by the quantizer's
    node.
    """

    def __init__(
        self,
        act_quant=FixedPointPerTensorActivationQuant,
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=return_quant_tensor, **kwargs
        )

    def forward(self, x):
        return self.output_quant(Relu6Fn.apply(x))


class QuantSiLU(nn.Module):
    """SiLU (Swish) with a fixed-point (by default) output quantizer.

    `nn.SiLU`/`F.silu` decomposes into `Sigmoid, Mul` on export, so this uses
    `SiLUFn` (a custom `torch.autograd.Function`) to force a single
    `Quantify::SiLU` ONNX node, followed by the quantizer's node.

    See the module docstring for how this differs from
    `quantizers.silu_quant.QuantSiLUActivationQuant`.
    """

    def __init__(
        self,
        act_quant=FixedPointPerTensorActivationQuant,
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=return_quant_tensor, **kwargs
        )

    def forward(self, x):
        return self.output_quant(SiLUFn.apply(x))


class QuantGELU(nn.Module):
    """GELU with a fixed-point (by default) output quantizer.

    `nn.GELU`/`F.gelu` decomposes into a long Erf- (or Tanh-)based chain on
    export, so this uses `GELUFn` (a custom `torch.autograd.Function`) to
    force a single `Quantify::GELU` ONNX node, followed by the quantizer's
    node.
    """

    def __init__(
        self,
        approximate: str = "none",
        act_quant=FixedPointPerTensorActivationQuant,
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.approximate = approximate
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=return_quant_tensor, **kwargs
        )

    def forward(self, x):
        return self.output_quant(GELUFn.apply(x, self.approximate))
