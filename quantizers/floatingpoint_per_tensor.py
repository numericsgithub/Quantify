"""
FloPoCo floating-point per-tensor quantizer for Brevitas.

Models the FloPoCo `FPFloat(wE, wF)` number format (see ~/FLOPOCO_FP_FORMAT.md):
`wE + wF + 3` bits = 2 exception bits (zero / normal / inf / NaN), a sign bit,
`wE` exponent bits and `wF` fraction bits.

    value = (-1)^s * (1 + F / 2^wF) * 2^(E - bias),   bias = 2^(wE-1) - 1

Differences from IEEE-754 that matter here:
  * no subnormals: anything that rounds to below `2^-bias` is flushed to a
    (signed) zero;
  * no reserved exponent codes: the largest finite value is
    `(2 - 2^-wF) * 2^(2^(wE-1))`, the smallest positive one `2^-bias`.

Quantization is "round to nearest even to `wF + 1` significand bits with an
unbounded exponent, THEN range-check": underflow (|y| < 2^-bias) -> signed zero,
overflow (|y| >= 2^(emax+1)) -> infinity. Overflow happens after rounding, so
e.g. 31.5 in (wE=3, wF=4) ties up to 32 and overflows although 31.5 < 32.

Unlike the fixed-point quantizer there is nothing to calibrate -- the format
is fully given by `(wE, wF)` -- so `search_done` is simply set at construction
(and again on the first forward after a harness reset). There is no scale: the
returned 4-tuple is `(quantized, 1, 0, wE + wF + 3)`.

Overflow handling (`overflow`):
  "saturate" (default): clamp to +-max_value. Training-friendly (never produces
      inf/NaN) and consistent with the fixed-point quantizers' clipping.
  "inf": the hardware-exact behaviour, +-inf. Use it to reproduce FloPoCo
      results bit for bit.
The clipped STE (`clipped_ste="clip"`, the default) zeroes the gradient of
values whose rounded magnitude overflows; underflowing values keep slope 1 (a
flush to zero is not a saturation).

Rounding is computed in float64 internally (exact for any wF <= 52); the result
is cast back to the input dtype.
"""

import math
from typing import Any, Tuple, Union

import torch
from torch.autograd import Function

from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector

from quantizers.base_injector import BaseWeightQuant, BaseActivationQuant
from quantizers.base_quantizer import BaseQuantizer

# wE <= 10 keeps 2^(emax+1) finite in float64 (the overflow threshold of wE=11
# would be 2^1024).
MAX_EXPONENT_BITS = 10
OVERFLOW_MODES = ("saturate", "inf")


def flopoco_format_info(exponent_bits: int, fraction_bits: int) -> dict:
    """Derived constants of FPFloat(wE, wF)."""
    bias = (1 << (exponent_bits - 1)) - 1
    emax = (1 << exponent_bits) - 1 - bias  # == 2^(wE-1)
    return {
        "bias": bias,
        "emin": -bias,
        "emax": emax,
        "total_bits": exponent_bits + fraction_bits + 3,
        "min_value": 2.0 ** (-bias),
        "max_value": (2.0 - 2.0 ** (-fraction_bits)) * 2.0 ** emax,
    }


def quantize_floating_point(
    x: torch.Tensor, exponent_bits: int, fraction_bits: int, overflow: str = "saturate"
) -> torch.Tensor:
    """Round `x` onto the FloPoCo FPFloat(wE, wF) grid (see module docstring).

    Pure value math: gradients are zero through the rounding (use the
    quantizer module for a straight-through backward). NaN passes through;
    +-inf is treated as overflow.
    """
    if overflow not in OVERFLOW_MODES:
        raise ValueError(f"overflow must be one of {OVERFLOW_MODES}, got {overflow!r}")
    info = flopoco_format_info(exponent_bits, fraction_bits)
    xd = x.to(torch.float64)

    m, e = torch.frexp(xd)  # x = m * 2^e, |m| in [0.5, 1); inf/nan pass through
    shift = fraction_bits + 1
    # torch.round is half-to-even; the ldexp scalings are exact powers of two.
    y = torch.ldexp(torch.round(torch.ldexp(m, torch.full_like(e, shift))), e - shift)
    mag = y.abs()

    over_value = math.inf if overflow == "inf" else info["max_value"]
    y = torch.where(mag >= 2.0 ** (info["emax"] + 1),
                    torch.copysign(torch.full_like(y, over_value), xd), y)
    # Flush to signed zero AFTER rounding (rounding can lift a value to 2^emin).
    y = torch.where(mag < info["min_value"], torch.copysign(torch.zeros_like(y), xd), y)
    return y.to(x.dtype)


class _STEQuantize(Function):
    """quantize_floating_point with a plain straight-through backward."""

    @staticmethod
    def forward(ctx, x, exponent_bits, fraction_bits, overflow):
        return quantize_floating_point(x, exponent_bits, fraction_bits, overflow)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class FloatingPointQuantFn(Function):
    """Straight-through shim; emits a single `Quantify::FloatingPointQuant`
    ONNX node during export."""

    @staticmethod
    def symbolic(g, x, exponent_bits, fraction_bits, overflow):
        info = flopoco_format_info(int(exponent_bits), int(fraction_bits))
        # exponent_bias_i / emax_i / fraction_bits_i / overflow_inf_i are what the
        # embedded FunctionProto body consumes (ints, so the float64 powers of
        # two it builds are exact -- an ONNX float attribute is float32 and
        # could not hold 2^128). min_value_f / max_value_f / total_bits_i /
        # overflow_s are for graph readability only (Netron).
        quantized = g.op(
            "Quantify::FloatingPointQuant",
            x,
            exponent_bits_i=int(exponent_bits),
            fraction_bits_i=int(fraction_bits),
            exponent_bias_i=info["bias"],
            emax_i=info["emax"],
            overflow_inf_i=int(overflow == "inf"),
            overflow_s=str(overflow),
            total_bits_i=info["total_bits"],
            min_value_f=float(info["min_value"]),
            max_value_f=float(info["max_value"]),
        ).setType(x.type())
        scale = g.op("Constant", value_t=torch.tensor(1.0))
        zero_point = g.op("Constant", value_t=torch.tensor(0.0))
        bw = g.op("Constant", value_t=torch.tensor(float(info["total_bits"])))
        return quantized, scale, zero_point, bw

    @staticmethod
    def forward(ctx, x, exponent_bits, fraction_bits, overflow):
        quantized = quantize_floating_point(x, int(exponent_bits), int(fraction_bits), overflow)
        total = flopoco_format_info(int(exponent_bits), int(fraction_bits))["total_bits"]
        return (
            quantized,
            torch.tensor(1.0, dtype=x.dtype, device=x.device),
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
            torch.tensor(float(total), dtype=x.dtype, device=x.device),
        )

    @staticmethod
    def backward(ctx, grad_quantized, grad_scale, grad_zero_point, grad_bw):
        # Plain STE; clipped variants are layered on in BaseQuantizer.forward.
        return grad_quantized, None, None, None


class FloatingPointPerTensorQuantizer(BaseQuantizer):
    """
    A self-contained FloPoCo floating-point per-tensor quantizer.

    Parameters
    ----------
    exponent_bits : int
        wE, 2..10. Bias is 2^(wE-1)-1.
    fraction_bits : int
        wF, >= 1 (the significand has wF + 1 bits).
    overflow : str
        "saturate" (default) or "inf" -- see the module docstring.
    clipped_ste : bool or str
        As for the fixed-point quantizer ("clip" default, "not_clip",
        "single_direction_clip"); "clipped" here means magnitude overflow.
    """

    def __init__(
        self,
        exponent_bits: int = 5,
        fraction_bits: int = 10,
        overflow: str = "saturate",
        clipped_ste: Union[bool, str] = "clip",
    ):
        if not 2 <= exponent_bits <= MAX_EXPONENT_BITS:
            raise ValueError(f"exponent_bits must be in [2, {MAX_EXPONENT_BITS}], got {exponent_bits}")
        if fraction_bits < 1:
            raise ValueError(f"fraction_bits must be >= 1, got {fraction_bits}")
        if overflow not in OVERFLOW_MODES:
            raise ValueError(f"overflow must be one of {OVERFLOW_MODES}, got {overflow!r}")
        info = flopoco_format_info(exponent_bits, fraction_bits)
        super().__init__(bit_width=info["total_bits"], clipped_ste=clipped_ste)
        self.exponent_bits = exponent_bits
        self.fraction_bits = fraction_bits
        self.overflow = overflow
        self.format_info = info
        # Nothing to calibrate: the format is fixed, so the quantizer is
        # usable (also in eval mode / export) right away.
        self.set_search_done(True)

    def _calibrate(self, x: torch.Tensor) -> Any:
        return {}

    def _save_calibration(self, params: Any) -> None:
        self.set_search_done(True)

    def _load_calibration(self) -> Any:
        return {}

    def _quantize(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        # The custom Function only during ONNX export (emits the custom node);
        # the lighter STE shim otherwise.
        if torch.onnx.is_in_onnx_export():
            return FloatingPointQuantFn.apply(
                x, self.exponent_bits, self.fraction_bits, self.overflow)[0]
        return _STEQuantize.apply(x, self.exponent_bits, self.fraction_bits, self.overflow)

    def _get_metadata(self, params: Any, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(1.0, dtype=x.dtype, device=x.device),
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
            torch.tensor(float(self.bit_width), dtype=x.dtype, device=x.device),
        )

    def _overflows(self, x: torch.Tensor) -> torch.Tensor:
        """True where the ROUNDED magnitude of `x` is out of range (the values
        `quantize_floating_point` saturates / sends to inf). NaN -> False."""
        y = quantize_floating_point(x.detach(), self.exponent_bits, self.fraction_bits, "inf")
        return y.abs() > self.format_info["max_value"]

    def _in_range_mask(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        return ~self._overflows(x)

    def _clip_side(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        over = self._overflows(x)
        return (over & (x > 0)).to(torch.int8) - (over & (x < 0)).to(torch.int8)


# ------ Brevitas injectors ------

class FloatingPointPerTensorWeightQuant(BaseWeightQuant):
    """
    Brevitas-compatible Injector for the floating-point weight quantizer.

        layer = QuantLinear(64, 32, weight_quant=FloatingPointPerTensorWeightQuant)

    Customise via a subclass (`exponent_bits = 4; fraction_bits = 3`).
    """
    exponent_bits = 5
    fraction_bits = 10
    overflow = "saturate"
    signed = True
    tensor_quant = FloatingPointPerTensorQuantizer


class FloatingPointPerTensorActivationQuant(BaseActivationQuant):
    """Brevitas-compatible Injector for the floating-point activation quantizer."""
    exponent_bits = 5
    fraction_bits = 10
    overflow = "saturate"
    signed = True  # a float format always carries a sign bit
    tensor_quant = FloatingPointPerTensorQuantizer


class FloatingPointPerTensorBiasQuant(BaseWeightQuant):
    """Brevitas-compatible Injector for the floating-point bias quantizer."""
    proxy_class = BiasQuantProxyFromInjector
    requires_input_scale = False
    exponent_bits = 5
    fraction_bits = 10
    overflow = "saturate"
    signed = True
    tensor_quant = FloatingPointPerTensorQuantizer
    quantizer_role = "bias"
