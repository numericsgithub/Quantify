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

Calibration works like the fixed-point quantizer's: you give a total bit budget
`bit_width` (= wE + wF + 3, default 16) and `_calibrate` searches the split.
Every candidate `wE` in 2..min(10, bit_width - 4) (so `wF = bit_width - 3 - wE
>= 1`) is evaluated on the calibration tensor -- SAD of the quantized tensor
and the percentage of values whose rounded magnitude overflows. Candidates
overflowing more than the role's ceiling (`max_clip_pct`: weights 15%, biases
and activations 0%, unknown role unconstrained -- same ceilings as the
fixed-point quantizer) are rejected; among the rest the lowest SAD wins (ties
-> fewer exponent bits). If every candidate overflows too much, the
least-overflowing one wins. Pin `exponent_bits` and/or `fraction_bits` to
restrict or skip the search; with BOTH given the format is fixed (`bit_width`
is derived and ignored) and there is nothing to calibrate. There is no scale:
the returned 4-tuple is `(quantized, 1, 0, wE + wF + 3)`.

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
from typing import Any, Optional, Tuple, Union

import torch
from torch.autograd import Function

from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector

from quantizers.base_injector import BaseWeightQuant, BaseActivationQuant
from quantizers.base_quantizer import BaseQuantizer
from quantizers.fixedpoint_per_tensor import _MAX_CLIP_PCT_BY_ROLE as _FIXED_MAX_CLIP_PCT_BY_ROLE

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


def _overflow_mask(x: torch.Tensor, exponent_bits: int, fraction_bits: int) -> torch.Tensor:
    """True where the ROUNDED magnitude of `x` is out of range (the values
    `quantize_floating_point` saturates / sends to inf). NaN -> False."""
    y = quantize_floating_point(x.detach(), exponent_bits, fraction_bits, "inf")
    return y.abs() > flopoco_format_info(exponent_bits, fraction_bits)["max_value"]


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


# Overflow ceiling (percent of values) the search tolerates, per role. Weights /
# biases share the fixed-point quantizer's numbers; activations are
# coverage-first there, i.e. no clipping.
_MAX_CLIP_PCT_BY_ROLE = {**_FIXED_MAX_CLIP_PCT_BY_ROLE, "activation": 0.0}


class FloatingPointPerTensorQuantizer(BaseQuantizer):
    """
    A self-contained FloPoCo floating-point per-tensor quantizer with a
    calibration search over the exponent/fraction split (see module docstring).

    Parameters
    ----------
    bit_width : int
        Total FloPoCo width wE + wF + 3 (>= 6). Ignored when both
        `exponent_bits` and `fraction_bits` are given.
    exponent_bits, fraction_bits : Optional[int]
        Pin wE (2..10) and/or wF (>= 1). One pinned -> the other follows from
        `bit_width`; both pinned -> fixed format, no search.
    overflow : str
        "saturate" (default) or "inf" -- see the module docstring.
    quantizer_role : str
        "weight" / "bias" / "activation" / "unknown": selects the overflow
        ceiling of the search.
    max_clip_pct : Optional[float]
        Overrides the role's ceiling (percent of values allowed to overflow).
    clipped_ste : bool or str
        As for the fixed-point quantizer ("clip" default, "not_clip",
        "single_direction_clip"); "clipped" here means magnitude overflow.
    """

    def __init__(
        self,
        bit_width: int = 16,
        exponent_bits: Optional[int] = None,
        fraction_bits: Optional[int] = None,
        overflow: str = "saturate",
        quantizer_role: str = "unknown",
        max_clip_pct: Optional[float] = None,
        clipped_ste: Union[bool, str] = "clip",
    ):
        if overflow not in OVERFLOW_MODES:
            raise ValueError(f"overflow must be one of {OVERFLOW_MODES}, got {overflow!r}")
        if exponent_bits is not None and not 2 <= exponent_bits <= MAX_EXPONENT_BITS:
            raise ValueError(f"exponent_bits must be in [2, {MAX_EXPONENT_BITS}], got {exponent_bits}")
        if fraction_bits is not None and fraction_bits < 1:
            raise ValueError(f"fraction_bits must be >= 1, got {fraction_bits}")
        if exponent_bits is not None and fraction_bits is not None:
            bit_width = exponent_bits + fraction_bits + 3
        if bit_width < 6:
            raise ValueError(f"bit_width must be >= 6 (wE >= 2, wF >= 1, +3), got {bit_width}")

        # Candidate exponent widths; wF = bit_width - 3 - wE must stay >= 1.
        if exponent_bits is not None:
            candidates = [exponent_bits]
        elif fraction_bits is not None:
            candidates = [bit_width - 3 - fraction_bits]
        else:
            candidates = list(range(2, min(MAX_EXPONENT_BITS, bit_width - 4) + 1))
        candidates = [c for c in candidates if 2 <= c <= MAX_EXPONENT_BITS and bit_width - 3 - c >= 1]
        if not candidates:
            raise ValueError(
                f"no valid (exponent_bits, fraction_bits) for bit_width={bit_width}, "
                f"exponent_bits={exponent_bits}, fraction_bits={fraction_bits}")

        super().__init__(bit_width=bit_width, clipped_ste=clipped_ste)
        self.overflow = overflow
        self.quantizer_role = quantizer_role
        self.max_clip_pct = (max_clip_pct if max_clip_pct is not None
                             else _MAX_CLIP_PCT_BY_ROLE.get(quantizer_role))
        self.candidate_exponent_bits = candidates

        # Search result. Default: the candidate closest to IEEE-ish 5.
        default_we = min(candidates, key=lambda c: abs(c - 5))
        self.register_buffer("search_result_exponent_bits", torch.tensor(default_we, dtype=torch.long))
        # Python-side mirror -- see BaseQuantizer._cached_scalar.
        self._exponent_bits_cached: int = default_we
        self._exponent_bits_version: int = -1
        # A single candidate means there is nothing to search.
        if len(candidates) == 1:
            self.set_search_done(True)

    @property
    def _exponent_bits_value(self) -> int:
        return self._cached_scalar(
            self.search_result_exponent_bits, "_exponent_bits_cached", "_exponent_bits_version", int)

    @property
    def exponent_bits(self) -> int:
        """wE of the current (calibrated, or pinned) format."""
        return self._exponent_bits_value

    @property
    def fraction_bits(self) -> int:
        return self.bit_width - 3 - self._exponent_bits_value

    @property
    def format_info(self) -> dict:
        return flopoco_format_info(self.exponent_bits, self.fraction_bits)

    def _calibrate(self, x: torch.Tensor) -> Any:
        """Pick the exponent width: lowest SAD among candidates whose overflow
        percentage is within `max_clip_pct`; if none is, the least-overflowing.
        Everything stays on-device (0-dim tensors, one stack/argmin) with a
        single `.item()` for the final pick."""
        if len(self.candidate_exponent_bits) == 1:
            return {"exponent_bits": self.candidate_exponent_bits[0]}
        xd = x.detach()
        finite = torch.isfinite(xd)
        if not bool(finite.all()):
            xd = torch.where(finite, xd, torch.zeros_like(xd))
        sads, clips = [], []
        for we in self.candidate_exponent_bits:
            wf = self.bit_width - 3 - we
            q = quantize_floating_point(xd, we, wf, "saturate")
            sads.append((xd - q).abs().sum().to(torch.float64))
            clips.append(_overflow_mask(xd, we, wf).to(torch.float64).mean() * 100.0)
        sad, clip = torch.stack(sads), torch.stack(clips)
        if self.max_clip_pct is None:
            best = sad.argmin()
        else:
            allowed = clip <= self.max_clip_pct
            best_allowed = torch.where(allowed, sad, torch.full_like(sad, float("inf"))).argmin()
            best = torch.where(allowed.any(), best_allowed, clip.argmin())
        return {"exponent_bits": self.candidate_exponent_bits[int(best.item())]}

    def _save_calibration(self, params: Any) -> None:
        self.search_result_exponent_bits.fill_(params["exponent_bits"])
        self._exponent_bits_cached = int(params["exponent_bits"])
        self._exponent_bits_version = self.search_result_exponent_bits._version
        self.set_search_done(True)

    def _load_calibration(self) -> Any:
        return {"exponent_bits": self._exponent_bits_value}

    def _formats(self, params: Any):
        we = int(params["exponent_bits"])
        return we, self.bit_width - 3 - we

    def _quantize(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        we, wf = self._formats(params)
        # The custom Function only during ONNX export (emits the custom node);
        # the lighter STE shim otherwise.
        if torch.onnx.is_in_onnx_export():
            return FloatingPointQuantFn.apply(x, we, wf, self.overflow)[0]
        return _STEQuantize.apply(x, we, wf, self.overflow)

    def _get_metadata(self, params: Any, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(1.0, dtype=x.dtype, device=x.device),
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
            torch.tensor(float(self.bit_width), dtype=x.dtype, device=x.device),
        )

    def _in_range_mask(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        return ~_overflow_mask(x, *self._formats(params))

    def _clip_side(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        over = _overflow_mask(x, *self._formats(params))
        return (over & (x > 0)).to(torch.int8) - (over & (x < 0)).to(torch.int8)


# ------ Brevitas injectors ------
# `bit_width` is the TOTAL FloPoCo width (wE + wF + 3) and is searched over;
# add `exponent_bits = ..` / `fraction_bits = ..` in a subclass to pin.

class FloatingPointPerTensorWeightQuant(BaseWeightQuant):
    """
    Brevitas-compatible Injector for the floating-point weight quantizer.

        layer = QuantLinear(64, 32, weight_quant=FloatingPointPerTensorWeightQuant)

    Customise via a subclass (`bit_width = 12`, or pin `exponent_bits = 4;
    fraction_bits = 3`).
    """
    bit_width = 16
    overflow = "saturate"
    signed = True
    tensor_quant = FloatingPointPerTensorQuantizer


class FloatingPointPerTensorActivationQuant(BaseActivationQuant):
    """Brevitas-compatible Injector for the floating-point activation quantizer."""
    bit_width = 16
    overflow = "saturate"
    signed = True  # a float format always carries a sign bit
    tensor_quant = FloatingPointPerTensorQuantizer


class FloatingPointPerTensorBiasQuant(BaseWeightQuant):
    """Brevitas-compatible Injector for the floating-point bias quantizer."""
    proxy_class = BiasQuantProxyFromInjector
    requires_input_scale = False
    bit_width = 16
    overflow = "saturate"
    signed = True
    tensor_quant = FloatingPointPerTensorQuantizer
    quantizer_role = "bias"
