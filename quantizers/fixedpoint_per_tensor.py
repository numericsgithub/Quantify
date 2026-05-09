"""
Fixed-Point Per-Tensor Weight Quantizer for Brevitas.

This quantizer represents weights using fixed-point arithmetic with configurable
bit-width and rounding mode. The MSB/LSB positions and signed/unsigned mode
are automatically determined from the weight tensor to maximize the number of
unique representable values while minimizing quantization error.

Fixed-point representation:
    Given msb and lsb (both integers), the step size is 2^lsb.
    For unsigned with bit_width bits: representable values are
        k * 2^lsb  for k in [0, 2^bit_width - 1]
    For signed (two's complement) with bit_width bits:
        k * 2^lsb  for k in [-2^(bit_width-1), 2^(bit_width-1) - 1]

    The relationship: msb = lsb + bit_width - 1 (unsigned)
                      msb = lsb + bit_width - 1 (signed, MSB is sign bit)

Example (unsigned, bit_width=3, lsb=-1):
    step = 0.5, codes 0..7, values: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5

Example (signed, bit_width=4, lsb=-1):
    step = 0.5, codes -8..7, values: -4.0, -3.5, ..., 3.0, 3.5
    (narrow range excludes -4.0)
"""

import math
from enum import Enum
from typing import Tuple, Optional, Any

import torch
import torch.nn as nn

from quantizers.base_injector import BaseWeightQuant, BaseActivationQuant
from torch.autograd import Function
from torch.onnx import symbolic_helper
from quantizers.base_quantizer import BaseQuantizer

# ---------------------------------------------------------------------------
# Core fixed-point quantization
# ---------------------------------------------------------------------------

def quantize_fixed_point(
    inputs: torch.Tensor,
    lsb: int,
    bit_width: int,
    signed: bool,
    rounding_mode: "RoundingMode",
    narrow_range: bool = True,
) -> torch.Tensor:
    """
    Quantize a input tensor to a fixed-point grid.

    Parameters
    ----------
    inputs : torch.Tensor
        The floating-point input tensor.
    lsb : int
        Position of the least-significant bit (can be negative for fractional steps).
    bit_width : int
        Total number of bits (including sign bit for signed mode).
    signed : bool
        Whether to use signed two's-complement representation.
    rounding_mode : RoundingMode
        ROUND_TO_NEAREST_EVEN or FLOOR.
    narrow_range : bool
        If True and signed, exclude the most negative value (e.g. -4.0 for 4-bit)
        to make the range symmetric.  Default True.

    Returns
    -------
    torch.Tensor
        Quantized (dequantized) input tensor on the fixed-point grid.
    """
    step = 2.0 ** lsb

    if signed:
        code_min = -(2 ** (bit_width - 1))
        if narrow_range:
            code_min += 1  # exclude most-negative code
        code_max = 2 ** (bit_width - 1) - 1
    else:
        code_min = 0
        code_max = 2 ** bit_width - 1

    # Quantize: map to integer codes, round, clamp, scale back
    codes = inputs / step
    codes = _round(codes, rounding_mode)
    codes = torch.clamp(codes, code_min, code_max)
    quantized = codes * step

    return quantized


# ---------------------------------------------------------------------------
# Optimal LSB search
# ---------------------------------------------------------------------------

def find_optimal_lsb(
    inputs: torch.Tensor,
    bit_width: int,
    signed: bool,
    rounding_mode: "RoundingMode",
    narrow_range: bool = True,
) -> Tuple[int, int]:
    """
    Search over LSB positions to find the one that maximises the number of
    unique quantised values.  Ties are broken by smallest SAD (Sum of Absolute Differences).

    Parameters
    ----------
    inputs : torch.Tensor
        The floating-point input tensor.
    bit_width : int
        Total number of bits.
    signed : bool
        Whether to use signed representation.
    rounding_mode : RoundingMode
        Rounding mode for quantization.
    narrow_range : bool
        Whether to exclude the most-negative code in signed mode.

    Returns
    -------
    int
        The optimal LSB position.
    """
    w_min = inputs.min().item()
    w_max = inputs.max().item()
    abs_max = max(abs(w_min), abs(w_max))

    if abs_max == 0.0:
        return 0, 1  # all-zero tensor, LSB doesn't matter

    if signed:
        n_positive_codes = 2 ** (bit_width - 1) - 1
        if n_positive_codes <= 0:
            n_positive_codes = 1
    else:
        n_positive_codes = 2 ** bit_width - 1

    ideal_lsb = math.log2(abs_max / n_positive_codes) if n_positive_codes > 0 else 0
    search_lo = math.floor(ideal_lsb) - 12
    search_hi = math.ceil(ideal_lsb) + 12

    best_lsb = search_lo
    best_unique = -1
    best_sad = float("inf")

    for lsb in reversed(range(search_lo, search_hi + 1)):
        q = quantize_fixed_point(inputs, lsb, bit_width, signed, rounding_mode, narrow_range)
        n_unique = int(torch.unique(q).numel())
        sad = float(torch.sum(torch.abs(inputs - q)).item())

        if n_unique > best_unique or (n_unique == best_unique and sad < best_sad):
            best_lsb = lsb
            best_unique = n_unique
            best_sad = sad

    return best_lsb, best_unique


# ---------------------------------------------------------------------------
# Rounding helpers
# ---------------------------------------------------------------------------

class RoundingMode(Enum):
    ROUND_TO_NEAREST_EVEN = "round_to_nearest_even"
    FLOOR = "floor"


def _round(x: torch.Tensor, mode: RoundingMode) -> torch.Tensor:
    """Round tensor according to the selected rounding mode."""
    if mode is RoundingMode.FLOOR:
        return torch.floor(x)
    # PyTorch's torch.round uses "round half to even" (banker's rounding)
    return torch.round(x)


# ---------------------------------------------------------------------------
# ONNX Custom Node Shim
# ---------------------------------------------------------------------------

class FixedPointQuantFn(Function):
    """Symbolic shim: emits a single `mydomain::FixedPointQuant` ONNX node."""

    @staticmethod
    def symbolic(g, x, scale, zero_point, lsb, bit_width, signed, narrow_range, rounding_mode):
        # Extract scale and zero_point values to embed as attributes instead of separate constant nodes
        # During ONNX export, scale and zero_point are torch._C.Value objects, not tensors.
        scale_val = symbolic_helper._maybe_get_const(scale, "t")
        zero_point_val = symbolic_helper._maybe_get_const(zero_point, "t")
        
        quantized = g.op(
            "mydomain::FixedPointQuant",
            x,
            scale_f=scale_val,
            zero_point_f=zero_point_val,
            lsb_i=int(lsb),
            bit_width_i=int(bit_width),
            signed_i=int(signed),
            narrow_range_i=int(narrow_range),
            rounding_mode_s=str(rounding_mode.value),
        ).setType(x.type())
        
        # Brevitas expects a 4-tuple output; create bw constant
        bw = g.op("Constant", value_t=torch.tensor(float(bit_width)))
        return quantized, scale, zero_point, bw

    @staticmethod
    def forward(ctx, x, scale, zero_point, lsb, bit_width, signed, narrow_range, rounding_mode):
        ctx.save_for_backward(x)
        # Compute quantization for PyTorch inference/tracing
        quantized = quantize_fixed_point(x, int(lsb), int(bit_width), signed, rounding_mode, narrow_range)
        bw = torch.tensor(float(bit_width), dtype=x.dtype, device=x.device)
        return quantized, scale, zero_point, bw

    @staticmethod
    def backward(ctx, grad_quantized, grad_scale, grad_zero_point, grad_bw):
        # Straight-Through Estimator: pass gradient through for the first input
        return grad_quantized, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Torch Module — usable as a standalone quantizer
# ---------------------------------------------------------------------------

class FixedPointPerTensorQuantizer(BaseQuantizer):
    """
    A self-contained fixed-point per-tensor quantizer.

    Parameters
    ----------
    bit_width : int
        Number of bits for the quantized representation.
    signed : bool
        Whether to use signed two's-complement representation. Explicitly set
        to match Brevitas proxy expectations and avoid QuantTensor validity errors.
    rounding_mode : RoundingMode
        ROUND_TO_NEAREST_EVEN (default) or FLOOR.
    narrow_range : bool
        Exclude most-negative code in signed mode (default True).
    """

    def __init__(
        self,
        bit_width: int = 8,
        signed: bool = True,
        rounding_mode: RoundingMode = RoundingMode.ROUND_TO_NEAREST_EVEN,
        narrow_range: bool = True,
    ):
        super().__init__(bit_width=bit_width)
        self.signed = signed
        self.rounding_mode = rounding_mode
        self.narrow_range = narrow_range
        
        # Register search results as buffers to ensure they are serialized in state_dict
        self.register_buffer('search_result_is_signed', torch.tensor(signed, dtype=torch.bool))
        self.register_buffer('search_result_lsb', torch.tensor(0, dtype=torch.long))

    def _calibrate(self, x: torch.Tensor) -> Any:
        """Run calibration/search logic and return a params dict."""
        # Use explicit signed setting from injector/constructor to align with Brevitas proxy
        signed = self.signed
        lsb, num_unique = find_optimal_lsb(
            x,
            self.bit_width,
            signed,
            self.rounding_mode,
            self.narrow_range,
        )
        return {'lsb': lsb, 'signed': signed, 'num_unique': num_unique}

    def _save_calibration(self, params: Any) -> None:
        """Save calibration results to buffers."""
        self.search_result_is_signed.fill_(params['signed'])
        self.search_result_lsb.fill_(params['lsb'])
        if params['num_unique'] > 1:
            self.search_done.fill_(True)
        else:
            self.search_done.fill_(False)

    def _load_calibration(self) -> Any:
        """Load calibration results from buffers."""
        return {
            'lsb': self.search_result_lsb.item(),
            'signed': self.search_result_is_signed.item()
        }

    def _quantize(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        """Apply quantization using the provided parameters."""
        quantized, _, _, _ = FixedPointQuantFn.apply(
            x,
            torch.tensor(2.0 ** params['lsb'], dtype=x.dtype, device=x.device),
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
            params['lsb'],
            self.bit_width,
            params['signed'],
            self.narrow_range,
            self.rounding_mode
        )
        return quantized

    def _get_metadata(self, params: Any, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scale, zero_point, and bit_width tensors matching x's dtype/device."""
        scale = torch.tensor(2.0 ** params['lsb'], dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        bit_width = torch.tensor(float(self.bit_width), dtype=x.dtype, device=x.device)
        return scale, zero_point, bit_width

    def detect_signed(self, inputs: torch.Tensor) -> bool:
        """Return True if any input is negative. (Kept for backward compatibility / manual checks)"""
        return bool((inputs < 0).any().item())


# ---------------------------------------------------------------------------
# Brevitas Injector — plug into QuantLinear, QuantConv2d, etc.
# ---------------------------------------------------------------------------

class FixedPointPerTensorWeightQuant(BaseWeightQuant):
    """
    Brevitas-compatible Injector for the fixed-point per-tensor weight
    quantizer.

    Usage::

        from brevitas.nn import QuantLinear
        layer = QuantLinear(
            in_features=64,
            out_features=32,
            bias=True,
            weight_quant=FixedPointPerTensorWeightQuant,
        )

    Override class attributes to customise::

        class My4bitQuant(FixedPointPerTensorWeightQuant):
            bit_width = 4
            rounding_mode = RoundingMode.FLOOR
    """

    rounding_mode = RoundingMode.ROUND_TO_NEAREST_EVEN
    narrow_range = True
    signed = True  # Explicitly declared to match proxy expectation and avoid QuantTensor validity errors
    tensor_quant = FixedPointPerTensorQuantizer


# ------ Brevitas Injector ------
class FixedPointPerTensorActivationQuant(BaseActivationQuant):
    """
    Brevitas-compatible Injector for the fixed-point per-tensor activation
    quantizer.

    Usage::
        act = QuantReLU(act_quant=FixedPointPerTensorActivationQuant)
    """

    rounding_mode = RoundingMode.ROUND_TO_NEAREST_EVEN
    narrow_range = True
    signed = False  # Explicitly declared to match proxy expectation
    tensor_quant = FixedPointPerTensorQuantizer
