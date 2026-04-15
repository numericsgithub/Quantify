"""
Fixed-Point Per-Tensor Weight Quantizer for Brevitas.

This quantizer represents weights using fixed-point arithmetic with configurable
bit-width and rounding mode. The MSB/LSB positions and signed/unsigned mode
are automatically determined from the weight tensor to maximize the number
of unique representable values while minimizing quantization error.

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
from typing import Tuple, Optional

import torch
import torch.nn as nn

from brevitas.inject import BaseInjector as Injector
from brevitas.inject.enum import QuantType
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector


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
# Core fixed-point quantization
# ---------------------------------------------------------------------------

def quantize_fixed_point(
    weights: torch.Tensor,
    lsb: int,
    bit_width: int,
    signed: bool,
    rounding_mode: RoundingMode,
    narrow_range: bool = True,
) -> torch.Tensor:
    """
    Quantize a weight tensor to a fixed-point grid.

    Parameters
    ----------
    weights : torch.Tensor
        The floating-point weight tensor.
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
        Quantized (dequantized) weight tensor on the fixed-point grid.
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

    val_min = code_min * step
    val_max = code_max * step

    # Quantize: map to integer codes, round, clamp, scale back
    codes = weights / step
    codes = _round(codes, rounding_mode)
    codes = torch.clamp(codes, code_min, code_max)
    quantized = codes * step

    return quantized


# ---------------------------------------------------------------------------
# Optimal LSB search
# ---------------------------------------------------------------------------

def find_optimal_lsb(
    weights: torch.Tensor,
    bit_width: int,
    signed: bool,
    rounding_mode: RoundingMode,
    narrow_range: bool = True,
) -> int:
    """
    Search over LSB positions to find the one that maximises the number of
    unique quantised values.  Ties are broken by smallest MSE.

    Parameters
    ----------
    weights : torch.Tensor
        The floating-point weight tensor.
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
    w_min = weights.min().item()
    w_max = weights.max().item()
    abs_max = max(abs(w_min), abs(w_max))

    if abs_max == 0.0:
        return 0  # all-zero tensor, LSB doesn't matter

    # Determine the search range for LSB.
    # The step = 2^lsb must be small enough to resolve the weights but large
    # enough that the representable range covers them.
    #
    # Upper bound on LSB: the full range must at least cover abs_max.
    #   For unsigned: (2^bw - 1) * 2^lsb >= abs_max
    #                 lsb >= log2(abs_max / (2^bw - 1))
    #   (similar for signed)
    # We search a generous window around the "ideal" LSB.

    if signed:
        n_positive_codes = 2 ** (bit_width - 1) - 1
        if n_positive_codes <= 0:
            n_positive_codes = 1
    else:
        n_positive_codes = 2 ** bit_width - 1

    ideal_lsb = math.log2(abs_max / n_positive_codes) if n_positive_codes > 0 else 0
    search_lo = math.floor(ideal_lsb) - 4
    search_hi = math.ceil(ideal_lsb) + 4

    best_lsb = search_lo
    best_unique = -1
    best_mse = float("inf")

    for lsb in range(search_lo, search_hi + 1):
        q = quantize_fixed_point(weights, lsb, bit_width, signed, rounding_mode, narrow_range)
        n_unique = int(torch.unique(q).numel())
        mse = float(torch.mean((weights - q) ** 2).item())

        if n_unique > best_unique or (n_unique == best_unique and mse < best_mse):
            best_lsb = lsb
            best_unique = n_unique
            best_mse = mse

    return best_lsb


# ---------------------------------------------------------------------------
# Torch Module — usable as a standalone quantizer
# ---------------------------------------------------------------------------

class FixedPointPerTensorWeightQuantizer(nn.Module):
    """
    A self-contained fixed-point per-tensor weight quantizer.

    Usage::

        quantizer = FixedPointPerTensorWeightQuantizer(bit_width=4)
        q_weights, scale, zero_point, bw = quantizer(linear.weight)

    Parameters
    ----------
    bit_width : int
        Number of bits for the quantized representation.
    rounding_mode : RoundingMode
        ROUND_TO_NEAREST_EVEN (default) or FLOOR.
    narrow_range : bool
        Exclude most-negative code in signed mode (default True).
    """

    def __init__(
        self,
        bit_width: int = 8,
        rounding_mode: RoundingMode = RoundingMode.ROUND_TO_NEAREST_EVEN,
        narrow_range: bool = True,
    ):
        super().__init__()
        self.bit_width = bit_width
        self.rounding_mode = rounding_mode
        self.narrow_range = narrow_range

    # ---- public helpers --------------------------------------------------

    def detect_signed(self, weights: torch.Tensor) -> bool:
        """Return True if any weight is negative."""
        return bool((weights < 0).any().item())

    # ---- forward ---------------------------------------------------------

    def forward(
        self, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize *weights* and return a Brevitas-style 4-tuple.

        Returns
        -------
        quantized : torch.Tensor
            Weights snapped to the fixed-point grid (dequantized form).
        scale : torch.Tensor
            Scalar step size ``2 ** lsb``.
        zero_point : torch.Tensor
            Always 0 for this quantizer.
        bit_width : torch.Tensor
            The bit-width as a float tensor.
        """
        signed = self.detect_signed(weights)

        lsb = find_optimal_lsb(
            weights,
            self.bit_width,
            signed,
            self.rounding_mode,
            self.narrow_range,
        )

        quantized = quantize_fixed_point(
            weights, lsb, self.bit_width, signed, self.rounding_mode, self.narrow_range
        )

        step = 2.0 ** lsb
        scale = torch.tensor(step, dtype=weights.dtype, device=weights.device)
        zero_point = torch.tensor(0.0, dtype=weights.dtype, device=weights.device)
        bw = torch.tensor(float(self.bit_width), device=weights.device)

        return quantized, scale, zero_point, bw


# ---------------------------------------------------------------------------
# Brevitas Injector — plug into QuantLinear, QuantConv2d, etc.
# ---------------------------------------------------------------------------

class FixedPointPerTensorWeightQuant(Injector):
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

    quant_type = QuantType.INT
    proxy_class = WeightQuantProxyFromInjector
    bit_width = 8
    rounding_mode = RoundingMode.ROUND_TO_NEAREST_EVEN
    narrow_range = True
    tensor_quant = FixedPointPerTensorWeightQuantizer
    
    # For Brevitas compatibility, we need to ensure signed is a proper boolean value
    # The actual signedness is determined by the tensor_quant module at runtime
    # This is a workaround to avoid the None/property issue
    @property
    def signed(self):
        # Return a default boolean value to satisfy Brevitas requirements
        # The actual signedness is handled by the tensor_quant module
        return True
