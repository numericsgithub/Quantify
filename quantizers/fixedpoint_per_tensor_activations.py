"""
Fixed-Point Per-Tensor Activation Quantizer for Brevitas.

This quantizer represents activations using fixed-point arithmetic with configurable
bit-width and rounding mode. The MSB/LSB positions and signed/unsigned mode
are automatically determined from the activation tensor to maximize the number
of unique representable values while minimizing quantization error.

Fixed-point representation:
    Given msb and lsb (both integers), the step size is 2^lsb.
    For unsigned with bit_width bits: representable values are
        k * 2^lsb  for k in [0, 2^bit_width - 1]
    For signed (two's complement) with bit_width bits:
        k * 2^lsb  for k in [-2^(bit_width-1), 2^(bit_width-1) - 1]
"""

import math
from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn

from brevitas.inject import BaseInjector as Injector
from brevitas.inject.enum import QuantType

# Robust import for Brevitas activation quant proxy
try:
    from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector as ActivationQuantProxyFromInjector
except ImportError:
    try:
        from brevitas.proxy.activation_quant import ActivationQuantProxyFromInjector
    except ImportError:
        try:
            from brevitas.proxy.activation import ActivationQuantProxyFromInjector
        except ImportError:
            raise ImportError(
                "Could not find ActivationQuantProxyFromInjector. "
                "Please ensure you have a compatible version of Brevitas installed."
            )


# ------ Rounding helpers ------
class RoundingMode(Enum):
    ROUND_TO_NEAREST_EVEN = "round_to_nearest_even"
    FLOOR = "floor"


def _round(x: torch.Tensor, mode: RoundingMode) -> torch.Tensor:
    """Round tensor according to the selected rounding mode."""
    if mode is RoundingMode.FLOOR:
        return torch.floor(x)
    # PyTorch's torch.round uses "round half to even" (banker's rounding)
    return torch.round(x)


# ------ Core fixed-point quantization ------
def quantize_fixed_point(
    tensor: torch.Tensor,
    lsb: int,
    bit_width: int,
    signed: bool,
    rounding_mode: RoundingMode,
    narrow_range: bool = True,
) -> torch.Tensor:
    """
    Quantize a tensor to a fixed-point grid.

    Parameters
    ----------
    tensor : torch.Tensor
        The floating-point tensor (weights or activations).
    lsb : int
        Position of the least-significant bit.
    bit_width : int
        Total number of bits.
    signed : bool
        Whether to use signed two's-complement representation.
    rounding_mode : RoundingMode
        ROUND_TO_NEAREST_EVEN or FLOOR.
    narrow_range : bool
        If True and signed, exclude the most negative value.

    Returns
    -------
    torch.Tensor
        Quantized (dequantized) tensor on the fixed-point grid.
    """
    step = 2.0 ** lsb

    if signed:
        code_min = -(2 ** (bit_width - 1))
        if narrow_range:
            code_min += 1
        code_max = 2 ** (bit_width - 1) - 1
    else:
        code_min = 0
        code_max = 2 ** bit_width - 1

    # Quantize: map to integer codes, round, clamp, scale back
    codes = tensor / step
    codes = _round(codes, rounding_mode)
    codes = torch.clamp(codes, code_min, code_max)
    quantized = codes * step

    return quantized


# ------ Optimal LSB search ------
def find_optimal_lsb(
    tensor: torch.Tensor,
    bit_width: int,
    signed: bool,
    rounding_mode: RoundingMode,
    narrow_range: bool = True,
) -> int:
    """
    Search over LSB positions to find the one that maximises the number of
    unique quantised values. Ties are broken by smallest SAD.

    Parameters
    ----------
    tensor : torch.Tensor
        The floating-point tensor.
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
    t_min = tensor.min().item()
    t_max = tensor.max().item()
    abs_max = max(abs(t_min), abs(t_max))

    if abs_max == 0.0:
        return 0, 1

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
        q = quantize_fixed_point(tensor, lsb, bit_width, signed, rounding_mode, narrow_range)
        n_unique = int(torch.unique(q).numel())
        sad = float(torch.sum(torch.abs(tensor - q)).item())

        if n_unique > best_unique or (n_unique == best_unique and sad < best_sad):
            best_lsb = lsb
            best_unique = n_unique
            best_sad = sad

    return best_lsb, best_unique


# ------ Torch Module ------
class FixedPointPerTensorActivationQuantizer(nn.Module):
    """
    A self-contained fixed-point per-tensor activation quantizer.

    Usage::

        quantizer = FixedPointPerTensorActivationQuantizer(bit_width=4)
        q_act, scale, zero_point, bw = quantizer(input_tensor)
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

        # Register search results as buffers to ensure they are serialized in state_dict
        self.register_buffer('search_done', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('search_result_is_signed', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('search_result_lsb', torch.tensor(0, dtype=torch.long))

    def detect_signed(self, tensor: torch.Tensor) -> bool:
        """Return True if any value is negative."""
        return bool((tensor < 0).any().item())

    def forward(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize *tensor* and return a Brevitas-style 4-tuple.

        Returns
        -------
        quantized : torch.Tensor
            Tensor snapped to the fixed-point grid.
        scale : torch.Tensor
            Scalar step size ``2 ** lsb``.
        zero_point : torch.Tensor
            Always 0 for this quantizer.
        bit_width : torch.Tensor
            The bit-width as a float tensor.
        """
        if self.search_done.device != tensor.device:
            self.search_done = self.search_done.to(tensor.device)
            self.search_result_is_signed = self.search_result_is_signed.to(tensor.device)
            self.search_result_lsb = self.search_result_lsb.to(tensor.device)

        if not self.search_done.item():
            signed = self.detect_signed(tensor)
            lsb, num_unique = find_optimal_lsb(
                tensor,
                self.bit_width,
                signed,
                self.rounding_mode,
                self.narrow_range,
            )
            self.search_result_is_signed.fill_(signed)
            self.search_result_lsb.fill_(lsb)
            if num_unique > 1:
                self.search_done.fill_(True)
            else:
                self.search_done.fill_(False)
        else:
            signed = self.search_result_is_signed.item()
            lsb = self.search_result_lsb.item()

        quantized = quantize_fixed_point(
            tensor, lsb, self.bit_width, signed, self.rounding_mode, self.narrow_range
        )

        step = 2.0 ** lsb
        scale = torch.tensor(step, dtype=tensor.dtype, device=tensor.device)
        zero_point = torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)
        bw = torch.tensor(float(self.bit_width), device=tensor.device)

        return quantized, scale, zero_point, bw


# ------ Brevitas Injector ------
class FixedPointPerTensorActivationQuant(Injector):
    """
    Brevitas-compatible Injector for the fixed-point per-tensor activation
    quantizer.

    Usage::

        from brevitas.nn import QuantReLU
        act = QuantReLU(act_quant=FixedPointPerTensorActivationQuant)
    """

    quant_type = QuantType.INT
    proxy_class = ActivationQuantProxyFromInjector
    bit_width = 8
    rounding_mode = RoundingMode.ROUND_TO_NEAREST_EVEN
    narrow_range = True
    tensor_quant = FixedPointPerTensorActivationQuantizer
    signed = False  # Activations are often unsigned, but detection handles it dynamically
