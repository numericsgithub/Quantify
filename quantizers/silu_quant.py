"""
Quantized SiLU Activation for Brevitas.

Provides a custom quantized SiLU module that applies SiLU followed by
fixed-point quantization. Emits a custom ONNX node (`Quantify::QuantSiLU`)
during export to preserve exact quantization semantics.
"""

import torch
import torch.nn as nn
from typing import Tuple, Any

from quantizers.base_quantizer import BaseQuantizer
from quantizers.base_injector import BaseActivationQuant
from quantizers.fixedpoint_per_tensor import quantize_fixed_point, find_optimal_lsb, RoundingMode
from torch.autograd import Function
from torch.onnx import symbolic_helper


class SiLUQuantFn(Function):
    """Symbolic shim: emits a single `Quantify::QuantSiLU` ONNX node."""

    @staticmethod
    def symbolic(g, x, scale, zero_point, lsb, bit_width, signed, rounding_mode):
        scale_val = symbolic_helper._maybe_get_const(scale, "t")
        zero_point_val = symbolic_helper._maybe_get_const(zero_point, "t")
        
        quantized = g.op(
            "Quantify::QuantSiLU",
            x,
            scale_f=scale_val,
            zero_point_f=zero_point_val,
            lsb_i=int(lsb),
            bit_width_i=int(bit_width),
            signed_i=int(signed),
            rounding_mode_s=str(rounding_mode.value),
        ).setType(x.type())
        
        return quantized

    @staticmethod
    def forward(ctx, x, scale, zero_point, lsb, bit_width, signed, rounding_mode):
        ctx.save_for_backward(x)
        # Apply SiLU then quantize
        x_silu = torch.nn.functional.silu(x)
        quantized = quantize_fixed_point(x_silu, int(lsb), int(bit_width), signed, rounding_mode)
        return quantized

    @staticmethod
    def backward(ctx, grad_quantized, grad_scale, grad_zero_point, grad_lsb, grad_bit_width, grad_signed, grad_rounding_mode):
        # Straight-Through Estimator: pass gradient through for the first input
        return grad_quantized, None, None, None, None, None, None


class SiLUTensorQuant(BaseQuantizer):
    """
    Tensor quantizer for SiLU activation.
    Applies SiLU, then searches for the optimal fixed-point grid scale.
    """

    def __init__(
        self,
        bit_width: int = 8,
        signed: bool = False,
        rounding_mode: RoundingMode = RoundingMode.ROUND_TO_NEAREST_EVEN,
    ):
        super().__init__(bit_width=bit_width)
        self.signed = signed
        self.rounding_mode = rounding_mode
        self.register_buffer('search_result_lsb', torch.tensor(0, dtype=torch.long))

    def _calibrate(self, x: torch.Tensor) -> Any:
        """Calibrate by finding the optimal LSB for the SiLU output."""
        x_silu = torch.nn.functional.silu(x)
        lsb, _ = find_optimal_lsb(
            x_silu, self.bit_width, self.signed, self.rounding_mode
        )
        return {'lsb': lsb}

    def _save_calibration(self, params: Any) -> None:
        """Save calibration results to buffers."""
        self.search_result_lsb.fill_(params['lsb'])
        self.search_done.fill_(True)

    def _load_calibration(self) -> Any:
        """Load calibration results from buffers."""
        return {'lsb': self.search_result_lsb.item()}

    def _quantize(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        """Apply SiLU and quantization."""
        if torch.onnx.is_in_onnx_export():
            scale = torch.tensor(2.0 ** params['lsb'], dtype=x.dtype, device=x.device)
            zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            quantized = SiLUQuantFn.apply(
                x, scale, zero_point, params['lsb'], self.bit_width, self.signed, self.rounding_mode
            )
            return quantized
            
        x_silu = torch.nn.functional.silu(x)
        return quantize_fixed_point(
            x_silu, int(params['lsb']), self.bit_width, self.signed, self.rounding_mode
        )

    def _get_metadata(self, params: Any, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scale, zero_point, and bit_width tensors."""
        scale = torch.tensor(2.0 ** params['lsb'], dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        bit_width = torch.tensor(float(self.bit_width), dtype=x.dtype, device=x.device)
        return scale, zero_point, bit_width


class QuantSiLUActivationQuant(BaseActivationQuant):
    """
    Brevitas-compatible Injector for the quantized SiLU activation.
    
    Usage::
        from brevitas.nn import QuantConv2d
        layer = QuantConv2d(3, 16, 3, act_quant=QuantSiLUActivationQuant)
    """
    rounding_mode = RoundingMode.ROUND_TO_NEAREST_EVEN
    tensor_quant = SiLUTensorQuant
