"""
Coefficient Per-Tensor Weight Quantizer for Brevitas.

This quantizer rounds weights to the nearest value from a set of predefined 
coefficients provided in a text file. It searches for the optimal coefficient 
set and a power-of-two scaling factor (2^n) that minimizes the Sum of 
Absolute Differences (SAD) between the original and quantized weights.

The text file format:
    Each line contains one set of coefficients.
    Coefficients within a set are separated by spaces.

Example:
    -1.0 0.0 1.0
    -0.5 -0.25 0.0 0.25 0.5
"""

import torch
import torch.nn as nn
from typing import Tuple, Any

from quantizers.base_injector import BaseWeightQuant
from quantizers.base_quantizer import BaseQuantizer
from torch.autograd import Function
from torch.onnx import symbolic_helper
from collections import deque

def apply_non_uniform_quantization(weights, coefficients, bit_shift_scale):
    scale = 2.0 ** bit_shift_scale
    scaled_coeffs = coefficients * scale

    diffs = torch.abs(weights.unsqueeze(-1) - scaled_coeffs)
    min_indices = torch.argmin(diffs, dim=-1)
    quantized = scaled_coeffs[min_indices]
    return quantized, scale, min_indices


class CoefficientQuantFn(Function):

    _queue: deque = deque()

    @staticmethod
    def symbolic(g, x, coefficients, bit_shift_scale, bit_width):
        captured_indices, captured_quantized = CoefficientQuantFn._queue.popleft()

        quantized = g.op(
            "Quantify::CoefficientQuant",
            x,
            coefficients,
            bit_shift_scale_i=int(bit_shift_scale),
            chosen_indices_t=captured_indices,
            quantized_values_t=captured_quantized,
        ).setType(x.type())

        scale = g.op("Constant", value_t=torch.tensor(2.0 ** bit_shift_scale))
        zero_point = g.op("Constant", value_t=torch.tensor(0.0))
        bw = g.op("Constant", value_t=torch.tensor(float(bit_width)))
        return quantized, scale, zero_point, bw

    @staticmethod
    def forward(ctx, x, coefficients, bit_shift_scale, bit_width):
        ctx.save_for_backward(x)
        quantized, scale, _ = apply_non_uniform_quantization(x, coefficients, bit_shift_scale)

        if torch.onnx.is_in_onnx_export():
            with torch.no_grad():
                _, __, indices = apply_non_uniform_quantization(x, coefficients, bit_shift_scale)
                CoefficientQuantFn._queue.append((
                    indices.cpu().to(torch.long),
                    quantized.cpu(),
                ))

        bw = torch.tensor(float(bit_width), dtype=x.dtype, device=x.device)
        return (
            quantized,
            torch.tensor(scale, dtype=x.dtype, device=x.device),
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
            bw,
        )

    @staticmethod
    def backward(ctx, grad_quantized, grad_scale, grad_zero_point, grad_bw):
        return grad_quantized, None, None, None

    @classmethod
    def reset_capture_state(cls):
        cls._queue.clear()

class CoefficientPerTensorWeightQuantizer(BaseQuantizer):
    """
    A self-contained coefficient-based per-tensor weight quantizer.
    Inherits infrastructure from BaseQuantizer (gating, calibration state, ONNX guards).
    """

    def __init__(self, filepath: str, bit_width: int = 8, clipped_ste: bool = False):
        # clipped_ste is stored by BaseQuantizer (shared by all quantizers).
        super().__init__(bit_width=bit_width, clipped_ste=clipped_ste)
        self.filepath = filepath
        
        # Read coefficient sets from the text file during initialization
        self.coefficient_sets = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    coeffs = torch.tensor([float(x) for x in line.split()], dtype=torch.float32)
                    self.coefficient_sets.append(coeffs)

        if not self.coefficient_sets:
            raise ValueError(f"No valid coefficient sets found in file: {filepath}")

        # Register search results as buffers (handled by base class for state-dict)
        self.register_buffer('best_set_idx', torch.tensor(0, dtype=torch.long))
        self.register_buffer('best_bit_shift_scale', torch.tensor(0, dtype=torch.long))

    def _calibrate(self, x: torch.Tensor) -> Any:
        """Run calibration/search logic and return a params dict."""
        device = x.device
        best_sad = float("inf")
        best_set_idx = 0
        best_bit_shift_scale = 0

        for idx, coeffs in enumerate(self.coefficient_sets):
            coeffs_dev = coeffs.to(device)
            for bit_shift_scale in range(-12, 13):
                quantized_temp, scale, _ = apply_non_uniform_quantization(x, coeffs_dev, bit_shift_scale)
                sad = torch.sum(torch.abs(x - quantized_temp)).item()
                
                if sad < best_sad:
                    best_sad = sad
                    best_set_idx = idx
                    best_bit_shift_scale = bit_shift_scale

        return {'set_idx': best_set_idx, 'bit_shift_scale': best_bit_shift_scale}

    def _save_calibration(self, params: Any) -> None:
        """Save calibration results to buffers."""
        self.best_set_idx.fill_(params['set_idx'])
        self.best_bit_shift_scale.fill_(params['bit_shift_scale'])
        self.search_done.fill_(True)

    def _load_calibration(self) -> Any:
        """Load calibration results from buffers."""
        return {
            'set_idx': self.best_set_idx.item(),
            'bit_shift_scale': self.best_bit_shift_scale.item()
        }

    def _quantize(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        chosen_coeffs = self.coefficient_sets[params['set_idx']].to(x.device)
        bit_shift_scale = params['bit_shift_scale']

        # Route through CoefficientQuantFn in ALL paths, not just ONNX export.
        # apply_non_uniform_quantization() alone returns scaled_coeffs[indices],
        # a tensor derived from the (constant) coefficient set and detached from
        # x -> no gradient reaches the weights during training. CoefficientQuantFn
        # supplies the straight-through-estimator backward (and handles the
        # ONNX-export integer capture internally when exporting). Clipped STE, if
        # enabled, is layered on top in BaseQuantizer.forward via _in_range_mask.
        quantized, _, _, _ = CoefficientQuantFn.apply(
            x,
            chosen_coeffs,
            bit_shift_scale,
            len(self.coefficient_sets[params['set_idx']]),
        )
        return quantized

    def _get_metadata(self, params: Any, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scale, zero_point, and bit_width tensors matching x's dtype/device."""
        scale = torch.tensor(2.0 ** params['bit_shift_scale'], dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        # Bit width corresponds to the number of coefficients in the chosen set
        bit_width = torch.tensor(float(len(self.coefficient_sets[params['set_idx']])), dtype=x.dtype, device=x.device)
        return scale, zero_point, bit_width

    def _in_range_mask(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        """Clipped-STE support: True where x lands inside the representable
        span of the chosen (scaled) coefficient set, False where it saturated.

        Nearest-coefficient quantization maps any input beyond the outermost
        scaled coefficients onto that extreme coefficient — the analog of a
        clamp. So the in-range span is [min(scaled_coeffs), max(scaled_coeffs)],
        inclusive at both ends (a weight sitting exactly on the outer
        coefficient still receives gradient). Inputs strictly outside are
        zeroed.
        """
        scaled = self.coefficient_sets[params['set_idx']].to(x.device) * (2.0 ** params['bit_shift_scale'])
        lower = scaled.min()
        upper = scaled.max()
        return (x >= lower) & (x <= upper)


class CoefficientPerTensorWeightQuant(BaseWeightQuant):
    """
    Brevitas-compatible Injector for the coefficient-based per-tensor weight quantizer.
    """
    tensor_quant = CoefficientPerTensorWeightQuantizer
    filepath = "coefficients.txt"
    # signed inherited from BaseWeightQuant (True)
