"""
Coefficient Per-Tensor Weight Quantizer for Brevitas.

This quantizer rounds weights to the nearest value from a set of predefined 
coefficients provided in a text file. It searches for the optimal coefficient 
set and a power-of-two scaling factor (2^n) that minimizes the Sum of 
Absolute Differences (SAD) between the original and quantized weights.

Calibration, per coefficient set: the shift that just covers the weights'
largest magnitude is computed directly from the weight range, and only the
`SHIFT_SEARCH_RADIUS` shifts on either side of it are tried. Candidates that
clip more than `max_clip_pct` percent of the weights are rejected (the same
ceiling the fixed-point weight quantizer uses); if every candidate clips more
than that, the least-clipping one wins.

The text file format:
    Each line contains one set of coefficients.
    Coefficients within a set are separated by spaces.

Example:
    -1.0 0.0 1.0
    -0.5 -0.25 0.0 0.25 0.5
"""

import math
from collections import deque
from typing import Any, Tuple, Union

import torch
from torch.autograd import Function

from quantizers.base_injector import BaseWeightQuant
from quantizers.base_quantizer import BaseQuantizer

# Weight clipping ceiling, shared with the fixed-point weight quantizer.
from quantizers.fixedpoint_per_tensor import MAX_CLIP_PCT_BY_ROLE

# Calibration tries the shift that just covers the weight range, plus this many
# shifts to either side (2 * radius + 1 candidates per coefficient set).
SHIFT_SEARCH_RADIUS = 6

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
    def symbolic(g, x, coefficients, bit_shift_scale):
        (captured_indices, captured_quantized, captured_coefficients,
         num_coefficients, min_value, max_value) = CoefficientQuantFn._queue.popleft()

        # The coefficient set is an ATTRIBUTE of the node (coefficients_t), not
        # a graph input: it is part of the quantizer, like chosen_indices_t.
        # The traced `coefficients` Value is deliberately left unused, so no
        # stray Constant node survives export.
        quantized = g.op(
            "Quantify::CoefficientQuant",
            x,
            coefficients_t=captured_coefficients,
            bit_shift_scale_i=int(bit_shift_scale),
            # scale_f: redundant precomputed float (== 2**bit_shift_scale),
            # so the embedded ONNX FunctionProto body
            # (utils/onnx_self_contained.py) can reference it directly
            # instead of computing a power-of-two from the int attribute
            # in-graph.
            scale_f=float(2.0 ** bit_shift_scale),
            # num_coefficients_i / min_value_f / max_value_f: the size of the
            # chosen set and the representable span in the tensor's own units
            # (min/max of coefficients * 2**bit_shift_scale) -- graph
            # readability only (e.g. Netron), like FixedPointQuant's
            # min_value_f/max_value_f. Not consumed by the embedded
            # FunctionProto body (utils/onnx_self_contained.py).
            num_coefficients_i=num_coefficients,
            min_value_f=min_value,
            max_value_f=max_value,
            chosen_indices_t=captured_indices,
            quantized_values_t=captured_quantized,
        ).setType(x.type())

        scale = g.op("Constant", value_t=torch.tensor(2.0 ** bit_shift_scale))
        zero_point = g.op("Constant", value_t=torch.tensor(0.0))
        # Brevitas expects a 4-tuple; its "bit width" slot carries the number
        # of coefficients.
        bw = g.op("Constant", value_t=torch.tensor(float(num_coefficients)))
        return quantized, scale, zero_point, bw

    @staticmethod
    def forward(ctx, x, coefficients, bit_shift_scale):
        ctx.save_for_backward(x)
        quantized, scale, _ = apply_non_uniform_quantization(x, coefficients, bit_shift_scale)

        if torch.onnx.is_in_onnx_export():
            with torch.no_grad():
                _, __, indices = apply_non_uniform_quantization(x, coefficients, bit_shift_scale)
                span_scale = 2.0 ** bit_shift_scale
                CoefficientQuantFn._queue.append((
                    indices.cpu().to(torch.long),
                    # .detach() -- this cached copy is embedded into the
                    # exported node as a plain tensor attribute purely for
                    # introspection (symbolic()'s quantized_values_t); it
                    # must never carry grad-tracking (x is typically a
                    # Parameter with requires_grad=True, e.g. a conv
                    # weight), or embedding it as a g.op() attribute hits
                    # PyTorch's "!v.requires_grad()" internal assert. The
                    # REAL output (`quantized`, returned below, used for the
                    # STE backward) is untouched -- only this cached copy
                    # is detached.
                    quantized.detach().cpu(),
                    coefficients.detach().cpu(),
                    int(coefficients.numel()),
                    float(coefficients.min()) * span_scale,
                    float(coefficients.max()) * span_scale,
                ))

        bw = torch.tensor(float(coefficients.numel()), dtype=x.dtype, device=x.device)
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

    def __init__(
        self,
        filepath: str,
        clipped_ste: Union[bool, str] = "clip",
        max_clip_pct: float = MAX_CLIP_PCT_BY_ROLE["weight"],
    ):
        # There is no bit_width: the representable values are given by the
        # coefficient sets, whose sizes differ. (BaseQuantizer still stores
        # one for its pass-through return; the largest set's size is used.)
        # Read coefficient sets from the text file during initialization
        coefficient_sets = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    coeffs = torch.tensor([float(x) for x in line.split()], dtype=torch.float32)
                    coefficient_sets.append(coeffs)

        if not coefficient_sets:
            raise ValueError(f"No valid coefficient sets found in file: {filepath}")

        # clipped_ste is stored by BaseQuantizer (shared by all quantizers).
        super().__init__(
            bit_width=max(len(c) for c in coefficient_sets), clipped_ste=clipped_ste)
        self.filepath = filepath
        self.coefficient_sets = coefficient_sets
        self.max_clip_pct = max_clip_pct

        # Register search results as buffers (handled by base class for state-dict)
        self.register_buffer('best_set_idx', torch.tensor(0, dtype=torch.long))
        self.register_buffer('best_bit_shift_scale', torch.tensor(0, dtype=torch.long))
        # Python-side mirrors -- see BaseQuantizer._cached_scalar / the note
        # in BaseQuantizer.__init__. _load_calibration() runs on every
        # quantizing forward call once calibrated.
        self._best_set_idx_cached: int = 0
        self._best_set_idx_version: int = -1
        self._best_bit_shift_scale_cached: int = 0
        self._best_bit_shift_scale_version: int = -1

    @property
    def _set_idx_value(self) -> int:
        return self._cached_scalar(
            self.best_set_idx, "_best_set_idx_cached", "_best_set_idx_version", int)

    @property
    def _bit_shift_scale_value(self) -> int:
        return self._cached_scalar(
            self.best_bit_shift_scale, "_best_bit_shift_scale_cached", "_best_bit_shift_scale_version", int)

    @staticmethod
    def _covering_shift(abs_max: float, coeffs: torch.Tensor) -> int:
        """Smallest shift whose scaled coefficient set spans `abs_max`, i.e.
        ceil(log2(abs_max / max|coeffs|)). 0 for all-zero weights / sets."""
        coeff_max = float(coeffs.abs().max())
        if abs_max <= 0.0 or coeff_max <= 0.0:
            return 0
        return math.ceil(math.log2(abs_max / coeff_max))

    def _calibrate(self, x: torch.Tensor) -> Any:
        """Run calibration/search logic and return a params dict.

        Per coefficient set, only the `2 * SHIFT_SEARCH_RADIUS + 1` shifts
        around the one that just covers `max|x|` (`_covering_shift`) are tried.
        Candidates clipping more than `max_clip_pct` percent of `x` (outside
        the scaled set's [min, max]) are rejected; among the rest the lowest
        SAD wins. If every candidate clips too much, the least-clipping one
        is taken instead.

        Every candidate's SAD and clip fraction stay 0-dim tensors -- no
        `.item()` -- and are compared on-device (`torch.stack`, one `argmin`);
        the only syncs are `max|x|` up front and the final pick. The loop
        itself still runs in Python (batching all candidates into one tensor
        op would multiply peak memory by the number of candidates).
        """
        device = x.device
        abs_max = float(x.detach().abs().max())  # sync 1 of 2
        sads = []
        clip_pcts = []
        combos = []

        for idx, coeffs in enumerate(self.coefficient_sets):
            coeffs_dev = coeffs.to(device=device, dtype=x.dtype)
            center = self._covering_shift(abs_max, coeffs)
            for bit_shift_scale in range(center - SHIFT_SEARCH_RADIUS,
                                         center + SHIFT_SEARCH_RADIUS + 1):
                quantized_temp, scale, _ = apply_non_uniform_quantization(x, coeffs_dev, bit_shift_scale)
                sads.append(torch.sum(torch.abs(x - quantized_temp)))  # stays a 0-dim tensor
                lower, upper = coeffs_dev.min() * scale, coeffs_dev.max() * scale
                clip_pcts.append(((x < lower) | (x > upper)).float().mean() * 100.0)
                combos.append((idx, bit_shift_scale))

        sad = torch.stack(sads)
        clip = torch.stack(clip_pcts)
        allowed = clip <= self.max_clip_pct
        best_allowed = torch.where(allowed, sad, torch.full_like(sad, float("inf"))).argmin()
        best_flat = torch.where(allowed.any(), best_allowed, clip.argmin())
        best_set_idx, best_bit_shift_scale = combos[int(best_flat.item())]  # sync 2 of 2

        return {'set_idx': best_set_idx, 'bit_shift_scale': best_bit_shift_scale}

    def _save_calibration(self, params: Any) -> None:
        """Save calibration results to buffers (and their Python-side mirrors)."""
        self.best_set_idx.fill_(params['set_idx'])
        self.best_bit_shift_scale.fill_(params['bit_shift_scale'])
        self._best_set_idx_cached = int(params['set_idx'])
        self._best_set_idx_version = self.best_set_idx._version
        self._best_bit_shift_scale_cached = int(params['bit_shift_scale'])
        self._best_bit_shift_scale_version = self.best_bit_shift_scale._version
        self.set_search_done(True)

    def _load_calibration(self) -> Any:
        """Load calibration results, reading the cache when the buffers
        haven't changed since the last read -- see BaseQuantizer._cached_scalar."""
        return {
            'set_idx': self._set_idx_value,
            'bit_shift_scale': self._bit_shift_scale_value,
        }

    def _quantize(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        chosen_coeffs = self.coefficient_sets[params['set_idx']].to(device=x.device, dtype=x.dtype)
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
        lower, upper = self._range_limits(x, params)
        return (x >= lower) & (x <= upper)

    def _clip_side(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        """Single-direction clipped-STE support: -1/+1 below/above the scaled
        coefficient span, 0 inside (same bounds as _in_range_mask)."""
        lower, upper = self._range_limits(x, params)
        return (x > upper).to(torch.int8) - (x < lower).to(torch.int8)

    def _range_limits(self, x: torch.Tensor, params: Any):
        scaled = self.coefficient_sets[params['set_idx']].to(device=x.device, dtype=x.dtype) * (2.0 ** params['bit_shift_scale'])
        return scaled.min(), scaled.max()


class CoefficientPerTensorWeightQuant(BaseWeightQuant):
    """
    Brevitas-compatible Injector for the coefficient-based per-tensor weight quantizer.
    """
    tensor_quant = CoefficientPerTensorWeightQuantizer
    filepath = "coefficients.txt"
    # signed inherited from BaseWeightQuant (True)
