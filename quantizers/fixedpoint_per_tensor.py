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
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from torch.autograd import Function
from torch.onnx import symbolic_helper
from quantizers.base_quantizer import BaseQuantizer
from collections import deque

# ---------------------------------------------------------------------------
# Core fixed-point quantization
# ---------------------------------------------------------------------------

def quantize_fixed_point_with_integers(
    inputs: torch.Tensor,
    lsb: int,
    bit_width: int,
    signed: bool,
    rounding_mode: "RoundingMode",
    narrow_range: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        to make the range symmetric.  Default False.

    Returns
    -------
    torch.Tensor
        Quantized (dequantized) input tensor on the fixed-point grid.
    torch.Tensor
        Quantized (dequantized) input tensor on the fixed-point grid as integers.
    """
    orig_dtype = inputs.dtype
    # float32 has 24 bits of exact integer mantissa -- comfortably enough for
    # any bit_width used in this project (8-16b). Upcasting all the way to
    # float64 here used to roughly double the wall-clock cost of every
    # quantize call (extra memory traffic + cast kernels) for no precision
    # benefit at these bit-widths; only stay at float64 if the caller's input
    # tensor actually already is float64.
    calc_dtype = torch.float32 if orig_dtype != torch.float64 else torch.float64
    inputs_calc = inputs.to(calc_dtype)

    step = 2.0 ** lsb  # lsb is a plain Python int -- no tensor needed here

    if signed:
        integer_min = -(2 ** (bit_width - 1))
        if narrow_range:
            integer_min += 1  # exclude most-negative integer
        integer_max = 2 ** (bit_width - 1) - 1
    else:
        integer_min = 0
        integer_max = 2 ** bit_width - 1

    # Quantize: map to integer, round, clamp, scale back
    integers = inputs_calc / step
    integers = _round(integers, rounding_mode)
    integers = torch.clamp(integers, integer_min, integer_max)
    quantized = integers * step

    # Cast back to the caller's dtype
    quantized = quantized.to(orig_dtype)

    return quantized, integers

def quantize_fixed_point(inputs: torch.Tensor, lsb: int, bit_width: int, signed: bool, rounding_mode: "RoundingMode", narrow_range: bool = False) -> torch.Tensor:
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
        to make the range symmetric.  Default False.

    Returns
    -------
    torch.Tensor
        Quantized (dequantized) input tensor on the fixed-point grid.

    Note
    ----
    Backward is plain STE (slope 1 everywhere). Clipped STE is applied one level
    up in BaseQuantizer.forward via ClippedSTEFn + _in_range_mask, so it is
    shared by all quantizers rather than baked into this function.
    """
    quantized, _, _, _ = FixedPointQuantFnTestingThings.apply(inputs, lsb, bit_width, signed, rounding_mode, narrow_range)
    # quantized, _ =  quantize_fixed_point_with_integers(inputs, lsb, bit_width, signed, rounding_mode, narrow_range)

    return quantized


# ---------------------------------------------------------------------------
# Optimal LSB search
# ---------------------------------------------------------------------------

# Robust-sigma selection constants. The LSB is really a choice of clipping
# threshold (range = 2^lsb * integer_max), and these set where that threshold
# lands, in units of a robust (outlier-immune) estimate of the distribution's
# spread.
#
# Activations (k=16): much wider than the weights, so most of the tail stays
# representable. Post-ReLU distributions carry real information in a long right
# tail, and a clipped activation corrupts the signal directly.
#
# Weights (k=12). Deliberately trims outliers: a tighter range costs PTQ accuracy
# up front but gives the bulk of the distribution finer resolution, which QAT can
# then adapt to (it can pull the clipped outliers back inside). Measured
# weight-only top-1 across k (examples/sweep_weight_k.py, 40 val batches):
#
#   k=4: 0.00%   k=8: 2.25%   k=12: 45.90%   k=13: 48.24%   k=16: 39.96%
#   k=18: 58.91% k=24: 31.06% k=32: 12.31%   k=64: 0.35%    float: 70.08%
#
# Note that curve is a STEP function, not a smooth optimum: the LSB is an integer,
# so each layer flips at its own k and accuracy only moves where flips happen (4
# layers flipping between k=17 and k=18 moves the model 24 points). k=18 scores
# highest but only because those flips happen to align — a single global k cannot
# express what is really a per-layer choice. k=12 sits in a stable band (43-48%
# across k=12..15) rather than on a spike.
ROBUST_SIGMA_K_WEIGHT = 12.0
ROBUST_SIGMA_K_ACTIVATION = 16.0

# Biases use a COVERAGE target instead of a sigma multiple: pick the finest LSB
# whose range holds at least this percent of the bias values, so at most
# (100 - this) percent are clipped.
#
# A sigma rule is the wrong tool here. It measures how WIDE a cluster is, which
# says nothing about WHERE it sits, and folded biases are
# beta - gamma*mu/sqrt(var+eps) — no reason to be zero-mean, while the grid is
# symmetric about zero with no zero-point. features.20's biases sit at -5.43 with
# sigma=0.69: a 4-sigma window is the right width in entirely the wrong place, and
# clipped 92.5% of the tensor. A percentile of |bias| asks the question that
# actually matters — "does the range reach the data?" — and gets the centring
# right for free.
#
# 99.9 is measured, not chosen by feel. Biases tolerate almost no clipping: a
# clipped weight's error is averaged across a whole filter, but a clipped bias is
# a constant offset on EVERY activation of its channel, with nothing to average it
# away. With weights fixed at k=12 (40 val batches, weight-only top-1):
#
#   cover  80% -> 0.00%    cover 99%   -> 33.13%    biases float -> 50.12%
#   cover  90% -> 0.00%    cover 99.9% -> 47.72%
#   cover  95% -> 4.18%    cover 100%  -> 44.06%
#
# Note 99.9 beats 100: trimming the last 0.1% IS worth it, because covering the
# absolute max forces a coarser step on everything else. The useful trim for a
# bias is 0.1%, not 20%.
BIAS_COVERAGE_PCT = 99.9


def integer_range(bit_width: int, signed: bool, narrow_range: bool = False) -> Tuple[int, int]:
    """The integer codes available at this bit width — the exact values
    quantize_fixed_point_with_integers clamps against."""
    if signed:
        integer_min = -(2 ** (bit_width - 1))
        if narrow_range:
            integer_min += 1
        integer_max = 2 ** (bit_width - 1) - 1
    else:
        integer_min = 0
        integer_max = 2 ** bit_width - 1
    return integer_min, integer_max


def robust_center_and_sigma(inputs: torch.Tensor,
                            ignore_zeros: bool = False) -> Tuple[float, float]:
    """Outlier-immune (center, spread) of a distribution: (median, 1.4826 * MAD).

    1.4826 makes the spread equal the standard deviation for Gaussian data. Unlike
    std, MAD is computed from the middle of the distribution, so no outlier —
    however extreme — can move it.

    The CENTER matters because the fixed-point grid is symmetric about zero and has
    no zero-point: it can only represent [integer_min*step, integer_max*step]. A
    spread of 0.69 tells you how wide a cluster is, but not where it sits. Weights
    are centered on zero (median measured at exactly 0.0000 across all 53
    MobileNetV2 tensors), so the distinction never surfaces for them — but folded
    biases are beta - gamma*mu/sqrt(var+eps), which has no reason to be zero-mean,
    and features.20's biases sit at -5.43 with a spread of only 0.69.

    ignore_zeros: compute over non-zero elements only. REQUIRED for post-ReLU
    activations: they are typically 40-70% exact zeros, which drags the median to
    0 and collapses MAD to exactly 0 (a degenerate range). It is already unstable
    before that — at 45% zeros MAD swings ~2x while the true spread is unchanged.
    Weights do not need it (float weights are essentially never exactly zero).
    """
    x = inputs.detach().flatten().float()
    if ignore_zeros:
        x = x[x != 0]
    if x.numel() == 0:
        return 0.0, 0.0
    med = x.median()
    mad = (x - med).abs().median()
    return float(med.item()), float(1.4826 * mad.item())


def robust_sigma(inputs: torch.Tensor, ignore_zeros: bool = False) -> float:
    """Spread only — see robust_center_and_sigma."""
    return robust_center_and_sigma(inputs, ignore_zeros=ignore_zeros)[1]


def find_optimal_lsb(
    inputs: torch.Tensor,
    bit_width: int,
    signed: bool,
    rounding_mode: "RoundingMode",
    narrow_range: bool = False,
    prefer_high_lsb: bool = False,
    robust_sigma_k: Optional[float] = None,
    robust_ignore_zeros: bool = False,
    coverage_pct: Optional[float] = None,
) -> Tuple[int, int, list]:
    """
    Search over LSB positions and select one.

    coverage_pct (biases): choose the FINEST lsb whose representable range holds
      at least this percent of |inputs|, so at most (100 - coverage_pct) percent
      is clipped. Takes precedence over robust_sigma_k. Because it works on
      |inputs| directly it needs no centring term — see BIAS_COVERAGE_PCT.

    robust_sigma_k: choose the FINEST lsb whose representable range still covers
      +/- (|center| + robust_sigma_k * sigma), where (center, sigma) is the
      outlier-immune (median, 1.4826*MAD) of the distribution. This sizes the
      range from the core and deliberately clips the tail beyond it. See
      ROBUST_SIGMA_K_WEIGHT / ROBUST_SIGMA_K_ACTIVATION.

    robust_sigma_k=None (legacy): maximise the number of unique quantised values.
      Kept for backward compatibility. Note this objective is a grid-UTILISATION
      metric, not an error metric, and it is anti-correlated with accuracy: the
      only way to score higher is to put a finer grid on the dense centre and
      clip the tails off. Measured over the 53 MobileNetV2 weight tensors it
      picked a finer lsb than every error-based rule on 53/53 layers and clipped
      12.9% of weights. Do not use it for new work.
      Its two sub-rules are, in practice, one rule: prefer_high_lsb=True
      (activations, widest range among the winners) and prefer_high_lsb=False
      (weights, lowest SAD among the winners) selected an identical lsb on all
      53 tensors — the max-unique constraint leaves the tie-break nothing to
      decide.

    Returns
    -------
    (best_lsb, best_unique, search_records)
        search_records is a list of (lsb, n_unique, sad) for every position tested,
        ordered high→low, used by the diagnostic plot. best_unique is the unique
        count AT best_lsb (callers gate search_done on it being > 1).
    """
    w_min = inputs.min().item()
    w_max = inputs.max().item()
    abs_max = max(abs(w_min), abs(w_max))

    if abs_max == 0.0:
        return 0, 1, []  # all-zero tensor, LSB doesn't matter

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
    search_records: list = []  # (lsb, n_unique, sad) — high to low
    unique_by_lsb: dict = {}

    for lsb in reversed(range(search_lo, search_hi + 1)):
        q = quantize_fixed_point(inputs, lsb, bit_width, signed, rounding_mode, narrow_range)
        n_unique = int(torch.unique(q).numel())
        sad = float(torch.sum(torch.abs(inputs - q)).item())
        search_records.append((lsb, n_unique, sad))
        unique_by_lsb[lsb] = n_unique

        if prefer_high_lsb:
            # Strict improvement only: first (highest) LSB with global max unique wins
            if n_unique > best_unique:
                best_lsb = lsb
                best_unique = n_unique
                best_sad = sad
        else:
            # Weight mode: ties broken by minimum SAD (finer grid preferred)
            if n_unique > best_unique or (n_unique == best_unique and sad < best_sad):
                best_lsb = lsb
                best_unique = n_unique
                best_sad = sad

    if coverage_pct is not None:
        # Finest lsb holding >= coverage_pct of |inputs|. Note the lsb is a power
        # of two, so the range can only halve between candidates: actual coverage
        # will usually EXCEED the target rather than hit it.
        flat = inputs.detach().flatten().float().abs()
        threshold = float(torch.quantile(flat, coverage_pct / 100.0).item())
        _, integer_max = integer_range(bit_width, signed, narrow_range)
        covering = [lsb for lsb in unique_by_lsb
                    if integer_max * (2.0 ** lsb) >= threshold]
        chosen = min(covering) if covering else max(unique_by_lsb)
        return chosen, unique_by_lsb[chosen], search_records

    if robust_sigma_k is not None:
        center, sigma = robust_center_and_sigma(inputs, ignore_zeros=robust_ignore_zeros)
        # sigma == 0 means the middle of the distribution is a single repeated
        # value (e.g. an activation that is >50% zeros when ignore_zeros was not
        # set). No core to size the range from, so fall back to covering the
        # 99.9th percentile rather than emitting a degenerate range.
        if sigma > 0.0:
            # The core spans [center - k*sigma, center + k*sigma]. The grid is
            # symmetric about zero with no zero-point, so to represent that span
            # the range must reach |center| + k*sigma — the far edge of the core,
            # not merely its half-width. Omitting |center| asks for a window the
            # right WIDTH in the wrong PLACE: features.20's biases sit at -5.43
            # with sigma=0.69, so k*sigma alone gave a +/-4 grid and clipped 92.5%
            # of the tensor. Weights are centered on zero (median 0.0000 on all 53
            # MobileNetV2 tensors), so this term is a no-op for them.
            threshold = abs(center) + robust_sigma_k * sigma
        else:
            threshold = float(torch.quantile(inputs.detach().flatten().float().abs(),
                                             0.999).item())
        _, integer_max = integer_range(bit_width, signed, narrow_range)
        # Finest (smallest) lsb whose range still covers +/-threshold. |integer_min|
        # >= integer_max, so integer_max is the binding side.
        covering = [lsb for lsb in unique_by_lsb
                    if integer_max * (2.0 ** lsb) >= threshold]
        robust_lsb = min(covering) if covering else max(unique_by_lsb)
        return robust_lsb, unique_by_lsb[robust_lsb], search_records

    return best_lsb, best_unique, search_records


# ---------------------------------------------------------------------------
# Rounding helpers
# ---------------------------------------------------------------------------

class RoundingMode(Enum):
    ROUND_TO_NEAREST_EVEN = "round_to_nearest_even"
    FLOOR = "floor"
    ROUND = "round"


def _round(x: torch.Tensor, mode: RoundingMode) -> torch.Tensor:
    """Round tensor according to the selected rounding mode."""
    if mode is RoundingMode.FLOOR:
        return torch.floor(x)
    if mode is RoundingMode.ROUND:
        return torch.floor(x + 0.5)
    if mode is RoundingMode.ROUND_TO_NEAREST_EVEN:
        return torch.round(x)
    raise Exception(f"Unknown rounding mode! {mode}")


# ---------------------------------------------------------------------------
# ONNX Custom Node Shim
# ---------------------------------------------------------------------------

class FixedPointQuantFnTestingThings(Function):
    """Symbolic shim: emits a single `Quantify::FixedPointQuant` ONNX node."""
    # inputs, lsb, bit_width, signed, rounding_mode, narrow_range
    @staticmethod
    def forward(ctx, x, lsb, bit_width, signed, rounding_mode, narrow_range):
        # Compute quantization for PyTorch inference/tracing
        quantized, integers = quantize_fixed_point_with_integers(
            x, int(lsb), int(bit_width), signed, rounding_mode, narrow_range
        )

        bw = torch.tensor(float(bit_width), dtype=x.dtype, device=x.device)
        return quantized, 0.0, 0.0, bw

    @staticmethod
    def backward(ctx, grad_quantized, grad_scale, grad_zero_point, grad_bw):
        # Plain Straight-Through Estimator (slope 1 everywhere). Clipped STE is
        # applied in BaseQuantizer.forward (shared across quantizers), not here.
        return grad_quantized, None, None, None, None, None, None


class FixedPointQuantFn(Function):
    """Symbolic shim: emits a single `Quantify::FixedPointQuant` ONNX node."""
    
    _integer_queue: deque = deque()

    @staticmethod
    def symbolic(g, x, scale, zero_point, lsb, bit_width, signed, narrow_range, rounding_mode):
        # Extract scale and zero_point values to embed as attributes instead of separate constant nodes
        # During ONNX export, scale and zero_point are torch._C.Value objects, not tensors.
        scale_val = symbolic_helper._maybe_get_const(scale, "t")
        zero_point_val = symbolic_helper._maybe_get_const(zero_point, "t")

        # Pop the integers that the corresponding forward() enqueued
        captured = FixedPointQuantFn._integer_queue.popleft()
        
        quantized = g.op(
            "Quantify::FixedPointQuant",
            x,
            scale_f=scale_val,
            zero_point_f=zero_point_val,
            lsb_i=int(lsb),
            bit_width_i=int(bit_width),
            signed_i=int(signed),
            narrow_range_i=int(narrow_range),
            rounding_mode_s=str(rounding_mode.value),
            quantized_ints_t=captured,
        ).setType(x.type())
        
        # Brevitas expects a 4-tuple output; create bw constant
        bw = g.op("Constant", value_t=torch.tensor(float(bit_width)))
        return quantized, scale, zero_point, bw

    @staticmethod
    def forward(ctx, x, scale, zero_point, lsb, bit_width, signed, narrow_range, rounding_mode):
        ctx.save_for_backward(x)
        # Compute quantization for PyTorch inference/tracing
        quantized, integers = quantize_fixed_point_with_integers(
            x, int(lsb), int(bit_width), signed, rounding_mode, narrow_range
        )
        
        if torch.onnx.is_in_onnx_export():
            with torch.no_grad():
                # Enqueue; symbolic() will dequeue in the same order
                FixedPointQuantFn._integer_queue.append(integers.cpu().to(torch.long))
                
        bw = torch.tensor(float(bit_width), dtype=x.dtype, device=x.device)
        return quantized, scale, zero_point, bw

    @staticmethod
    def backward(ctx, grad_quantized, grad_scale, grad_zero_point, grad_bw):
        # Straight-Through Estimator: pass gradient through for the first input
        return grad_quantized, None, None, None, None, None, None, None

    @classmethod
    def reset_capture_state(cls):
        cls._integer_queue.clear()


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
        Exclude most-negative code in signed mode (default False).
    clipped_ste : bool
        If True, use a clipped Straight-Through Estimator: gradient passes
        through (slope 1) only for weights inside the representable range and is
        zeroed for weights the forward clamp saturated. If False (default),
        plain STE (slope 1 everywhere) is used. This is a toggle so plain vs
        clipped STE can be ablated. The flag and the masking mechanism live in
        BaseQuantizer; this class supplies the fixed-point range via
        _in_range_mask(). Only affects the live training/inference path; the
        ONNX-export path is unchanged.
    """

    def __init__(
        self,
        bit_width: int = 8,
        signed: bool = True,
        rounding_mode: RoundingMode = RoundingMode.ROUND,
        narrow_range: bool = False,
        quantizer_role: str = "unknown",
        clipped_ste: bool = False,
    ):
        # clipped_ste is stored by BaseQuantizer (shared by all quantizers).
        super().__init__(bit_width=bit_width, clipped_ste=clipped_ste)
        self.signed = signed
        self.rounding_mode = rounding_mode
        self.narrow_range = narrow_range
        self.quantizer_role = quantizer_role

        # Register search results as buffers to ensure they are serialized in state_dict
        self.register_buffer('search_result_is_signed', torch.tensor(signed, dtype=torch.bool))
        self.register_buffer('search_result_lsb', torch.tensor(0, dtype=torch.long))

    def _calibrate(self, x: torch.Tensor) -> Any:
        """Run calibration/search logic and return a params dict."""
        if torch.all(x >= 0.0):
            self.signed = False
        else:
            self.signed = True
        is_act = self.quantizer_role == "activation"
        is_bias = self.quantizer_role == "bias"
        # The three roles get three rules; their distributions are not alike.
        #   bias       -> coverage target (see BIAS_COVERAGE_PCT): the folded bias
        #                 carries a DC offset, so what matters is whether the range
        #                 reaches the data, not how wide the cluster is.
        #   activation -> sigma rule, wide k, and must ignore the ReLU zero-spike
        #                 (>50% zeros collapses MAD to exactly 0).
        #   weight     -> sigma rule.
        lsb, num_unique, search_records = find_optimal_lsb(
            x,
            self.bit_width,
            self.signed,
            self.rounding_mode,
            self.narrow_range,
            prefer_high_lsb=is_act,
            coverage_pct=(BIAS_COVERAGE_PCT if is_bias else None),
            robust_sigma_k=(None if is_bias else
                            (ROBUST_SIGMA_K_ACTIVATION if is_act
                             else ROBUST_SIGMA_K_WEIGHT)),
            robust_ignore_zeros=is_act,
        )
        return {
            'lsb': lsb,
            'signed': self.signed,
            'num_unique': num_unique,
            'search_records': search_records,
        }

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
        # Use the custom Function only during ONNX export to emit the custom node.
        # During training_harness/inference, call the direct math function to avoid overhead.
        if torch.onnx.is_in_onnx_export():
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
        return quantize_fixed_point(
            x,
            int(params['lsb']),
            self.bit_width,
            params['signed'],
            self.rounding_mode,
            self.narrow_range
        )

    def representable_range(self, params: Any = None) -> Optional[Tuple[float, float]]:
        """(lower, upper) float bounds of this quantizer's fixed-point grid.

        These are the exact values torch.clamp() saturates against in
        quantize_fixed_point_with_integers: integer_min*step and integer_max*step
        with step = 2**lsb. Anything outside cannot be represented at all — it is
        pinned to a bound, and (under plain STE) keeps receiving gradient that can
        never change its quantized value.

        params: a calibration dict; defaults to the stored calibration.
        Returns None if this quantizer has not been calibrated yet.
        """
        if params is None:
            if not bool(self.search_done.item()):
                return None
            params = self._load_calibration()
        step = 2.0 ** int(params['lsb'])
        integer_min, integer_max = integer_range(
            self.bit_width, params['signed'], self.narrow_range)
        return integer_min * step, integer_max * step

    def _in_range_mask(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        """Clipped-STE support: True where x lands inside the fixed-point grid's
        representable range, False where the forward clamp saturated it.

        Boundary convention is inclusive (>= lower AND <= upper): a weight sitting
        exactly on the bottom or top code counts as in-range and keeps slope 1.
        """
        lower, upper = self.representable_range(params)
        return (x >= lower) & (x <= upper)

    def _get_metadata(self, params: Any, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scale, zero_point, and bit_width tensors matching x's dtype/device."""
        scale = torch.tensor(2.0 ** params['lsb'], dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        bit_width = torch.tensor(float(self.bit_width), dtype=x.dtype, device=x.device)
        return scale, zero_point, bit_width

    def _get_diagnostics_params(self, params) -> dict:
        d = {
            "lsb":            int(params["lsb"]),
            "bit_width":      self.bit_width,
            "signed":         bool(params.get("signed", self.signed)),
            "quantizer_role": self.quantizer_role,
        }
        if "search_records" in params:
            d["search_records"] = params["search_records"]
        return d

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

    rounding_mode = RoundingMode.ROUND
    narrow_range = False
    signed = True  # Explicitly declared to match proxy expectation and avoid QuantTensor validity errors
    tensor_quant = FixedPointPerTensorQuantizer


# ------ Brevitas Injector ------
class FixedPointPerTensorActivationQuant(BaseActivationQuant):
    """
    Brevitas-compatible Injector for the fixed-point per-tensor activation
    quantizer.

    Usage:
        act = QuantReLU(act_quant=FixedPointPerTensorActivationQuant)
    """

    rounding_mode = RoundingMode.FLOOR
    narrow_range = False
    signed = False  # Explicitly declared to match proxy expectation
    tensor_quant = FixedPointPerTensorQuantizer


# ------ Brevitas Injector for Bias ------
class FixedPointPerTensorBiasQuant(BaseWeightQuant):
    """
    Brevitas-compatible Injector for the fixed-point per-tensor bias quantizer.

    Usage:
        layer = QuantLinear(
            in_features=64, out_features=32, bias=True,
            weight_quant=FixedPointPerTensorWeightQuant,
            bias_quant=FixedPointPerTensorBiasQuant,
        )
    """

    proxy_class = BiasQuantProxyFromInjector
    requires_input_scale = False  # Bias scale is computed from bias data itself
    rounding_mode = RoundingMode.ROUND
    narrow_range = False
    signed = True  # Explicitly declared to match proxy expectation
    tensor_quant = FixedPointPerTensorQuantizer
    quantizer_role = "bias"
