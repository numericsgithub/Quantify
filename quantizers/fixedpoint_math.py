"""
Fixed-point math: rounding modes, the quantize function, and the LSB search.

Pure quantization helpers shared by `FixedPointPerTensorQuantizer`
(`quantizers/fixedpoint_per_tensor.py`, which re-exports everything here) and
`CoefficientPerTensorWeightQuant`. See that module's docstring for the
fixed-point representation (step 2^lsb, signed/unsigned code ranges).
"""

import math
from enum import Enum
from typing import Optional, Tuple

import torch
from torch.autograd import Function

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
    quantized, _, _, _ = FixedPointQuantSTEFn.apply(inputs, lsb, bit_width, signed, rounding_mode, narrow_range)

    return quantized


# ---------------------------------------------------------------------------
# Optimal LSB search
# ---------------------------------------------------------------------------

def find_optimal_lsb(
    inputs: torch.Tensor,
    bit_width: int,
    signed: bool,
    rounding_mode: "RoundingMode",
    narrow_range: bool = False,
    prefer_high_lsb: bool = False,
    max_clip_pct: Optional[float] = None,
    max_abs_value: Optional[float] = None,
    min_lsb: Optional[int] = None,
) -> Tuple[int, int, list]:
    """
    Search over LSB positions to find the best fixed-point step size.

    Two selection rules:
      prefer_high_lsb=False (weights, biases): maximise the number of unique
        quantised output values, ties broken by smallest SAD — prefers a
        finer grid when multiple LSBs reach the same unique count.
      prefer_high_lsb=True (activations): COVERAGE-FIRST. Maximising unique
        quantised values (the weight-mode objective) systematically clips
        outliers: a narrow grid packs many more distinct codes into a dense
        bulk-of-distribution cluster than a wide grid does, so it always wins
        the "maximise unique count" objective even though it saturates rare
        extreme values. Instead, pick the FINEST (lowest) LSB whose
        representable range still covers abs_max without clipping it — this
        guarantees the full observed range (including outliers) survives
        calibration, and is the finest resolution possible subject to that
        constraint.

    max_clip_pct : only consulted when prefer_high_lsb=False. A hard ceiling
        on the percentage of input values that may fall outside the
        representable range (and so get clipped) at the selected LSB. Any
        candidate LSB exceeding it is disqualified from the "maximise unique
        count" selection above, regardless of how good its unique-count/SAD
        would otherwise be -- a bit-width that technically resolves the bulk
        of the distribution very finely is useless if it clips too much of
        it. `max_clip_pct=0.0` means NO clipping at all is tolerated (used
        for biases). The search range (`search_lo`..`search_hi`, centered on
        `ideal_lsb` -- the LSB that exactly spans `abs_max` with zero
        clipping) always includes LSBs coarse enough to clip nothing, so
        there is always at least one qualifying candidate; if every tested
        LSB were somehow disqualified, the one that clipped the least is
        used as a defensive fallback instead of leaving the quantizer
        uncalibrated.

    max_abs_value : a hard ceiling on the observed `abs_max` BEFORE it drives
        LSB selection (either rule). Used for the input quantizer of a
        saturating activation (e.g. ReLU6, Sigmoid, Tanh -- see
        `quantizers/activations.py` and pitfall #21 in
        `docs/llm/pitfalls/brevitas_pitfalls.md`): without this, a huge
        pre-activation outlier would make coverage-first size the ENTIRE
        grid to cover it, even though the activation itself clips/saturates
        far below that -- wasting resolution on a range the activation
        throws away anyway. Capping `abs_max` here means any real input
        beyond the cap gets clipped by the quantizer (intentional: it would
        have been clipped/saturated by the activation right afterward
        regardless). `None` (default) leaves `abs_max` as observed, matching
        every other use of this function (weights, biases, non-saturating
        activations).

    min_lsb : a hard FLOOR on the selected LSB (the grid may be this fine or
        coarser, never finer). Used for the output quantizer of a
        value-preserving activation (ReLU, ReLU6 -- see
        `quantizers/activations.py`): its output values are a subset of its
        input quantizer's grid, so any LSB finer than the input's can only
        add codes that never occur ("empty LSBs"), wasting bits that could
        have gone to range. The floor is applied on top of whichever
        selection rule is active; `None` (default) leaves the search
        unconstrained.

    Returns
    -------
    (best_lsb, best_unique, search_records)
        search_records is a list of (lsb, n_unique, sad) for every position tested,
        ordered high→low, used by the diagnostic plot.
    """
    w_min = inputs.min().item()
    w_max = inputs.max().item()
    abs_max = max(abs(w_min), abs(w_max))
    if max_abs_value is not None:
        abs_max = min(abs_max, max_abs_value)

    if abs_max == 0.0:
        # all-zero tensor, LSB doesn't matter (but still honour the floor)
        return (0 if min_lsb is None else max(0, min_lsb)), 1, []

    if signed:
        n_positive_codes = 2 ** (bit_width - 1) - 1
        if n_positive_codes <= 0:
            n_positive_codes = 1
    else:
        n_positive_codes = 2 ** bit_width - 1
    integer_max = n_positive_codes
    if signed:
        integer_min = -(2 ** (bit_width - 1))
        if narrow_range:
            integer_min += 1
    else:
        integer_min = 0

    ideal_lsb = math.log2(abs_max / n_positive_codes) if n_positive_codes > 0 else 0
    search_lo = math.floor(ideal_lsb) - 12
    search_hi = math.ceil(ideal_lsb) + 12
    if min_lsb is not None:
        search_hi = max(search_hi, min_lsb)  # the floor itself must be a candidate

    if prefer_high_lsb:
        # Coverage-first: smallest step whose range still contains abs_max.
        coverage_lsb = math.ceil(ideal_lsb)
        while integer_max * (2.0 ** (coverage_lsb - 1)) >= abs_max:
            coverage_lsb -= 1  # a finer step still covers abs_max -- keep sharpening
        while integer_max * (2.0 ** coverage_lsb) < abs_max:
            coverage_lsb += 1  # float rounding left us a hair too fine -- back off
        if min_lsb is not None:
            coverage_lsb = max(coverage_lsb, min_lsb)  # never finer than the floor

    best_lsb = search_lo
    best_unique = -1
    best_sad = float("inf")
    search_records: list = []  # (lsb, n_unique, sad) — high to low

    n_total = inputs.numel()
    found_qualifying = False
    fallback_lsb = search_lo
    fallback_clip_pct = float("inf")

    for lsb in reversed(range(search_lo, search_hi + 1)):
        q = quantize_fixed_point(inputs, lsb, bit_width, signed, rounding_mode, narrow_range)
        n_unique = int(torch.unique(q).numel())
        sad = float(torch.sum(torch.abs(inputs - q)).item())
        search_records.append((lsb, n_unique, sad))

        if min_lsb is not None and lsb < min_lsb:
            continue  # finer than the floor -- recorded for the plot, never selectable

        if prefer_high_lsb:
            if lsb == coverage_lsb:
                best_lsb = lsb
                best_unique = n_unique
                best_sad = sad
        else:
            if max_clip_pct is not None:
                step = 2.0 ** lsb
                q_min = integer_min * step
                q_max = integer_max * step
                n_clip = int(((inputs < q_min) | (inputs > q_max)).sum().item())
                clip_pct = 100.0 * n_clip / max(n_total, 1)
                if clip_pct < fallback_clip_pct:
                    fallback_lsb = lsb
                    fallback_clip_pct = clip_pct
                if clip_pct > max_clip_pct:
                    continue  # disqualified -- clips more than the allowed ceiling
                found_qualifying = True
            # Weight/bias mode: ties broken by minimum SAD (finer grid preferred)
            if n_unique > best_unique or (n_unique == best_unique and sad < best_sad):
                best_lsb = lsb
                best_unique = n_unique
                best_sad = sad

    if min_lsb is not None and best_lsb < min_lsb:
        best_lsb = min_lsb  # defensive: nothing selectable (shouldn't happen)

    if max_clip_pct is not None and not found_qualifying:
        # Defensive fallback: every tested LSB clipped more than max_clip_pct
        # allows (shouldn't happen -- the search range always reaches an
        # LSB coarse enough for 0% clipping). Use the least-clipping one
        # rather than silently calibrating with an over-clipping grid.
        for lsb, n_unique, sad in search_records:
            if lsb == fallback_lsb:
                best_lsb, best_unique, best_sad = lsb, n_unique, sad
                break

    return best_lsb, best_unique, search_records


# ---------------------------------------------------------------------------
# Rounding helpers
# ---------------------------------------------------------------------------

class RoundingMode(Enum):
    ROUND_TO_NEAREST_EVEN = "round_to_nearest_even"
    FLOOR = "floor"
    ROUND = "round"


# Integer encoding of RoundingMode used ONLY for the `rounding_mode_i`
# attribute emitted alongside `rounding_mode_s` on exported ONNX nodes --
# see utils/onnx_self_contained.py. The embedded ONNX FunctionProto bodies
# need to pick a rounding operator per node *instance* (every node's
# attributes are fixed, but the function body is shared across all nodes of
# that op_type in the model), and there's no standard ONNX op for comparing
# strings, so the mode is also encoded numerically for a `Where`-based
# selection inside the function body. `rounding_mode_s` remains the
# human-readable source of truth; this is a derived, redundant attribute.
ROUNDING_MODE_TO_INT = {
    RoundingMode.FLOOR: 0,
    RoundingMode.ROUND: 1,
    RoundingMode.ROUND_TO_NEAREST_EVEN: 2,
}


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

class FixedPointQuantSTEFn(Function):
    """Plain fake-quantize whose only job is the straight-through backward; NO `symbolic()`.

    The rounding in `quantize_fixed_point_with_integers` has zero gradient, so
    this Function is what makes `quantize_fixed_point` differentiable.

    Used by `quantize_fixed_point` (e.g. the LSB calibration search). Without a
    `symbolic()` it cannot be exported -- `FixedPointQuantFn` below is the
    variant that emits the `Quantify::FixedPointQuant` node.
    """
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
