"""
Diagnostic tool for the saturation-aware input-range guardrail (pitfall #21
in docs/llm/pitfalls/brevitas_pitfalls.md): for each quantized activation in
quantizers/activations.py, sweep candidate LSB settings at a fixed
bit-width and show which ones the guardrail would allow or reject for that
activation's INPUT quantizer.

Run directly for a printed report:
    python -m utils.activation_lsb_sweep

Each row is one candidate (lsb, bit_width) grid setting:
    lsb         -- LSB position (see quantizers/fixedpoint_per_tensor.py's
                   module docstring for the fixed-point convention used
                   throughout this repo).
    msb         -- msb = lsb + bit_width - 1 (same convention).
    bit_width   -- fixed per sweep (8, by default, per the table this tool
                   was written to reproduce).
    min_value   -- the grid's most negative representable value at this
                   (lsb, bit_width), assuming signed=True (pre-activation
                   values are typically signed for every one of these
                   activations, including ones whose OUTPUT is forced
                   non-negative -- e.g. ReLU's whole job is to zero out a
                   signed input).
    max_value   -- the grid's most positive representable value.
    allowed     -- whether this setting's representable magnitude
                   (max(|min_value|, |max_value|)) is within the
                   activation's input `max_abs_value` cap -- `None` cap
                   (ReLU, SiLU, GELU, LeakyReLU, Softmax: no natural
                   saturation point) allows everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn

from quantizers.base_quantizer import BaseQuantizer
from quantizers.activations import (
    QuantReLU,
    QuantReLU6,
    QuantSigmoid,
    QuantTanh,
    QuantSiLU,
    QuantGELU,
    QuantLeakyReLU,
    QuantSoftmax,
)

# (name, class) -- order matches the module docstring / CLAUDE.md listing.
ACTIVATIONS: List[Tuple[str, Type[nn.Module]]] = [
    ("relu", QuantReLU),
    ("relu6", QuantReLU6),
    ("sigmoid", QuantSigmoid),
    ("tanh", QuantTanh),
    ("silu", QuantSiLU),
    ("gelu", QuantGELU),
    ("leaky_relu", QuantLeakyReLU),
    ("softmax", QuantSoftmax),
]

DEFAULT_LSB_VALUES: range = range(-4, 7)  # -4..6 inclusive, brackets the cap for every activation here
DEFAULT_BIT_WIDTH = 8


@dataclass(frozen=True)
class LsbRow:
    lsb: int
    msb: int
    bit_width: int
    min_value: float
    max_value: float
    allowed: bool


def find_input_quantizer(module: nn.Module, sample_input: Optional[torch.Tensor] = None) -> BaseQuantizer:
    """The quantizer with the lowest `inference_sequence_id` is the one
    whose `forward()` ran first -- for every activation in
    quantizers/activations.py that is always the input quantizer (see
    tests/test_activation_io_quantizers.py's `_find_by_role_position`,
    which this mirrors). Runs a calibrating forward pass as a side effect.
    """
    module.train()
    module(sample_input if sample_input is not None else torch.randn(8, 4))
    quantizers = [m for m in module.modules() if isinstance(m, BaseQuantizer)]
    if not quantizers:
        raise AssertionError(f"No BaseQuantizer found inside {type(module).__name__}")
    return min(quantizers, key=lambda q: q.inference_sequence_id)


def get_input_cap(activation_cls: Type[nn.Module], bit_width: int = DEFAULT_BIT_WIDTH) -> Optional[float]:
    """The REAL, configured `max_abs_value` of a fresh instance's input
    quantizer -- read off the live object (not recomputed from the
    activation's private saturation constant) so this also exercises the
    actual wiring (Brevitas kwarg-prefix forwarding for QuantReLU/Sigmoid/
    Tanh, direct construction for the rest) rather than just restating it.
    """
    module = activation_cls(bit_width=bit_width)
    q_in = find_input_quantizer(module)
    return q_in.max_abs_value


def representable_range(lsb: int, bit_width: int, signed: bool = True) -> Tuple[float, float]:
    """(min_value, max_value) of the fixed-point grid at this (lsb,
    bit_width) -- see quantizers/fixedpoint_per_tensor.py's module
    docstring for the convention (narrow_range=False, matching every
    activation's default)."""
    step = 2.0 ** lsb
    if signed:
        integer_min = -(2 ** (bit_width - 1))
        integer_max = 2 ** (bit_width - 1) - 1
    else:
        integer_min = 0
        integer_max = 2 ** bit_width - 1
    return integer_min * step, integer_max * step


def build_lsb_table(
    max_abs_value: Optional[float],
    bit_width: int = DEFAULT_BIT_WIDTH,
    lsb_values: Sequence[int] = DEFAULT_LSB_VALUES,
    signed: bool = True,
) -> List[LsbRow]:
    rows = []
    for lsb in lsb_values:
        msb = lsb + bit_width - 1
        min_value, max_value = representable_range(lsb, bit_width, signed)
        # find_optimal_lsb's coverage-first search (quantizers/fixedpoint_per_tensor.py)
        # sizes the grid using ONLY the positive-code count (n_positive_codes
        # = 2**(bit_width-1)-1 for signed), i.e. against `max_value` here --
        # it never looks at the asymmetric extra negative code (`min_value`
        # has one more representable magnitude than `max_value` in two's
        # complement). So "allowed" must be judged against `max_value`, not
        # `max(|min_value|, |max_value|)`, to match what calibration
        # actually does -- confirmed in tests/test_activation_lsb_sweep.py.
        allowed = True if max_abs_value is None else max_value <= max_abs_value
        rows.append(LsbRow(lsb, msb, bit_width, min_value, max_value, allowed))
    return rows


def format_table(rows: Sequence[LsbRow]) -> str:
    header = f"{'lsb':>4} {'msb':>4} {'bit_width':>9} {'min_value':>12} {'max_value':>12} {'allowed':>7}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r.lsb:>4} {r.msb:>4} {r.bit_width:>9} {r.min_value:>12.5f} "
            f"{r.max_value:>12.5f} {'yes' if r.allowed else 'no':>7}"
        )
    return "\n".join(lines)


def print_all_tables(bit_width: int = DEFAULT_BIT_WIDTH, lsb_values: Sequence[int] = DEFAULT_LSB_VALUES) -> None:
    for name, cls in ACTIVATIONS:
        cap = get_input_cap(cls, bit_width=bit_width)
        rows = build_lsb_table(cap, bit_width=bit_width, lsb_values=lsb_values)
        cap_str = f"{cap:g}" if cap is not None else "None (uncapped)"
        print(f"\n=== {name}  (input max_abs_value cap = {cap_str}) ===")
        print(format_table(rows))


if __name__ == "__main__":
    print_all_tables()
