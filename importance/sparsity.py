"""Which weights / kernels / filters are zero?

Quantized weights are easy: if the layer's weight quantizer is active, a
weight is zero iff its quantized value is exactly 0.

Float weights are never exactly zero, so "zero" needs a tolerance. A fixed
fraction of the largest weight (``max|w| / 1000``) is a good start, but it
gets unfair for big layers: a 3x3x512x512 conv can have hundreds of
thousands of weights below that line, and together they can still carry a
real share of the layer's signal. So the float threshold is the *smaller*
of two bounds:

- ``rel_tol * max|w|`` (default ``1e-3``) -- no weight within 1/1000 of the
  layer's largest one is ever kept as "non-zero" *just because it is tiny*,
  and nothing bigger than that is ever called zero;
- the *mass budget*: the largest threshold ``t`` such that all weights with
  ``|w| <= t`` *together* hold at most ``mass_tol`` (default ``1e-3``) of the
  layer's total L1 mass ``sum |w|``. Zeroing all of them changes every
  output pre-activation by at most that fraction of what the full weight
  tensor could contribute, no matter how many of them there are.

Example from the design discussion: ``[3000, 0.1, 5]`` -> rel bound 3.0,
mass budget 3.005 admits only 0.1 -> threshold 0.1, so 0.1 counts as zero
and 5 does not. With a million weights of 0.1 next to the 3000, the rel
bound alone would zero all of them, but they hold ~97% of the mass, so the
budget keeps them.

Thresholds are per weight tensor (layer), not per filter, so a filter whose
weights are all tiny compared to the rest of the layer is flagged as a zero
filter.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

DEFAULT_REL_TOL = 1e-3
DEFAULT_MASS_TOL = 1e-3


def float_zero_threshold(weight: np.ndarray, rel_tol: float = DEFAULT_REL_TOL,
                         mass_tol: float = DEFAULT_MASS_TOL) -> Dict[str, float]:
    """Return {"threshold", "rel_bound", "mass_bound"}; a weight is zero iff |w| <= threshold."""
    a = np.sort(np.abs(np.asarray(weight, dtype=np.float64)).ravel())
    if a.size == 0 or a[-1] == 0.0:
        return {"threshold": 0.0, "rel_bound": 0.0, "mass_bound": 0.0}
    rel_bound = rel_tol * float(a[-1])
    budget = mass_tol * float(a.sum())
    k = int(np.searchsorted(np.cumsum(a), budget, side="right"))  # first k smallest fit the budget
    if 0 < k < a.size and a[k] == a[k - 1]:
        # `|w| <= a[k-1]` would also catch the tied values past the budget;
        # back off to the largest value strictly below the tie group
        k = int(np.searchsorted(a, a[k - 1], side="left"))
    mass_bound = float(a[k - 1]) if k > 0 else 0.0
    return {"threshold": min(rel_bound, mass_bound), "rel_bound": rel_bound, "mass_bound": mass_bound}


def zero_mask(weight: np.ndarray, qweight: Optional[np.ndarray] = None,
              rel_tol: float = DEFAULT_REL_TOL, mass_tol: float = DEFAULT_MASS_TOL):
    """Return (mask, info). `mask` has the weight's shape, True where zero."""
    if qweight is not None:
        q = np.asarray(qweight)
        return q == 0, {"method": "quantized", "threshold": 0.0}
    w = np.asarray(weight)
    th = float_zero_threshold(w, rel_tol, mass_tol)
    mask = np.abs(w) <= th["threshold"]
    return mask, {"method": "float", "rel_tol": rel_tol, "mass_tol": mass_tol, **th}


def layer_sparsity(weight: np.ndarray, qweight: Optional[np.ndarray] = None,
                   bias: Optional[np.ndarray] = None, groups: int = 1,
                   in_channels: Optional[int] = None, **tol) -> dict:
    """Summary + per-level masks for one Conv/Linear weight ``[F, C/groups, *k]``.

    Returns a dict with:
      mask_weight  bool [F, C, *k]
      mask_kernel  bool [F, C]     (every weight of that kernel is zero)
      mask_filter  bool [F]        (every weight of that filter is zero)
      mask_input   bool [in_channels]  real input channels no filter reads
      zero_weights_per_filter / zero_kernels_per_filter  int [F]
      info         threshold/method details + totals (JSON-friendly)
    """
    w = np.asarray(weight)
    mask, info = zero_mask(w, qweight, **tol)
    F, C = w.shape[0], w.shape[1] if w.ndim > 1 else 1
    flat = mask.reshape(F, C, -1)
    mask_kernel = flat.all(axis=2)
    mask_filter = mask_kernel.all(axis=1)

    in_channels = in_channels or C * groups
    per_group_f = max(1, F // max(groups, 1))
    mask_input = np.ones(in_channels, dtype=bool)
    for g in range(max(groups, 1)):
        rows = mask_kernel[g * per_group_f:(g + 1) * per_group_f]  # [F/g, C]
        cols = rows.all(axis=0) if rows.size else np.ones(C, dtype=bool)
        mask_input[g * C:(g + 1) * C] = cols[: max(0, min(C, in_channels - g * C))]

    bias_zero = None
    if bias is not None:
        b = np.abs(np.asarray(bias))
        bias_zero = b <= (info["threshold"] if info["method"] == "float" else 0.0)

    info.update({
        "n_weights": int(mask.size), "zero_weights": int(mask.sum()),
        "n_kernels": int(mask_kernel.size), "zero_kernels": int(mask_kernel.sum()),
        "n_filters": int(F), "zero_filters": int(mask_filter.sum()),
        # a "zero" filter with a non-zero bias still emits a constant
        "zero_filters_with_bias": int((mask_filter & ~bias_zero).sum()) if bias_zero is not None else 0,
        "n_inputs": int(in_channels), "unused_inputs": int(mask_input.sum()),
        "max_abs": float(np.abs(w).max()) if w.size else 0.0,
    })
    return {
        "mask_weight": mask, "mask_kernel": mask_kernel, "mask_filter": mask_filter,
        "mask_input": mask_input, "bias_zero": bias_zero,
        "zero_weights_per_filter": flat.reshape(F, -1).sum(axis=1).astype(np.int64),
        "zero_kernels_per_filter": mask_kernel.sum(axis=1).astype(np.int64),
        "info": info,
    }
