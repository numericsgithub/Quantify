"""
Descriptive, location-based quantizer naming.

`QuantizerManager.register_quantizer()` (quantizers/manager.py) assigns every
quantizer a generic `quant_N` id at construction time -- it has no idea where
in the model the quantizer sits, so the id says nothing useful on its own
(e.g. in a lifecycle log line, a diagnostics filename, or a PTQ search plot).

`assign_descriptive_quant_ids()` replaces those generic ids with ones derived
from the model's own `named_modules()` paths, e.g.:

    features.3.conv.0.weight_quant.tensor_quant  ->  quant_id "features_3_conv_0_weight"
                                                       display_name "features.3.conv.0 [weight]"

This only works after the quantizer is actually wired into a real model (the
name comes from where Brevitas attached its proxy, e.g. `.weight_quant`,
`.bias_quant`, `.act_quant`), so call it once the model is fully built --
not from inside a quantizer's own `__init__`.

Consolidates what used to be two near-identical private implementations
(`training_harness/trainer_v2.py::_make_quant_id` +
`examples/find_perfect_lsbs_imagenet_ptq.py::_assign_descriptive_ids`) into
one shared, public utility both now delegate to.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch.nn as nn

from quantizers.base_quantizer import BaseQuantizer
from quantizers.manager import QuantizerManager


# Maps the Brevitas proxy suffix in a named_modules() path to an explicit
# role label. Checked in order so longer (more specific) suffixes win over
# shorter ones that would also match (e.g. the fused-activation path before
# the plain ".act_quant.tensor_quant" one).
_PROXY_SUFFIX_TO_ROLE = [
    (".weight_quant.tensor_quant", "weight"),
    (".bias_quant.tensor_quant", "bias"),
    (".act_quant.fused_activation_quant_proxy.tensor_quant", "act"),
    (".act_quant.tensor_quant", "act"),
    (".input_quant.tensor_quant", "act_in"),
    (".output_quant.tensor_quant", "act_out"),
    # Proxies without a nested tensor_quant (non-standard direct attachment).
    (".weight_quant", "weight"),
    (".bias_quant", "bias"),
    (".input_quant", "act_in"),
    (".output_quant", "act_out"),
    (".act_quant", "act"),
]


def split_quant_path(path: str) -> Tuple[str, str]:
    """Split a `named_modules()` dotted path into `(parent_path, role)`.

    `role` is `""` when no known Brevitas proxy suffix matched -- callers
    should fall back to using `path` itself as the identifier in that case.
    """
    for suffix, role in _PROXY_SUFFIX_TO_ROLE:
        if path.endswith(suffix):
            return path[: -len(suffix)], role
    return path, ""


def assign_descriptive_quant_ids(
    model: nn.Module, manager: Optional[QuantizerManager] = None
) -> None:
    """
    Replace every `BaseQuantizer` submodule's generic `quant_N` id with a
    location-based one derived from `model`'s `named_modules()` paths, and
    keep `manager.quantizers`'s registry keys in sync.

    Each quantizer gets two attributes:
      `quant_id`     -- log-/filename-safe (underscores, no dots), e.g.
                        `"features_3_conv_0_weight"`. Used in lifecycle log
                        messages, diagnostics filenames, and as the
                        `QuantizerManager.quantizers` registry key.
      `display_name` -- human-readable, keeps the original dots and shows
                        the role in brackets, e.g. `"features.3.conv.0 [weight]"`.

    Safe to call at any point in a quantizer's lifecycle -- it only renames,
    it never touches calibration or annealing state. Call it once the model
    is fully constructed (after every `QuantConv2d`/`QuantLinear`/etc. has
    attached its proxies), typically right after building the model and
    before training starts.

    Duplicate-id collisions (two proxies resolving to the same parent+role --
    rare, but possible with hand-built module trees) are disambiguated with a
    numeric suffix rather than raising, since this is a display/filename
    convenience, not a correctness-critical identifier.

    Any quantizer registered with `manager` but not reachable via
    `model.named_modules()` (e.g. a throwaway Brevitas-internal proxy object
    that registers but is never wired into the real module tree -- see
    pitfall #9 in docs/llm/pitfalls/brevitas_pitfalls.md) keeps its existing
    `quant_id` as its `display_name`.
    """
    mgr = manager if manager is not None else QuantizerManager()

    seen: dict = {}  # qid -> collision count, for the numeric-suffix dedup
    for path, module in model.named_modules():
        if not isinstance(module, BaseQuantizer):
            continue

        parent, role = split_quant_path(path)
        if role:
            parent_us = parent.replace(".", "_")
            qid = f"{parent_us}_{role}" if parent_us else role
            display_name = f"{parent} [{role}]" if parent else f"[{role}]"
        else:
            qid = path.replace(".", "_")
            display_name = path

        if qid in seen:
            seen[qid] += 1
            dedup = f"_{seen[qid]}"
            qid += dedup
            display_name += dedup
        else:
            seen[qid] = 0

        module.quant_id = qid
        module.display_name = display_name

    # Keep the manager's registry keys in sync with the descriptive names.
    mgr.quantizers = {q.quant_id: q for q in mgr.quantizers.values()}

    # Ghost quantizers unreachable via named_modules() keep whatever quant_id
    # they already had (assigned at registration time) as their display name.
    for q in mgr.quantizers.values():
        if not hasattr(q, "display_name"):
            q.display_name = q.quant_id
