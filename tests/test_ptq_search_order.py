"""
Regression test for the PTQ search-order bug: QuantResNet18 declares
input_quant AFTER layer1..layer4 in __init__ (models/resnet_quant.py), but
forward() calls it FIRST. mgr.quantizers iterates in declaration order, so
the per-quantizer search loop in find_perfect_lsbs_imagenet_ptq.py used to
search input_quant last (~29/30) instead of first — meaning 28 other
activation quantizers were searched against an unoptimized input_quant range,
then went stale once input_quant's own search later changed it.

QuantizerManager.quantizers_in_execution_order() fixes this by sorting on
inference_sequence_id, which base_quantizer.py assigns via a monotonic counter
on each quantizer's first forward() call — i.e. true forward execution order.
"""

import pytest
import torch

from quantizers.manager import QuantizerManager
from quantizers.base_quantizer import BaseQuantizer
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorActivationQuant
from models.resnet_quant import QuantResNet18
from examples.find_perfect_lsbs_imagenet_ptq import _assign_descriptive_ids


class _AQ(FixedPointPerTensorActivationQuant):
    bit_width = 8


def _build_and_run_forward():
    QuantizerManager().reset()
    model = QuantResNet18(num_classes=10, weight_quant=None, act_quant=_AQ)
    _assign_descriptive_ids(model)
    # Quantizers are freshly built (search_done=False); disable quantization
    # so eval-mode forward doesn't trip the uncalibrated-quantizer guard. The
    # forward pass itself still assigns inference_sequence_id to every
    # quantizer it actually reaches, which is all this test needs.
    QuantizerManager().disable_quantization()
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, 3, 64, 64))
    return model


def test_named_modules_order_puts_input_quant_last():
    """Sanity check on the bug itself: declaration order really does put
    input_quant after the layer blocks."""
    model = _build_and_run_forward()
    quant_names = [
        name for name, m in model.named_modules() if isinstance(m, BaseQuantizer)
    ]
    input_idx = next(i for i, n in enumerate(quant_names) if n.startswith("input_quant"))
    layer1_idx = next(i for i, n in enumerate(quant_names) if n.startswith("layer1"))
    assert input_idx > layer1_idx, (
        "expected named_modules() declaration order to put input_quant after "
        "layer1 (the bug this test guards against) — if this now fails, "
        "models/resnet_quant.py's __init__ order may have changed"
    )


def test_quantizers_in_execution_order_puts_input_quant_first():
    model = _build_and_run_forward()
    ordered = QuantizerManager().quantizers_in_execution_order()

    ordered_names = [q.quant_id for q in ordered]
    input_idx = ordered_names.index("input_quant_act")
    layer1_idx = next(i for i, n in enumerate(ordered_names) if n.startswith("layer1"))
    post_pool_idx = ordered_names.index("post_pool_quant_act")

    assert input_idx == 0, f"input_quant must be searched first, got position {input_idx}"
    assert input_idx < layer1_idx < post_pool_idx, (
        "execution order must be input -> layer1 -> post_pool"
    )


def test_quantizers_in_execution_order_is_monotonic_in_sequence_id():
    model = _build_and_run_forward()
    ordered = QuantizerManager().quantizers_in_execution_order()
    seq_ids = [q.inference_sequence_id for q in ordered]
    assert seq_ids == sorted(seq_ids)


def test_quantizers_in_execution_order_raises_before_any_forward_pass():
    QuantizerManager().reset()
    model = QuantResNet18(num_classes=10, weight_quant=None, act_quant=_AQ)
    _assign_descriptive_ids(model)
    with pytest.raises(RuntimeError, match="no quantizer has been reached"):
        QuantizerManager().quantizers_in_execution_order()
