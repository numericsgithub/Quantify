"""
test_quant_id_naming.py — _make_quant_id role tagging for activation proxies.

Brevitas wraps both `act_quant` AND `input_quant`/`output_quant` tensor_quants
in a `fused_activation_quant_proxy`. If the suffix table only knows the fused
shape for `act_quant`, output/input activations fall through to a raw,
role-less name — so a non-fixed-point activation quantizer (which also lacks the
`quantizer_role` attribute) would resolve to "unknown" and be silently skipped
by group control. These tests lock in that the fused input/output shapes are
tagged from the structure alone.
"""

import pytest

from training_harness.trainer_v2 import _make_quant_id, _reset_and_register
from quantizers.manager import QuantizerManager


# ---------------------------------------------------------------------------
# _make_quant_id: fused activation proxy variants
# ---------------------------------------------------------------------------

def test_fused_output_and_input_get_role_tags():
    # The gap that was fixed: these used to fall through to the raw dotted name.
    assert _make_quant_id(
        "conv1.output_quant.fused_activation_quant_proxy.tensor_quant") == "conv1_act_out"
    assert _make_quant_id(
        "stem.input_quant.fused_activation_quant_proxy.tensor_quant") == "stem_act_in"


def test_existing_shapes_still_map():
    assert _make_quant_id("conv1.weight_quant.tensor_quant") == "conv1_weight"
    assert _make_quant_id("conv1.bias_quant.tensor_quant") == "conv1_bias"
    assert _make_quant_id("id.act_quant.fused_activation_quant_proxy.tensor_quant") == "id_act"
    assert _make_quant_id("conv1.output_quant.tensor_quant") == "conv1_act_out"


# ---------------------------------------------------------------------------
# End-to-end on the real MNIST V2 model
# ---------------------------------------------------------------------------

@pytest.fixture
def mnist_registered():
    from examples.mnist_qat_v2 import MNISTQuantNet
    model = MNISTQuantNet()
    _reset_and_register(model)
    mgr = QuantizerManager()
    yield mgr
    mgr.reset()


def test_no_raw_fallback_names(mnist_registered):
    # No quant_id should still carry the un-stripped proxy noise.
    bad = [qid for qid in mnist_registered.quantizers
           if "fused_activation_quant_proxy" in qid or "tensor_quant" in qid]
    assert bad == [], f"quant_ids fell through to the raw fallback: {bad}"


def test_output_activations_tagged_act_out(mnist_registered):
    # The conv output activations must now carry the _act_out tag.
    outs = [qid for qid in mnist_registered.quantizers if qid.endswith("_act_out")]
    assert len(outs) >= 2   # conv1 + conv2 (+ fc) output activations


def test_role_resolves_from_quant_id_without_attribute(mnist_registered):
    # resolve_role() consults the quant_id FIRST, so it returns "activation" for
    # every activation position regardless of the quantizer_role attribute — this
    # is what makes group targeting robust for non-fixed-point quantizers.
    mgr = mnist_registered
    acts = [q for qid, q in mgr.quantizers.items()
            if qid.endswith(("_act", "_act_in", "_act_out"))]
    assert acts
    for q in acts:
        assert mgr.resolve_role(q) == "activation"


def test_histogram_has_no_unknown(mnist_registered):
    assert mnist_registered.role_histogram()["unknown"] == 0
