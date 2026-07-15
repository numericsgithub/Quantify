"""
test_select_quantizers.py — QuantizerManager role resolution + group selection.

Covers the Phase-1 keystone: select_quantizers(group), role_histogram(),
resolve_role()/stamp_roles(), and the "unknown role" accounting that guards
every group-targeted control from silently skipping quantizers.

These tests use lightweight stub quantizers (a quant_id and optionally a
quantizer_role attribute) registered directly on the singleton — the selection
logic only reads those two attributes, so no real Brevitas model is needed.
"""

from types import SimpleNamespace

import pytest

from quantizers.manager import QuantizerManager


def _stub(quant_id=None, quantizer_role=None):
    q = SimpleNamespace()
    if quant_id is not None:
        q.quant_id = quant_id
    if quantizer_role is not None:
        q.quantizer_role = quantizer_role
    return q


@pytest.fixture
def mgr():
    m = QuantizerManager()
    m.reset()
    yield m
    m.reset()


def _register(m, quantizers):
    """Populate the manager registry keyed by quant_id (mirrors V2)."""
    m.quantizers = {q.quant_id: q for q in quantizers}


# ---------------------------------------------------------------------------
# Role resolution
# ---------------------------------------------------------------------------

def test_resolve_role_from_quant_id_suffix(mgr):
    assert mgr.resolve_role(_stub("conv1_weight")) == "weight"
    assert mgr.resolve_role(_stub("conv1_bias")) == "bias"
    assert mgr.resolve_role(_stub("conv1_act_in")) == "activation"
    assert mgr.resolve_role(_stub("conv1_act_out")) == "activation"
    assert mgr.resolve_role(_stub("relu_act")) == "activation"


def test_resolve_role_falls_back_to_attribute(mgr):
    # V1-style id has no role suffix -> fall back to quantizer_role attribute.
    assert mgr.resolve_role(_stub("quant_0", quantizer_role="weight")) == "weight"
    assert mgr.resolve_role(_stub("quant_1", quantizer_role="activation")) == "activation"


def test_resolve_role_unknown(mgr):
    # No suffix and no usable attribute -> unknown.
    assert mgr.resolve_role(_stub("weird_module")) == "unknown"
    assert mgr.resolve_role(_stub("quant_2", quantizer_role="unknown")) == "unknown"


def test_quant_id_suffix_wins_over_attribute(mgr):
    # Structure-derived suffix is the primary signal even if the attribute disagrees.
    assert mgr.resolve_role(_stub("conv1_weight", quantizer_role="activation")) == "weight"


# ---------------------------------------------------------------------------
# select_quantizers
# ---------------------------------------------------------------------------

def test_filtering_per_role(mgr):
    _register(mgr, [
        _stub("c1_weight"), _stub("c2_weight"),
        _stub("c1_bias"),
        _stub("in_act"), _stub("c1_act_out"), _stub("fc_act_out"),
    ])
    assert {q.quant_id for q in mgr.select_quantizers("weights")} == {"c1_weight", "c2_weight"}
    assert {q.quant_id for q in mgr.select_quantizers("biases")} == {"c1_bias"}
    assert {q.quant_id for q in mgr.select_quantizers("activations")} == {
        "in_act", "c1_act_out", "fc_act_out"
    }


def test_all_returns_everything(mgr):
    quantizers = [_stub("c1_weight"), _stub("c1_bias"), _stub("in_act"), _stub("weird")]
    _register(mgr, quantizers)
    assert len(mgr.select_quantizers("all")) == len(quantizers)


def test_empty_group_returns_empty_list(mgr):
    _register(mgr, [_stub("c1_weight"), _stub("in_act")])
    assert mgr.select_quantizers("biases") == []


def test_empty_registry(mgr):
    assert mgr.select_quantizers("all") == []
    assert mgr.select_quantizers("weights") == []


def test_unknown_group_raises(mgr):
    _register(mgr, [_stub("c1_weight")])
    with pytest.raises(ValueError):
        mgr.select_quantizers("weight")   # singular is not a valid group
    with pytest.raises(ValueError):
        mgr.select_quantizers("garbage")


def test_unknown_role_excluded_from_named_groups(mgr):
    _register(mgr, [_stub("c1_weight"), _stub("weird_thing")])
    selected = mgr.select_quantizers("weights")
    assert {q.quant_id for q in selected} == {"c1_weight"}
    # The unknown-role stub is in "all" but no named group.
    assert len(mgr.select_quantizers("all")) == 2


# ---------------------------------------------------------------------------
# Histogram / unknown accounting
# ---------------------------------------------------------------------------

def test_role_histogram_counts(mgr):
    _register(mgr, [
        _stub("c1_weight"), _stub("c2_weight"),
        _stub("c1_bias"),
        _stub("in_act"), _stub("c1_act_out"),
        _stub("weird"), _stub("quant_9", quantizer_role="unknown"),
    ])
    hist = mgr.role_histogram()
    assert hist == {"weight": 2, "bias": 1, "activation": 2, "unknown": 2, "total": 7}


def test_unknown_role_count_accurate(mgr):
    _register(mgr, [_stub("c1_weight"), _stub("weird1"), _stub("weird2")])
    assert mgr.unknown_role_count() == 2


# ---------------------------------------------------------------------------
# stamp_roles
# ---------------------------------------------------------------------------

def test_stamp_roles_caches_role(mgr):
    q = _stub("c1_weight")
    _register(mgr, [q])
    assert not hasattr(q, "role")
    mgr.stamp_roles()
    assert q.role == "weight"


def test_stamped_role_is_used_by_selection(mgr):
    # Stamp an explicit role, then corrupt the source signals: selection must
    # trust the stamped value (the cached resolution).
    q = _stub("c1_weight")
    _register(mgr, [q])
    mgr.stamp_roles()
    q.quant_id = "now_looks_like_act"   # would resolve to activation if re-derived
    assert {x.quant_id for x in mgr.select_quantizers("weights")} == {"now_looks_like_act"}
