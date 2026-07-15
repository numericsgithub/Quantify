"""
test_qat_group_controls.py — Phase 4 group-targeted QAT controls.

Exercises the QuantizerManager group mutators and the ControlManager commands
that wrap them, on the REAL MNIST fixed-point quantizers (9: 3 weight, 2 bias,
4 activation). The central guarantee is precision: a group op changes ONLY that
group and leaves the others untouched, and unknown-role quantizers are counted
and never silently mutated.
"""

from types import SimpleNamespace

import pytest

from quantizers.manager import QuantizerManager
from training_harness.trainer_v2 import _reset_and_register
from training_harness.api.control import ControlManager, ControlValidationError


@pytest.fixture
def mgr():
    from examples.mnist_qat_v2 import MNISTQuantNet
    model = MNISTQuantNet()
    _reset_and_register(model)          # register + stamp roles (alpha:=1.0 each)
    m = QuantizerManager()
    yield m
    m.reset()


@pytest.fixture
def ctrl(mgr):
    # QAT appliers operate on the singleton, not the trainer, so a trainer-less
    # ControlManager is enough to drive submit/drain.
    return ControlManager(trainer=None, collector=None, callbacks=None)


def _alphas(mgr, group):
    return [float(q.annealing_alpha) for q in mgr.select_quantizers(group)]


# ---------------------------------------------------------------------------
# Manager-level: group precision
# ---------------------------------------------------------------------------

def test_group_counts(mgr):
    assert len(mgr.select_quantizers("weights")) == 3
    assert len(mgr.select_quantizers("biases")) == 2
    assert len(mgr.select_quantizers("activations")) == 4
    assert len(mgr.select_quantizers("all")) == 9


def test_set_alpha_only_affects_target_group(mgr):
    mgr.set_group_annealing_alpha("all", 1.0)
    res = mgr.set_group_annealing_alpha("weights", 0.3)
    assert res["count"] == 3 and res["unknown_role"] == 0
    assert all(a == pytest.approx(0.3) for a in _alphas(mgr, "weights"))
    assert all(a == pytest.approx(1.0) for a in _alphas(mgr, "activations"))
    assert all(a == pytest.approx(1.0) for a in _alphas(mgr, "biases"))


def test_disable_group_zeros_alpha_and_step_only_for_group(mgr):
    mgr.set_group_annealing_alpha("all", 1.0)
    for q in mgr.select_quantizers("all"):
        q.annealing_alpha_step = 0.1
    mgr.disable_group("activations")
    for q in mgr.select_quantizers("activations"):
        assert float(q.annealing_alpha) == 0.0 and q.annealing_alpha_step == 0.0
    # weights untouched — the headline "quantize weights, leave activations float"
    for q in mgr.select_quantizers("weights"):
        assert float(q.annealing_alpha) == pytest.approx(1.0)
        assert q.annealing_alpha_step == pytest.approx(0.1)


def test_ramp_sets_zero_and_step(mgr):
    mgr.set_group_annealing_ramp("biases", 4)
    for q in mgr.select_quantizers("biases"):
        assert float(q.annealing_alpha) == 0.0
        assert q.annealing_alpha_step == pytest.approx(0.25)


def test_set_step_only(mgr):
    mgr.set_group_annealing_alpha("all", 0.5)
    mgr.set_group_annealing_step("weights", 0.2)
    for q in mgr.select_quantizers("weights"):
        assert q.annealing_alpha_step == pytest.approx(0.2)
        assert float(q.annealing_alpha) == pytest.approx(0.5)   # alpha untouched


def test_recalibrate_group_clears_search_done_only_for_group(mgr):
    for q in mgr.select_quantizers("all"):
        q.search_done.fill_(True)
    mgr.recalibrate_group("weights")
    assert all(not bool(q.search_done) for q in mgr.select_quantizers("weights"))
    assert all(bool(q.search_done) for q in mgr.select_quantizers("activations"))


def test_unknown_role_counted_and_not_mutated(mgr):
    # Inject an unclassifiable quantizer; a "weights" op must not touch it and
    # must report it as unknown.
    mgr.quantizers["mystery"] = SimpleNamespace()   # no role, no buffers
    res = mgr.set_group_annealing_alpha("weights", 0.1)
    assert res["count"] == 3 and res["unknown_role"] == 1
    assert not hasattr(mgr.quantizers["mystery"], "annealing_alpha")


# ---------------------------------------------------------------------------
# Manager-level: LSB by id
# ---------------------------------------------------------------------------

def test_set_lsb_by_id(mgr):
    qid = next(q for q in mgr.quantizers if q.endswith("_weight"))
    res = mgr.set_lsb(qid, -5)
    assert res == {"quant_id": qid, "role": "weight", "lsb": -5}
    assert int(mgr.quantizers[qid].search_result_lsb) == -5


def test_set_lsb_unknown_id_raises(mgr):
    with pytest.raises(KeyError):
        mgr.set_lsb("does_not_exist", -3)


def test_set_lsb_non_fixedpoint_raises(mgr):
    mgr.quantizers["fake"] = SimpleNamespace()      # no search_result_lsb
    with pytest.raises(TypeError):
        mgr.set_lsb("fake", -3)


def test_describe_quantizers_shape(mgr):
    rows = mgr.describe_quantizers()
    assert len(rows) == 9
    r = rows[0]
    assert {"quant_id", "role", "alpha", "alpha_step", "lsb"} <= set(r)


# ---------------------------------------------------------------------------
# ControlManager command flow
# ---------------------------------------------------------------------------

def test_set_annealing_command_absolute(mgr, ctrl):
    mgr.set_group_annealing_alpha("all", 1.0)
    cmd = ctrl.submit("set_annealing", {"group": "weights", "mode": "absolute", "alpha": 0.25})
    ctrl.drain("step")
    assert ctrl.get_command(cmd.id)["status"] == "applied"
    assert all(a == pytest.approx(0.25) for a in _alphas(mgr, "weights"))
    assert all(a == pytest.approx(1.0) for a in _alphas(mgr, "activations"))


def test_disable_command_needs_confirm(ctrl):
    with pytest.raises(ControlValidationError):
        ctrl.submit("disable_quant", {"group": "activations"})


def test_disable_command_epoch_boundary(mgr, ctrl):
    cmd = ctrl.submit("disable_quant", {"group": "activations", "confirm": True})
    ctrl.drain("step")                                  # wrong boundary -> not applied
    assert ctrl.get_command(cmd.id)["status"] == "pending"
    ctrl.drain("epoch")
    assert ctrl.get_command(cmd.id)["status"] == "applied"
    assert all(a == 0.0 for a in _alphas(mgr, "activations"))
    assert all(a == pytest.approx(1.0) for a in _alphas(mgr, "weights"))


def test_recalibrate_command_needs_confirm(ctrl):
    with pytest.raises(ControlValidationError):
        ctrl.submit("recalibrate", {"group": "all"})


def test_set_lsb_command(mgr, ctrl):
    qid = next(q for q in mgr.quantizers if q.endswith("_weight"))
    cmd = ctrl.submit("set_lsb", {"quant_id": qid, "lsb": -7})
    ctrl.drain("step")
    assert ctrl.get_command(cmd.id)["status"] == "applied"
    assert int(mgr.quantizers[qid].search_result_lsb) == -7


def test_invalid_group_rejected(ctrl):
    with pytest.raises(ControlValidationError):
        ctrl.submit("set_annealing", {"group": "weight", "mode": "absolute", "alpha": 1.0})


def test_invalid_mode_rejected(ctrl):
    with pytest.raises(ControlValidationError):
        ctrl.submit("set_annealing", {"group": "weights", "mode": "bogus"})


def test_lsb_out_of_range_rejected(ctrl):
    with pytest.raises(ControlValidationError):
        ctrl.submit("set_lsb", {"quant_id": "x", "lsb": 999})


def test_unknown_warning_in_command_result(mgr, ctrl):
    mgr.quantizers["mystery"] = SimpleNamespace()
    cmd = ctrl.submit("set_annealing", {"group": "weights", "mode": "absolute", "alpha": 0.5})
    ctrl.drain("step")
    result = ctrl.get_command(cmd.id)["result"]
    assert "WARNING" in result and "unknown role" in result
