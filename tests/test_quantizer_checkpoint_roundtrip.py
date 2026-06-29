"""
Checkpoint round-trip tests for quantizer calibration state.

The core property under test: if a model has some quantizers calibrated and
some not, saving its state_dict and loading it into a FRESH instance of the
same model must reproduce that exact per-quantizer state (search_done,
search_result_lsb, search_result_is_signed, annealing_alpha) — and a fully
calibrated model must actually produce quantized output after the round trip,
not just report the right flags.

Also covers a real, empirically-confirmed risk: Brevitas recreates the
tensor_quant proxy submodule on every load_state_dict() call (verified
directly — object identity differs before vs. after, even when reloading a
model's own state_dict onto itself). Two consequences:

  1. A quantizer object reference captured BEFORE load_state_dict() is a
     "ghost" afterward — it is no longer the object actually wired into the
     model, and inspecting it post-load shows stale, pre-load state.
  2. QuantizerManager.register_quantizer() never removes superseded objects,
     so the old (now-orphaned, unreachable-via-named_modules) proxies stay in
     QuantizerManager().quantizers indefinitely after a load — empirically
     confirmed: a 3-quantizer model's registry grows to 6 entries after one
     load_state_dict() call onto itself. Any code that iterates
     mgr.quantizers.values() directly (instead of re-deriving from
     model.named_modules()) after a load is therefore at risk of touching
     dead objects with zero effect on the real model, or of double-counting
     in progress/diagnostics output. The mitigation already used by the real
     training flow is training_harness.trainer_v2._reset_and_register(model),
     which wipes and rebuilds the registry purely from the model's current
     module tree — this file locks that mitigation in with a test.
"""

import pytest
import torch
import torch.nn as nn
import brevitas.nn as qnn

from quantizers.manager import QuantizerManager
from quantizers.base_quantizer import BaseQuantizer
from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorWeightQuant,
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorBiasQuant,
)
from training_harness.trainer_v2 import _reset_and_register


class _WQ(FixedPointPerTensorWeightQuant):
    bit_width = 8


class _AQ(FixedPointPerTensorActivationQuant):
    bit_width = 8


class _BQ(FixedPointPerTensorBiasQuant):
    bit_width = 8


class _RoundTripModel(nn.Module):
    """One quantizer of each role. act_out is the final op, so the model's
    return value IS the activation quantizer's output directly — letting
    tests check "is the model's output quantized" without reaching into
    internals."""

    def __init__(self):
        super().__init__()
        self.conv = qnn.QuantConv2d(3, 4, 3, bias=True, weight_quant=_WQ, bias_quant=_BQ)
        self.act_out = qnn.QuantIdentity(act_quant=_AQ)

    def forward(self, x):
        return self.act_out(self.conv(x))


def _quantizers_by_path(model: nn.Module) -> dict[str, BaseQuantizer]:
    return {name: m for name, m in model.named_modules() if isinstance(m, BaseQuantizer)}


def _by_role(quantizers: dict[str, BaseQuantizer]) -> dict[str, BaseQuantizer]:
    return {q.quantizer_role: q for q in quantizers.values()}


@pytest.fixture(autouse=True)
def reset_manager():
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


class TestMixedCalibrationRoundtrip:
    """Some quantizers calibrated, some not -- saved, then loaded into a
    fresh model instance. Every quantizer's state must come through exactly
    as saved, matched by structural position, not just by chance."""

    def test_calibrated_and_uncalibrated_state_both_round_trip_correctly(self):
        model_a = _RoundTripModel()
        by_role_a = _by_role(_quantizers_by_path(model_a))

        # Calibrate weight + activation with distinct, deliberately chosen
        # values; leave bias untouched (fresh/uncalibrated).
        by_role_a["weight"].search_result_lsb.fill_(-9)
        by_role_a["weight"].search_result_is_signed.fill_(True)
        by_role_a["weight"].search_done.fill_(True)
        by_role_a["weight"].annealing_alpha.data.fill_(1.0)

        by_role_a["activation"].search_result_lsb.fill_(-4)
        by_role_a["activation"].search_result_is_signed.fill_(False)
        by_role_a["activation"].search_done.fill_(True)
        by_role_a["activation"].annealing_alpha.data.fill_(1.0)

        assert by_role_a["bias"].search_done.item() is False

        saved_state = model_a.state_dict()

        # Fresh model instance + fresh manager registration.
        QuantizerManager().reset()
        model_b = _RoundTripModel()
        incompatible = model_b.load_state_dict(saved_state, strict=False)
        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []

        by_role_b = _by_role(_quantizers_by_path(model_b))

        assert by_role_b["weight"].search_done.item() is True
        assert int(by_role_b["weight"].search_result_lsb.item()) == -9
        assert bool(by_role_b["weight"].search_result_is_signed.item()) is True
        assert by_role_b["weight"].annealing_alpha.item() == pytest.approx(1.0)

        assert by_role_b["activation"].search_done.item() is True
        assert int(by_role_b["activation"].search_result_lsb.item()) == -4
        assert bool(by_role_b["activation"].search_result_is_signed.item()) is False
        assert by_role_b["activation"].annealing_alpha.item() == pytest.approx(1.0)

        # The uncalibrated quantizer must stay uncalibrated -- not flipped to
        # True, and not silently inheriting another role's LSB.
        assert by_role_b["bias"].search_done.item() is False

    def test_eval_mode_forward_raises_for_the_still_uncalibrated_quantizer(self):
        """Proves the loaded 'uncalibrated' state is real, not just a flag
        that happens to read False: running the model in eval mode (where
        BaseQuantizer.forward() raises if alpha > 0 and search_done is False)
        must actually trip the existing uncalibrated-quantizer guard."""
        model_a = _RoundTripModel()
        by_role_a = _by_role(_quantizers_by_path(model_a))
        for role in ("weight", "activation", "bias"):
            by_role_a[role].search_done.fill_(True)
            by_role_a[role].annealing_alpha.data.fill_(1.0)
        # Now un-calibrate just the bias quantizer again, but leave alpha=1.0
        # (active) -- exactly the "preserved roles + one fresh quantizer"
        # shape seen with --init-from-ptq in the real pipeline.
        by_role_a["bias"].search_done.fill_(False)

        saved_state = model_a.state_dict()

        QuantizerManager().reset()
        model_b = _RoundTripModel()
        model_b.load_state_dict(saved_state, strict=False)
        model_b.eval()

        with pytest.raises(RuntimeError, match="has not been calibrated"):
            with torch.no_grad():
                model_b(torch.randn(2, 3, 8, 8))


class TestFullyCalibratedRoundtripProducesQuantizedOutput:

    def test_model_output_lands_on_the_activation_quantizer_grid_after_load(self):
        model_a = _RoundTripModel()
        by_role_a = _by_role(_quantizers_by_path(model_a))
        lsbs = {"weight": -9, "bias": -11, "activation": -4}
        signed = {"weight": True, "bias": True, "activation": False}
        for role, q in by_role_a.items():
            q.search_result_lsb.fill_(lsbs[role])
            q.search_result_is_signed.fill_(signed[role])
            q.search_done.fill_(True)
            q.annealing_alpha.data.fill_(1.0)

        saved_state = model_a.state_dict()

        QuantizerManager().reset()
        model_b = _RoundTripModel()
        model_b.load_state_dict(saved_state, strict=False)

        model_b.eval()
        x = torch.randn(2, 3, 8, 8) * 5.0
        with torch.no_grad():
            out = model_b(x)

        step = 2.0 ** lsbs["activation"]
        ratio = out / step
        assert torch.allclose(ratio, ratio.round(), atol=1e-3), (
            "model output is not aligned to the activation quantizer's grid "
            "-- the loaded calibration state was not actually applied"
        )

    def test_quantized_output_actually_differs_from_float_reference(self):
        """Guards against a no-op that coincidentally lands on-grid: with
        quantization disabled, the model must produce different output."""
        model_a = _RoundTripModel()
        by_role_a = _by_role(_quantizers_by_path(model_a))
        for role, lsb in (("weight", -9), ("bias", -11), ("activation", -4)):
            by_role_a[role].search_result_lsb.fill_(lsb)
            by_role_a[role].search_done.fill_(True)
            by_role_a[role].annealing_alpha.data.fill_(1.0)

        saved_state = model_a.state_dict()

        QuantizerManager().reset()
        model_b = _RoundTripModel()
        model_b.load_state_dict(saved_state, strict=False)
        model_b.eval()

        x = torch.randn(2, 3, 8, 8) * 5.0
        with torch.no_grad():
            out_quantized = model_b(x)

        QuantizerManager().disable_quantization()
        with torch.no_grad():
            out_float = model_b(x)
        QuantizerManager().enable_quantization()

        assert not torch.allclose(out_quantized, out_float), (
            "quantized and float outputs are identical -- quantization had "
            "no effect after loading the checkpoint"
        )


class TestStaleQuantizerReferencesAfterLoad:
    """The 'ghost copy' risk: a reference captured before load_state_dict()
    is silently wrong afterward."""

    def test_quantizer_object_identity_changes_after_load(self):
        model_a = _RoundTripModel()
        for q in _quantizers_by_path(model_a).values():
            q.search_result_lsb.fill_(-9)
            q.search_done.fill_(True)
            q.annealing_alpha.data.fill_(1.0)
        saved_state = model_a.state_dict()

        QuantizerManager().reset()
        model_b = _RoundTripModel()
        stale_refs = _quantizers_by_path(model_b)
        assert all(not q.search_done.item() for q in stale_refs.values())

        model_b.load_state_dict(saved_state, strict=False)
        fresh_refs = _quantizers_by_path(model_b)

        for name, fresh_q in fresh_refs.items():
            stale_q = stale_refs[name]
            assert stale_q is not fresh_q, (
                f"expected load_state_dict to recreate the tensor_quant "
                f"proxy for {name!r} (this is the Brevitas behavior this "
                f"test documents); if this now fails, the underlying "
                f"behavior changed and the 'stale reference' risk may no "
                f"longer apply -- re-check before relaxing this test"
            )
            # The reference captured BEFORE the load does not reflect it.
            assert stale_q.search_done.item() is False
            # Only a reference re-fetched AFTER the load does.
            assert fresh_q.search_done.item() is True
            assert int(fresh_q.search_result_lsb.item()) == -9


class TestManagerRegistryOrphansAfterLoad:
    """The deeper version of the 'ghost copy' risk: superseded quantizer
    objects are never removed from QuantizerManager().quantizers."""

    def test_load_state_dict_orphans_old_quantizers_in_the_registry(self):
        model = _RoundTripModel()
        n_before = len(QuantizerManager().quantizers)
        assert n_before == 3

        model.load_state_dict(model.state_dict(), strict=False)

        reachable_ids = {id(m) for m in _quantizers_by_path(model).values()}
        registered = list(QuantizerManager().quantizers.values())
        orphans = [q for q in registered if id(q) not in reachable_ids]

        assert len(registered) == 6, (
            "expected the registry to grow by exactly the original 3 "
            "quantizers (now orphaned) plus the 3 freshly recreated ones; "
            "if this changed, re-validate the premise of this whole test class"
        )
        assert len(orphans) == 3

    def test_reset_and_register_eliminates_the_orphans(self):
        """The mitigation already used by the real training flow
        (training_harness/trainer_v2.py calls this early in fit(), after any
        --init-from-ptq checkpoint load and before _activate_qat())."""
        model = _RoundTripModel()
        model.load_state_dict(model.state_dict(), strict=False)
        assert len(QuantizerManager().quantizers) == 6  # orphans present

        _reset_and_register(model)

        reachable_ids = {id(m) for m in _quantizers_by_path(model).values()}
        registered = list(QuantizerManager().quantizers.values())
        assert len(registered) == len(reachable_ids) == 3
        assert all(id(q) in reachable_ids for q in registered)

    def test_real_pipeline_shape_weights_then_activations_load_has_no_orphans(self):
        """Integration check mirroring find_perfect_lsbs_imagenet_ptq.py's
        --init-from-ckpt flow: build, calibrate, save; build fresh, load;
        confirm _reset_and_register (the mitigation) leaves the registry in
        exact 1:1 correspondence with the model's real quantizers."""
        model_a = _RoundTripModel()
        for q in _quantizers_by_path(model_a).values():
            q.search_result_lsb.fill_(-9)
            q.search_done.fill_(True)
            q.annealing_alpha.data.fill_(1.0)
        saved_state = model_a.state_dict()

        QuantizerManager().reset()
        model_b = _RoundTripModel()
        model_b.load_state_dict(saved_state, strict=False)  # creates 3 orphans
        _reset_and_register(model_b)  # the required resync after any load

        reachable_ids = {id(m) for m in _quantizers_by_path(model_b).values()}
        registered = list(QuantizerManager().quantizers.values())
        assert len(registered) == 3
        assert all(id(q) in reachable_ids for q in registered)


class TestDoubleLoadStateDictIsUnsafe:
    """Regression for a previously-discovered real bug in this codebase: two
    sequential load_state_dict() calls on the same model silently lose the
    first call's effect on any buffer key absent from the second payload,
    because each call recreates the tensor_quant proxy from scratch. The safe
    pattern is exactly one load_state_dict() call per model lifecycle (merge
    dicts in Python first if combining two sources is ever needed)."""

    def test_second_load_wipes_first_loads_buffers_not_present_in_second_payload(self):
        # Source A: calibrates only the weight quantizer.
        model_a = _RoundTripModel()
        _by_role(_quantizers_by_path(model_a))["weight"].search_result_lsb.fill_(-9)
        _by_role(_quantizers_by_path(model_a))["weight"].search_done.fill_(True)
        full_state_a = model_a.state_dict()
        # Simulate a "partial" checkpoint containing only the weight quantizer's keys.
        weight_only_keys = {
            k: v for k, v in full_state_a.items()
            if "weight_quant" in k
        }

        # Source B: calibrates only the activation quantizer, to a DIFFERENT LSB.
        QuantizerManager().reset()
        model_src_b = _RoundTripModel()
        _by_role(_quantizers_by_path(model_src_b))["activation"].search_result_lsb.fill_(-4)
        _by_role(_quantizers_by_path(model_src_b))["activation"].search_done.fill_(True)
        full_state_b = model_src_b.state_dict()
        act_only_keys = {
            k: v for k, v in full_state_b.items()
            if "act_quant" in k
        }

        QuantizerManager().reset()
        target = _RoundTripModel()
        target.load_state_dict(weight_only_keys, strict=False)
        by_role = _by_role(_quantizers_by_path(target))
        assert by_role["weight"].search_done.item() is True  # first load took effect

        # Second load_state_dict() call recreates ALL proxies again, including
        # the weight one that isn't present in this second, smaller payload --
        # it silently reverts to its freshly-constructed (uncalibrated) default.
        target.load_state_dict(act_only_keys, strict=False)
        by_role_after = _by_role(_quantizers_by_path(target))
        assert by_role_after["activation"].search_done.item() is True
        assert by_role_after["weight"].search_done.item() is False, (
            "expected the second load_state_dict() call to silently wipe the "
            "first call's effect on the weight quantizer -- if this now "
            "passes, Brevitas's proxy-recreation behavior changed and the "
            "single-load-per-lifecycle convention in this codebase "
            "(find_perfect_lsbs_imagenet_ptq.py's --init-from-ckpt chaining) "
            "may no longer be strictly necessary"
        )

    def test_merging_state_dicts_before_a_single_load_is_the_safe_pattern(self):
        """The actual mitigation: merge both partial payloads in Python
        first, then call load_state_dict() exactly once."""
        model_a = _RoundTripModel()
        _by_role(_quantizers_by_path(model_a))["weight"].search_result_lsb.fill_(-9)
        _by_role(_quantizers_by_path(model_a))["weight"].search_done.fill_(True)
        weight_only_keys = {
            k: v for k, v in model_a.state_dict().items() if "weight_quant" in k
        }

        QuantizerManager().reset()
        model_src_b = _RoundTripModel()
        _by_role(_quantizers_by_path(model_src_b))["activation"].search_result_lsb.fill_(-4)
        _by_role(_quantizers_by_path(model_src_b))["activation"].search_done.fill_(True)
        act_only_keys = {
            k: v for k, v in model_src_b.state_dict().items() if "act_quant" in k
        }

        merged = {**weight_only_keys, **act_only_keys}

        QuantizerManager().reset()
        target = _RoundTripModel()
        target.load_state_dict(merged, strict=False)

        by_role = _by_role(_quantizers_by_path(target))
        assert by_role["weight"].search_done.item() is True
        assert int(by_role["weight"].search_result_lsb.item()) == -9
        assert by_role["activation"].search_done.item() is True
        assert int(by_role["activation"].search_result_lsb.item()) == -4
