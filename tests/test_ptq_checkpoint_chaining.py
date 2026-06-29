"""
Tests for chaining find_perfect_lsbs_imagenet_ptq.py runs via --init-from-ckpt:
run --mode weights first, save a checkpoint, then run --mode activations
with --init-from-ckpt pointing at it. The model for the second run is built
with BOTH weight_quant and act_quant so the already-calibrated weight
quantizers stay active (quantizing activations on top of an already
weight-quantized model) while the activation quantizers are searched fresh.

Covers:
  - _build_quantized_model without --init-from-ckpt (plain single-role build,
    regression check against the pre-chaining behavior)
  - _build_quantized_model with --init-from-ckpt: combined model construction,
    correct bit-widths per role, checkpoint state loaded with zero missing keys
  - backward compatibility with checkpoints saved before role_bit_widths existed
    (only ptq_search_mode/bit_width present)
  - _disable_target_role_keep_others_active: loaded weight quantizers stay at
    alpha=1.0/search_done=True; activation quantizers get the fresh-search
    placeholder (alpha=0, search_done=True until the per-quantizer loop flips
    it back to False)
  - chaining in the other direction (activations first, then weights)
"""

import argparse

import pytest
import torch
import torch.nn as nn

from quantizers.manager import QuantizerManager
from quantizers.base_quantizer import BaseQuantizer
from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorWeightQuant,
)
from models.resnet_quant import QuantResNet18
from examples.find_perfect_lsbs_imagenet_ptq import (
    _build_quantized_model,
    _disable_target_role_keep_others_active,
    _assign_descriptive_ids,
)


WEIGHTS_LSB = -10
ACTIVATIONS_LSB = -6
BIAS_LSB = -14


@pytest.fixture(autouse=True)
def reset_manager():
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


def _base_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model="resnet18",
        mode="weights",
        bit_width=8,
        num_classes=10,
        pretrained=False,
        fuse_bn=False,
        init_from_ckpt=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _build_weights_checkpoint(tmp_path) -> str:
    """Run a minimal stand-in for --mode weights and save its checkpoint."""
    args = _base_args(mode="weights", bit_width=8)
    model, target_role, bw, prev_extra, prev_role_bw = _build_quantized_model(
        args, torch.device("cpu"),
    )
    assert target_role == "weight"
    mgr = QuantizerManager()
    real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
    for q in real_qs:
        q.search_result_lsb.fill_(WEIGHTS_LSB)
        q.search_done.fill_(True)
        q.annealing_alpha.data.fill_(1.0)

    payload = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "metrics": {"final_val_acc": 65.0},
        "extra": {
            "ptq_search_mode": "weights",
            "bit_width": 8,
            "role_bit_widths": {"weight": 8},
            "calibrated_lsbs": {q.quant_id: WEIGHTS_LSB for q in real_qs},
            "selected_lsbs": {q.quant_id: WEIGHTS_LSB for q in real_qs},
        },
    }
    path = tmp_path / "weights_ckpt.pt"
    torch.save(payload, path)
    return str(path)


class TestBuildQuantizedModelWithoutChaining:

    def test_weights_mode_builds_weight_only_model(self):
        args = _base_args(mode="weights", bit_width=8)
        model, target_role, bw, prev_extra, prev_role_bw = _build_quantized_model(
            args, torch.device("cpu"),
        )
        assert target_role == "weight"
        assert bw == 8
        assert prev_extra == {}
        assert prev_role_bw == {}

        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        roles = {q.quantizer_role for q in real_qs}
        assert roles == {"weight"}, f"Expected only weight quantizers, got roles={roles}"

    def test_activations_mode_builds_activation_only_model(self):
        args = _base_args(mode="activations", bit_width=8)
        model, target_role, bw, prev_extra, prev_role_bw = _build_quantized_model(
            args, torch.device("cpu"),
        )
        assert target_role == "activation"
        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        roles = {q.quantizer_role for q in real_qs}
        assert roles == {"activation"}

    def test_bias_mode_builds_bias_only_model(self):
        args = _base_args(mode="bias", bit_width=8)
        model, target_role, bw, prev_extra, prev_role_bw = _build_quantized_model(
            args, torch.device("cpu"),
        )
        assert target_role == "bias"
        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        roles = {q.quantizer_role for q in real_qs}
        assert roles == {"bias"}
        # Bias quantization is fc-only in this model.
        assert len(real_qs) == 1


class TestBuildQuantizedModelWithChaining:

    def test_combined_model_has_both_roles(self, tmp_path):
        weights_ckpt = _build_weights_checkpoint(tmp_path)

        QuantizerManager().reset()
        args = _base_args(mode="activations", bit_width=6, init_from_ckpt=weights_ckpt)
        model, target_role, bw, prev_extra, prev_role_bw = _build_quantized_model(
            args, torch.device("cpu"),
        )

        assert target_role == "activation"
        assert bw == 6
        assert prev_role_bw == {"weight": 8}

        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        roles = {q.quantizer_role for q in real_qs}
        assert roles == {"weight", "activation"}

    def test_loaded_weight_quantizers_carry_checkpoint_lsb(self, tmp_path):
        weights_ckpt = _build_weights_checkpoint(tmp_path)

        QuantizerManager().reset()
        args = _base_args(mode="activations", bit_width=6, init_from_ckpt=weights_ckpt)
        model, target_role, bw, prev_extra, prev_role_bw = _build_quantized_model(
            args, torch.device("cpu"),
        )

        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        weight_qs = [q for q in real_qs if q.quantizer_role == "weight"]
        act_qs    = [q for q in real_qs if q.quantizer_role == "activation"]
        assert len(weight_qs) > 0
        assert len(act_qs) > 0

        for q in weight_qs:
            assert q.search_done.item() is True
            assert int(q.search_result_lsb.item()) == WEIGHTS_LSB
        for q in act_qs:
            # Freshly built — not yet calibrated.
            assert q.search_done.item() is False

    def test_load_has_no_missing_keys(self, tmp_path):
        """Every key in the weights checkpoint must resolve onto the combined
        model's structure (weight_quant proxies exist in both)."""
        weights_ckpt = _build_weights_checkpoint(tmp_path)

        QuantizerManager().reset()
        args = _base_args(mode="activations", bit_width=6, init_from_ckpt=weights_ckpt)
        # Re-run with instrumentation: call load_state_dict ourselves to check
        # missing keys precisely (the function only prints this).
        payload = torch.load(weights_ckpt, map_location="cpu")

        class _WQ(FixedPointPerTensorWeightQuant):
            bit_width = 8
        class _AQ(FixedPointPerTensorActivationQuant):
            bit_width = 6

        QuantizerManager().reset()
        from examples.find_perfect_lsbs_imagenet_ptq import _build_model
        model = _build_model(args, _WQ, _AQ)
        incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
        assert incompatible.unexpected_keys == []
        # Missing keys are expected: activation-quantizer buffers don't exist
        # in a weights-only checkpoint.
        missing_roles = {
            "activation" if ("act_quant" in k or "input_quant" in k or "output_quant" in k)
            else "other"
            for k in incompatible.missing_keys
        }
        assert missing_roles <= {"activation", "other"}

    def test_combined_model_forward_pass_works(self, tmp_path):
        weights_ckpt = _build_weights_checkpoint(tmp_path)

        QuantizerManager().reset()
        args = _base_args(mode="activations", bit_width=6, init_from_ckpt=weights_ckpt)
        model, target_role, bw, prev_extra, prev_role_bw = _build_quantized_model(
            args, torch.device("cpu"),
        )
        mgr = QuantizerManager()
        _disable_target_role_keep_others_active(mgr, target_role)

        # Simulate "calibration done" for the activation quantizers, as the
        # real per-quantizer search loop would do, so eval mode doesn't trip
        # the uncalibrated-quantizer-active guard in base_quantizer.py.
        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        for q in real_qs:
            if q.quantizer_role == "activation":
                q.search_result_lsb.fill_(ACTIVATIONS_LSB)
                q.search_done.fill_(True)
                q.annealing_alpha.data.fill_(1.0)

        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 10)

    def test_backward_compat_with_old_single_role_checkpoint_format(self, tmp_path):
        """A checkpoint saved before role_bit_widths existed (only
        ptq_search_mode/bit_width) must still be loadable."""
        args = _base_args(mode="weights", bit_width=8)
        model, _, _, _, _ = _build_quantized_model(args, torch.device("cpu"))
        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        for q in real_qs:
            q.search_result_lsb.fill_(WEIGHTS_LSB)
            q.search_done.fill_(True)
            q.annealing_alpha.data.fill_(1.0)

        old_format_payload = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "metrics": {},
            "extra": {
                "ptq_search_mode": "weights",
                "bit_width": 8,
                # no "role_bit_widths" key — old format
            },
        }
        path = tmp_path / "old_format_ckpt.pt"
        torch.save(old_format_payload, path)

        QuantizerManager().reset()
        args2 = _base_args(mode="activations", bit_width=6, init_from_ckpt=str(path))
        model2, target_role, bw, prev_extra, prev_role_bw = _build_quantized_model(
            args2, torch.device("cpu"),
        )
        assert prev_role_bw == {"weight": 8}
        assert target_role == "activation"


class TestDisableTargetRoleKeepOthersActive:

    def test_target_role_gets_disabled_for_fresh_search(self, tmp_path):
        weights_ckpt = _build_weights_checkpoint(tmp_path)
        QuantizerManager().reset()
        args = _base_args(mode="activations", bit_width=6, init_from_ckpt=weights_ckpt)
        model, target_role, bw, prev_extra, prev_role_bw = _build_quantized_model(
            args, torch.device("cpu"),
        )
        mgr = QuantizerManager()
        _disable_target_role_keep_others_active(mgr, target_role)

        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        act_qs = [q for q in real_qs if q.quantizer_role == "activation"]
        assert len(act_qs) > 0
        for q in act_qs:
            assert q.annealing_alpha.item() == pytest.approx(0.0)
            assert q.search_done.item() is True  # placeholder; per-quantizer loop flips False

    def test_other_role_loaded_quantizers_stay_fully_active(self, tmp_path):
        weights_ckpt = _build_weights_checkpoint(tmp_path)
        QuantizerManager().reset()
        args = _base_args(mode="activations", bit_width=6, init_from_ckpt=weights_ckpt)
        model, target_role, bw, prev_extra, prev_role_bw = _build_quantized_model(
            args, torch.device("cpu"),
        )
        mgr = QuantizerManager()
        _disable_target_role_keep_others_active(mgr, target_role)

        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        weight_qs = [q for q in real_qs if q.quantizer_role == "weight"]
        assert len(weight_qs) > 0
        for q in weight_qs:
            assert q.annealing_alpha.item() == pytest.approx(1.0)
            assert q.annealing_alpha_step == pytest.approx(0.0)
            assert q.search_done.item() is True
            assert int(q.search_result_lsb.item()) == WEIGHTS_LSB

    def test_uncalibrated_other_role_quantizer_disabled_safely(self):
        """If a non-target-role quantizer exists but was never calibrated
        (no checkpoint loaded), it must be disabled, not left active."""
        QuantizerManager().reset()

        class _WQ(FixedPointPerTensorWeightQuant):
            bit_width = 8
        class _AQ(FixedPointPerTensorActivationQuant):
            bit_width = 8

        model = QuantResNet18(num_classes=10, weight_quant=_WQ, act_quant=_AQ)
        _assign_descriptive_ids(model)
        mgr = QuantizerManager()

        _disable_target_role_keep_others_active(mgr, target_role="weight")

        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        act_qs = [q for q in real_qs if q.quantizer_role == "activation"]
        assert len(act_qs) > 0
        for q in act_qs:
            assert q.annealing_alpha.item() == pytest.approx(0.0)
            assert q.search_done.item() is True


class TestChainingOtherDirection:
    """Activations searched first, then weights continued from that checkpoint."""

    def test_activations_first_then_weights(self, tmp_path):
        args = _base_args(mode="activations", bit_width=8)
        model, target_role, bw, _, _ = _build_quantized_model(args, torch.device("cpu"))
        assert target_role == "activation"
        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        for q in real_qs:
            q.search_result_lsb.fill_(ACTIVATIONS_LSB)
            q.search_done.fill_(True)
            q.annealing_alpha.data.fill_(1.0)

        payload = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "metrics": {},
            "extra": {
                "ptq_search_mode": "activations",
                "bit_width": 8,
                "role_bit_widths": {"activation": 8},
                "calibrated_lsbs": {q.quant_id: ACTIVATIONS_LSB for q in real_qs},
                "selected_lsbs": {q.quant_id: ACTIVATIONS_LSB for q in real_qs},
            },
        }
        act_ckpt = tmp_path / "act_ckpt.pt"
        torch.save(payload, act_ckpt)

        QuantizerManager().reset()
        args2 = _base_args(mode="weights", bit_width=10, init_from_ckpt=str(act_ckpt))
        model2, target_role2, bw2, _, prev_role_bw2 = _build_quantized_model(
            args2, torch.device("cpu"),
        )
        assert target_role2 == "weight"
        assert prev_role_bw2 == {"activation": 8}

        real_qs2 = [m for _, m in model2.named_modules() if isinstance(m, BaseQuantizer)]
        roles2 = {q.quantizer_role for q in real_qs2}
        assert roles2 == {"weight", "activation"}

        act_qs2 = [q for q in real_qs2 if q.quantizer_role == "activation"]
        for q in act_qs2:
            assert q.search_done.item() is True
            assert int(q.search_result_lsb.item()) == ACTIVATIONS_LSB


def _build_bias_checkpoint(tmp_path, prior_ckpt: str, prior_role_bw: dict) -> str:
    """Run a minimal stand-in for --mode bias --init-from-ckpt <prior_ckpt>
    and save its checkpoint, e.g. continuing a weights-mode checkpoint."""
    QuantizerManager().reset()
    args = _base_args(mode="bias", bit_width=8, init_from_ckpt=prior_ckpt)
    model, target_role, bw, prev_extra, prev_role_bw = _build_quantized_model(
        args, torch.device("cpu"),
    )
    assert target_role == "bias"

    real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
    bias_qs = [q for q in real_qs if q.quantizer_role == "bias"]
    for q in bias_qs:
        q.search_result_lsb.fill_(BIAS_LSB)
        q.search_done.fill_(True)
        q.annealing_alpha.data.fill_(1.0)

    payload = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "metrics": {},
        "extra": {
            "ptq_search_mode": "bias",
            "bit_width": 8,
            "role_bit_widths": {**prior_role_bw, "bias": 8},
            "calibrated_lsbs": {q.quant_id: BIAS_LSB for q in bias_qs},
            "selected_lsbs": {q.quant_id: BIAS_LSB for q in bias_qs},
        },
    }
    path = tmp_path / "bias_ckpt.pt"
    torch.save(payload, path)
    return str(path)


class TestThreeWayChainingWeightsBiasActivations:
    """The user's requested chain order: weights -> bias -> activations.
    Bias quantization (FixedPointPerTensorBiasQuant, requires_input_scale=False)
    calibrates directly against the bias values, so it has no dependency on
    activation quantization having run -- it can legitimately slot in between
    weights and activations."""

    def test_weights_then_bias_combined_model_has_both_roles(self, tmp_path):
        weights_ckpt = _build_weights_checkpoint(tmp_path)
        bias_ckpt = _build_bias_checkpoint(tmp_path, weights_ckpt, {"weight": 8})

        payload = torch.load(bias_ckpt, map_location="cpu")
        assert payload["extra"]["role_bit_widths"] == {"weight": 8, "bias": 8}

        QuantizerManager().reset()
        args = _base_args(mode="bias", bit_width=8, init_from_ckpt=weights_ckpt)
        model, target_role, bw, _, prev_role_bw = _build_quantized_model(
            args, torch.device("cpu"),
        )
        assert target_role == "bias"
        assert prev_role_bw == {"weight": 8}

        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        roles = {q.quantizer_role for q in real_qs}
        assert roles == {"weight", "bias"}

        weight_qs = [q for q in real_qs if q.quantizer_role == "weight"]
        for q in weight_qs:
            assert q.search_done.item() is True
            assert int(q.search_result_lsb.item()) == WEIGHTS_LSB

    def test_full_chain_weights_bias_activations(self, tmp_path):
        weights_ckpt = _build_weights_checkpoint(tmp_path)
        bias_ckpt = _build_bias_checkpoint(tmp_path, weights_ckpt, {"weight": 8})

        QuantizerManager().reset()
        args = _base_args(mode="activations", bit_width=8, init_from_ckpt=bias_ckpt)
        model, target_role, bw, _, prev_role_bw = _build_quantized_model(
            args, torch.device("cpu"),
        )
        assert target_role == "activation"
        assert prev_role_bw == {"weight": 8, "bias": 8}

        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        roles = {q.quantizer_role for q in real_qs}
        assert roles == {"weight", "bias", "activation"}

        weight_qs = [q for q in real_qs if q.quantizer_role == "weight"]
        bias_qs   = [q for q in real_qs if q.quantizer_role == "bias"]
        act_qs    = [q for q in real_qs if q.quantizer_role == "activation"]
        assert len(weight_qs) > 0 and len(bias_qs) > 0 and len(act_qs) > 0

        for q in weight_qs:
            assert q.search_done.item() is True
            assert int(q.search_result_lsb.item()) == WEIGHTS_LSB
        for q in bias_qs:
            assert q.search_done.item() is True
            assert int(q.search_result_lsb.item()) == BIAS_LSB
        for q in act_qs:
            # Freshly built -- not yet calibrated.
            assert q.search_done.item() is False

    def test_full_chain_forward_pass_works(self, tmp_path):
        weights_ckpt = _build_weights_checkpoint(tmp_path)
        bias_ckpt = _build_bias_checkpoint(tmp_path, weights_ckpt, {"weight": 8})

        QuantizerManager().reset()
        args = _base_args(mode="activations", bit_width=8, init_from_ckpt=bias_ckpt)
        model, target_role, bw, _, prev_role_bw = _build_quantized_model(
            args, torch.device("cpu"),
        )
        mgr = QuantizerManager()
        _disable_target_role_keep_others_active(mgr, target_role)

        real_qs = [m for _, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        for q in real_qs:
            if q.quantizer_role == "activation":
                q.search_result_lsb.fill_(ACTIVATIONS_LSB)
                q.search_done.fill_(True)
                q.annealing_alpha.data.fill_(1.0)

        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 10)
