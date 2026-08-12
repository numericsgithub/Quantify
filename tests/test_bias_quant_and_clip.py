"""
Tests for two related fixes:

1. Bias quantization after BN fusion.
   Every conv is built bias=False (each feeds a BatchNorm), and
   fuse_bn_into_conv() then folds BN away and CREATES conv.bias. Previously
   bias_quant was wired to the classifier only, so those 52 folded biases were
   raw nn.Parameters that no quantizer owned -- they shipped in float while the
   model claimed to be fully quantized. Passing bias_quant to the convs fixes it:
   Brevitas hooks __setattr__ on 'bias' and calls bias_quant.init_tensor_quant()
   at the moment fusion assigns it.

2. The one-time clamp into the representable range.
   A loaded PTQ checkpoint routinely holds parameters far outside the grid their
   quantizer can represent (classifier.1 was measured at |w|max=1.251 against a
   +/-0.25 range). Those are pinned at the clip bound -- their quantized value is
   frozen -- while plain STE keeps leaking gradient and pushing them further out.
   The clamp CLAMPS ONLY; it must never round onto the grid.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import brevitas.nn as qnn

from models.mobilenetv2_quant import QuantMobileNetV2
from quantizers import (
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorBiasQuant,
    FixedPointPerTensorWeightQuant,
)
from quantizers.base_quantizer import BaseQuantizer
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer, integer_range
from quantizers.manager import QuantizerManager
from utils.bn_fusion import fuse_bn_into_conv
from examples.train_imagenet_qat import _clip_params_to_quant_range


@pytest.fixture(autouse=True)
def _isolate_quantizer_manager():
    """QuantizerManager is a singleton, so quantizers built by one test stay
    registered and are counted by the next. Reset around every test."""
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


# ============================================================ bias quantization

def _mnv2(bias_quant=FixedPointPerTensorBiasQuant, trained_bn=False):
    model = QuantMobileNetV2(
        num_classes=10,
        weight_quant=FixedPointPerTensorWeightQuant,
        act_quant=FixedPointPerTensorActivationQuant,
        bias_quant=bias_quant,
    )
    if trained_bn:
        _give_bn_trained_stats(model)
    return model


def _give_bn_trained_stats(model):
    """Make every BatchNorm look trained rather than freshly initialised.

    Matters because a fresh BN has bias=0 and running_mean=0, so folding it
    yields a conv bias of ALL ZEROS -- one unique value, which
    _save_calibration() refuses to mark calibrated (it gates search_done on
    num_unique > 1). The real pipeline loads pretrained weights BEFORE fusing, so
    its folded biases are non-zero; a test that fuses an untrained model is
    testing a situation that never occurs.
    """
    torch.manual_seed(0)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.normal_(1.0, 0.1)
                m.bias.normal_(0.0, 0.1)
                m.running_mean.normal_(0.0, 0.1)
                m.running_var.uniform_(0.5, 1.5)


def _role_counts():
    from collections import Counter
    return Counter(q.quantizer_role for q in QuantizerManager().quantizers.values())


def test_every_folded_bias_gets_a_quantizer():
    """The bug: 52 of 53 biases were raw float after fusion."""
    model = _mnv2()
    before = _role_counts()["bias"]
    n_fused = fuse_bn_into_conv(model)
    after = _role_counts()["bias"]

    assert n_fused == 52, f"expected 52 BN layers to fold, got {n_fused}"
    assert before == 1, "only the classifier should have a bias before fusion"
    assert after == before + n_fused, (
        f"fusion created {n_fused} biases but only {after - before} gained a "
        "quantizer -- the rest would ship as raw float")


def test_no_layer_keeps_an_unquantized_bias():
    """Stated as the property that actually matters, independent of counts."""
    model = _mnv2()
    fuse_bn_into_conv(model)

    unquantized = [
        name for name, m in model.named_modules()
        if isinstance(m, (qnn.QuantConv2d, qnn.QuantLinear))
        and m.bias is not None
        and not m.bias_quant.is_quant_enabled
    ]
    assert not unquantized, f"these layers have a float bias: {unquantized}"


def test_bias_quant_is_actually_applied_in_forward():
    """is_quant_enabled is necessary but not sufficient -- check the value moves."""
    conv = qnn.QuantConv2d(3, 4, 3, padding=1, bias=False,
                           weight_quant=FixedPointPerTensorWeightQuant,
                           bias_quant=FixedPointPerTensorBiasQuant)
    assert not conv.bias_quant.is_quant_enabled, "no bias yet -> nothing to quantize"

    conv.bias = nn.Parameter(torch.randn(4) * 3.0)   # what fuse_bn_into_conv does
    assert conv.bias_quant.is_quant_enabled, (
        "assigning .bias must trigger Brevitas' __setattr__ hook -> init_tensor_quant()")

    conv.train()
    x = torch.randn(2, 3, 8, 8)
    for _ in range(12):
        conv(x)
    conv.eval()

    qb = conv.bias_quant(conv.bias)
    qb = qb.value if hasattr(qb, "value") else qb
    assert not torch.allclose(conv.bias.data, qb.detach()), \
        "bias passed through unquantized"


def test_bias_quant_none_still_builds():
    """bias_quant=None must remain valid (float-bias baseline runs)."""
    model = _mnv2(bias_quant=None)
    fuse_bn_into_conv(model)
    assert _role_counts()["bias"] == 0


def test_model_still_runs_after_fusion_with_bias_quant():
    """End-to-end on a model whose BN looks trained (see _give_bn_trained_stats):
    every bias quantizer must calibrate and the model must still run in eval."""
    model = _mnv2(trained_bn=True)
    fuse_bn_into_conv(model)
    model.train()
    x = torch.randn(2, 3, 32, 32)
    for _ in range(12):
        model(x)

    uncal = [q.quantizer_role for q in QuantizerManager().quantizers.values()
             if not q.search_done.item()]
    assert not uncal, f"{len(uncal)} quantizer(s) never calibrated: {set(uncal)}"

    model.eval()
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_all_zero_folded_bias_does_not_calibrate():
    """Documents a sharp edge this change exposes.

    _save_calibration() gates search_done on num_unique > 1, so an all-zero bias
    never calibrates -- and an uncalibrated-but-active quantizer raises in eval.
    Folding a FRESHLY INITIALISED BatchNorm produces exactly that (bias=0,
    running_mean=0 -> folded bias is all zeros).

    This does not bite the real pipeline (pretrained weights are loaded before
    fusion, and _load_ptq_checkpoint fuses then immediately load_state_dict's the
    real biases and search_done buffers over the top), but it is the reason a
    stale pre-bias-quant checkpoint must not be silently reused: its state dict
    has no search_done for these 52 new quantizers, so they would stay
    uncalibrated and raise at eval.
    """
    model = _mnv2()          # fresh BN on purpose
    fuse_bn_into_conv(model)
    model.train()
    x = torch.randn(2, 3, 32, 32)
    for _ in range(12):
        model(x)

    biases = [m.bias for m in model.modules()
              if isinstance(m, qnn.QuantConv2d) and m.bias is not None]
    assert all(float(b.abs().max()) == 0.0 for b in biases), \
        "folding a fresh BN should yield all-zero conv biases"

    uncal = [q for q in QuantizerManager().quantizers.values()
             if not q.search_done.item()]
    assert len(uncal) == 52 and {q.quantizer_role for q in uncal} == {"bias"}


# ================================================== representable_range contract

def test_representable_range_matches_the_forward_clamp():
    """The range must be exactly what the forward saturates against, or the clamp
    would move values the quantizer would not have clipped (or miss ones it does)."""
    q = FixedPointPerTensorQuantizer(bit_width=4, signed=True, quantizer_role="weight")
    q.train()
    with torch.no_grad():
        q(torch.linspace(-2, 2, 100))

    lo, hi = q.representable_range()
    params = q._load_calibration()
    step = 2.0 ** int(params["lsb"])
    imin, imax = integer_range(4, params["signed"], q.narrow_range)
    assert (lo, hi) == (imin * step, imax * step)

    # empirically: nothing quantizes outside [lo, hi]
    q.eval()
    out, _, _, _ = q(torch.linspace(-50, 50, 1000))
    assert float(out.min()) >= lo - 1e-9
    assert float(out.max()) <= hi + 1e-9


def test_representable_range_is_none_before_calibration():
    q = FixedPointPerTensorQuantizer(bit_width=8, quantizer_role="weight")
    assert q.representable_range() is None


def test_base_quantizer_range_defaults_to_none():
    """Quantizers with no uniform grid must report None so callers SKIP them,
    rather than having an LSB-derived clamp wrongly applied. The live case is
    CoefficientPerTensorWeightQuantizer: a non-uniform grid with no LSB at all.
    """
    class _NoGridQuant(BaseQuantizer):
        """Minimal concrete subclass that does not override representable_range."""
        def _calibrate(self, x): return {}
        def _save_calibration(self, params): pass
        def _load_calibration(self): return {}
        def _quantize(self, x, params): return x
        def _get_metadata(self, params, x):
            z = torch.tensor(0.0)
            return z, z, z

    q = _NoGridQuant(bit_width=8)
    assert q.representable_range() is None, (
        "a quantizer that does not define a uniform grid must report None so the "
        "clamp skips it instead of inventing bounds for it")


def test_in_range_mask_agrees_with_representable_range():
    """_in_range_mask was refactored to call representable_range -- pin that they
    cannot drift apart."""
    q = FixedPointPerTensorQuantizer(bit_width=4, signed=True, quantizer_role="weight")
    q.train()
    with torch.no_grad():
        q(torch.linspace(-2, 2, 100))
    params = q._load_calibration()
    lo, hi = q.representable_range()

    x = torch.linspace(-10, 10, 500)
    mask = q._in_range_mask(x, params)
    assert torch.equal(mask, (x >= lo) & (x <= hi))


# ================================================================== the clamp

def _tiny_model_with_out_of_range_params():
    """A calibrated conv whose weight/bias then get shoved far outside the grid."""
    model = nn.Sequential(
        qnn.QuantConv2d(3, 4, 3, padding=1, bias=False,
                        weight_quant=FixedPointPerTensorWeightQuant,
                        bias_quant=FixedPointPerTensorBiasQuant)
    )
    model[0].bias = nn.Parameter(torch.randn(4) * 0.1)
    model.train()
    x = torch.randn(2, 3, 8, 8)
    for _ in range(12):
        model(x)
    return model, x


def test_clamp_brings_out_of_range_params_inside():
    model, _ = _tiny_model_with_out_of_range_params()
    conv = model[0]
    wq = conv.weight_quant.tensor_quant
    lo, hi = wq.representable_range()

    with torch.no_grad():
        conv.weight[0, 0, 0, 0] = hi * 100.0     # far outside
        conv.weight[0, 0, 0, 1] = lo * 100.0
    assert float(conv.weight.max()) > hi

    _clip_params_to_quant_range(model)

    assert float(conv.weight.max()) <= hi + 1e-9
    assert float(conv.weight.min()) >= lo - 1e-9


def test_clamp_does_not_quantize():
    """Clamp only. In-range values must keep their exact float value -- if this
    rounded onto the grid it would destroy the latent weights QAT trains."""
    model, _ = _tiny_model_with_out_of_range_params()
    conv = model[0]
    lo, hi = conv.weight_quant.tensor_quant.representable_range()

    with torch.no_grad():
        conv.weight.mul_(0.5)                  # ensure everything is inside
    assert float(conv.weight.abs().max()) < hi

    before = conv.weight.data.clone()
    _clip_params_to_quant_range(model)
    assert torch.equal(before, conv.weight.data), \
        "in-range weights were modified -- the clamp must not round onto the grid"


def test_clamp_is_idempotent():
    model, _ = _tiny_model_with_out_of_range_params()
    conv = model[0]
    with torch.no_grad():
        conv.weight[0, 0, 0, 0] = 1e6

    _clip_params_to_quant_range(model)
    once = conv.weight.data.clone()
    _clip_params_to_quant_range(model)
    assert torch.equal(once, conv.weight.data)


def test_clamp_covers_biases_too():
    model, _ = _tiny_model_with_out_of_range_params()
    conv = model[0]
    bq = conv.bias_quant.tensor_quant
    lo, hi = bq.representable_range()

    with torch.no_grad():
        conv.bias[0] = hi * 50.0
    _clip_params_to_quant_range(model)
    assert float(conv.bias.max()) <= hi + 1e-9, "bias was not clamped"


def test_clamp_warns_and_is_noop_on_uncalibrated_model(capsys):
    """A silent no-op is the main failure mode of this helper: if the mapping
    breaks, it must say so rather than quietly clamping nothing."""
    model = nn.Sequential(
        qnn.QuantConv2d(3, 4, 3, padding=1, bias=False,
                        weight_quant=FixedPointPerTensorWeightQuant)
    )
    _clip_params_to_quant_range(model)
    assert "WARNING" in capsys.readouterr().out


def test_clamp_leaves_quantized_output_unchanged_for_in_range_model():
    """Sanity: clamping a model whose params are all in range must not move the
    forward output at all."""
    model, x = _tiny_model_with_out_of_range_params()
    with torch.no_grad():
        model[0].weight.mul_(0.3)
    model.eval()
    before = model(x).detach().clone()
    _clip_params_to_quant_range(model)
    after = model(x).detach()
    assert torch.allclose(before, after, atol=1e-6)
