"""
Gradient-equivalence tests: a fully quantized model must backpropagate the
SAME gradients as an identical plain-PyTorch model with no quantizers at all,
as long as no quantizer actually changes any value in the forward pass.

Every test builds a small two-layer model twice -- once from Brevitas/Quantify
quantized modules (input, weights, biases, activation input AND output, and
the model output all quantized), once from plain `torch.nn` modules -- with
identical parameters, then compares the gradient of every parameter and of the
input after one backward pass.

The data is chosen so the quantizers are no-ops in the forward pass: every
input, weight and bias is a small dyadic rational (k / 2**n with few
significant bits), so every intermediate value lands exactly on the
fixed-point grid each quantizer calibrates to. In float64, sums/products of
such values are computed exactly, so for piecewise-linear activations the
quantized and the reference model must agree BIT-EXACTLY (`torch.equal`), in
values and in gradients. A forward hook on every quantizer verifies this
premise directly (quantizer output == quantizer input), so a failure can only
mean the backward pass differs, not that the test data was quantized.

Smooth activations (Sigmoid, Tanh, SiLU, GELU, Softmax) produce transcendental
outputs that no fixed-point grid can represent exactly (except at a handful
of points like 0), so their OUTPUT quantizer -- and the model-output
quantizer downstream of it -- necessarily rounds by at most one LSB. All
activation quantizers run at 32 bits so that rounding is ~1e-9, and smooth
activations' gradients are compared with a tight tolerance instead. Every quantizer upstream of the
nonlinearity (model input, layer-1 weight/bias, activation input) and the
layer-2 weight/bias quantizers are still verified to be exact no-ops.

Each case runs in three quantizer states:
  - "quantizing":   annealing_alpha = 1 (fully fake-quantized, STE backward)
  - "mid_anneal":   annealing_alpha = 0.5 (AnnealingBlendFn backward)
  - "disabled":     QuantizerManager().disable_quantization() (passthrough)
plus a clipped-STE variant (all values are in range, so the mask must be all
ones and change nothing).

Known deviations found by these tests are marked xfail (strict where the
failure is deterministic), each with the reason -- see KNOWN_BUGS below.
"""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn

from quantizers import (
    FixedPointPerTensorWeightQuant,
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorBiasQuant,
    CoefficientPerTensorWeightQuant,
    QuantSiLUActivationQuant,
    QuantReLU,
    QuantReLU6,
    QuantSigmoid,
    QuantTanh,
    QuantSiLU,
    QuantGELU,
    QuantLeakyReLU,
    QuantSoftmax,
)
from quantizers.base_quantizer import BaseQuantizer
from quantizers.silu_quant import SiLUTensorQuant
from quantizers.manager import QuantizerManager

DTYPE = torch.float64

# Activation (input/output/model-input) quantizers run wide enough that every
# dyadic intermediate value below fits on their grid; weights and biases stay
# at a realistic 8 bits (the test data is designed to fit them exactly).
ACT_BW = 32
OUT_BW = 32

# Upper bound on |pre-activation| per activation, so saturating activations'
# capped input quantizers (ReLU6/Sigmoid -> 6, Tanh -> 3) never clip. ReLU6's
# bound sits above 6 on purpose so its saturated region (gradient 0) is
# exercised too.
PREACT_LIMIT = {
    "relu6": 7.5,
    "sigmoid": 5.0,
    "tanh": 2.5,
}
DEFAULT_PREACT_LIMIT = 4.0
OUTPUT_LIMIT = 4.0

COEFF_FILE = __file__.rsplit("/", 1)[0] + "/dummy_coeffs.txt"

# CoefficientPerTensorWeightQuantizer only works in float32 (see KNOWN_BUGS);
# float32 still computes every dyadic test value exactly.
FLOAT32_ARCHS = {"linear_coeff_weights"}

KNOWN_BUGS = {
    "fused_silu_passthrough": (
        "SiLUTensorQuant applies SiLU inside _quantize() only, so the passthrough "
        "(quantization disabled/gated off) returns x instead of silu(x) and "
        "AnnealingBlendFn blends x (not silu(x)) with the quantized silu(x), "
        "with a slope-1 backward that skips silu'(x)."),
    "coeff_float64": (
        "CoefficientPerTensorWeightQuantizer returns float32 weights for a float64 "
        "model (coefficient sets are float32 tensors and never cast to x.dtype), "
        "so a float64 QuantLinear fails with a Double/Float matmul error."),
}


class _SignedQuantSiLUActivationQuant(QuantSiLUActivationQuant):
    # The default (signed=False) clips SiLU's negative lobe to 0, which would
    # change values by design; signed keeps the quantizer a no-op here.
    signed = True


class _DummyCoeffWeightQuant(CoefficientPerTensorWeightQuant):
    filepath = COEFF_FILE


# ---------------------------------------------------------------------------
# Activations: (quantized module, reference module, exact?)
# ---------------------------------------------------------------------------

def _qid(bw=ACT_BW, act_quant=FixedPointPerTensorActivationQuant):
    return qnn.QuantIdentity(act_quant=act_quant, bit_width=bw, return_quant_tensor=False)


ACTIVATIONS = {
    # piecewise-linear with dyadic slopes -> bit-exact
    "identity": (lambda: _qid(), lambda: nn.Identity(), True),
    "relu": (lambda: QuantReLU(bit_width=ACT_BW), lambda: nn.ReLU(), True),
    "relu6": (lambda: QuantReLU6(bit_width=ACT_BW), lambda: nn.ReLU6(), True),
    "leaky_relu": (lambda: QuantLeakyReLU(negative_slope=0.25, bit_width=ACT_BW),
                   lambda: nn.LeakyReLU(0.25), True),
    # smooth -> output quantizer rounds by <= 1 LSB, compared with tolerance
    "sigmoid": (lambda: QuantSigmoid(bit_width=ACT_BW), lambda: nn.Sigmoid(), False),
    "tanh": (lambda: QuantTanh(bit_width=ACT_BW), lambda: nn.Tanh(), False),
    "silu": (lambda: QuantSiLU(bit_width=ACT_BW), lambda: nn.SiLU(), False),
    "gelu": (lambda: QuantGELU(bit_width=ACT_BW), lambda: nn.GELU(), False),
    "gelu_tanh": (lambda: QuantGELU(approximate="tanh", bit_width=ACT_BW),
                  lambda: nn.GELU(approximate="tanh"), False),
    "softmax": (lambda: QuantSoftmax(dim=1, bit_width=ACT_BW), lambda: nn.Softmax(dim=1), False),
    "fused_silu_act_quant": (lambda: _qid(act_quant=_SignedQuantSiLUActivationQuant),
                             lambda: nn.SiLU(), False),
}

# ---------------------------------------------------------------------------
# Two-layer architectures
# ---------------------------------------------------------------------------

def _wq(weight_quant=FixedPointPerTensorWeightQuant):
    return dict(weight_quant=weight_quant, bias_quant=FixedPointPerTensorBiasQuant,
                return_quant_tensor=False)


class _BN(nn.Module):
    """Eval-mode BatchNorm with eps=0 and power-of-four running_var, so the
    normalization divides by an exact power of two and stays dyadic. Kept in
    eval mode even when the surrounding model is put in train mode."""

    def __init__(self, bn):
        super().__init__()
        self.bn = bn.eval()
        self.bn.eps = 0.0

    def train(self, mode=True):
        super().train(mode)
        self.bn.eval()
        return self

    def forward(self, x):
        return self.bn(x)


ARCHS = {
    # name: (build(quant) -> dict(l1, pre, post, l2), make_input(), pow2_scale_only)
    "linear_linear": (
        lambda q: dict(
            l1=qnn.QuantLinear(6, 5, bias=True, **_wq()) if q else nn.Linear(6, 5),
            l2=qnn.QuantLinear(5, 3, bias=True, **_wq()) if q else nn.Linear(5, 3)),
        lambda: _dyadic((4, 6), 4, 2), False),
    "linear_coeff_weights": (
        lambda q: dict(
            l1=qnn.QuantLinear(6, 5, bias=True, **_wq(_DummyCoeffWeightQuant)) if q else nn.Linear(6, 5),
            l2=qnn.QuantLinear(5, 3, bias=True, **_wq(_DummyCoeffWeightQuant)) if q else nn.Linear(5, 3)),
        lambda: _dyadic((4, 6), 4, 2), True),
    "conv1d_conv1d": (
        lambda q: dict(
            l1=qnn.QuantConv1d(2, 3, 3, padding=1, bias=True, **_wq()) if q else nn.Conv1d(2, 3, 3, padding=1),
            l2=qnn.QuantConv1d(3, 2, 3, stride=2, bias=True, **_wq()) if q else nn.Conv1d(3, 2, 3, stride=2)),
        lambda: _dyadic((2, 2, 8), 4, 2), False),
    "conv2d_conv2d": (
        lambda q: dict(
            l1=qnn.QuantConv2d(2, 3, 3, padding=1, bias=True, **_wq()) if q else nn.Conv2d(2, 3, 3, padding=1),
            l2=qnn.QuantConv2d(3, 2, 3, stride=2, bias=True, **_wq()) if q else nn.Conv2d(3, 2, 3, stride=2)),
        lambda: _dyadic((2, 2, 6, 6), 4, 2), False),
    "grouped_conv2d_maxpool_linear": (
        lambda q: dict(
            l1=qnn.QuantConv2d(2, 4, 3, padding=1, groups=2, bias=True, **_wq()) if q
            else nn.Conv2d(2, 4, 3, padding=1, groups=2),
            post=nn.Sequential(nn.MaxPool2d(2), nn.Flatten()),
            l2=qnn.QuantLinear(4 * 3 * 3, 3, bias=True, **_wq()) if q else nn.Linear(4 * 3 * 3, 3)),
        lambda: _dyadic((2, 2, 6, 6), 4, 2), False),
    "conv2d_nobias_avgpool_linear": (
        lambda q: dict(
            l1=qnn.QuantConv2d(2, 3, 3, padding=1, bias=False, **_wq()) if q
            else nn.Conv2d(2, 3, 3, padding=1, bias=False),
            post=nn.Sequential(nn.AvgPool2d(2), nn.Flatten()),
            l2=qnn.QuantLinear(3 * 2 * 2, 3, bias=True, **_wq()) if q else nn.Linear(3 * 2 * 2, 3)),
        lambda: _dyadic((2, 2, 4, 4), 4, 2), False),
    "conv2d_batchnorm_conv2d": (
        lambda q: dict(
            l1=qnn.QuantConv2d(2, 3, 3, padding=1, bias=True, **_wq()) if q else nn.Conv2d(2, 3, 3, padding=1),
            pre=_BN(nn.BatchNorm2d(3)),
            l2=qnn.QuantConv2d(3, 2, 3, bias=True, **_wq()) if q else nn.Conv2d(3, 2, 3)),
        lambda: _dyadic((2, 2, 5, 5), 4, 2), False),
    "convtranspose1d_conv1d": (
        lambda q: dict(
            l1=qnn.QuantConvTranspose1d(2, 3, 3, stride=2, bias=True, **_wq()) if q
            else nn.ConvTranspose1d(2, 3, 3, stride=2),
            l2=qnn.QuantConv1d(3, 2, 3, bias=True, **_wq()) if q else nn.Conv1d(3, 2, 3)),
        lambda: _dyadic((2, 2, 5), 4, 2), False),
    "convtranspose2d_convtranspose2d": (
        lambda q: dict(
            l1=qnn.QuantConvTranspose2d(2, 3, 2, stride=2, bias=True, **_wq()) if q
            else nn.ConvTranspose2d(2, 3, 2, stride=2),
            l2=qnn.QuantConvTranspose2d(3, 2, 3, padding=1, bias=True, **_wq()) if q
            else nn.ConvTranspose2d(3, 2, 3, padding=1)),
        lambda: _dyadic((2, 2, 3, 3), 4, 2), False),
    "conv3d_conv3d": (
        lambda q: dict(
            l1=qnn.QuantConv3d(1, 2, 2, bias=True, **_wq()) if q else nn.Conv3d(1, 2, 2),
            l2=qnn.QuantConv3d(2, 2, 2, bias=True, **_wq()) if q else nn.Conv3d(2, 2, 2)),
        lambda: _dyadic((2, 1, 4, 4, 4), 4, 2), False),
    "embedding_linear": (
        lambda q: dict(
            l1=qnn.QuantEmbedding(10, 4, weight_quant=FixedPointPerTensorWeightQuant,
                                  return_quant_tensor=False) if q else nn.Embedding(10, 4),
            l2=qnn.QuantLinear(4, 3, bias=True, **_wq()) if q else nn.Linear(4, 3)),
        lambda: torch.tensor([[0, 3, 7, 3], [9, 1, 4, 6]]), False),
}


class TwoLayer(nn.Module):
    """input quant -> l1 -> [pre] -> activation -> [post] -> l2 -> output quant.
    Built identically for the quantized and the reference model, so parameter
    names match one-to-one."""

    def __init__(self, arch, act_name, quant):
        super().__init__()
        build, _, _ = ARCHS[arch]
        q_act, ref_act, _ = ACTIVATIONS[act_name]
        parts = build(quant)
        is_embedding = arch == "embedding_linear"
        self.inp = _qid() if quant and not is_embedding else nn.Identity()
        self.l1 = parts["l1"]
        self.pre = parts.get("pre", nn.Identity())
        self.act = q_act() if quant else ref_act()
        self.post = parts.get("post", nn.Identity())
        self.l2 = parts["l2"]
        self.out = _qid(OUT_BW) if quant else nn.Identity()

    def preact(self, x):
        return self.pre(self.l1(self.inp(x)))

    def forward(self, x):
        return self.out(self.l2(self.post(self.act(self.preact(x)))))


# ---------------------------------------------------------------------------
# Dyadic test data
# ---------------------------------------------------------------------------

def _dyadic(shape, max_k, frac_bits, gen=None):
    """Integers in [-max_k, max_k] / 2**frac_bits."""
    k = torch.randint(-max_k, max_k + 1, shape, generator=gen)
    return (k / 2 ** frac_bits).to(DTYPE)


def _dyadic_scale(ratio, pow2_only):
    """Largest f <= ratio of the form 2**e * (m / 8), m in 8..15 (or m == 8 if
    pow2_only). Three mantissa bits keep scaled 8-bit weights representable."""
    e = math.floor(math.log2(ratio))
    if pow2_only:
        return 2.0 ** e
    m = math.floor(ratio / 2.0 ** e * 8)
    return 2.0 ** e * min(m, 15) / 8


@torch.no_grad()
def _init_params(model, arch, act_name, x, seed):
    """Fill the reference model with dyadic parameters, then rescale the
    module feeding the activation (and layer 2) so the pre-activation and the
    output stay inside every quantizer's range."""
    gen = torch.Generator().manual_seed(seed)
    pow2_only = ARCHS[arch][2]
    for name, p in model.named_parameters():
        if pow2_only:  # coefficient set {-0.5, -0.25, 0, 0.25, 0.5}
            p.copy_(_dyadic(p.shape, 2, 2, gen))
        elif name.endswith("bias"):
            # Distinct values: a quantizer that sees a single unique value
            # never marks itself calibrated (search_done stays False).
            k = torch.randperm(17, generator=gen)[:p.numel()] - 8
            p.copy_((k / 16).reshape(p.shape))
        else:
            p.copy_(_dyadic(p.shape, 4, 2, gen))
    if isinstance(model.pre, _BN):
        bn = model.pre.bn
        bn.weight.copy_(torch.randint(1, 5, bn.weight.shape, generator=gen) / 4)
        bn.bias.copy_(_dyadic(bn.bias.shape, 4, 3, gen))
        bn.running_mean.copy_(_dyadic(bn.running_mean.shape, 4, 2, gen))
        bn.running_var.copy_(4.0 ** torch.randint(-1, 2, bn.running_var.shape, generator=gen))

    # Pre-activation: scale (and sign-flip so the extreme value is positive,
    # which pushes ReLU6 into its saturated region) the module feeding it.
    feeder = model.pre if isinstance(model.pre, _BN) else model.l1
    h = model.preact(x)
    peak = h.flatten()[h.abs().argmax()].item()
    f = _dyadic_scale(PREACT_LIMIT.get(act_name, DEFAULT_PREACT_LIMIT) / abs(peak), pow2_only)
    f = math.copysign(f, peak)
    for p in feeder.parameters():
        p.mul_(f)

    y = model(x)
    f = _dyadic_scale(OUTPUT_LIMIT / y.abs().max().item(), pow2_only=True)
    for p in model.l2.parameters():
        p.mul_(f)


def _copy_params(src, dst):
    src_params = dict(src.named_parameters())
    dst_params = dict(dst.named_parameters())
    assert src_params.keys() == dst_params.keys()
    with torch.no_grad():
        for name, p in dst_params.items():
            p.copy_(src_params[name])
        for (_, b_src), (_, b_dst) in zip(src.pre.named_buffers(), dst.pre.named_buffers()):
            b_dst.copy_(b_src)


def _quantizers(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, BaseQuantizer)}


def _record_quantizer_io(model):
    """Forward hooks recording (expected output, output, scale) of every
    quantizer. A no-op quantizer returns its input unchanged -- except the
    fused SiLU quantizer, which applies SiLU itself, so returns silu(input)."""
    records, handles = {}, []
    for name, q in _quantizers(model).items():
        def hook(mod, inp, out, name=name):
            expected = inp[0].detach()
            if isinstance(mod, SiLUTensorQuant):
                expected = F.silu(expected)
            records[name] = (expected, out[0].detach(), out[1].detach())
        handles.append(q.register_forward_hook(hook))
    return records, handles


def _is_downstream_of_smooth_act(qname):
    # The smooth activation's output quantizer and the model-output quantizer.
    # (QuantIdentity-based activations name it output_quant, Brevitas'
    # QuantSigmoid/QuantTanh name it act_quant; the fused SiLU quantizes its
    # own output via act_quant.)
    return qname.startswith(("act.output_quant", "act.act_quant", "out."))


def _loss(y, target):
    return 0.5 * ((y - target) ** 2).sum()


def _run_case(arch, act_name, state, clipped_ste=False, seed=0, dtype=None):
    torch.manual_seed(seed)
    if dtype is None:
        dtype = torch.float32 if arch in FLOAT32_ARCHS else DTYPE
    exact = ACTIVATIONS[act_name][2]
    x = ARCHS[arch][1]()
    if x.is_floating_point():
        x = x.to(dtype)

    ref = TwoLayer(arch, act_name, quant=False).to(dtype)
    _init_params(ref, arch, act_name, x, seed)

    QuantizerManager().reset()
    qmodel = TwoLayer(arch, act_name, quant=True).to(dtype)
    _copy_params(ref, qmodel)
    quantizers = _quantizers(qmodel)
    assert quantizers, "quantized model has no quantizers"
    for q in quantizers.values():
        q.clipped_ste = clipped_ste

    # Calibration pass (train mode, like the harness's calibration phase).
    qmodel.train()
    ref.train()
    with torch.no_grad():
        qmodel(x)
    for name, q in quantizers.items():
        assert q.search_done_value, f"{name} did not calibrate"

    if state == "mid_anneal":
        for q in quantizers.values():
            q.set_annealing_alpha(0.5)
    elif state == "disabled":
        QuantizerManager().disable_quantization()
    else:
        assert state == "quantizing"
        assert QuantizerManager().is_quantizing_everything_fully

    records, handles = _record_quantizer_io(qmodel)
    x_q = x.clone().requires_grad_() if x.is_floating_point() else x
    x_r = x.clone().requires_grad_() if x.is_floating_point() else x
    y_q = qmodel(x_q)
    y_r = ref(x_r)
    for h in handles:
        h.remove()

    # --- Premise: no quantizer changed any value in the forward pass. -------
    assert records.keys() == quantizers.keys(), "not every quantizer ran"
    for name, (expected, q_out, scale) in records.items():
        if exact or not _is_downstream_of_smooth_act(name):
            assert torch.equal(q_out, expected), (
                f"premise broken: quantizer {name} changed its input "
                f"(max |diff| = {(q_out - expected).abs().max().item()})")
        else:
            assert (q_out - expected).abs().max() <= scale, (
                f"{name} is off by {(q_out - expected).abs().max().item()}, "
                f"more than 1 LSB ({scale.item()})")

    target = _dyadic(y_r.shape, 8, 3, torch.Generator().manual_seed(seed + 1)).to(dtype)
    _loss(y_q, target).backward()
    _loss(y_r, target).backward()

    # --- The actual check: identical gradients. -----------------------------
    grads_q = {n: p.grad for n, p in qmodel.named_parameters()}
    grads_r = {n: p.grad for n, p in ref.named_parameters()}
    assert grads_q.keys() == grads_r.keys()
    if x.is_floating_point():
        grads_q["<input>"], grads_r["<input>"] = x_q.grad, x_r.grad

    for name, g_r in grads_r.items():
        g_q = grads_q[name]
        assert g_q is not None, f"no gradient reached {name} in the quantized model"
        assert g_r.abs().sum() > 0 or name.startswith("pre."), f"degenerate test: zero gradient for {name}"
        if exact:
            assert torch.equal(g_q, g_r), (
                f"gradient of {name} differs from the float reference: "
                f"max |diff| = {(g_q - g_r).abs().max().item()}")
        else:
            torch.testing.assert_close(g_q, g_r, rtol=1e-6, atol=1e-7,
                                       msg=lambda m: f"gradient of {name}: {m}")
    if exact:
        assert torch.equal(y_q, y_r)
    else:
        torch.testing.assert_close(y_q, y_r, rtol=1e-6, atol=1e-7)


@pytest.fixture(autouse=True)
def reset_manager():
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


def _known_bug_marks(act_name, state="quantizing"):
    """xfail markers for known deviations."""
    marks = []
    if act_name == "fused_silu_act_quant" and state != "quantizing":
        marks.append(pytest.mark.xfail(reason=KNOWN_BUGS["fused_silu_passthrough"], strict=True))
    return marks


@pytest.mark.parametrize("arch, act_name, state", [
    pytest.param(arch, act, state, marks=_known_bug_marks(act, state), id=f"{arch}-{act}-{state}")
    for arch in ARCHS
    for act in ACTIVATIONS
    for state in ["quantizing", "mid_anneal", "disabled"]
])
def test_gradients_match_float_reference(arch, act_name, state):
    _run_case(arch, act_name, state)


@pytest.mark.parametrize("ste_mode", ["clip", "single_direction_clip"])
@pytest.mark.parametrize("arch, act_name", [
    pytest.param(arch, act, marks=_known_bug_marks(act), id=f"{arch}-{act}")
    for arch in ["linear_linear", "conv2d_conv2d", "embedding_linear"]
    for act in ACTIVATIONS
])
def test_gradients_match_float_reference_clipped_ste(arch, act_name, ste_mode):
    """All values are in range, so either clipping mode must pass every
    gradient unchanged."""
    _run_case(arch, act_name, "quantizing", clipped_ste=ste_mode)


@pytest.mark.parametrize("act_name, seed", [
    pytest.param(act, seed, marks=_known_bug_marks(act), id=f"{act}-{seed}")
    for act in ["relu", "relu6", "leaky_relu", "tanh", "gelu"]
    for seed in [1, 2, 3]
])
def test_gradients_match_float_reference_other_seeds(act_name, seed):
    _run_case("conv2d_conv2d", act_name, "quantizing", seed=seed)


def test_relu6_gradient_at_clip_boundaries_matches_pytorch():
    """Regression: QuantReLU6's gradient exactly at the clip points 0 and 6 must
    be nn.ReLU6's (0), not torch.clamp's (1). Quantized pre-activations land
    on these points exactly whenever they round to code 0."""
    act = QuantReLU6(bit_width=8).to(DTYPE).train()
    x = torch.tensor([-1.0, 0.0, 3.0, 6.0, 7.0], dtype=DTYPE, requires_grad=True)
    act(x).sum().backward()
    x_ref = x.detach().clone().requires_grad_()
    nn.ReLU6()(x_ref).sum().backward()
    assert torch.equal(x.grad, x_ref.grad), f"{x.grad.tolist()} != {x_ref.grad.tolist()}"


@pytest.mark.xfail(reason=KNOWN_BUGS["coeff_float64"], strict=True, raises=RuntimeError)
def test_coefficient_weight_quant_float64():
    _run_case("linear_coeff_weights", "identity", "quantizing", dtype=torch.float64)


def test_negative_control_off_grid_data_is_detected():
    """Sanity check of the harness itself: with off-grid (random float) input
    data, the quantizers are NOT no-ops, and the premise check must catch it.
    Guards against the main tests passing vacuously (e.g. quantizers silently
    gated off)."""
    QuantizerManager().reset()
    qmodel = TwoLayer("linear_linear", "relu", quant=True).to(DTYPE).train()
    x = torch.randn(4, 6, dtype=DTYPE)
    with torch.no_grad():
        qmodel(x)
    records, handles = _record_quantizer_io(qmodel)
    with torch.no_grad():
        qmodel(x)
    for h in handles:
        h.remove()
    changed = [n for n, (q_in, q_out, _) in records.items() if not torch.equal(q_in, q_out)]
    assert "inp.act_quant.fused_activation_quant_proxy.tensor_quant" in changed
    assert "l1.weight_quant.tensor_quant" in changed
