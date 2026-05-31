"""
Verify that a bit-width-annealed model is actually quantized end-to-end.

Trains SimpleMNISTNet for a short number of epochs with `annealing_mode='bit_width'`
(enough to complete the 16→8 ramp + a couple QAT epochs), exports to ONNX, then
runs four diagnostic checks:

  1. Quantizer state    — effective_bit_width, alpha, search_done all correct.
  2. Weight grid        — every layer's quantized weights lie on the 8-bit grid.
  3. Activation grid    — runtime activations after each quantizer are on-grid.
  4. ONNX export        — Quantify::FixedPointQuant nodes present, bit_width=8.

Prints a per-quantizer table and a single-line VERDICT. Throwaway script.
"""

import collections
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, ".")

import brevitas.nn as qnn

from examples.simple_mnist_qat_bitwidth import SimpleMNISTNet
from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorQuantizer,
    quantize_fixed_point,
)
from quantizers.manager import QuantizerManager
from training_harness.config import QuantScheduleConfig, TrainerConfig
from training_harness.engine_utils import set_seed
from training_harness.trainer import Trainer
from utils.onnx_export import export_onnx_with_io


# ---------------------------------------------------------------------------
# Train (short)
# ---------------------------------------------------------------------------

def train_short(epochs: int = 12) -> tuple[nn.Module, torch.Tensor, str]:
    set_seed(42, deterministic=True)
    QuantizerManager().reset()

    tx = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tx)
    val_ds = datasets.MNIST("./data", train=False, transform=tx)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleMNISTNet().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    cfg = TrainerConfig(
        experiment_name="bw_verify",
        output_dir="logs/_bw_verify",
        epochs=epochs,
        batch_size=256,
        learning_rate=1e-3,
        quant_schedule=QuantScheduleConfig(
            float_warmup_epochs=5,
            annealing_mode="bit_width",
            start_bit_width=16,
            track_scale_factors=False,
        ),
    )
    cfg.logging.save_plots = False

    Trainer(config=cfg, model=model, optimizer=opt,
            train_loader=train_loader, val_loader=val_loader,
            loss_fn=nn.CrossEntropyLoss()).fit()

    dummy = train_ds[0][0].unsqueeze(0).to(device)
    onnx_path = "_bw_verify.onnx"
    model.eval()
    export_onnx_with_io(model, dummy, onnx_path)
    return model, dummy, onnx_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_step(q: FixedPointPerTensorQuantizer) -> float:
    return 2.0 ** int(q.search_result_lsb.item())


def _is_on_grid(values: torch.Tensor, step: float, atol: float = 1e-6) -> bool:
    """Every value must be an integer multiple of `step` (within atol)."""
    if values.numel() == 0:
        return True
    integers = values.detach().cpu().double() / step
    return bool(((integers - integers.round()).abs() <= atol).all())


def _named_quantizers(model: nn.Module) -> dict[str, FixedPointPerTensorQuantizer]:
    """Map fully-qualified attribute path → FixedPointPerTensorQuantizer instance."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, FixedPointPerTensorQuantizer):
            out[name] = mod
    return out


# ---------------------------------------------------------------------------
# Check 1: quantizer state
# ---------------------------------------------------------------------------

def check_quantizer_state(model: nn.Module, target_bw: int = 8) -> list[str]:
    failures = []
    print(f"\n{'Quantizer':<55} {'eff_bw':>7} {'alpha':>6} {'cal':>4} {'LSB':>5}")
    print("=" * 85)
    for name, q in sorted(_named_quantizers(model).items()):
        eff_bw = int(q.effective_bit_width.item())
        alpha = float(q.annealing_alpha.item())
        done = bool(q.search_done.item())
        lsb = int(q.search_result_lsb.item())
        print(f"{name:<55} {eff_bw:>7} {alpha:>6.2f} {str(done):>4} {lsb:>5}")
        if eff_bw != target_bw:
            failures.append(f"{name}: eff_bw={eff_bw} != target={target_bw}")
        if abs(alpha - 1.0) > 1e-6:
            failures.append(f"{name}: alpha={alpha} != 1.0 (bit-width mode should pin to 1)")
        if not done:
            failures.append(f"{name}: search_done=False — never calibrated")
    return failures


# ---------------------------------------------------------------------------
# Check 2: weight grid
# ---------------------------------------------------------------------------

def check_weight_grid(model: nn.Module) -> list[str]:
    failures = []
    print(f"\n{'Layer':<25} {'n_unique':>10} {'|q|_max':>10} {'on-grid':>8} {'in-range':>10}")
    print("=" * 70)
    for name, mod in model.named_modules():
        if not isinstance(mod, (qnn.QuantConv2d, qnn.QuantLinear)):
            continue
        # Find this layer's weight quantizer (it's a Brevitas proxy wrapping our FixedPoint quantizer)
        proxy = mod.weight_quant
        if proxy is None or not hasattr(proxy, "tensor_quant"):
            continue
        tq = proxy.tensor_quant
        if not isinstance(tq, FixedPointPerTensorQuantizer):
            continue

        raw = mod.weight.detach()
        # Use the same code path the model uses at forward time
        with torch.no_grad():
            out = proxy(raw)
        q_weight = out.value if hasattr(out, "value") else out
        q_weight = q_weight.detach()
        step = _grid_step(tq)
        n_unique = int(torch.unique(q_weight).numel())
        on_grid = _is_on_grid(q_weight, step)

        eff_bw = int(tq.effective_bit_width.item())
        signed = bool(tq.search_result_is_signed.item())
        int_min = -(2 ** (eff_bw - 1)) if signed else 0
        int_max = (2 ** (eff_bw - 1) - 1) if signed else (2 ** eff_bw - 1)
        lo, hi = int_min * step, int_max * step
        q_max_abs = float(q_weight.abs().max())
        in_range = bool((q_weight >= lo - 1e-6).all() and (q_weight <= hi + 1e-6).all())

        print(f"{name+'.weight':<25} {n_unique:>10} {q_max_abs:>10.4f} {('✓' if on_grid else '✗'):>8} {('✓' if in_range else '✗'):>10}")
        if n_unique > 2 ** eff_bw:
            failures.append(f"{name}.weight: {n_unique} unique values > 2^{eff_bw}")
        if not on_grid:
            failures.append(f"{name}.weight: values not on fixed-point grid (step={step:.2e})")
        if not in_range:
            failures.append(f"{name}.weight: quantized values escape [{lo:.4f}, {hi:.4f}]")
    return failures


# ---------------------------------------------------------------------------
# Check 3: activation grid
# ---------------------------------------------------------------------------

def check_activation_grid(model: nn.Module, sample: torch.Tensor) -> list[str]:
    """Hook every quant-emitting layer (QuantIdentity / QuantConv2d / QuantLinear)
    and inspect its post-quantization output."""
    failures = []
    captured: dict[str, torch.Tensor] = {}

    def make_hook(layer_name: str):
        def hook(_mod, _inp, output):
            t = output.value if hasattr(output, "value") else output
            captured[layer_name] = t.detach()
        return hook

    handles = []
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, (qnn.QuantIdentity, qnn.QuantConv2d, qnn.QuantLinear)):
            targets.append(name)
            handles.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        _ = model(sample)

    for h in handles:
        h.remove()

    quantizer_map = _named_quantizers(model)
    print(f"\n{'Activation source':<25} {'n_unique':>10} {'on-grid':>8} {'in-range':>10}")
    print("=" * 60)
    for layer_name in targets:
        act = captured.get(layer_name)
        if act is None:
            continue
        # Pull the matching activation quantizer (output_quant or act_quant).
        # Brevitas wraps activation quantizers in a FusedActivationQuantProxy,
        # which holds the real tensor_quant — drill in if present.
        layer = dict(model.named_modules())[layer_name]
        out_q = getattr(layer, "output_quant", None) or getattr(layer, "act_quant", None)
        if out_q is None:
            continue
        if hasattr(out_q, "fused_activation_quant_proxy") and out_q.fused_activation_quant_proxy is not None:
            tq = out_q.fused_activation_quant_proxy.tensor_quant
        elif hasattr(out_q, "tensor_quant"):
            tq = out_q.tensor_quant
        else:
            continue
        if not isinstance(tq, FixedPointPerTensorQuantizer):
            continue
        step = _grid_step(tq)
        n_unique = int(torch.unique(act).numel())
        on_grid = _is_on_grid(act, step)

        eff_bw = int(tq.effective_bit_width.item())
        signed = bool(tq.search_result_is_signed.item())
        int_min = -(2 ** (eff_bw - 1)) if signed else 0
        int_max = (2 ** (eff_bw - 1) - 1) if signed else (2 ** eff_bw - 1)
        lo, hi = int_min * step, int_max * step
        in_range = bool((act >= lo - 1e-6).all() and (act <= hi + 1e-6).all())

        print(f"{layer_name:<25} {n_unique:>10} {('✓' if on_grid else '✗'):>8} {('✓' if in_range else '✗'):>10}")
        if n_unique > 2 ** eff_bw:
            failures.append(f"{layer_name}: {n_unique} unique activation values > 2^{eff_bw}")
        if not on_grid:
            failures.append(f"{layer_name}: activation values not on grid")
        if not in_range:
            failures.append(f"{layer_name}: activations outside [{lo:.4f}, {hi:.4f}]")
    return failures


# ---------------------------------------------------------------------------
# Check 4: ONNX inspection
# ---------------------------------------------------------------------------

def check_onnx(path: str, target_bw: int = 8) -> list[str]:
    import onnx
    failures = []
    m = onnx.load(path)

    ops = collections.Counter(n.op_type for n in m.graph.node)
    doms = collections.Counter(n.domain for n in m.graph.node)
    print(f"\nONNX ops:      {dict(ops)}")
    print(f"ONNX domains:  {dict(doms)}")

    fp_nodes = [n for n in m.graph.node if n.op_type == "FixedPointQuant"]
    print(f"FixedPointQuant nodes: {len(fp_nodes)}")
    if len(fp_nodes) == 0:
        failures.append("No FixedPointQuant ops in exported ONNX")
        return failures

    # bit_width attribute should be the target everywhere. Torch translates
    # `bit_width_i=...` to an ONNX attribute named `bit_width` (the `_i` is a
    # type hint, not part of the name).
    bws = []
    int_extremes = []
    for n in fp_nodes:
        attrs = {a.name: a for a in n.attribute}
        if "bit_width" not in attrs:
            failures.append(
                f"node {n.name}: missing bit_width attribute "
                f"(found: {sorted(attrs.keys())})"
            )
            continue
        bws.append(int(attrs["bit_width"].i))
        if "quantized_ints" in attrs:
            t = attrs["quantized_ints"].t
            arr = onnx.numpy_helper.to_array(t)
            int_extremes.append((int(arr.min()), int(arr.max()), arr.size))

    bw_set = set(bws)
    print(f"FixedPointQuant.bit_width values: {sorted(bw_set)}")
    if bw_set != {target_bw}:
        failures.append(f"Mixed/non-target bit_width in ONNX: {sorted(bw_set)}; expected {{{target_bw}}}")

    # Sample integer range
    print(f"quantized_ints summary (first 3 nodes): {int_extremes[:3]}")
    for lo, hi, n in int_extremes:
        if hi > 2 ** (target_bw - 1) - 1 or lo < -(2 ** (target_bw - 1)):
            failures.append(f"quantized_ints out of signed-{target_bw} range: [{lo}, {hi}]")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 85)
    print("Training short bit-width-annealing run …")
    print("=" * 85)
    model, dummy, onnx_path = train_short(epochs=12)

    print("\n" + "=" * 85)
    print("CHECK 1 — quantizer state at end of training")
    print("=" * 85)
    f1 = check_quantizer_state(model, target_bw=8)

    print("\n" + "=" * 85)
    print("CHECK 2 — weights on the 8-bit fixed-point grid")
    print("=" * 85)
    f2 = check_weight_grid(model)

    print("\n" + "=" * 85)
    print("CHECK 3 — activations on the 8-bit fixed-point grid")
    print("=" * 85)
    f3 = check_activation_grid(model, dummy)

    print("\n" + "=" * 85)
    print(f"CHECK 4 — ONNX export ({onnx_path})")
    print("=" * 85)
    f4 = check_onnx(onnx_path, target_bw=8)

    all_failures = f1 + f2 + f3 + f4
    print("\n" + "=" * 85)
    if not all_failures:
        print("VERDICT: PASS — model is fully 8-bit quantized end-to-end.")
    else:
        print(f"VERDICT: FAIL — {len(all_failures)} issue(s):")
        for f in all_failures:
            print(f"  - {f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
