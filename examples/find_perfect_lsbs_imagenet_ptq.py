"""
find_perfect_lsbs_imagenet_ptq.py — PTQ per-quantizer LSB search on ImageNet.

Sequentially processes each quantizer in forward-pass order. For each one:
  1. Calibrate using a single training batch (near-zero LR so weights don't
     move). The quantizer manager's quantization_start_gap=2 means the
     quantizer may need to see a few steps before it activates; the script
     loops with training steps until q.search_done flips to True, then stops.
     Annealing is disabled by setting annealing_alpha_step=1 so the quantizer
     jumps from passthrough to fully-quantized in one active step.
  2. Evaluate the validation set at each LSB candidate in the range
     [calibrated_lsb - search_radius, ..., calibrated_lsb + search_radius].
  3. Select the LSB with the lowest validation loss and lock it in.
  4. Move to the next quantizer (all previously-locked quantizers remain
     active so their effect is visible during later searches).

All previous quantizers stay active throughout the search, so each new
quantizer is optimised in the context of the already-quantized network.

Outputs per quantizer:
  - SVG + PNG plot: val_loss bars + val_acc line vs. LSB candidate,
    with the calibrated and selected LSBs highlighted
  - Text log entry: per-candidate metrics and selection decision

One summary plot is written at the end showing how val metrics evolve
as each quantizer is added.

Usage examples
--------------
# ResNet-18, search weight LSBs (±2 positions), pretrained, local ImageFolder:
python examples/find_perfect_lsbs_imagenet_ptq.py \\
    --model resnet18 --mode weights --weight-bits 8 --pretrained \\
    --data-dir /data/imagenet --search-radius 2

# ResNet-18, search activation LSBs on HuggingFace, quick eval (100 batches):
python examples/find_perfect_lsbs_imagenet_ptq.py \\
    --model resnet18 --mode activations --act-bits 8 --pretrained \\
    --hf-dataset ILSVRC/imagenet-1k --eval-batches 100

# ResNet-50, weight search with radius 1 (only direct neighbours):
python examples/find_perfect_lsbs_imagenet_ptq.py \\
    --model resnet50 --mode weights --weight-bits 8 --pretrained \\
    --data-dir /data/imagenet --search-radius 1
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from models.resnet_quant import QuantResNet18, QuantResNet50
from models.mobilenetv1_quant import QuantMobileNetV1
from models.mobilenetv2_quant import QuantMobileNetV2
from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorWeightQuant,
)
from quantizers.base_quantizer import BaseQuantizer
from quantizers.manager import QuantizerManager
from utils.weight_mapping import load_pretrained_weights


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PTQ per-quantizer LSB search on ImageNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model",
        choices=["resnet18", "resnet50", "mobilenetv1", "mobilenetv2"],
        default="resnet18",
    )
    p.add_argument(
        "--mode",
        choices=["weights", "activations"],
        required=True,
        help="Quantize weights or activations (not both — biases are skipped)",
    )
    p.add_argument("--weight-bits", type=int, default=8, help="Bit width for weight quantizers")
    p.add_argument("--act-bits",    type=int, default=8, help="Bit width for activation quantizers")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="Load torchvision pretrained weights (supported: resnet18, resnet50, mobilenetv2)",
    )

    d = p.add_argument_group("data")
    d.add_argument(
        "--data-dir", type=str, default=None,
        metavar="PATH",
        help="ImageFolder root (train/ + val/ subdirs). Uses NVIDIA DALI when set.",
    )
    d.add_argument(
        "--hf-dataset", type=str, default="ILSVRC/imagenet-1k",
        help="HuggingFace dataset name (used when --data-dir is not set)",
    )
    d.add_argument("--num-workers",  type=int, default=8)
    d.add_argument("--dali-threads", type=int, default=4)
    d.add_argument("--batch-size",   type=int, default=128)

    s = p.add_argument_group("search")
    s.add_argument(
        "--search-radius", type=int, default=2,
        help="Number of LSB positions to test on each side of the calibrated value. "
             "0 = evaluate the calibrated position only (no search).",
    )
    s.add_argument(
        "--eval-batches", type=int, default=None,
        help="Validation batches per LSB candidate evaluation. "
             "None = full validation set. Use a smaller number for speed.",
    )

    p.add_argument("--output-dir",      type=str, default="output/ptq_lsb_search")
    p.add_argument("--experiment-name", type=str, default=None,
                   help="Override the auto-generated experiment name")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_model(args, weight_quant, act_quant) -> nn.Module:
    nc = args.num_classes
    if args.model == "resnet18":
        return QuantResNet18(nc, weight_quant, act_quant)
    if args.model == "resnet50":
        return QuantResNet50(nc, weight_quant, act_quant)
    if args.model == "mobilenetv1":
        return QuantMobileNetV1(nc, weight_quant, act_quant)
    if args.model == "mobilenetv2":
        return QuantMobileNetV2(nc, weight_quant=weight_quant, act_quant=act_quant)
    raise ValueError(f"Unknown model: {args.model!r}")


def _load_pretrained(model: nn.Module, model_name: str) -> nn.Module:
    from torchvision.models import (
        resnet18, ResNet18_Weights,
        resnet50, ResNet50_Weights,
        mobilenet_v2, MobileNet_V2_Weights,
    )
    if model_name == "resnet18":
        print("[pretrained] resnet18 (IMAGENET1K_V1) …")
        float_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif model_name == "resnet50":
        print("[pretrained] resnet50 (IMAGENET1K_V2) …")
        float_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    elif model_name == "mobilenetv2":
        print("[pretrained] mobilenet_v2 …")
        float_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    else:
        print(f"[pretrained] No torchvision weights for {model_name!r}, skipping.")
        return model
    return load_pretrained_weights(model, float_model)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class _HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        return self.transform(item["image"].convert("RGB")), item["label"]


def _build_dataloaders(args) -> tuple:
    if args.data_dir:
        from utils.dali_pipeline import build_dali_loaders
        print(f"[data] DALI from {args.data_dir} …")
        return build_dali_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_threads=args.dali_threads,
        )

    from datasets import load_dataset
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                          T.ToTensor(), norm])
    val_tf   = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])

    print(f"[data] HuggingFace ({args.hf_dataset}) …")
    hf_train = load_dataset(args.hf_dataset, split="train")
    hf_val   = load_dataset(args.hf_dataset, split="validation")

    persistent = args.num_workers > 0
    prefetch   = 3 if args.num_workers > 0 else None
    train_loader = DataLoader(
        _HFDatasetWrapper(hf_train, train_tf),
        batch_size=args.batch_size, shuffle=True, pin_memory=True,
        num_workers=args.num_workers, persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        _HFDatasetWrapper(hf_val, val_tf),
        batch_size=args.batch_size, shuffle=False, pin_memory=True,
        num_workers=args.num_workers, persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: Optional[int],
    label: str = "",
) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy_percent) over up to max_batches batches.

    Prints a live \r progress line while running so the user can track how
    far through the validation set each LSB evaluation has progressed.
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    n_batches  = max_batches if max_batches is not None else "?"
    prefix     = f"    {label}" if label else "   "
    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images  = images.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True).long()
        outputs = model(images)
        total_loss += loss_fn(outputs, labels).item() * images.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += images.size(0)
        running_loss = total_loss / total
        running_acc  = 100.0 * correct / total
        print(
            f"{prefix}  batch [{i+1}/{n_batches}]"
            f"  loss={running_loss:.4f}  acc={running_acc:.2f}%   ",
            end="\r", flush=True,
        )
    print()  # newline after the \r progress line
    if total == 0:
        return float("inf"), 0.0
    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Descriptive quantizer naming
# ---------------------------------------------------------------------------

_PROXY_SUFFIXES: list[tuple[str, str]] = [
    (".weight_quant.tensor_quant",                            "_weight"),
    (".bias_quant.tensor_quant",                              "_bias"),
    (".act_quant.fused_activation_quant_proxy.tensor_quant",  "_act"),
    (".act_quant.tensor_quant",                               "_act"),
    (".input_quant.tensor_quant",                             "_act_in"),
    (".output_quant.tensor_quant",                            "_act_out"),
]


def _assign_descriptive_ids(model: nn.Module) -> None:
    """
    Replace generic quant_N ids with location-based names derived from
    model.named_modules() paths, then sync the QuantizerManager registry.
    """
    mgr  = QuantizerManager()
    seen: dict[str, int] = {}
    for path, module in model.named_modules():
        if not isinstance(module, BaseQuantizer):
            continue
        qid = path
        for suffix, role in _PROXY_SUFFIXES:
            if path.endswith(suffix):
                parent = path[: -len(suffix)].replace(".", "_")
                qid = f"{parent}{role}" if parent else f"root{role}"
                break
        else:
            qid = path.replace(".", "_")
        # Deduplicate with a counter suffix
        if qid in seen:
            seen[qid] += 1
            qid = f"{qid}_{seen[qid]}"
        else:
            seen[qid] = 0
        module.quant_id = qid
    mgr.quantizers = {q.quant_id: q for q in mgr.quantizers.values()}


# ---------------------------------------------------------------------------
# Per-quantizer PTQ search plot
# ---------------------------------------------------------------------------

def _save_ptq_search_plot(
    *,
    results: List[Tuple[int, float, float]],
    calib_lsb: int,
    selected_lsb: int,
    quant_id: str,
    quantizer_role: str,
    bit_width: int,
    quantizer_index: int,
    n_quantizers: int,
    out_dir: Path,
) -> None:
    """
    Dual-axis bar/line chart: val_loss bars (left) + val_acc line (right).
    Bar colours:
      orangered  = selected LSB (min val_loss)
      gold       = calibrated LSB (when different from selected)
      steelblue  = all other candidates
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    lsbs   = [r[0] for r in results]
    losses = [r[1] for r in results]
    accs   = [r[2] for r in results]

    def _bar_color(lsb: int) -> str:
        if lsb == selected_lsb:
            return "orangered"
        if lsb == calib_lsb:
            return "gold"
        return "steelblue"

    fig, ax_loss = plt.subplots(figsize=(max(10, len(lsbs) + 4), 5))
    ax_acc = ax_loss.twinx()

    ax_loss.bar(lsbs, losses, color=[_bar_color(l) for l in lsbs], alpha=0.80, width=0.55)
    ax_loss.set_xlabel("LSB position")
    ax_loss.set_ylabel("Val loss", color="steelblue")
    ax_loss.tick_params(axis="y", labelcolor="steelblue")

    ax_acc.plot(lsbs, accs, color="green", linewidth=2.0, marker="o", markersize=7, zorder=5)
    ax_acc.set_ylabel("Val acc (%)", color="green")
    ax_acc.tick_params(axis="y", labelcolor="green")

    # Vertical marker for selected LSB (red dashed) and calibrated LSB (navy dotted)
    ax_loss.axvline(selected_lsb, color="red",  linestyle="--", linewidth=1.8, alpha=0.9)
    if calib_lsb != selected_lsb:
        ax_loss.axvline(calib_lsb, color="navy", linestyle=":",  linewidth=1.4, alpha=0.8)

    ax_loss.set_xticks(lsbs)
    ax_loss.tick_params(axis="x", rotation=45)

    selected_r = next(r for r in results if r[0] == selected_lsb)
    calib_r    = next(r for r in results if r[0] == calib_lsb)
    delta_loss = selected_r[1] - calib_r[1]
    delta_acc  = selected_r[2] - calib_r[2]

    legend_handles = [
        Patch(facecolor="orangered", alpha=0.80,
              label=f"Selected  LSB={selected_lsb}  (min val_loss)"),
        Patch(facecolor="gold",      alpha=0.80,
              label=f"Calibrated  LSB={calib_lsb}"),
        Patch(facecolor="steelblue", alpha=0.80,
              label="Other candidates"),
        Line2D([0], [0], color="green", linewidth=2.0, marker="o", markersize=6,
               label="Val acc (%)"),
    ]
    ax_loss.legend(handles=legend_handles, loc="upper left", fontsize=8)

    info = (
        f"Quantizer [{quantizer_index}/{n_quantizers}]: {quant_id}\n"
        f"Role: {quantizer_role}  |  Bit width: {bit_width}b\n"
        f"Calibrated  LSB={calib_lsb:>4}  val_loss={calib_r[1]:.4f}  acc={calib_r[2]:.2f}%\n"
        f"Selected    LSB={selected_lsb:>4}  val_loss={selected_r[1]:.4f}  acc={selected_r[2]:.2f}%\n"
        f"Δ vs calibrated:  val_loss={delta_loss:+.4f}  acc={delta_acc:+.2f}%"
    )
    ax_loss.text(
        0.01, 0.98, info,
        transform=ax_loss.transAxes, fontsize=8.5, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.92),
    )

    change_str = (f"  (Δacc={delta_acc:+.2f}%)" if calib_lsb != selected_lsb
                  else "  (same as calibrated)")
    fig.suptitle(
        f"PTQ LSB Search  [{quantizer_index}/{n_quantizers}]  —  {quant_id}  "
        f"[{quantizer_role}]  [{bit_width}b]\n"
        f"Calibrated LSB={calib_lsb}  →  Selected LSB={selected_lsb}{change_str}",
        fontsize=10,
    )
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"ptq_{quant_id.replace('/', '_')}"
    fig.savefig(base.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def _save_summary_plot(
    *,
    quant_ids: List[str],
    selected_lsbs: List[int],
    val_losses: List[float],
    val_accs: List[float],
    baseline_loss: float,
    baseline_acc: float,
    out_dir: Path,
) -> None:
    """
    One point per quantizer: val_loss (left, dashed) + val_acc (right, solid).
    Each point represents the best-LSB evaluation AFTER adding that quantizer,
    with all previous quantizers already active at their selected LSBs.
    Dotted baseline reference lines show the float-precision starting point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(quant_ids)
    x = list(range(n))

    fig, ax_acc = plt.subplots(figsize=(max(10, n * 0.7 + 3), 5))
    ax_loss = ax_acc.twinx()

    ax_acc.plot(x, val_accs,   color="green",     marker="o", linewidth=2.0, markersize=6,
                label="Val acc (%)")
    ax_loss.plot(x, val_losses, color="steelblue", marker="s", linewidth=2.0, markersize=6,
                 linestyle="--", label="Val loss")

    # Dotted baseline reference lines
    ax_acc.axhline(baseline_acc,  color="green",     linestyle=":", linewidth=1.0, alpha=0.55,
                   label=f"Float baseline acc={baseline_acc:.2f}%")
    ax_loss.axhline(baseline_loss, color="steelblue", linestyle=":", linewidth=1.0, alpha=0.55,
                    label=f"Float baseline loss={baseline_loss:.4f}")

    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(
        [f"{qid}\nLSB={lsb}" for qid, lsb in zip(quant_ids, selected_lsbs)],
        rotation=45, ha="right", fontsize=7,
    )
    ax_acc.set_ylabel("Val acc (%)",  color="green")
    ax_loss.set_ylabel("Val loss",    color="steelblue")
    ax_acc.tick_params(axis="y",  labelcolor="green")
    ax_loss.tick_params(axis="y", labelcolor="steelblue")
    ax_acc.set_xlabel("Quantizer (in forward-pass order)")

    lines1, labels1 = ax_acc.get_legend_handles_labels()
    lines2, labels2 = ax_loss.get_legend_handles_labels()
    ax_acc.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

    fig.suptitle(
        f"PTQ LSB Search — Cumulative Val Metrics  ({n} quantizers optimized)\n"
        "Each point: val metrics after adding & optimizing that quantizer "
        "(all earlier ones already active)",
        fontsize=10,
    )
    plt.tight_layout()

    base = out_dir / "ptq_summary"
    fig.savefig(base.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text log
# ---------------------------------------------------------------------------

def _log_quantizer_result(
    log_path: Path,
    *,
    quant_id: str,
    quantizer_role: str,
    bit_width: int,
    calib_lsb: int,
    selected_lsb: int,
    results: List[Tuple[int, float, float]],
) -> None:
    selected_r = next(r for r in results if r[0] == selected_lsb)
    calib_r    = next(r for r in results if r[0] == calib_lsb)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "",
        "=" * 68,
        f"  Quantizer : {quant_id}   Role: {quantizer_role}   [{ts}]",
        "=" * 68,
        f"  Bit width        : {bit_width}b",
        f"  Calibrated LSB   : {calib_lsb:>4}  "
        f"(val_loss={calib_r[1]:.6f}  val_acc={calib_r[2]:.3f}%)",
        f"  Selected LSB     : {selected_lsb:>4}  "
        f"(val_loss={selected_r[1]:.6f}  val_acc={selected_r[2]:.3f}%)",
        "",
        f"  {'LSB':>5}  {'Val Loss':>12}  {'Val Acc (%)':>12}  Note",
        f"  {'───':>5}  {'────────':>12}  {'──────────':>12}  ────────────────────",
    ]
    for lsb, loss, acc in sorted(results, key=lambda r: r[0]):
        if lsb == selected_lsb and lsb == calib_lsb:
            note = "◄ selected  (also calibrated)"
        elif lsb == selected_lsb:
            note = "◄ selected"
        elif lsb == calib_lsb:
            note = "  calibrated"
        else:
            note = ""
        lines.append(f"  {lsb:>5}  {loss:>12.6f}  {acc:>12.3f}%  {note}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Quantizer injector classes ───────────────────────────────────────────
    if args.mode == "weights":
        bw = args.weight_bits
        class _WQ(FixedPointPerTensorWeightQuant):
            bit_width = bw
        weight_quant, act_quant = _WQ, None
        target_role = "weight"
    else:
        bw = args.act_bits
        class _AQ(FixedPointPerTensorActivationQuant):
            bit_width = bw
        weight_quant, act_quant = None, _AQ
        target_role = "activation"

    # ── Build model ──────────────────────────────────────────────────────────
    QuantizerManager().reset()
    model = _build_model(args, weight_quant, act_quant).to(device)
    if args.pretrained:
        model = _load_pretrained(model, args.model)
    _assign_descriptive_ids(model)

    # ── Identify quantizers to optimise (in forward-pass order) ─────────────
    mgr = QuantizerManager()
    target_items: list[tuple[str, BaseQuantizer]] = [
        (qid, q) for qid, q in mgr.quantizers.items()
        if q.quantizer_role == target_role
    ]
    if not target_items:
        print(f"[ERROR] No {target_role} quantizers found in this model. "
              "Check model construction arguments.")
        return

    n_total = len(target_items)

    # ── Manager-level PTQ settings ───────────────────────────────────────────
    mgr.quantization_start_gap = 2   # each quantizer at position N gates for N*2 steps before
                                     # activating — cleared naturally by prior calibration loops

    # ── Disable all quantizers; prevent accidental calibration triggers ───────
    for _, q in mgr.quantizers.items():
        q.annealing_alpha.data.fill_(0.0)
        q.search_done.fill_(True)

    # ── Output directory & log ───────────────────────────────────────────────
    exp_name = (args.experiment_name
                or f"{args.model}_{args.mode}_{bw}b_r{args.search_radius}")
    out_dir  = Path(args.output_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ptq_search.log"

    with open(log_path, "w") as fh:
        fh.write(
            f"PTQ LSB Search Log\n"
            f"{'='*68}\n"
            f"  Model          : {args.model}\n"
            f"  Mode           : {args.mode}  ({bw}b)\n"
            f"  Pretrained     : {args.pretrained}\n"
            f"  Search radius  : ±{args.search_radius}\n"
            f"  Eval batches   : {args.eval_batches or 'full'}\n"
            f"  Quant gap      : {mgr.quantization_start_gap}\n"
            f"  N quantizers   : {n_total}\n"
            f"  Device         : {device}\n"
            f"  Started        : {datetime.now()}\n"
            f"{'='*68}\n"
        )

    print(f"\n{'═'*68}")
    print(f"  PTQ LSB Search — {exp_name}")
    print(f"  Model: {args.model}  |  Mode: {args.mode}  |  {bw}b")
    print(f"  Radius: ±{args.search_radius}  |  Gap: {mgr.quantization_start_gap}"
          f"  |  Eval: {args.eval_batches or 'full'} batches")
    print(f"  Quantizers to search: {n_total}")
    print(f"{'═'*68}\n")

    # ── Data & optimiser ─────────────────────────────────────────────────────
    print("Loading data …")
    train_loader, val_loader = _build_dataloaders(args)
    loss_fn = nn.CrossEntropyLoss()

    # Near-zero LR: weights barely move, but the training-mode forward+backward
    # pass is what allows quantizer calibration to fire (base_quantizer guards
    # on self.training before running _calibrate).
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)

    # ── Baseline evaluation (all quantizers disabled = float precision) ───────
    print("Baseline evaluation (no quantization) …")
    baseline_loss, baseline_acc = _evaluate(model, val_loader, loss_fn, device,
                                             args.eval_batches, label="baseline")
    print(f"  val_loss={baseline_loss:.4f}  val_acc={baseline_acc:.2f}%\n")

    with open(log_path, "a") as fh:
        fh.write(
            f"\nBaseline (no quantization):\n"
            f"  val_loss={baseline_loss:.6f}  val_acc={baseline_acc:.3f}%\n"
        )

    # Pre-fetch one calibration batch from the training set and reuse it for
    # every quantizer.  Weights stay nearly fixed (lr=1e-10), so one batch is
    # representative enough for PTQ calibration.
    for _calib_images, _calib_labels in train_loader:
        calib_images = _calib_images.to(device)
        calib_labels = _calib_labels.to(device).long()
        break

    # ── Per-quantizer search ──────────────────────────────────────────────────
    summary_qids:   List[str]   = []
    summary_lsbs:   List[int]   = []
    summary_accs:   List[float] = []
    summary_losses: List[float] = []

    sep = "─" * 68

    for qi, (qid, q) in enumerate(target_items, start=1):
        t0 = time.time()
        print(sep)
        print(f"[{qi}/{n_total}]  {qid}  ({target_role}, {bw}b)")
        print(sep)

        # ── Calibrate ─────────────────────────────────────────────────────────
        # Start as passthrough (alpha=0); annealing_alpha_step=1 makes the
        # quantizer jump to fully-quantized (alpha=1) in one active training
        # step, effectively disabling gradual annealing.
        q.search_done.fill_(False)
        q.annealing_alpha.data.fill_(0.0)
        q.annealing_alpha_step = 1.0

        # Advance inference_counter to exactly the gap threshold so calibration
        # fires on the very first training step.  The gap mechanism is designed
        # for QAT multi-epoch training where steps accumulate naturally; for PTQ
        # we force the counter to the threshold instead of waiting for it to
        # accumulate organically (which may require many steps per quantizer).
        gap_threshold = q.inference_sequence_id * mgr.quantization_start_gap
        if q.inference_counter < gap_threshold:
            q.inference_counter = gap_threshold

        # BN stats are frozen; only the quantizer calibration needs train mode.
        model.train()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()

        # Run training steps until search_done flips True.  With the counter
        # pre-advanced above, calibration fires on step 1 for all quantizers.
        # The loop guards against the degenerate case where find_optimal_lsb
        # sees only 1 unique value in the calibration data and refuses to mark
        # calibration as done (_save_calibration only sets search_done=True when
        # num_unique > 1); a hard limit prevents an infinite loop in that case.
        MAX_CALIB_STEPS = 10
        calib_steps = 0
        while not q.search_done.item():
            calib_steps += 1
            print(f"  calib step {calib_steps}/{MAX_CALIB_STEPS}"
                  f"  (counter={q.inference_counter}"
                  f"  threshold={gap_threshold}"
                  f"  training={q.training}) …",
                  end="\r", flush=True)
            if calib_steps > MAX_CALIB_STEPS:
                print(f"\n  [WARNING] calibration did not complete after {MAX_CALIB_STEPS} steps"
                      f" — search_done={q.search_done.item()}"
                      f"  counter={q.inference_counter}"
                      f"  threshold={gap_threshold}"
                      f"  training={q.training}"
                      f"  alpha={q.annealing_alpha.item()}")
                print(f"  Forcing search_done=True with calibrated LSB={q.search_result_lsb.item()}")
                q.search_done.fill_(True)
                break
            optimizer.zero_grad()
            outputs = model(calib_images)
            loss = loss_fn(outputs, calib_labels)
            loss.backward()
            optimizer.step()
        print()  # newline after \r

        calib_lsb    = int(q.search_result_lsb.item())
        calib_signed = bool(q.search_result_is_signed.item())
        print(f"  Calibrated in {calib_steps} step(s): LSB={calib_lsb}  signed={calib_signed}")

        # ── Sweep LSB candidates ───────────────────────────────────────────────
        candidates = list(range(calib_lsb - args.search_radius,
                                calib_lsb + args.search_radius + 1))
        results: List[Tuple[int, float, float]] = []

        n_candidates = len(candidates)
        for ci, candidate_lsb in enumerate(candidates, start=1):
            q.search_result_lsb.fill_(candidate_lsb)
            q.search_done.fill_(True)

            tag = " (calibrated)" if candidate_lsb == calib_lsb else ""
            print(f"  [{ci}/{n_candidates}] evaluating LSB={candidate_lsb}{tag} …")
            v_loss, v_acc = _evaluate(model, val_loader, loss_fn, device, args.eval_batches,
                                      label=f"LSB={candidate_lsb}")
            results.append((candidate_lsb, v_loss, v_acc))

            result_tag = " ← calibrated" if candidate_lsb == calib_lsb else ""
            print(f"  LSB={candidate_lsb:4d}  val_loss={v_loss:.4f}  val_acc={v_acc:.2f}%{result_tag}")

        # ── Select best LSB (min val_loss) ─────────────────────────────────────
        best_lsb = min(results, key=lambda r: r[1])[0]
        q.search_result_lsb.fill_(best_lsb)
        q.search_done.fill_(True)
        # alpha=1.0, alpha_step=0.0 — quantizer stays fully active for all subsequent steps

        best_r   = next(r for r in results if r[0] == best_lsb)
        calib_r  = next(r for r in results if r[0] == calib_lsb)
        delta_l  = best_r[1] - calib_r[1]
        delta_a  = best_r[2] - calib_r[2]
        elapsed  = time.time() - t0

        print(f"  → Selected LSB={best_lsb}  "
              f"val_loss={best_r[1]:.4f}  val_acc={best_r[2]:.2f}%  "
              f"(Δ vs calib: loss={delta_l:+.4f}, acc={delta_a:+.2f}%)  "
              f"[{elapsed:.0f}s]")

        # ── Log + plot ─────────────────────────────────────────────────────────
        _log_quantizer_result(
            log_path,
            quant_id=qid, quantizer_role=target_role, bit_width=bw,
            calib_lsb=calib_lsb, selected_lsb=best_lsb, results=results,
        )
        _save_ptq_search_plot(
            results=results,
            calib_lsb=calib_lsb,
            selected_lsb=best_lsb,
            quant_id=qid,
            quantizer_role=target_role,
            bit_width=bw,
            quantizer_index=qi,
            n_quantizers=n_total,
            out_dir=out_dir,
        )

        summary_qids.append(qid)
        summary_lsbs.append(best_lsb)
        summary_accs.append(best_r[2])
        summary_losses.append(best_r[1])

    # ── Final evaluation ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Final evaluation (all optimized quantizers active) …")
    final_loss, final_acc = _evaluate(model, val_loader, loss_fn, device, args.eval_batches,
                                      label="final")
    print(f"  val_loss={final_loss:.4f}  val_acc={final_acc:.2f}%")
    print(f"  Baseline:  val_loss={baseline_loss:.4f}  val_acc={baseline_acc:.2f}%")
    print(f"  Δ:         val_loss={final_loss - baseline_loss:+.4f}"
          f"  val_acc={final_acc - baseline_acc:+.2f}%")

    with open(log_path, "a") as fh:
        fh.write(
            f"\n{'='*68}\n"
            f"  FINAL RESULTS\n"
            f"{'='*68}\n"
            f"  Baseline  : val_loss={baseline_loss:.6f}  val_acc={baseline_acc:.3f}%\n"
            f"  Quantized : val_loss={final_loss:.6f}  val_acc={final_acc:.3f}%\n"
            f"  Δ         : val_loss={final_loss - baseline_loss:+.6f}"
            f"  val_acc={final_acc - baseline_acc:+.3f}%\n"
        )

    # ── Summary plot ──────────────────────────────────────────────────────────
    if summary_qids:
        _save_summary_plot(
            quant_ids=summary_qids,
            selected_lsbs=summary_lsbs,
            val_losses=summary_losses,
            val_accs=summary_accs,
            baseline_loss=baseline_loss,
            baseline_acc=baseline_acc,
            out_dir=out_dir,
        )

    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    main()
