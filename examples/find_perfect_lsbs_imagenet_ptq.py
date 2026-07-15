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
    --model resnet18 --mode weights --pretrained \\
    --data-dir /data/imagenet --search-radius 2

# ResNet-18, search activation LSBs on HuggingFace, quick eval (100 batches):
python examples/find_perfect_lsbs_imagenet_ptq.py \\
    --model resnet18 --mode activations --pretrained \\
    --hf-dataset ILSVRC/imagenet-1k --eval-batches 100

# ResNet-50, weight search at 8 bits with radius 1 (only direct neighbours):
python examples/find_perfect_lsbs_imagenet_ptq.py \\
    --model resnet50 --mode weights --bit-width 8 --pretrained \\
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
    FixedPointPerTensorBiasQuant,
)
from quantizers.base_quantizer import BaseQuantizer
from quantizers.manager import QuantizerManager
from utils.weight_mapping import load_pretrained_weights
from utils.bn_fusion import fuse_bn_into_conv


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
        choices=["weights", "activations", "bias"],
        required=True,
        help="Quantize weights, activations, or bias (one role per run — "
             "chain the others in via --init-from-ckpt). Bias quantization "
             "(fc layer only) calibrates against the bias values themselves "
             "(FixedPointPerTensorBiasQuant sets requires_input_scale=False), "
             "so it has no dependency on activation quantization having run.",
    )
    p.add_argument("--bit-width", type=int, default=10,
                   help="Bit width applied to whichever quantizer role is selected by --mode")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="Load torchvision pretrained weights (supported: resnet18, resnet50, mobilenetv2)",
    )
    p.add_argument(
        "--fuse-bn",
        action="store_true",
        help="Fuse BatchNorm into the preceding conv/linear weights before the "
             "quantizer search. Calibrates against the same weight distribution "
             "a BN-folded deployment graph will actually use.",
    )
    p.add_argument(
        "--init-from-ckpt",
        type=str, default=None, metavar="PATH",
        help="Checkpoint from a previous run of this script (e.g. --mode "
             "weights), loaded with strict=False before this run's search "
             "begins. The model is built with a quantizer for every role "
             "already calibrated in the checkpoint (extra.role_bit_widths) "
             "PLUS --mode's role: previously-calibrated roles keep their "
             "LSBs and stay fully active throughout, while --mode's role is "
             "freshly searched on top (e.g. quantize bias on an already "
             "weight-quantized model, then activations on top of both). If "
             "the checkpoint's role set already includes --mode's role, this "
             "just warm-starts the same search from the checkpoint's "
             "weights. Must use the same --fuse-bn setting as the "
             "checkpoint's run, since fusing changes the model's module "
             "structure.",
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
    d.add_argument("--batch-size",   type=int, default=512)

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

def _build_model(args, weight_quant, act_quant, bias_quant=None) -> nn.Module:
    nc = args.num_classes
    if args.model == "resnet18":
        return QuantResNet18(nc, weight_quant, act_quant, bias_quant)
    if args.model == "resnet50":
        return QuantResNet50(nc, weight_quant, act_quant, bias_quant)
    if args.model == "mobilenetv1":
        return QuantMobileNetV1(nc, weight_quant, act_quant, bias_quant)
    if args.model == "mobilenetv2":
        return QuantMobileNetV2(nc, weight_quant=weight_quant, act_quant=act_quant,
                                 bias_quant=bias_quant)
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

    Each quantizer gets two names:
      quant_id     — filesystem-safe (underscores, no spaces), used for filenames
      display_name — human-readable with original dots and role in brackets,
                     e.g. "layer1.0.conv1 [weight]"
    """
    mgr  = QuantizerManager()
    seen: dict[str, int] = {}
    for path, module in model.named_modules():
        if not isinstance(module, BaseQuantizer):
            continue
        for suffix, role in _PROXY_SUFFIXES:
            if path.endswith(suffix):
                parent_dots = path[: -len(suffix)]
                parent_us   = parent_dots.replace(".", "_")
                role_label  = role.lstrip("_")
                qid          = f"{parent_us}_{role_label}" if parent_us else f"root_{role_label}"
                display_name = (f"{parent_dots} [{role_label}]"
                                if parent_dots else f"[{role_label}]")
                break
        else:
            qid          = path.replace(".", "_")
            display_name = path
        # Deduplicate with a counter suffix
        if qid in seen:
            seen[qid] += 1
            suffix_n      = f"_{seen[qid]}"
            qid          += suffix_n
            display_name += suffix_n
        else:
            seen[qid] = 0
        module.quant_id     = qid
        module.display_name = display_name
    mgr.quantizers = {q.quant_id: q for q in mgr.quantizers.values()}
    # Any quantizer registered in the manager but not reachable via named_modules()
    # (e.g. Brevitas internal proxy objects) gets a plain quant_id as its display name.
    for q in mgr.quantizers.values():
        if not hasattr(q, "display_name"):
            q.display_name = q.quant_id


# ---------------------------------------------------------------------------
# Per-quantizer PTQ search plot
# ---------------------------------------------------------------------------

def _save_ptq_search_plot(
    *,
    results: List[Tuple[int, float, float]],
    calib_lsb: int,
    selected_lsb: int,
    quant_id: str,
    display_name: str,
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
        f"Quantizer [{quantizer_index}/{n_quantizers}]: {display_name}\n"
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
        f"PTQ LSB Search  [{quantizer_index}/{n_quantizers}]  —  {display_name}  [{bit_width}b]\n"
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
    display_names: List[str],
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
        [f"{name}\nLSB={lsb}" for name, lsb in zip(display_names, selected_lsbs)],
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
# LSB choice histogram (calibrated vs. actually-selected)
# ---------------------------------------------------------------------------

def _save_lsb_histogram_plot(
    *,
    calib_lsbs: List[int],
    selected_lsbs: List[int],
    out_dir: Path,
) -> None:
    """
    Two side-by-side bar charts, sharing a y-axis and a common x-axis (LSB
    position): how many quantizers calibration directly chose at each LSB
    (left, find_optimal_lsb's raw answer) vs. how many ended up selected
    after the full validation sweep (right, min val_loss). A large mismatch
    between the two panels means the calibration heuristic and the
    measured-best LSB disagree often.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    calib_counts = Counter(calib_lsbs)
    sel_counts   = Counter(selected_lsbs)
    all_lsbs = sorted(set(calib_counts) | set(sel_counts))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(max(10, len(all_lsbs) * 0.7 + 4), 5), sharey=True,
    )

    ax1.bar(all_lsbs, [calib_counts.get(l, 0) for l in all_lsbs],
            color="gold", alpha=0.85, width=0.6)
    ax1.set_title(f"Calibrated LSB  (find_optimal_lsb)\n{len(calib_lsbs)} quantizers")
    ax1.set_xlabel("LSB position")
    ax1.set_ylabel("Number of quantizers")
    ax1.set_xticks(all_lsbs)
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(all_lsbs, [sel_counts.get(l, 0) for l in all_lsbs],
            color="orangered", alpha=0.85, width=0.6)
    ax2.set_title(f"Selected LSB  (min val_loss sweep)\n{len(selected_lsbs)} quantizers")
    ax2.set_xlabel("LSB position")
    ax2.set_xticks(all_lsbs)
    ax2.tick_params(axis="x", rotation=45)

    n_match = sum(1 for c, s in zip(calib_lsbs, selected_lsbs) if c == s)
    fig.suptitle(
        f"LSB Choice Distribution — calibrated vs. selected"
        f"  ({n_match}/{len(calib_lsbs)} quantizers agree)",
        fontsize=11,
    )
    plt.tight_layout()

    base = out_dir / "ptq_lsb_histogram"
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
    display_name: str,
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
        f"  Quantizer : {display_name}   [{ts}]",
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
# Sanity check
# ---------------------------------------------------------------------------

def _sanity_check_quantizer(q: BaseQuantizer, lsb: int, bit_width: int) -> str:
    """
    Run the quantizer on a synthetic linspace input spanning the full
    representable range plus a 20% margin on each side, then verify that the
    number of distinct quantized codes and the output range match what the
    (lsb, bit_width, signed) triple predicts.

    Returns a one-line summary string (also printed to console).
    """
    step   = 2.0 ** lsb
    signed = bool(q.search_result_is_signed.item())
    if signed:
        q_min = -(2 ** (bit_width - 1)) * step
        q_max = (2 ** (bit_width - 1) - 1) * step
    else:
        q_min = 0.0
        q_max = (2 ** bit_width - 1) * step
    n_expected = 2 ** bit_width

    span   = q_max - q_min
    margin = 0.20 * span if span > 0 else max(step, 1.0)
    n_points = min(max(n_expected * 4, 1000), 200_000)
    x = torch.linspace(float(q_min - margin), float(q_max + margin), steps=n_points)

    was_training = q.training
    q.eval()
    with torch.no_grad():
        out, _, _, _ = q(x)
    q.train(was_training)

    n_unique = int(torch.unique(out).numel())
    out_min  = out.min().item()
    out_max  = out.max().item()

    uniq_ok  = n_unique == n_expected
    range_ok = (abs(out_min - q_min) <= step * 1.5) and (abs(out_max - q_max) <= step * 1.5)

    msg = (
        f"  Sanity [{bit_width}b {'signed' if signed else 'unsigned'}]  "
        f"step=2^{lsb}={step:.6g}  "
        f"expected_range=[{q_min:.6g}, {q_max:.6g}]  "
        f"unique={n_unique}/{n_expected} {'✓' if uniq_ok else '✗'}  "
        f"actual_range=[{out_min:.6g}, {out_max:.6g}] {'✓' if range_ok else '✗'}"
    )
    print(msg)
    if not (uniq_ok and range_ok):
        print("    [WARNING] Sanity check failed — quantizer output does not "
              "match the expected grid for this LSB/bit-width.")
    return msg


# ---------------------------------------------------------------------------
# Model + checkpoint-chaining setup
# ---------------------------------------------------------------------------

_MODE_TO_ROLE = {"weights": "weight", "activations": "activation", "bias": "bias"}


def _build_quantized_model(
    args: argparse.Namespace, device: torch.device,
) -> tuple[nn.Module, str, int, dict, dict]:
    """
    Build the (possibly combined weight+activation+bias) quantized model for
    this run, optionally continuing from a checkpoint produced by a previous
    run of this script (--init-from-ckpt).

    Returns (model, target_role, bw, prev_extra, prev_role_bit_widths):
      target_role:          "weight", "activation", or "bias" — the role --mode searches
      bw:                   bit-width for target_role this run
      prev_extra:           loaded checkpoint's "extra" dict, or {} if none
      prev_role_bit_widths: {role: bit_width} already calibrated by a prior
                             run, or {} if --init-from-ckpt was not given
    """
    bw = args.bit_width
    target_role = _MODE_TO_ROLE[args.mode]

    prev_payload = None
    prev_extra: dict = {}
    prev_role_bit_widths: dict[str, int] = {}
    if args.init_from_ckpt:
        prev_payload = torch.load(args.init_from_ckpt, map_location="cpu")
        prev_extra = prev_payload.get("extra", {})
        prev_role_bit_widths = dict(prev_extra.get("role_bit_widths", {}))
        if not prev_role_bit_widths:
            # Backward compat with checkpoints that only recorded a single
            # role/bit_width (no chaining support yet when they were saved).
            single_mode = prev_extra.get("ptq_search_mode")
            single_bw   = prev_extra.get("bit_width")
            if single_mode is None or single_bw is None:
                raise ValueError(
                    f"{args.init_from_ckpt} is missing extra.role_bit_widths "
                    f"(or the older extra.ptq_search_mode / extra.bit_width) "
                    f"— was it produced by this script?"
                )
            prev_role_bit_widths = {_MODE_TO_ROLE[single_mode]: single_bw}
        print(f"[init-from-ckpt] {args.init_from_ckpt}  "
              f"(roles already calibrated: {prev_role_bit_widths})")

    # ── Quantizer injector classes ───────────────────────────────────────────
    # Always build the quantizer for --mode's role. Additionally build any
    # OTHER role(s) already calibrated in the loaded checkpoint (at their
    # original bit-width), so that already-calibrated role stays present and
    # active while the new role is searched on top of it.
    build_weights = (args.mode == "weights")     or ("weight" in prev_role_bit_widths)
    build_acts    = (args.mode == "activations") or ("activation" in prev_role_bit_widths)
    build_bias    = (args.mode == "bias")        or ("bias" in prev_role_bit_widths)

    weight_quant = None
    if build_weights:
        w_bw = bw if args.mode == "weights" else prev_role_bit_widths["weight"]
        class _WQ(FixedPointPerTensorWeightQuant):
            bit_width = w_bw
        weight_quant = _WQ

    act_quant = None
    if build_acts:
        a_bw = bw if args.mode == "activations" else prev_role_bit_widths["activation"]
        class _AQ(FixedPointPerTensorActivationQuant):
            bit_width = a_bw
        act_quant = _AQ

    bias_quant = None
    if build_bias:
        b_bw = bw if args.mode == "bias" else prev_role_bit_widths["bias"]
        class _BQ(FixedPointPerTensorBiasQuant):
            bit_width = b_bw
        bias_quant = _BQ

    # ── Build model ──────────────────────────────────────────────────────────
    QuantizerManager().reset()
    model = _build_model(args, weight_quant, act_quant, bias_quant).to(device)
    if args.pretrained:
        model = _load_pretrained(model, args.model)
    if args.fuse_bn:
        n_fused = fuse_bn_into_conv(model)
        print(f"Fused {n_fused} BatchNorm layer(s) into preceding conv/linear weights.")
    if prev_payload is not None:
        incompatible = model.load_state_dict(prev_payload["model_state_dict"], strict=False)
        print(f"[init-from-ckpt] missing keys: {len(incompatible.missing_keys)}"
              f"  unexpected keys: {len(incompatible.unexpected_keys)}")
        if incompatible.unexpected_keys:
            print(f"  {incompatible.unexpected_keys}")
    _assign_descriptive_ids(model)

    return model, target_role, bw, prev_extra, prev_role_bit_widths


def _set_search_states(
    mgr: QuantizerManager, target_role: str, active_roles: set[str]
) -> None:
    """
    Put every registered quantizer into the right state for a per-role LSB
    search of `target_role`:

      - target_role quantizers  -> disabled passthrough (alpha=0). The search
        loop re-enables and calibrates them one at a time.
      - quantizers whose role is in `active_roles` -> fully active (alpha=1,
        no further annealing). These are already calibrated — an earlier role
        in a weights->bias->activations sweep, or a role loaded via
        --init-from-ckpt — so their effect is visible while target_role is
        searched on top of them.
      - every other quantizer   -> disabled passthrough.

    search_done is forced True on the non-active quantizers so an eval-mode
    forward (the baseline eval / the LSB sweeps) treats them as calibrated
    passthrough instead of trying to calibrate mid-eval (which would raise in
    eval mode). The search loop flips search_done back to False on each target
    quantizer right before calibrating it.

    Passing `active_roles` explicitly (rather than inferring "already active"
    from search_done, as an earlier version did) is what makes a multi-role
    search in a single process correct: disabled non-target quantizers get
    search_done=True here, so a later role could no longer be distinguished
    from a genuinely-calibrated one by search_done alone.
    """
    for q in mgr.quantizers.values():
        if q.quantizer_role == target_role:
            q.annealing_alpha.data.fill_(0.0)
            q.search_done.fill_(True)
        elif q.quantizer_role in active_roles:
            q.annealing_alpha.data.fill_(1.0)
            q.annealing_alpha_step = 0.0
        else:
            q.annealing_alpha.data.fill_(0.0)
            q.search_done.fill_(True)


def _disable_target_role_keep_others_active(mgr: QuantizerManager, target_role: str) -> None:
    """Backward-compatible shim over _set_search_states that infers the roles to
    keep active from search_done (any non-target quantizer already calibrated)
    rather than taking them explicitly.

    Kept for the single-role --init-from-ckpt chaining path (and its tests):
    there, the roles carried in from the checkpoint are exactly the ones with
    search_done=True, so inferring is equivalent to passing them. The multi-role
    in-process search instead calls _set_search_states with an explicit
    active_roles set (see the module docstring on _set_search_states for why
    that distinction matters).
    """
    active_roles = {
        q.quantizer_role
        for q in mgr.quantizers.values()
        if q.quantizer_role != target_role and q.search_done.item()
    }
    _set_search_states(mgr, target_role, active_roles)


def search_role_lsbs(
    *,
    model: nn.Module,
    target_role: str,
    bit_width: int,
    val_loader,
    loss_fn: nn.Module,
    device: torch.device,
    search_radius: int,
    eval_batches: Optional[int],
    out_dir: Path,
    log_path: Path,
    calib_images: torch.Tensor,
    calib_labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    active_roles: Optional[set] = None,
) -> dict:
    """
    Greedy per-quantizer LSB search for every quantizer of `target_role`, in
    forward-pass order, on an already-built model whose quantizers are
    registered with the singleton QuantizerManager and have already been given
    descriptive ids (_assign_descriptive_ids).

    A forward pass must have run over the model before this is called so that
    inference_sequence_id — hence forward-execution order — is defined; the
    caller's baseline evaluation satisfies that.

    active_roles: roles already calibrated that stay fully active throughout
    (earlier roles in a weights->bias->activations sweep, or roles loaded via
    --init-from-ckpt). Defaults to empty.

    Writes per-quantizer plots + log entries under out_dir. Returns a summary
    dict: {qids, display_names, calib_lsbs, selected_lsbs, accs, losses}.
    """
    mgr = QuantizerManager()
    mgr.quantization_start_gap = 2   # each quantizer at position N gates for N*2
                                     # steps before activating — cleared naturally
                                     # by prior calibration loops
    mgr.diagnostics_dir = str(out_dir)
    active_roles = set(active_roles or ())

    _set_search_states(mgr, target_role, active_roles)

    # ── Identify quantizers to optimise (in forward-pass order) ─────────────
    # Search upstream -> downstream so each quantizer's optimum reflects the
    # FINAL settings of everything feeding into it. quantizers_in_execution_order
    # also drops Brevitas's internal "ghost" quantizer objects (registered but
    # never reached by forward()).
    ordered_ids = {
        q.quant_id: i for i, q in enumerate(mgr.quantizers_in_execution_order())
    }
    target_items: list[tuple[str, BaseQuantizer]] = [
        (qid, q) for qid, q in mgr.quantizers.items()
        if q.quantizer_role == target_role and q.quant_id in ordered_ids
    ]
    if not target_items:
        raise ValueError(
            f"No reachable {target_role} quantizers found in this model "
            "(after dropping objects never reached by the forward pass). "
            "Check model construction arguments."
        )
    target_items.sort(key=lambda item: ordered_ids[item[1].quant_id])
    n_total = len(target_items)
    print(f"  {target_role} quantizers in execution path: {n_total}\n")

    # ── Per-quantizer search ──────────────────────────────────────────────────
    summary_qids:          List[str]   = []
    summary_display_names: List[str]   = []
    summary_calib_lsbs:    List[int]   = []
    summary_lsbs:          List[int]   = []
    summary_accs:          List[float] = []
    summary_losses:        List[float] = []

    sep = "─" * 68

    for qi, (qid, q) in enumerate(target_items, start=1):
        t0 = time.time()
        print(sep)
        print(f"[{qi}/{n_total}]  {q.display_name}  ({target_role}, {bit_width}b)")
        print(sep)

        # ── Calibrate ─────────────────────────────────────────────────────────
        # Start as passthrough (alpha=0); annealing_alpha_step=1 makes the
        # quantizer jump to fully-quantized (alpha=1) in one active training
        # step, effectively disabling gradual annealing.
        q.search_done.fill_(False)
        q.annealing_alpha.data.fill_(0.0)
        q.annealing_alpha_step = 1.0

        # Advance inference_counter to exactly the gap threshold so calibration
        # fires on the very first training step.
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
        candidates = list(range(calib_lsb - search_radius,
                                calib_lsb + search_radius + 1))
        results: List[Tuple[int, float, float]] = []

        n_candidates = len(candidates)
        for ci, candidate_lsb in enumerate(candidates, start=1):
            q.search_result_lsb.fill_(candidate_lsb)
            q.search_done.fill_(True)

            tag = " (calibrated)" if candidate_lsb == calib_lsb else ""
            print(f"  [{ci}/{n_candidates}] evaluating LSB={candidate_lsb}{tag} …")
            v_loss, v_acc = _evaluate(model, val_loader, loss_fn, device, eval_batches,
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

        # ── Sanity check ──────────────────────────────────────────────────────
        _sanity_check_quantizer(q, best_lsb, bit_width)

        # ── Log + plot ─────────────────────────────────────────────────────────
        _log_quantizer_result(
            log_path,
            quant_id=qid, display_name=q.display_name,
            quantizer_role=target_role, bit_width=bit_width,
            calib_lsb=calib_lsb, selected_lsb=best_lsb, results=results,
        )
        _save_ptq_search_plot(
            results=results,
            calib_lsb=calib_lsb,
            selected_lsb=best_lsb,
            quant_id=qid,
            display_name=q.display_name,
            quantizer_role=target_role,
            bit_width=bit_width,
            quantizer_index=qi,
            n_quantizers=n_total,
            out_dir=out_dir,
        )
        plot_path = (out_dir / f"ptq_{qid.replace('/', '_')}.png").resolve()
        print(f"  Plot: {plot_path}")

        summary_qids.append(qid)
        summary_display_names.append(q.display_name)
        summary_calib_lsbs.append(calib_lsb)
        summary_lsbs.append(best_lsb)
        summary_accs.append(best_r[2])
        summary_losses.append(best_r[1])

    return {
        "qids": summary_qids,
        "display_names": summary_display_names,
        "calib_lsbs": summary_calib_lsbs,
        "selected_lsbs": summary_lsbs,
        "accs": summary_accs,
        "losses": summary_losses,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, target_role, bw, prev_extra, prev_role_bit_widths = _build_quantized_model(args, device)

    # Roles already calibrated via --init-from-ckpt stay fully active while the
    # target role is searched on top (the target role itself is (re)searched).
    active_roles = {r for r in prev_role_bit_widths if r != target_role}

    mgr = QuantizerManager()
    mgr.quantization_start_gap = 2   # each quantizer at position N gates for N*2 steps before
                                     # activating — cleared naturally by prior calibration loops

    # Rough pre-filter count for the header/log (ghost objects are dropped once
    # the search runs and forward-execution order is known).
    n_target = sum(1 for q in mgr.quantizers.values() if q.quantizer_role == target_role)
    if n_target == 0:
        print(f"[ERROR] No {target_role} quantizers found in this model. "
              "Check model construction arguments.")
        return

    # ── Output directory & log ───────────────────────────────────────────────
    exp_name = (args.experiment_name
                or f"{args.model}_{args.mode}_{bw}b_r{args.search_radius}")
    out_dir  = Path(args.output_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory : {out_dir.resolve()}")
    log_path = out_dir / "ptq_search.log"

    with open(log_path, "w") as fh:
        fh.write(
            f"PTQ LSB Search Log\n"
            f"{'='*68}\n"
            f"  Model          : {args.model}\n"
            f"  Mode           : {args.mode}  ({bw}b)\n"
            f"  Init from ckpt : {args.init_from_ckpt or 'none'}"
            f"{f' (roles already calibrated: {prev_role_bit_widths})' if prev_role_bit_widths else ''}\n"
            f"  Pretrained     : {args.pretrained}\n"
            f"  Search radius  : ±{args.search_radius}\n"
            f"  Eval batches   : {args.eval_batches or 'full'}\n"
            f"  Quant gap      : {mgr.quantization_start_gap}\n"
            f"  N quantizers   : {n_target}\n"
            f"  Device         : {device}\n"
            f"  Started        : {datetime.now()}\n"
            f"{'='*68}\n"
        )

    print(f"\n{'═'*68}")
    print(f"  PTQ LSB Search — {exp_name}")
    print(f"  Model: {args.model}  |  Mode: {args.mode}  |  {bw}b")
    print(f"  Radius: ±{args.search_radius}  |  Gap: {mgr.quantization_start_gap}"
          f"  |  Eval: {args.eval_batches or 'full'} batches")
    print(f"  Quantizers to search: {n_target}")
    print(f"{'═'*68}\n")

    # ── Data & optimiser ─────────────────────────────────────────────────────
    print("Loading data …")
    train_loader, val_loader = _build_dataloaders(args)
    loss_fn = nn.CrossEntropyLoss()

    # Near-zero LR: weights barely move, but the training-mode forward+backward
    # pass is what allows quantizer calibration to fire (base_quantizer guards
    # on self.training before running _calibrate).
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)

    # ── Baseline evaluation ───────────────────────────────────────────────────
    # Put quantizers in search state first: the target role and every
    # not-yet-calibrated role are passthrough, while roles carried in from
    # --init-from-ckpt stay active. This is also the first forward pass over the
    # model, so it establishes each quantizer's inference_sequence_id (hence the
    # forward-execution order the search relies on).
    _set_search_states(mgr, target_role, active_roles)
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
    summary = search_role_lsbs(
        model=model,
        target_role=target_role,
        bit_width=bw,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        search_radius=args.search_radius,
        eval_batches=args.eval_batches,
        out_dir=out_dir,
        log_path=log_path,
        calib_images=calib_images,
        calib_labels=calib_labels,
        optimizer=optimizer,
        active_roles=active_roles,
    )
    summary_qids          = summary["qids"]
    summary_display_names = summary["display_names"]
    summary_calib_lsbs    = summary["calib_lsbs"]
    summary_lsbs          = summary["selected_lsbs"]
    summary_accs          = summary["accs"]
    summary_losses        = summary["losses"]

    # ── Final evaluation ──────────────────────────────────────────────────────
    sep = "─" * 68
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
            display_names=summary_display_names,
            selected_lsbs=summary_lsbs,
            val_losses=summary_losses,
            val_accs=summary_accs,
            baseline_loss=baseline_loss,
            baseline_acc=baseline_acc,
            out_dir=out_dir,
        )
        _save_lsb_histogram_plot(
            calib_lsbs=summary_calib_lsbs,
            selected_lsbs=summary_lsbs,
            out_dir=out_dir,
        )

    # ── Save calibrated model checkpoint ─────────────────────────────────────
    # Matches the {epoch, model_state_dict, optimizer_state_dict, metrics,
    # config, extra} shape used by training_harness.checkpointing._build_payload
    # so it can be loaded with CheckpointManager.resume(..., reset_calibration=False)
    # to start QAT from these PTQ-found LSBs (reset_calibration=True, the
    # harness default, would wipe search_done and force re-calibration).
    ckpt_path = out_dir / "ptq_calibrated_model.pt"
    ckpt_payload = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": {
            "baseline_val_loss": baseline_loss,
            "baseline_val_acc": baseline_acc,
            "final_val_loss": final_loss,
            "final_val_acc": final_acc,
        },
        "config": vars(args),
        "extra": {
            "ptq_search_mode": args.mode,
            "bit_width": bw,
            "role_bit_widths": {**prev_role_bit_widths, target_role: bw},
            "fuse_bn": args.fuse_bn,
            "calibrated_lsbs": {
                **prev_extra.get("calibrated_lsbs", {}),
                **dict(zip(summary_qids, summary_calib_lsbs)),
            },
            "selected_lsbs": {
                **prev_extra.get("selected_lsbs", {}),
                **dict(zip(summary_qids, summary_lsbs)),
            },
        },
    }
    torch.save(ckpt_payload, ckpt_path)
    print(f"\nSaved PTQ-calibrated model checkpoint: {ckpt_path.resolve()}")
    print(f"  Load for QAT with reset_calibration=False to keep these LSBs, e.g.:")
    print(f"    CheckpointManager(...).resume(model, path={str(ckpt_path)!r}, reset_calibration=False)")

    print(f"\nAll results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
