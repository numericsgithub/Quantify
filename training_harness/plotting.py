"""
plotting.py — Training visualisation for the training_harness harness.

Produces publication-ready plots from a MetricsTracker:
  - Loss curves (train + val)
  - Accuracy curves
  - Learning rate schedule
  - Quantization scale factor evolution

All plots are saved as PNG files. Matplotlib is the only dependency.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Default colour palette (colourblind-friendly)
_COLORS = {
    "train":  "#2196F3",   # blue
    "val":    "#FF5722",   # orange-red
    "accent": "#4CAF50",   # green
    "muted":  "#9E9E9E",   # grey
}

_FIGSIZE_SINGLE = (8, 4)
_FIGSIZE_WIDE   = (14, 4)
_FIGSIZE_GRID   = (14, 8)
_DPI = 150


class TrainingPlotter:
    """
    Generates training_harness plots from a MetricsTracker.

    Usage::

        plotter = TrainingPlotter(save_dir="plots", experiment_name="my_exp")
        plotter.plot_all(tracker)       # save everything at once

        # Or individually:
        plotter.plot_loss(tracker)
        plotter.plot_accuracy(tracker)
        plotter.plot_scale_factors(tracker)
    """

    def __init__(
        self,
        save_dir: str = "plots",
        experiment_name: str = "experiment",
        show: bool = False,
    ):
        """
        Args:
            save_dir:          Directory to save plot PNG files.
            experiment_name:   Prefix for saved filenames.
            show:              If True, also call plt.show() (for notebooks).
        """
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.show = show
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Convenience: plot everything
    # ------------------------------------------------------------------

    def plot_all(self, tracker, lr_history: Optional[List[float]] = None) -> List[str]:
        """
        Generate all available plots and return a list of saved file paths.

        Args:
            tracker:     A MetricsTracker instance.
            lr_history:  Optional list of LR values per step.
        """
        paths = []

        paths.append(self.plot_loss(tracker))
        paths.append(self.plot_accuracy(tracker))
        paths.append(self.plot_overview(tracker, lr_history=lr_history))

        if tracker.scale_history:
            paths.append(self.plot_scale_factors(tracker))

        return [p for p in paths if p is not None]

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_loss(self, tracker) -> Optional[str]:
        """Plot train and validation loss curves."""
        train_ep, train_loss = tracker.get_metric_series("train_loss")
        val_ep,   val_loss   = tracker.get_metric_series("val_loss")

        if not train_loss and not val_loss:
            return None

        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
        _style_axes(ax)

        if train_loss:
            ax.plot(train_ep, train_loss, color=_COLORS["train"], label="Train", linewidth=1.8)
        if val_loss:
            ax.plot(val_ep, val_loss, color=_COLORS["val"], label="Val", linewidth=1.8)
            # Mark the best val loss
            best_idx = val_loss.index(min(val_loss))
            ax.axvline(val_ep[best_idx], color=_COLORS["val"], linestyle="--", alpha=0.4, linewidth=1)
            ax.scatter(
                [val_ep[best_idx]], [val_loss[best_idx]],
                color=_COLORS["val"], s=60, zorder=5,
                label=f"Best val: {val_loss[best_idx]:.4f}"
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{self.experiment_name} — Loss")
        ax.legend(framealpha=0.8)

        return self._save(fig, "loss")

    def plot_accuracy(self, tracker) -> Optional[str]:
        """Plot train and validation accuracy curves (if available)."""
        train_ep, train_acc = tracker.get_metric_series("train_acc")
        val_ep,   val_acc   = tracker.get_metric_series("val_acc")

        if not train_acc and not val_acc:
            return None

        fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
        _style_axes(ax)

        if train_acc:
            ax.plot(train_ep, train_acc, color=_COLORS["train"], label="Train", linewidth=1.8)
        if val_acc:
            ax.plot(val_ep, val_acc, color=_COLORS["val"], label="Val", linewidth=1.8)
            best_idx = val_acc.index(max(val_acc))
            ax.scatter(
                [val_ep[best_idx]], [val_acc[best_idx]],
                color=_COLORS["val"], s=60, zorder=5,
                label=f"Best val: {val_acc[best_idx]:.4f}"
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{self.experiment_name} — Accuracy")
        ax.legend(framealpha=0.8)

        return self._save(fig, "accuracy")

    def plot_scale_factors(self, tracker) -> Optional[str]:
        """
        Plot the evolution of quantization scale factors over epochs.

        If there are many layers (>12), splits them into subplots
        so the chart stays readable.
        """
        if not tracker.scale_history:
            return None

        # Reshape: layer_name → list of (epoch, scale) pairs
        layer_series: Dict[str, Dict[int, float]] = {}
        for epoch, scales in tracker.scale_history.items():
            for layer, scale in scales.items():
                layer_series.setdefault(layer, {})[epoch] = scale

        n_layers = len(layer_series)
        if n_layers == 0:
            return None

        # Use a grid layout for many layers
        n_cols = min(3, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        fig_h  = max(3, 3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, fig_h))
        if n_layers == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]

        cmap = plt.get_cmap("tab20")
        flat_axes = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

        for idx, (layer_name, ep_scale) in enumerate(sorted(layer_series.items())):
            ax = flat_axes[idx]
            _style_axes(ax)
            epochs = sorted(ep_scale.keys())
            values = [ep_scale[e] for e in epochs]
            ax.plot(epochs, values, color=cmap(idx % 20), linewidth=1.8)
            ax.set_title(layer_name, fontsize=7, pad=3)
            ax.set_xlabel("Epoch", fontsize=7)
            ax.set_ylabel("Scale", fontsize=7)
            ax.tick_params(labelsize=6)

        # Hide unused axes
        for idx in range(n_layers, len(flat_axes)):
            flat_axes[idx].set_visible(False)

        fig.suptitle(f"{self.experiment_name} — Quantization Scale Factors", y=1.01)
        fig.tight_layout()

        return self._save(fig, "scale_factors")

    def plot_overview(
        self,
        tracker,
        lr_history: Optional[List[float]] = None,
    ) -> Optional[str]:
        """
        Four-panel overview: loss, accuracy, LR schedule, and a run summary.
        """
        has_acc = bool(tracker.get_metric_series("val_acc")[0])
        has_lr  = bool(lr_history)

        n_panels = 2 + int(has_acc) + int(has_lr)
        fig = plt.figure(figsize=(6 * n_panels, 4))
        gs  = gridspec.GridSpec(1, n_panels, figure=fig, wspace=0.35)

        panel = 0

        # --- Loss panel ---
        ax_loss = fig.add_subplot(gs[panel]); panel += 1
        _style_axes(ax_loss)
        train_ep, train_loss = tracker.get_metric_series("train_loss")
        val_ep,   val_loss   = tracker.get_metric_series("val_loss")
        if train_loss:
            ax_loss.plot(train_ep, train_loss, color=_COLORS["train"], label="Train", lw=1.8)
        if val_loss:
            ax_loss.plot(val_ep, val_loss, color=_COLORS["val"], label="Val", lw=1.8)
        ax_loss.set_title("Loss"); ax_loss.set_xlabel("Epoch"); ax_loss.legend(fontsize=8)

        # --- Accuracy panel ---
        if has_acc:
            ax_acc = fig.add_subplot(gs[panel]); panel += 1
            _style_axes(ax_acc)
            train_ea, train_ac = tracker.get_metric_series("train_acc")
            val_ea,   val_ac   = tracker.get_metric_series("val_acc")
            if train_ac:
                ax_acc.plot(train_ea, train_ac, color=_COLORS["train"], label="Train", lw=1.8)
            if val_ac:
                ax_acc.plot(val_ea, val_ac, color=_COLORS["val"], label="Val", lw=1.8)
            ax_acc.set_title("Accuracy"); ax_acc.set_xlabel("Epoch"); ax_acc.legend(fontsize=8)

        # --- LR panel ---
        if has_lr:
            ax_lr = fig.add_subplot(gs[panel]); panel += 1
            _style_axes(ax_lr)
            ax_lr.plot(lr_history, color=_COLORS["accent"], lw=1.8)
            ax_lr.set_title("Learning Rate"); ax_lr.set_xlabel("Step")
            ax_lr.set_yscale("log")

        # --- Summary panel (text) ---
        ax_sum = fig.add_subplot(gs[panel])
        ax_sum.axis("off")
        summary = tracker.summary()
        lines = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                 for k, v in summary.items()]
        text = "\n".join(lines)
        ax_sum.text(
            0.05, 0.95, text,
            transform=ax_sum.transAxes,
            va="top", ha="left",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#BDBDBD"),
        )
        ax_sum.set_title("Run Summary")

        fig.suptitle(f"{self.experiment_name}", fontsize=12, y=1.01)

        return self._save(fig, "overview")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, fig, suffix: str) -> str:
        fname = f"{self.experiment_name}_{suffix}.png"
        path  = os.path.join(self.save_dir, fname)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Saved → {path}")
        return path


def _style_axes(ax) -> None:
    """Apply a clean, minimal style to a matplotlib axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.spines["bottom"].set_color("#BDBDBD")
    ax.tick_params(colors="#555555")
    ax.grid(True, linestyle="--", alpha=0.4, color="#BDBDBD")
    ax.set_facecolor("#FAFAFA")
