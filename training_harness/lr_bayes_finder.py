"""
lr_bayes_finder.py — Bayesian-optimization learning-rate finder.

``find_learning_rate`` searches for the learning rate that most reduces the
*fresh post-epoch training loss* when fine-tuning a (quantization-aware) model
for exactly one epoch from a fixed starting checkpoint. The search is Bayesian
optimization over ``log10(lr)`` with a Gaussian-Process surrogate
(scikit-optimize) — not a grid sweep or binary search.

Why one epoch from a fixed checkpoint: every trial answers the *identical*
question "one epoch of this constant LR from this exact starting state", so the
trials are directly comparable. Weights are reloaded fresh from the checkpoint
for every trial; trials never resume from each other.

Why the objective is the *fresh* training loss: the running loss/accuracy
accumulated during an epoch is an average over a model that changed throughout
the epoch and describes no single set of weights. Instead each trial's four
metrics (train loss/acc, val loss/acc) are measured in eval mode (BN/dropout
off) over a fixed subset (train) and the val loader (val) — of the *final*
weights, so they describe one concrete model and are comparable across trials.

Caveat (important). This optimizes "best LR after one epoch from the starting
checkpoint", which is a *proxy* for "best LR for a full fine-tune". A slightly
lower, more stable LR sometimes loses the one-epoch comparison but wins over a
long schedule. Treat the returned value as the *center of a good region* to
probe further when the real training run is many epochs long — see
``LRFinderResult.gp_predicted_optimum_lr`` for the neighborhood, not just the
single best point.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from skopt import Optimizer as SkoptOptimizer
from skopt.space import Real


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class LRTrial:
    """One trial: one epoch of a constant LR from the starting checkpoint."""
    trial_id: int
    lr: float
    log10_lr: float
    # fresh post-epoch metrics of the final weights (eval mode)
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    # deltas vs the baseline (starting checkpoint)
    d_train_loss: float
    d_train_acc: float
    d_val_loss: float
    d_val_acc: float
    catastrophic: bool
    objective: float                # what the GP minimises (train_loss, NaN→fallback)
    checkpoint_path: Optional[str]
    seed: int

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        # Exact, round-trip-safe representations (folder names are for humans only)
        d["lr_hex"] = float(self.lr).hex()
        d["log10_lr_hex"] = float(self.log10_lr).hex()
        return d


@dataclass
class LRFinderResult:
    recommended_lr: float
    baseline: Dict[str, float]              # train_loss/acc, val_loss/acc of the start checkpoint
    fallback_loss: float
    trials: List[LRTrial] = field(default_factory=list)
    overfit_flagged: bool = False           # best-train-loss LR worsened val loss → fell back
    gp_predicted_optimum_lr: Optional[float] = None
    gp_predicted_optimum_value: Optional[float] = None
    lr_min: float = 1e-8
    lr_max: float = 1e-3


# ---------------------------------------------------------------------------
# Small eval / train helpers
# ---------------------------------------------------------------------------

def _resolve_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _default_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    """Top-1: returns (n_correct, n_total)."""
    preds = outputs.argmax(dim=1)
    return int(preds.eq(targets).sum().item()), int(targets.numel())


@torch.no_grad()
def _eval_loss_acc(
    model: nn.Module,
    batches,
    loss_fn: nn.Module,
    device: torch.device,
    accuracy_fn: Callable,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Eval-mode (BN/dropout off) loss + accuracy over `batches`."""
    was_training = model.training
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for i, (x, y) in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        out = model(x)
        loss = loss_fn(out, y)
        n = int(y.numel())
        total_loss += float(loss.item()) * n
        c, t = accuracy_fn(out, y)
        correct += c
        total += t
    model.train(was_training)
    if total == 0:
        return float("nan"), 0.0
    return total_loss / total, correct / total


def _train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip_norm: Optional[float],
) -> bool:
    """Train exactly one epoch at the optimizer's (constant) LR.

    Returns True if training diverged (NaN/Inf loss) — in which case the epoch
    is aborted early since the weights are already ruined.
    """
    model.train()
    diverged = False
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        if not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
    return diverged


def _materialize_subset(train_loader: DataLoader, n_samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Pull a FIXED subset (kept on CPU, in a fixed batch order) from the train
    loader — reused verbatim on every trial so the fresh train-metric numbers
    are comparable across trials."""
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    seen = 0
    for x, y in train_loader:
        batches.append((x.detach().cpu().clone(), y.detach().cpu().clone()))
        seen += int(y.shape[0])
        if seen >= n_samples:
            break
    if not batches:
        raise ValueError("train_loader yielded no data for the fixed eval subset")
    return batches


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_learning_rate(
    model: nn.Module,
    starting_checkpoint_path: str,
    output_dir: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_factory: Callable[[float], torch.optim.Optimizer],
    *,
    loss_fn: Optional[nn.Module] = None,
    accuracy_fn: Optional[Callable] = None,
    lr_min: float = 1e-8,
    lr_max: float = 1e-3,
    n_initial_points: int = 4,
    n_calls: int = 16,
    train_eval_subset_size: int = 4096,
    val_eval_max_batches: Optional[int] = None,
    device: Any = "auto",
    seed: int = 42,
    grad_clip_norm: Optional[float] = 1.0,
    prepare_model_fn: Optional[Callable[[nn.Module], None]] = None,
    lr_tol_log: float = 1e-6,
    sanity_rel_tol: float = 0.5,
) -> LRFinderResult:
    """Bayesian-optimization learning-rate finder (GP surrogate over log10(lr)).

    Parameters
    ----------
    model : nn.Module
        Already-constructed, quantization-aware model with fixed (calibrated)
        quantizers. Its weights are overwritten from ``starting_checkpoint_path``
        at the start of every trial.
    starting_checkpoint_path : str
        Checkpoint every trial starts from. May be a raw ``state_dict`` or a
        payload dict containing ``"model_state_dict"``.
    output_dir : str
        Directory to create and write all trial outputs into. Existing trials
        found here are loaded and fed to the optimizer as prior observations
        (crash-resume) instead of being recomputed.
    train_loader, val_loader : DataLoader
        Training loader (one full epoch is trained per trial) and validation
        loader (used for the val-metric evaluation and the overfitting guard).
    optimizer_factory : Callable[[float], Optimizer]
        Given a learning rate, returns a *fresh* optimizer bound to the model's
        parameters. A factory (not an instance) is required because each trial
        must build its optimizer at the specific LR being tested.
    loss_fn : nn.Module, optional
        Defaults to ``nn.CrossEntropyLoss()``.
    accuracy_fn : Callable, optional
        ``(outputs, targets) -> (n_correct, n_total)``; defaults to top-1.
    lr_min, lr_max : float
        Search bounds (default 1e-8, 1e-3). The search dimension is
        ``[log10(lr_min), log10(lr_max)]``; LR is exponentiated inside each trial
        so sampling and the GP length scale stay uniform across orders of
        magnitude.
    n_initial_points : int
        Quasi-random seed points before the GP takes over (default 4).
    n_calls : int
        Total trials, including seeds and any resumed/cached trials (default 16).
    train_eval_subset_size : int
        Number of training samples for the fixed post-epoch train evaluation.
    prepare_model_fn : Callable[[model], None], optional
        Called after every fresh checkpoint load. Use it to (re)activate the
        quantizers so the reloaded model actually quantizes (e.g. bypass the
        staggered-activation gate for calibrated quantizers). Without it, a
        freshly-loaded model may run as float passthrough on its first forward.
    lr_tol_log : float
        Two LRs within this distance in log10 space are considered the same
        point; a re-proposal of an already-evaluated point is treated as a
        deliberate noise-reduction repeat and re-run with a *different* seed
        (the GP's observation-noise term averages the repeats — no manual
        repeat-and-average path).

    Returns
    -------
    LRFinderResult
        ``recommended_lr`` (min fresh train loss subject to the validation
        guardrail), the baseline metrics, every trial's LR + four metric deltas
        + catastrophic flag + checkpoint path, and the fitted GP's predicted
        optimum LR/value.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if accuracy_fn is None:
        accuracy_fn = _default_accuracy
    dev = _resolve_device(device)
    model.to(dev)
    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(output_dir, "trials.json")

    lo, hi = math.log10(lr_min), math.log10(lr_max)
    if not (lo < hi):
        raise ValueError(f"require lr_min < lr_max, got {lr_min} >= {lr_max}")

    # --- Checkpoint loading -------------------------------------------------
    _payload = torch.load(starting_checkpoint_path, map_location="cpu")
    _start_state = _payload.get("model_state_dict", _payload) if isinstance(_payload, dict) else _payload

    def _load_start_weights() -> None:
        model.load_state_dict(_start_state, strict=False)
        model.to(dev)
        if prepare_model_fn is not None:
            prepare_model_fn(model)

    # --- Fixed train-eval subset -------------------------------------------
    train_eval_batches = _materialize_subset(train_loader, train_eval_subset_size)

    def _eval_four() -> Dict[str, float]:
        tl, ta = _eval_loss_acc(model, train_eval_batches, loss_fn, dev, accuracy_fn)
        vl, va = _eval_loss_acc(model, val_loader, loss_fn, dev, accuracy_fn,
                                max_batches=val_eval_max_batches)
        return {"train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va}

    # --- Baseline (untouched starting checkpoint) --------------------------
    _seed_everything(seed)
    _load_start_weights()
    baseline = _eval_four()
    print(f"[lr-bo] baseline  train_loss={baseline['train_loss']:.4f} "
          f"train_acc={baseline['train_acc']:.4f} "
          f"val_loss={baseline['val_loss']:.4f} val_acc={baseline['val_acc']:.4f}")
    if not math.isfinite(baseline["train_loss"]):
        raise RuntimeError("baseline training loss is not finite — the data/loss/"
                           "precision or quantizer setup is broken, not the LR.")

    # --- Fallback loss (perturbed model) for NaN/Inf replacement -----------
    _seed_everything(seed + 7)
    _load_start_weights()
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.add_(torch.randn_like(p) * (p.detach().float().std() + 1e-8) * 3.0)
    if prepare_model_fn is not None:
        prepare_model_fn(model)
    fallback_loss, _ = _eval_loss_acc(model, train_eval_batches, loss_fn, dev, accuracy_fn)
    if not math.isfinite(fallback_loss):
        # Perturbation itself diverged; use a comfortably-bad finite constant.
        fallback_loss = max(baseline["train_loss"] * 10.0, 100.0)
    _load_start_weights()  # restore clean state
    print(f"[lr-bo] fallback loss (perturbed model) = {fallback_loss:.4f}")

    result = LRFinderResult(
        recommended_lr=float("nan"), baseline=baseline, fallback_loss=fallback_loss,
        lr_min=lr_min, lr_max=lr_max,
    )

    # --- Optimizer + resume from cached trials -----------------------------
    opt = SkoptOptimizer(
        dimensions=[Real(lo, hi, name="log10_lr")],
        base_estimator="GP", n_initial_points=n_initial_points,
        acq_func="EI", random_state=seed,
    )
    cached = _load_cached_trials(map_path)
    for t in cached:
        result.trials.append(t)
        opt.tell([t.log10_lr], t.objective)
    if cached:
        print(f"[lr-bo] resumed {len(cached)} cached trial(s) from {map_path}")
    next_id = (max((t.trial_id for t in result.trials), default=-1) + 1)

    # --- One trial ----------------------------------------------------------
    def _run_trial(trial_id: int, log10_lr: float, trial_seed: int) -> LRTrial:
        lr = float(10.0 ** log10_lr)
        sub = os.path.join(output_dir, f"{trial_id:04d}_lr_{lr:.2e}")
        os.makedirs(sub, exist_ok=True)
        print(f"[lr-bo] trial {trial_id}  lr={lr:.3e} (log10={log10_lr:.4f})  seed={trial_seed}")

        _seed_everything(trial_seed)
        _load_start_weights()
        optimizer = optimizer_factory(lr)
        diverged = _train_one_epoch(model, optimizer, train_loader, loss_fn, dev, grad_clip_norm)
        metrics = _eval_four()

        catastrophic = diverged or not math.isfinite(metrics["train_loss"])
        if catastrophic:
            objective = fallback_loss
            metrics = {"train_loss": fallback_loss, "train_acc": 0.0,
                       "val_loss": (metrics["val_loss"] if math.isfinite(metrics["val_loss"]) else fallback_loss),
                       "val_acc": 0.0}
            print(f"[lr-bo]   CATASTROPHIC (diverged) — objective set to fallback {fallback_loss:.4f}")
        else:
            objective = metrics["train_loss"]

        ckpt_path = os.path.join(sub, "model.pt")
        torch.save({"model_state_dict": model.state_dict(), "lr": lr, "log10_lr": log10_lr,
                    "metrics": metrics, "catastrophic": catastrophic}, ckpt_path)

        trial = LRTrial(
            trial_id=trial_id, lr=lr, log10_lr=log10_lr,
            train_loss=metrics["train_loss"], train_acc=metrics["train_acc"],
            val_loss=metrics["val_loss"], val_acc=metrics["val_acc"],
            d_train_loss=metrics["train_loss"] - baseline["train_loss"],
            d_train_acc=metrics["train_acc"] - baseline["train_acc"],
            d_val_loss=metrics["val_loss"] - baseline["val_loss"],
            d_val_acc=metrics["val_acc"] - baseline["val_acc"],
            catastrophic=catastrophic, objective=objective,
            checkpoint_path=ckpt_path, seed=trial_seed,
        )
        with open(os.path.join(sub, "metrics.json"), "w") as fh:
            json.dump(trial.to_json(), fh, indent=2)
        print(f"[lr-bo]   Δtrain_loss={trial.d_train_loss:+.4f}  Δtrain_acc={trial.d_train_acc:+.4f}"
              f"  Δval_loss={trial.d_val_loss:+.4f}  Δval_acc={trial.d_val_acc:+.4f}")
        return trial

    def _within_tol(a: float, b: float) -> bool:
        return abs(a - b) <= lr_tol_log

    # --- Lowest-LR sanity check: force lr_min early ------------------------
    if not any(_within_tol(t.log10_lr, lo) for t in result.trials):
        t_min = _run_trial(next_id, lo, seed); next_id += 1
        result.trials.append(t_min)
        opt.tell([t_min.log10_lr], t_min.objective)
        _persist_trials(map_path, result)
    t_min = next(t for t in result.trials if _within_tol(t.log10_lr, lo))
    _sanity_check_lr_min(t_min, baseline, sanity_rel_tol)

    # --- Bayesian-optimization loop ----------------------------------------
    n_repeats = 0
    while len(result.trials) < n_calls:
        x = opt.ask()
        log10_lr = float(x[0])
        is_repeat = any(_within_tol(t.log10_lr, log10_lr) for t in result.trials)
        # A re-proposal of an already-sampled point is a deliberate
        # noise-reduction repeat: re-run with a *different* seed for an
        # independent observation (skopt's GP noise term averages them).
        trial_seed = seed + (10_000 + n_repeats) if is_repeat else seed + len(result.trials)
        if is_repeat:
            n_repeats += 1
        trial = _run_trial(next_id, log10_lr, trial_seed); next_id += 1
        result.trials.append(trial)
        opt.tell(x, trial.objective)
        _persist_trials(map_path, result)

    # --- GP predicted optimum (neighborhood, not just the best point) ------
    try:
        res = opt.get_result()
        gp = res.models[-1] if getattr(res, "models", None) else None
        if gp is not None:
            grid = np.linspace(lo, hi, 400).reshape(-1, 1)
            Xt = res.space.transform(grid.tolist())
            mu = np.asarray(gp.predict(Xt)).reshape(-1)
            j = int(np.argmin(mu))
            result.gp_predicted_optimum_lr = float(10.0 ** grid[j, 0])
            result.gp_predicted_optimum_value = float(mu[j])
    except Exception as exc:  # GP prediction is a nicety, never fatal
        print(f"[lr-bo] GP optimum prediction skipped: {exc}")

    # --- Selection with the validation guardrail ---------------------------
    result.recommended_lr, result.overfit_flagged = _select_lr(result, baseline)
    _persist_result(output_dir, result)

    print(f"\n[lr-bo] recommended lr = {result.recommended_lr:.3e}"
          f"{'  (val-guardrail fallback: best-train-loss LR overfit)' if result.overfit_flagged else ''}")
    if result.gp_predicted_optimum_lr is not None:
        print(f"[lr-bo] GP predicted optimum ≈ {result.gp_predicted_optimum_lr:.3e} "
              f"(predicted train_loss {result.gp_predicted_optimum_value:.4f})")
    print("[lr-bo] NOTE: this is the best LR after ONE epoch — a proxy for a full "
          "fine-tune. Treat it as the center of a good region for long runs.")
    return result


# ---------------------------------------------------------------------------
# Selection, sanity, persistence
# ---------------------------------------------------------------------------

def _select_lr(result: LRFinderResult, baseline: Dict[str, float]) -> Tuple[float, bool]:
    """Recommend the LR minimising fresh training loss, subject to the
    validation guardrail: it must also improve validation loss vs baseline. If
    the best-train-loss LR worsens val loss, flag overfitting and fall back to
    the next-best candidate whose val loss also improved."""
    usable = [t for t in result.trials if not t.catastrophic]
    if not usable:
        # Everything diverged — return the least-bad by objective.
        best = min(result.trials, key=lambda t: t.objective)
        return best.lr, False

    by_train = sorted(usable, key=lambda t: t.train_loss)
    best_train = by_train[0]
    if best_train.val_loss <= baseline["val_loss"]:
        return best_train.lr, False

    # best-train-loss LR overfits (val loss worse) → next-best that improves val
    for t in by_train[1:]:
        if t.val_loss <= baseline["val_loss"]:
            return t.lr, True
    # None improved val loss; keep the best-train-loss LR but flag it.
    return best_train.lr, True


def _sanity_check_lr_min(t_min: LRTrial, baseline: Dict[str, float], rel_tol: float) -> None:
    """A near-zero LR must change almost nothing. If lr_min produces NaN/Inf or
    a clear degradation, the problem is the data/loss/precision, not the LR —
    raise so no untrustworthy result is returned."""
    if t_min.catastrophic or not math.isfinite(t_min.train_loss):
        raise RuntimeError(
            f"lr_min={t_min.lr:.1e} produced a non-finite / catastrophic training "
            f"loss. A near-zero LR should barely move the model — this points to a "
            f"broken data pipeline, loss, precision, or quantizer setup, not the LR."
        )
    base = baseline["train_loss"]
    if t_min.train_loss > base * (1.0 + rel_tol) + 1e-6:
        raise RuntimeError(
            f"lr_min={t_min.lr:.1e} degraded training loss {base:.4f} -> "
            f"{t_min.train_loss:.4f} (> {rel_tol:.0%}). A near-zero LR should be ~a "
            f"no-op; something other than the LR is wrong."
        )
    if t_min.val_acc < baseline["val_acc"] - 0.05:
        raise RuntimeError(
            f"lr_min={t_min.lr:.1e} dropped val_acc {baseline['val_acc']:.4f} -> "
            f"{t_min.val_acc:.4f}. A near-zero LR should be ~a no-op; the pipeline "
            f"is suspect."
        )
    print(f"[lr-bo] lr_min sanity OK: train_loss {base:.4f} -> {t_min.train_loss:.4f} "
          f"(≈ baseline, as expected for a near-zero LR).")


def _load_cached_trials(map_path: str) -> List[LRTrial]:
    if not os.path.exists(map_path):
        return []
    with open(map_path) as fh:
        raw = json.load(fh)
    trials: List[LRTrial] = []
    for _id, d in sorted(raw.items(), key=lambda kv: int(kv[0])):
        # Prefer the exact hex round-trip if present.
        lr = float.fromhex(d["lr_hex"]) if "lr_hex" in d else float(d["lr"])
        log10_lr = float.fromhex(d["log10_lr_hex"]) if "log10_lr_hex" in d else float(d["log10_lr"])
        trials.append(LRTrial(
            trial_id=int(d["trial_id"]), lr=lr, log10_lr=log10_lr,
            train_loss=d["train_loss"], train_acc=d["train_acc"],
            val_loss=d["val_loss"], val_acc=d["val_acc"],
            d_train_loss=d["d_train_loss"], d_train_acc=d["d_train_acc"],
            d_val_loss=d["d_val_loss"], d_val_acc=d["d_val_acc"],
            catastrophic=d["catastrophic"], objective=d["objective"],
            checkpoint_path=d.get("checkpoint_path"), seed=d.get("seed", 0),
        ))
    return trials


def _persist_trials(map_path: str, result: LRFinderResult) -> None:
    """The authoritative map file: keyed by integer trial ID, with exact
    (hex) LR representations — never rely on folder names for correctness."""
    out = {str(t.trial_id): t.to_json() for t in result.trials}
    tmp = map_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, map_path)


def _persist_result(output_dir: str, result: LRFinderResult) -> None:
    summary = {
        "recommended_lr": result.recommended_lr,
        "recommended_lr_hex": float(result.recommended_lr).hex()
                              if math.isfinite(result.recommended_lr) else None,
        "overfit_flagged": result.overfit_flagged,
        "baseline": result.baseline,
        "fallback_loss": result.fallback_loss,
        "gp_predicted_optimum_lr": result.gp_predicted_optimum_lr,
        "gp_predicted_optimum_value": result.gp_predicted_optimum_value,
        "lr_min": result.lr_min, "lr_max": result.lr_max,
        "n_trials": len(result.trials),
    }
    with open(os.path.join(output_dir, "result.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
