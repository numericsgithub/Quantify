"""
Tests for training_harness.lr_bayes_finder.find_learning_rate.

Uses a tiny MLP + synthetic classification data so "one full epoch per trial"
is fast on CPU. The finder itself is model-agnostic.
"""

import json
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training_harness.lr_bayes_finder import find_learning_rate, LRFinderResult


def _make(n=256, d=8, c=3, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    w = torch.randn(d, c)
    y = (X @ w).argmax(1)
    ds = TensorDataset(X, y)
    train = DataLoader(ds, batch_size=32, shuffle=True)
    val = DataLoader(ds, batch_size=32, shuffle=False)
    model = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, c))
    return model, train, val


def _save_start(model, path):
    torch.save({"model_state_dict": model.state_dict()}, path)


def test_find_lr_basic(tmp_path):
    model, train, val = _make()
    ckpt = tmp_path / "start.pt"
    _save_start(model, ckpt)
    fac = lambda lr: torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    out = tmp_path / "lrbo"
    res = find_learning_rate(
        model, str(ckpt), str(out), train, val, fac,
        lr_min=1e-6, lr_max=1.0, n_initial_points=3, n_calls=6,
        train_eval_subset_size=128, device="cpu", seed=1,
    )

    assert isinstance(res, LRFinderResult)
    assert len(res.trials) == 6
    assert res.lr_min <= res.recommended_lr <= res.lr_max
    # searched in log space -> recommended is one of the trials' LRs
    assert any(abs(math.log10(t.lr) - math.log10(res.recommended_lr)) < 1e-9 for t in res.trials)
    # baseline + all four metrics present
    for k in ("train_loss", "train_acc", "val_loss", "val_acc"):
        assert k in res.baseline
    # GP predicted optimum reported
    assert res.gp_predicted_optimum_lr is not None
    assert res.lr_min <= res.gp_predicted_optimum_lr <= res.lr_max

    # persistence: map file keyed by int id, with exact hex round-trip
    tj = json.loads((out / "trials.json").read_text())
    assert set(tj.keys()) == {str(i) for i in range(6)}
    for _id, d in tj.items():
        assert float.fromhex(d["lr_hex"]) == d["lr"] or abs(float.fromhex(d["lr_hex"]) - d["lr"]) < 1e-12
        assert "d_train_loss" in d and "d_val_acc" in d and "catastrophic" in d
        # a checkpoint was saved per trial
        assert d["checkpoint_path"] and (tmp_path in (tmp_path,))  # sanity
    assert (out / "result.json").exists()
    # every trial folder has a saved model
    for i in range(6):
        subs = list(out.glob(f"{i:04d}_lr_*"))
        assert len(subs) == 1 and (subs[0] / "model.pt").exists()


def test_resume_reuses_cached_trials(tmp_path):
    model, train, val = _make()
    ckpt = tmp_path / "start.pt"
    _save_start(model, ckpt)
    fac = lambda lr: torch.optim.SGD(model.parameters(), lr=lr)
    out = tmp_path / "lrbo"

    res1 = find_learning_rate(model, str(ckpt), str(out), train, val, fac,
                              lr_min=1e-6, lr_max=1.0, n_initial_points=3, n_calls=4,
                              train_eval_subset_size=128, device="cpu", seed=3)
    assert len(res1.trials) == 4
    lrs1 = {t.trial_id: t.lr for t in res1.trials}

    # Resume with a larger budget: the 4 cached trials must be reused (same LRs),
    # only the extra trials get computed.
    res2 = find_learning_rate(model, str(ckpt), str(out), train, val, fac,
                              lr_min=1e-6, lr_max=1.0, n_initial_points=3, n_calls=6,
                              train_eval_subset_size=128, device="cpu", seed=3)
    assert len(res2.trials) == 6
    for tid, lr in lrs1.items():
        match = next(t for t in res2.trials if t.trial_id == tid)
        assert match.lr == lr  # cached, not recomputed


def test_lr_min_sanity_raises_on_broken_pipeline(tmp_path):
    """If a near-zero LR does NOT behave like a no-op (here: a loss that returns
    NaN), the finder must raise rather than return an untrustworthy result."""
    model, train, val = _make()
    ckpt = tmp_path / "start.pt"
    _save_start(model, ckpt)
    fac = lambda lr: torch.optim.SGD(model.parameters(), lr=lr)

    class NanLoss(nn.Module):
        def forward(self, out, tgt):
            return out.sum() * float("nan")

    with pytest.raises(RuntimeError):
        find_learning_rate(model, str(ckpt), str(tmp_path / "lrbo"), train, val, fac,
                           loss_fn=NanLoss(), lr_min=1e-6, lr_max=1.0,
                           n_initial_points=2, n_calls=4,
                           train_eval_subset_size=128, device="cpu", seed=5)


def test_catastrophic_lr_uses_fallback_not_nan(tmp_path):
    """A huge LR that diverges must be recorded as catastrophic with the
    fallback loss as objective (not NaN poisoning the GP)."""
    model, train, val = _make()
    ckpt = tmp_path / "start.pt"
    _save_start(model, ckpt)
    fac = lambda lr: torch.optim.SGD(model.parameters(), lr=lr)

    res = find_learning_rate(model, str(ckpt), str(tmp_path / "lrbo"), train, val, fac,
                             lr_min=1e-6, lr_max=1e6, n_initial_points=3, n_calls=8,
                             train_eval_subset_size=128, device="cpu", seed=2)
    # at least one catastrophic trial expected at the very high end
    cats = [t for t in res.trials if t.catastrophic]
    for t in cats:
        assert math.isfinite(t.objective)
        assert t.objective == pytest.approx(res.fallback_loss)
        assert t.train_acc == 0.0
    assert math.isfinite(res.recommended_lr)
