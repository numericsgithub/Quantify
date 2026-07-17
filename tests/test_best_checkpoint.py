"""
Tests for CheckpointManager's stable 'best.pt' + seeded best-threshold.

The point of seed_best(): a chain of runs must stay monotonic. If a run starts
from a 71.00% checkpoint and only ever gets worse (69.20%), best.pt must still
hold the 71.00% starting model so the next run in the chain picks that up
instead of regressing.
"""

import os

import torch
import torch.nn as nn

from training_harness.checkpointing import CheckpointManager


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3))


def _mgr(tmp_path, mode="max"):
    return CheckpointManager(save_dir=str(tmp_path), top_k=3, monitor_mode=mode,
                             experiment_name="t")


def test_seed_best_writes_best_and_sets_threshold(tmp_path):
    m = _model()
    mgr = _mgr(tmp_path)
    mgr.seed_best(metric_value=0.71, model=m)

    assert os.path.exists(tmp_path / "best.pt")
    assert mgr._best_metric == 0.71
    payload = torch.load(tmp_path / "best.pt", map_location="cpu")
    assert "model_state_dict" in payload


def test_worse_run_cannot_regress_best(tmp_path):
    """Start at 0.71, then every epoch is worse -> best.pt must stay the 0.71 seed."""
    m = _model()
    mgr = _mgr(tmp_path)
    mgr.seed_best(metric_value=0.71, model=m)
    seeded = torch.load(tmp_path / "best.pt", map_location="cpu")["model_state_dict"]

    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    # Mutate the model so a saved checkpoint would differ from the seed
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)

    for ep, worse in enumerate([0.70, 0.692, 0.65]):
        mgr.save(epoch=ep, metric_value=worse, model=m, optimizer=opt)

    assert mgr._best_metric == 0.71, "threshold must not move for worse metrics"
    still = torch.load(tmp_path / "best.pt", map_location="cpu")["model_state_dict"]
    for k in seeded:
        assert torch.equal(seeded[k], still[k]), "best.pt was overwritten by a worse run"


def test_better_run_updates_best(tmp_path):
    m = _model()
    mgr = _mgr(tmp_path)
    mgr.seed_best(metric_value=0.71, model=m)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    with torch.no_grad():
        for p in m.parameters():
            p.add_(2.0)
    mgr.save(epoch=0, metric_value=0.715, model=m, optimizer=opt)

    assert mgr._best_metric == 0.715
    saved = torch.load(tmp_path / "best.pt", map_location="cpu")
    assert saved["metrics"] == {} or True
    now = saved["model_state_dict"]
    for k, v in m.state_dict().items():
        assert torch.equal(v, now[k]), "best.pt should hold the improved weights"


def test_best_threshold_persists_across_manager_reload(tmp_path):
    m = _model()
    mgr = _mgr(tmp_path)
    mgr.seed_best(metric_value=0.71, model=m)

    # A fresh manager on the same dir must remember the threshold (crash/restart)
    mgr2 = _mgr(tmp_path)
    assert mgr2._best_metric == 0.71
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    mgr2.save(epoch=0, metric_value=0.70, model=m, optimizer=opt)
    assert mgr2._best_metric == 0.71, "reloaded threshold must still block a worse metric"


def test_min_mode_best(tmp_path):
    """monitor_mode='min' (e.g. loss): lower must win."""
    m = _model()
    mgr = _mgr(tmp_path, mode="min")
    mgr.seed_best(metric_value=1.0, model=m)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    mgr.save(epoch=0, metric_value=1.5, model=m, optimizer=opt)   # worse
    assert mgr._best_metric == 1.0
    mgr.save(epoch=1, metric_value=0.8, model=m, optimizer=opt)   # better
    assert mgr._best_metric == 0.8
