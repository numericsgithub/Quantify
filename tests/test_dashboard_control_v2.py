"""
test_dashboard_control_v2.py — ControlManager bound to the V2 trainer.

Verifies the Phase-2 migration by inspecting the trainer's OWN state after a
command drains (not merely that submit returned): LR + scheduler suspension,
add-epochs extending the re-read loop bound, callback toggles rejected on V2,
reload-best confirmation/graceful failure, and — via a real short fit() — that a
manually set LR survives the scheduler's next step.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytest

from training_harness.trainer_v2 import QATTrainerV2
from training_harness.config_v2 import TrainerConfigV2
from training_harness.config import CheckpointConfig, LoggingConfig
from training_harness.api.control import ControlValidationError


def _make_trainer(tmp_path, with_step_scheduler=False, plateau=False):
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = (torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
             if with_step_scheduler else None)
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    config = TrainerConfigV2(
        experiment_name="ctl_test",
        output_dir=str(tmp_path),
        epochs=1,
        num_classes=2,
        smoothing=0.0,               # avoid the timm import
        device="cpu",
        dry_run=True,
        dry_run_batches=2,
        api_port=None,               # no HTTP server; _control is still created
        reduce_lr_on_plateau=plateau,
        reduce_lr_patience=1,
        logging=LoggingConfig(save_plots=False, csv_log=False),
        checkpoint=CheckpointConfig(save_last=False, top_k=1),
    )
    return QATTrainerV2(
        config=config, model=model, optimizer=opt,
        train_loader=loader, val_loader=None,
        loss_fn=nn.CrossEntropyLoss(), scheduler=sched,
    )


# ---------------------------------------------------------------------------
# Control queue is created even without the HTTP server
# ---------------------------------------------------------------------------

def test_control_manager_always_created(tmp_path):
    t = _make_trainer(tmp_path)
    assert t._control is not None
    assert t.api_server is None          # API off, but the queue still exists


# ---------------------------------------------------------------------------
# LR / hyperparameter change + scheduler suspension
# ---------------------------------------------------------------------------

def test_lr_applies_and_suspends_step_scheduler(tmp_path):
    t = _make_trainer(tmp_path, with_step_scheduler=True)
    cmd = t._control.submit("set_hyperparams", {"lr": 0.05})
    assert t._control.get_command(cmd.id)["status"] == "pending"

    t._control.drain("step")                       # applied at the step boundary

    assert t.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert t._scheduler_suspended is True
    assert t._control.get_command(cmd.id)["status"] == "applied"


def test_lr_suspends_plateau_scheduler(tmp_path):
    # No per-step scheduler, but ReduceLROnPlateau is active -> still suspended.
    t = _make_trainer(tmp_path, with_step_scheduler=False, plateau=True)
    assert t._plateau_lr_sched is not None
    t._control.submit("set_hyperparams", {"lr": 0.02})
    t._control.drain("step")
    assert t.optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
    assert t._scheduler_suspended is True


def test_lr_without_any_scheduler_does_not_flag_suspension(tmp_path):
    t = _make_trainer(tmp_path)          # no scheduler at all
    t._control.submit("set_hyperparams", {"lr": 0.05})
    t._control.drain("step")
    assert t.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert t._scheduler_suspended is False


def test_resume_scheduler_via_flag(tmp_path):
    t = _make_trainer(tmp_path, with_step_scheduler=True)
    t._control.submit("set_hyperparams", {"lr": 0.05})
    t._control.drain("step")
    assert t._scheduler_suspended is True
    t._control.submit("set_hyperparams", {"suspend_scheduler": False})
    t._control.drain("step")
    assert t._scheduler_suspended is False


# ---------------------------------------------------------------------------
# add-epochs extends the re-read loop bound
# ---------------------------------------------------------------------------

def test_add_epochs_extends_end_epoch(tmp_path):
    t = _make_trainer(tmp_path)
    t._end_epoch = 5                     # as fit() would have set it
    old_budget = t.config.epochs
    cmd = t._control.submit("add_epochs", {"count": 3})
    t._control.drain("epoch")
    assert t._end_epoch == 8
    assert t.config.epochs == old_budget + 3
    assert t._control.get_command(cmd.id)["status"] == "applied"


# ---------------------------------------------------------------------------
# Callback toggles are rejected on V2
# ---------------------------------------------------------------------------

def test_v2_has_callback_registry(tmp_path):
    # V2 now has a registry (was the "not supported" regression, now restored).
    t = _make_trainer(tmp_path)
    assert t._control.callbacks is not None
    names = [c["name"] for c in t._control.callbacks.list()]
    assert "checkpointing" in names and "optimizer_step" in names


def test_toggle_callback_applies_on_v2(tmp_path):
    t = _make_trainer(tmp_path)
    assert t.callbacks.is_enabled("checkpointing") is True
    cmd = t._control.submit("toggle_callback", {"name": "checkpointing", "enabled": False})
    t._control.drain("epoch")
    assert t._control.get_command(cmd.id)["status"] == "applied"
    assert t.callbacks.is_enabled("checkpointing") is False


def test_toggle_core_callback_rejected(tmp_path):
    t = _make_trainer(tmp_path)
    with pytest.raises(ControlValidationError):
        t._control.submit("toggle_callback", {"name": "optimizer_step", "enabled": False})


def test_toggle_unknown_callback_rejected(tmp_path):
    t = _make_trainer(tmp_path)
    with pytest.raises(ControlValidationError):
        t._control.submit("toggle_callback", {"name": "nope", "enabled": False})


# ---------------------------------------------------------------------------
# A1: live ReduceLROnPlateau params
# ---------------------------------------------------------------------------

def test_scheduler_params_apply(tmp_path):
    t = _make_trainer(tmp_path, plateau=True)
    cmd = t._control.submit("set_scheduler_params", {"patience": 7, "factor": 0.25})
    t._control.drain("epoch")
    assert t._control.get_command(cmd.id)["status"] == "applied"
    assert t._plateau_lr_sched.patience == 7
    assert t._plateau_lr_sched.factor == pytest.approx(0.25)


def test_scheduler_params_without_plateau_fails(tmp_path):
    t = _make_trainer(tmp_path, plateau=False)   # no ReduceLROnPlateau
    cmd = t._control.submit("set_scheduler_params", {"patience": 3})
    t._control.drain("epoch")
    rec = t._control.get_command(cmd.id)
    assert rec["status"] == "failed" and "reducelronplateau" in rec["result"].lower()


def test_scheduler_params_validation(tmp_path):
    t = _make_trainer(tmp_path, plateau=True)
    with pytest.raises(ControlValidationError):
        t._control.submit("set_scheduler_params", {})                 # nothing given
    with pytest.raises(ControlValidationError):
        t._control.submit("set_scheduler_params", {"factor": 1.5})    # out of (0,1)


# ---------------------------------------------------------------------------
# reload-best: confirmation required; graceful failure with no checkpoint
# ---------------------------------------------------------------------------

def test_reload_best_requires_confirm(tmp_path):
    t = _make_trainer(tmp_path)
    with pytest.raises(ControlValidationError):
        t._control.submit("reload_best", {})


def test_reload_best_without_checkpoint_fails_at_apply(tmp_path):
    t = _make_trainer(tmp_path)
    cmd = t._control.submit("reload_best", {"confirm": True})   # valid -> queued
    t._control.drain("epoch")
    rec = t._control.get_command(cmd.id)
    assert rec["status"] == "failed"                            # no checkpoint yet
    assert "no best checkpoint" in rec["result"].lower()


# ---------------------------------------------------------------------------
# End-to-end: a manually set LR survives the scheduler's next step
# ---------------------------------------------------------------------------

def test_manual_lr_survives_scheduler_over_fit(tmp_path):
    t = _make_trainer(tmp_path, with_step_scheduler=True)
    # Pre-submit before fit(); the loop drains it on the first training step.
    t._control.submit("set_hyperparams", {"lr": 0.5})
    t.fit()   # dry_run: 1 epoch x 2 batches
    # StepLR(gamma=0.5) would have driven LR to ~0.025 across 2 steps; instead it
    # is suspended after the first drain, so the manual 0.5 stands.
    assert t.optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    assert t._scheduler_suspended is True


# ---------------------------------------------------------------------------
# Pause / resume (1b)
# ---------------------------------------------------------------------------

def test_pause_clears_event_resume_sets_it(tmp_path):
    t = _make_trainer(tmp_path)
    assert t._pause_event.is_set() and t._paused is False
    res = t._control.pause()                      # direct, off-queue (not a command)
    assert res == {"paused": True, "was_paused": False}
    assert t._paused is True
    assert not t._pause_event.is_set()            # the loop would block at the gate
    res = t._control.resume()                     # direct, off-queue
    assert res == {"resumed": True, "was_paused": True}
    assert t._paused is False and t._pause_event.is_set()


def test_pause_is_not_a_queued_command(tmp_path):
    # Guards the fix: pause must not go through submit/drain (that races resume).
    t = _make_trainer(tmp_path)
    with pytest.raises(ControlValidationError):
        t._control.submit("pause", {})


def test_resume_when_not_paused_is_noop(tmp_path):
    t = _make_trainer(tmp_path)
    res = t._control.resume()
    assert res["resumed"] is True and res["was_paused"] is False
    assert t._pause_event.is_set()


# ---------------------------------------------------------------------------
# End-epoch-early (2)
# ---------------------------------------------------------------------------

def test_end_epoch_early_sets_flag(tmp_path):
    t = _make_trainer(tmp_path)
    assert t._end_epoch_early is False
    t._control.submit("end_epoch_early", {})
    t._control.drain("step")
    assert t._end_epoch_early is True


# ---------------------------------------------------------------------------
# Halt after epoch (1a)
# ---------------------------------------------------------------------------

def test_halt_requires_confirm(tmp_path):
    t = _make_trainer(tmp_path)
    with pytest.raises(ControlValidationError):
        t._control.submit("halt", {})


def test_halt_sets_flag_at_epoch_boundary(tmp_path):
    t = _make_trainer(tmp_path)
    assert t._halt_requested is False
    cmd = t._control.submit("halt", {"confirm": True})
    t._control.drain("epoch")
    assert t._halt_requested is True
    assert t._control.get_command(cmd.id)["status"] == "applied"


# ---------------------------------------------------------------------------
# Reload-by-criterion with a second checkpoint pool (4 / 5)
# ---------------------------------------------------------------------------

def _make_trainer_with_pools(tmp_path):
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    config = TrainerConfigV2(
        experiment_name="pool_test", output_dir=str(tmp_path),
        epochs=1, num_classes=2, smoothing=0.0, device="cpu",
        dry_run=True, dry_run_batches=2, api_port=None,
        logging=LoggingConfig(save_plots=False, csv_log=False),
        checkpoint=CheckpointConfig(monitor_metric="val_acc", monitor_mode="max",
                                    top_k=1, save_last=False),
        secondary_checkpoint_metrics=[("train_loss", "min")],
    )
    return QATTrainerV2(config=config, model=model, optimizer=opt,
                        train_loader=loader, val_loader=loader,
                        loss_fn=nn.CrossEntropyLoss())


def test_secondary_pool_registered(tmp_path):
    t = _make_trainer_with_pools(tmp_path)
    assert set(t._checkpoint_pools) == {"val_acc", "train_loss"}


def test_reload_criterion_selects_pool(tmp_path):
    t = _make_trainer_with_pools(tmp_path)
    t.fit()   # populates both the val_acc and train_loss pools
    for crit in ("best_val_acc", "best_train_loss"):
        cmd = t._control.submit("reload_best", {"confirm": True, "criterion": crit})
        t._control.drain("epoch")
        rec = t._control.get_command(cmd.id)
        assert rec["status"] == "applied", f"{crit}: {rec['result']}"
        assert crit in rec["result"]


def test_reload_unknown_criterion_fails(tmp_path):
    t = _make_trainer_with_pools(tmp_path)
    t.fit()
    cmd = t._control.submit("reload_best", {"confirm": True, "criterion": "best_nope"})
    t._control.drain("epoch")
    rec = t._control.get_command(cmd.id)
    assert rec["status"] == "failed"
    assert "no checkpoint pool" in rec["result"].lower()


def test_reload_weights_only_must_be_bool(tmp_path):
    t = _make_trainer(tmp_path)
    with pytest.raises(ControlValidationError):
        t._control.submit("reload_best", {"confirm": True, "weights_only": "yes"})
