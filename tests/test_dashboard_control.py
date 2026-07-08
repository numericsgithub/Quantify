"""
Tests for the dashboard live-control layer (training_harness/api/control.py
and the write endpoints).

Covers: command validation (bad LR, unsafe callback toggle, reload without
confirm, bad epoch count), queue application at safe boundaries, scheduler
suspension, the new read endpoints (callbacks / commands / events), and an
end-to-end HTTP round trip through a live Trainer.
"""
import json
import tempfile
import time
import urllib.error
import urllib.request

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytest

from training_harness.trainer import Trainer
from training_harness.config import TrainerConfig, LoggingConfig, QuantScheduleConfig
from training_harness.api import create_app, ControlValidationError
from quantizers import FixedPointPerTensorWeightQuant


class TinyQuantNet(nn.Module):
    def __init__(self):
        super().__init__()
        import brevitas.nn as qnn
        self.conv = qnn.QuantConv2d(1, 4, kernel_size=3, padding=1,
                                    weight_quant=FixedPointPerTensorWeightQuant)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = qnn.QuantLinear(4 * 8 * 8, 10,
                                  weight_quant=FixedPointPerTensorWeightQuant)

    def forward(self, x):
        return self.fc(self.flatten(self.relu(self.conv(x))))


@pytest.fixture
def dummy_loader():
    X = torch.randn(32, 1, 8, 8)
    y = torch.randint(0, 10, (32,))
    return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)


def make_trainer(tmpdir, dummy_loader, scheduler=None, early_stop=False):
    """A Trainer with the control API wired (bound to an ephemeral port)."""
    model = TinyQuantNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    config = TrainerConfig(
        experiment_name="ctl_test",
        output_dir=tmpdir,
        epochs=2,
        dry_run=True,
        api_port=0,
        early_stopping_patience=3 if early_stop else None,
        logging=LoggingConfig(log_every_n_steps=1, save_plots=False),
        quant_schedule=QuantScheduleConfig(float_warmup_epochs=1),
    )
    return Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        loss_fn=nn.CrossEntropyLoss(),
        scheduler=scheduler,
    )


# ---------------------------------------------------------------------------
# Validation (submit-time, no application needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("params", [
    {"lr": -1.0},          # negative
    {"lr": 0.0},           # zero
    {"lr": 1e9},           # absurd
    {"lr": float("nan")},  # non-finite
    {"lr": "fast"},        # non-numeric
    {},                    # nothing to change
    {"weight_decay": -0.1},
])
def test_bad_hyperparams_rejected(tmp_path, dummy_loader, params):
    trainer = make_trainer(str(tmp_path), dummy_loader)
    with pytest.raises(ControlValidationError):
        trainer._control.submit("set_hyperparams", params)
    trainer.api_server.shutdown()


def test_unsafe_callback_toggle_rejected(tmp_path, dummy_loader):
    trainer = make_trainer(str(tmp_path), dummy_loader)
    # core, non-toggleable
    with pytest.raises(ControlValidationError):
        trainer._control.submit("toggle_callback", {"name": "optimizer_step", "enabled": False})
    with pytest.raises(ControlValidationError):
        trainer._control.submit("toggle_callback", {"name": "metrics_logging", "enabled": False})
    # unknown
    with pytest.raises(ControlValidationError):
        trainer._control.submit("toggle_callback", {"name": "does_not_exist", "enabled": False})
    trainer.api_server.shutdown()


def test_reload_without_confirm_rejected(tmp_path, dummy_loader):
    trainer = make_trainer(str(tmp_path), dummy_loader)
    with pytest.raises(ControlValidationError):
        trainer._control.submit("reload_best", {})
    with pytest.raises(ControlValidationError):
        trainer._control.submit("reload_best", {"confirm": False})
    trainer.api_server.shutdown()


@pytest.mark.parametrize("count", [0, -5, 1.5, "3", True])
def test_bad_add_epochs_rejected(tmp_path, dummy_loader, count):
    trainer = make_trainer(str(tmp_path), dummy_loader)
    with pytest.raises(ControlValidationError):
        trainer._control.submit("add_epochs", {"count": count})
    trainer.api_server.shutdown()


# ---------------------------------------------------------------------------
# Application at safe boundaries
# ---------------------------------------------------------------------------

def test_lr_change_applies_on_step_drain(tmp_path, dummy_loader):
    trainer = make_trainer(str(tmp_path), dummy_loader)
    cmd = trainer._control.submit("set_hyperparams", {"lr": 3e-4})
    assert cmd.status == "pending"
    assert cmd.apply_at == "step"

    # not applied until the loop drains the step queue
    assert trainer.optimizer.param_groups[0]["lr"] == 0.1
    trainer._control.drain("step")
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert trainer._control.get_command(cmd.id)["status"] == "applied"
    trainer.api_server.shutdown()


def test_lr_change_suspends_scheduler(tmp_path, dummy_loader):
    sched_holder = {}

    def make_sched(opt):
        s = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        sched_holder["s"] = s
        return s

    model = TinyQuantNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    config = TrainerConfig(
        experiment_name="ctl_sched", output_dir=str(tmp_path), epochs=1,
        dry_run=True, api_port=0,
        logging=LoggingConfig(log_every_n_steps=1, save_plots=False),
        quant_schedule=QuantScheduleConfig(float_warmup_epochs=1),
    )
    trainer = Trainer(config=config, model=model, optimizer=optimizer,
                      train_loader=dummy_loader, val_loader=dummy_loader,
                      loss_fn=nn.CrossEntropyLoss(),
                      scheduler=make_sched(optimizer))
    assert trainer._scheduler_suspended is False
    trainer._control.submit("set_hyperparams", {"lr": 5e-4})
    trainer._control.drain("step")
    assert trainer._scheduler_suspended is True
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)

    # explicit resume
    trainer._control.submit("set_hyperparams", {"suspend_scheduler": False})
    trainer._control.drain("step")
    assert trainer._scheduler_suspended is False
    trainer.api_server.shutdown()


def test_add_epochs_extends_budget(tmp_path, dummy_loader):
    trainer = make_trainer(str(tmp_path), dummy_loader)
    assert trainer.config.epochs == 2
    cmd = trainer._control.submit("add_epochs", {"count": 5})
    assert cmd.apply_at == "epoch"
    trainer._control.drain("epoch")
    assert trainer.config.epochs == 7
    assert trainer._control.get_command(cmd.id)["status"] == "applied"
    trainer.api_server.shutdown()


def test_toggle_callback_applies(tmp_path, dummy_loader):
    trainer = make_trainer(str(tmp_path), dummy_loader)
    assert trainer.callbacks.is_enabled("checkpointing") is True
    trainer._control.submit("toggle_callback", {"name": "checkpointing", "enabled": False})
    trainer._control.drain("epoch")
    assert trainer.callbacks.is_enabled("checkpointing") is False
    trainer.api_server.shutdown()


def test_reload_best_fails_without_checkpoint(tmp_path, dummy_loader):
    trainer = make_trainer(str(tmp_path), dummy_loader)
    cmd = trainer._control.submit("reload_best", {"confirm": True})
    trainer._control.drain("epoch")  # no checkpoint saved yet -> apply-time failure
    rec = trainer._control.get_command(cmd.id)
    assert rec["status"] == "failed"
    assert "no best checkpoint" in rec["result"].lower()
    trainer.api_server.shutdown()


# ---------------------------------------------------------------------------
# Read endpoints via the Flask test client
# ---------------------------------------------------------------------------

def test_callbacks_endpoint(tmp_path, dummy_loader):
    trainer = make_trainer(str(tmp_path), dummy_loader)
    client = create_app(trainer._api_collector, trainer._control).test_client()
    data = client.get("/api/v1/callbacks").get_json()
    names = {c["name"]: c for c in data["callbacks"]}
    assert names["checkpointing"]["toggleable"] is True
    assert names["optimizer_step"]["toggleable"] is False
    assert names["metrics_logging"]["toggleable"] is False
    trainer.api_server.shutdown()


def test_commands_and_events_endpoints(tmp_path, dummy_loader):
    trainer = make_trainer(str(tmp_path), dummy_loader)
    client = create_app(trainer._api_collector, trainer._control).test_client()

    # a submitted command shows up in history and produces an event
    resp = client.post("/api/v1/control/hyperparams", json={"lr": 1e-3})
    assert resp.status_code == 202
    cid = resp.get_json()["id"]

    cmds = client.get("/api/v1/commands").get_json()["commands"]
    assert any(c["id"] == cid for c in cmds)
    assert client.get(f"/api/v1/commands/{cid}").get_json()["status"] == "pending"
    assert client.get("/api/v1/commands/nope").status_code == 404

    events = client.get("/api/v1/events").get_json()["events"]
    assert any(e["kind"] == "command_submitted" for e in events)
    trainer.api_server.shutdown()


def test_http_validation_returns_400(tmp_path, dummy_loader):
    trainer = make_trainer(str(tmp_path), dummy_loader)
    client = create_app(trainer._api_collector, trainer._control).test_client()
    assert client.post("/api/v1/control/hyperparams", json={"lr": -1}).status_code == 400
    assert client.post("/api/v1/control/reload-best", json={}).status_code == 400
    assert client.post("/api/v1/control/add-epochs", json={"count": 0}).status_code == 400
    assert client.post("/api/v1/control/callbacks/optimizer_step",
                       json={"enabled": False}).status_code == 400
    trainer.api_server.shutdown()


def test_read_only_mode_rejects_control():
    """create_app without a ControlManager serves reads, 503s on control."""
    from training_harness.api import RunStateCollector

    class _Stub:
        config = TrainerConfig(experiment_name="ro", epochs=1)
        model = nn.Linear(2, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        _global_step = 0
    collector = RunStateCollector(_Stub(), jsonl_path=None)
    client = create_app(collector, control=None).test_client()
    assert client.get("/api/v1/status").status_code == 200
    assert client.get("/api/v1/callbacks").get_json() == {"callbacks": []}
    assert client.post("/api/v1/control/hyperparams", json={"lr": 1e-3}).status_code == 503


# ---------------------------------------------------------------------------
# End-to-end over real HTTP inside a live run
# ---------------------------------------------------------------------------

def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
        return json.loads(resp.read())


def test_e2e_add_epochs_over_http(tmp_path, dummy_loader):
    trainer = make_trainer(str(tmp_path), dummy_loader)
    port = trainer.api_server.port

    status, body = _post(port, "/api/v1/control/add-epochs", {"count": 3})
    assert status == 202
    assert body["status"] == "pending"

    trainer.fit()  # drains the epoch queue during the run

    cmd = _get(port, f"/api/v1/commands/{body['id']}")
    assert cmd["status"] == "applied"
    # started at 2 epochs, +3 => 5, reflected in /status
    assert _get(port, "/api/v1/status")["total_epochs"] == 5
    trainer.api_server.shutdown()
