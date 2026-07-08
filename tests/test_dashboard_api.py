"""
Tests for the read-only training monitoring API (training_harness/api/).

Covers: opt-in behavior, endpoint schemas, ?since_step/?since_epoch
filtering, and end-to-end serving from a live background thread for both
the V1 Trainer and QATTrainerV2.
"""
import json
import tempfile
import urllib.request

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytest

from training_harness.trainer import Trainer
from training_harness.config import TrainerConfig, LoggingConfig, QuantScheduleConfig
from training_harness.trainer_v2 import QATTrainerV2
from training_harness.config_v2 import TrainerConfigV2, QATScheduleConfigV2
from training_harness.api import RunStateCollector, create_app
from quantizers import FixedPointPerTensorWeightQuant


class TinyQuantNet(nn.Module):
    """Minimal quantized model to exercise the harness quickly."""
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


def _get_json(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
        assert resp.status == 200
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Opt-in behavior
# ---------------------------------------------------------------------------

def test_api_disabled_by_default(dummy_loader):
    """Without api_port, no server or collector is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            config=TrainerConfig(experiment_name="no_api", output_dir=tmpdir),
            model=TinyQuantNet(),
            optimizer=optim.Adam(TinyQuantNet().parameters()),
            train_loader=dummy_loader,
        )
        assert trainer.api_server is None
        assert trainer._api_collector is None


# ---------------------------------------------------------------------------
# Collector unit tests (?since= filtering, schemas) via Flask test client
# ---------------------------------------------------------------------------

class _StubTrainer:
    """Bare-minimum trainer stand-in for collector unit tests."""
    def __init__(self):
        self.config = TrainerConfig(experiment_name="stub", epochs=10)
        self.model = nn.Linear(4, 2)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.5)
        self._global_step = 0


@pytest.fixture
def stub_client():
    collector = RunStateCollector(_StubTrainer(), jsonl_path=None)
    for step in range(1, 6):
        collector.on_step(step * 10, {"loss": 1.0 / step}, phase="train")
    for epoch in range(3):
        collector.on_epoch(epoch, {"train_loss": 1.0 - epoch * 0.1, "val_acc": 0.5 + epoch * 0.1})
    app = create_app(collector)
    return app.test_client(), collector


def test_health(stub_client):
    client, _ = stub_client
    assert client.get("/api/v1/health").get_json() == {"ok": True}


def test_metrics_full_history(stub_client):
    client, _ = stub_client
    data = client.get("/api/v1/metrics").get_json()
    assert [s["step"] for s in data["steps"]] == [10, 20, 30, 40, 50]
    assert [e["epoch"] for e in data["epochs"]] == [0, 1, 2]
    assert data["steps"][0]["loss"] == 1.0
    assert data["steps"][0]["lr"] == 0.5
    assert "train_acc" in data["caveats"]


def test_metrics_since_filtering(stub_client):
    client, _ = stub_client
    data = client.get("/api/v1/metrics?since_step=30&since_epoch=1").get_json()
    assert [s["step"] for s in data["steps"]] == [40, 50]
    assert [e["epoch"] for e in data["epochs"]] == [2]

    # Cursor at the newest values -> empty increments
    data = client.get("/api/v1/metrics?since_step=50&since_epoch=2").get_json()
    assert data["steps"] == []
    assert data["epochs"] == []


def test_metrics_latest(stub_client):
    client, _ = stub_client
    data = client.get("/api/v1/metrics/latest").get_json()
    assert data["step"]["step"] == 50
    assert data["epoch"]["epoch"] == 2
    assert data["status"] == "running"


def test_status_schema(stub_client):
    client, collector = stub_client
    data = client.get("/api/v1/status").get_json()
    for key in ("status", "experiment_name", "phase", "epoch", "total_epochs",
                "global_step", "uptime_s", "eta_s", "current_lr", "pid",
                "model_class", "trainer_version", "last_update"):
        assert key in data, f"missing key: {key}"
    assert data["status"] == "running"
    assert data["experiment_name"] == "stub"
    assert data["total_epochs"] == 10
    assert data["current_lr"] == 0.5
    assert data["phase"]["name"] in ("float_warmup", "qat")

    collector.mark_finished()
    assert client.get("/api/v1/status").get_json()["status"] == "finished"


def test_config_endpoint(stub_client):
    client, _ = stub_client
    data = client.get("/api/v1/config").get_json()
    assert data["experiment_name"] == "stub"
    assert data["epochs"] == 10
    assert "quant_schedule" in data


# ---------------------------------------------------------------------------
# End-to-end: live server inside a real training run
# ---------------------------------------------------------------------------

def test_v1_trainer_live_server(dummy_loader):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainerConfig(
            experiment_name="api_e2e_v1",
            output_dir=tmpdir,
            epochs=2,
            dry_run=True,
            api_port=0,  # OS-assigned free port
            logging=LoggingConfig(log_every_n_steps=1, save_plots=False),
            quant_schedule=QuantScheduleConfig(float_warmup_epochs=1),
        )
        model = TinyQuantNet()
        trainer = Trainer(
            config=config,
            model=model,
            optimizer=optim.Adam(model.parameters(), lr=1e-3),
            train_loader=dummy_loader,
            val_loader=dummy_loader,
        )
        assert trainer.api_server is not None
        port = trainer.api_server.port
        assert port

        # API must respond before training starts
        assert _get_json(port, "/api/v1/health") == {"ok": True}
        assert _get_json(port, "/api/v1/status")["status"] == "running"

        trainer.fit()

        status = _get_json(port, "/api/v1/status")
        assert status["status"] == "finished"
        assert status["trainer_version"] == "v1"
        assert status["epochs_completed"] == 2
        assert status["phase"]["name"] == "qat"  # warmup was 1 of 2 epochs
        assert status["phase"]["quantizers"]["total"] >= 1

        metrics = _get_json(port, "/api/v1/metrics")
        assert len(metrics["epochs"]) == 2
        assert len(metrics["steps"]) == 4  # 2 epochs x 2 dry-run batches
        assert all("train_loss" in e for e in metrics["epochs"])
        assert all("val_acc" in e for e in metrics["epochs"])

        ckpts = _get_json(port, "/api/v1/checkpoints")
        assert ckpts["monitor_metric"] == "val_loss"
        assert len(ckpts["checkpoints"]) >= 1
        assert {"epoch", "metric_value", "path", "mtime"} <= set(ckpts["checkpoints"][0])

        # JSONL persistence next to the CSV log. Step/epoch records are
        # always present; audit `event` records (checkpoint saves, phase
        # changes, finish) are also written once the control layer exists.
        jsonl = f"{trainer.logger.run_dir}/api_metrics.jsonl"
        with open(jsonl) as f:
            lines = [json.loads(l) for l in f]
        types = {l["type"] for l in lines}
        assert {"step", "epoch"} <= types

        trainer.api_server.shutdown()


def test_v2_trainer_live_server(dummy_loader):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainerConfigV2(
            experiment_name="api_e2e_v2",
            output_dir=tmpdir,
            epochs=2,
            dry_run=True,
            api_port=0,
            num_classes=10,
            smoothing=0.0,  # avoid the timm import path; plain CE is enough here
            logging=LoggingConfig(log_every_n_steps=1, save_plots=False),
            qat=QATScheduleConfigV2(float_warmup_epochs=1),
        )
        model = TinyQuantNet()
        trainer = QATTrainerV2(
            config=config,
            model=model,
            optimizer=optim.Adam(model.parameters(), lr=1e-3),
            train_loader=dummy_loader,
            val_loader=dummy_loader,
        )
        port = trainer.api_server.port
        trainer.fit()

        status = _get_json(port, "/api/v1/status")
        assert status["status"] == "finished"
        assert status["trainer_version"] == "v2"
        assert status["phase"]["name"] == "qat"
        assert status["phase"]["quantizers"]["total"] >= 1

        config_data = _get_json(port, "/api/v1/config")
        assert config_data["trainer_class"] == "QATTrainerV2"
        assert "qat" in config_data

        trainer.api_server.shutdown()
