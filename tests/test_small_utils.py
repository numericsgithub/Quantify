"""Unit tests for the small helper modules: EMAModel, CSVLogger, model_info."""
import csv

import pytest
import torch
import torch.nn as nn

from training_harness.ema import EMAModel
from utils.csv_logger import CSVLogger
from utils.model_info import count_parameters, summarize_parameters


class _TinyBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x):
        return self.bn(self.fc(x))


class TestEMAModel:
    def test_shadow_is_frozen_eval_copy(self):
        model = _TinyBN()
        ema = EMAModel(model, decay=0.9)
        assert not ema._shadow.training
        assert all(not p.requires_grad for p in ema._shadow.parameters())
        # a copy, not an alias
        assert ema._shadow.fc.weight.data_ptr() != model.fc.weight.data_ptr()

    def test_update_blends_parameters_with_decay(self):
        model = _TinyBN()
        ema = EMAModel(model, decay=0.75)
        before = ema._shadow.fc.weight.clone()
        with torch.no_grad():
            model.fc.weight.add_(1.0)
        ema.update(model)
        expected = 0.75 * before + 0.25 * model.fc.weight
        assert torch.allclose(ema._shadow.fc.weight, expected)

    def test_update_copies_buffers_directly(self):
        model = _TinyBN()
        ema = EMAModel(model, decay=0.99)
        model.train()
        model(torch.randn(8, 4))  # moves BN running stats
        ema.update(model)
        assert torch.equal(ema._shadow.bn.running_mean, model.bn.running_mean)
        assert torch.equal(ema._shadow.bn.num_batches_tracked, model.bn.num_batches_tracked)

    def test_apply_and_restore_round_trip_leaves_buffers_alone(self):
        model = _TinyBN()
        ema = EMAModel(model, decay=0.5)
        with torch.no_grad():
            model.fc.weight.add_(2.0)
        ema.update(model)

        trained = model.fc.weight.clone()
        buffers_before = [b.clone() for b in model.buffers()]
        stash = ema.apply_to(model)
        assert torch.allclose(model.fc.weight, ema._shadow.fc.weight)
        assert not torch.allclose(model.fc.weight, trained)
        for b, saved in zip(model.buffers(), buffers_before):
            assert torch.equal(b, saved)

        ema.restore(model, stash)
        assert torch.equal(model.fc.weight, trained)

    def test_state_dict_round_trip(self):
        model = _TinyBN()
        ema = EMAModel(model)
        other = EMAModel(_TinyBN())
        other.load_state_dict(ema.state_dict())
        for a, b in zip(ema.state_dict().values(), other.state_dict().values()):
            assert torch.equal(a, b)

    def test_to_returns_self(self):
        ema = EMAModel(_TinyBN())
        assert ema.to("cpu") is ema


class TestCSVLogger:
    def test_header_rows_and_flush_before_close(self, tmp_path):
        path = tmp_path / "nested" / "log.csv"
        log = CSVLogger(path, ["epoch", "loss"])
        log.log(epoch=1, loss=0.5)
        # flushed per call: readable while the logger is still open
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert rows == [{"epoch": "1", "loss": "0.5"}]
        log.close()
        log.close()  # idempotent

    def test_context_manager_closes_file(self, tmp_path):
        with CSVLogger(tmp_path / "log.csv", ["a"]) as log:
            log.log(a=1)
        assert log._file.closed

    def test_unknown_field_is_rejected(self, tmp_path):
        with CSVLogger(tmp_path / "log.csv", ["a"]) as log:
            with pytest.raises(ValueError):
                log.log(b=1)


class TestModelInfo:
    def test_count_parameters_total_and_trainable(self):
        model = _TinyBN()
        total, trainable = count_parameters(model)
        assert total == 4 * 3 + 3 + 3 + 3  # fc weight+bias, bn weight+bias
        assert trainable == total
        for p in model.fc.parameters():
            p.requires_grad_(False)
        total2, trainable2 = count_parameters(model)
        assert total2 == total
        assert trainable2 == 6

    def test_summarize_prints_rows_and_totals(self, capsys):
        summarize_parameters(_TinyBN())
        out = capsys.readouterr().out
        assert "fc" in out and "Linear" in out
        assert "bn" in out and "BatchNorm1d" in out
        assert "Total" in out
        assert "Trainable" not in out  # all trainable -> line omitted

    def test_summarize_shows_trainable_line_when_frozen(self, capsys):
        model = _TinyBN()
        for p in model.fc.parameters():
            p.requires_grad_(False)
        summarize_parameters(model)
        assert "Trainable" in capsys.readouterr().out

    def test_summarize_parameterless_model(self, capsys):
        summarize_parameters(nn.Sequential(nn.ReLU()))
        assert "no parameters" in capsys.readouterr().out


class TestWorkspace:
    def test_at_accepts_str_root_and_creates_dirs(self, tmp_path):
        from utils.workspace import Workspace

        ws = Workspace.at(str(tmp_path), "run1")
        assert ws.root == (tmp_path / "run1").resolve()
        for d in (ws.root, ws.data, ws.checkpoints, ws.logs, ws.datasets):
            assert d.is_dir()
        # datasets are shared across runs, the rest is per run
        assert ws.datasets == tmp_path.resolve() / "cached_datasets"

    def test_workspace_from_args_uses_env_root(self, tmp_path, monkeypatch):
        import argparse

        from utils.workspace import add_workspace_args, workspace_from_args

        monkeypatch.setenv("QUANT_WORKDIR", str(tmp_path))
        parser = argparse.ArgumentParser()
        add_workspace_args(parser, name="exp")
        ws = workspace_from_args(parser.parse_args([]))
        assert ws.root == (tmp_path / "exp").resolve()

    def test_workspace_from_args_missing_env_has_helpful_error(self, monkeypatch):
        import argparse

        from utils.workspace import workspace_from_args

        monkeypatch.delenv("QUANT_WORKDIR", raising=False)
        with pytest.raises(Exception, match="QUANT_WORKDIR"):
            workspace_from_args(argparse.Namespace(workdir="exp"))

    def test_workdir_with_path_separator_is_rejected(self, tmp_path, monkeypatch):
        import argparse

        from utils.workspace import workspace_from_args

        monkeypatch.setenv("QUANT_WORKDIR", str(tmp_path))
        with pytest.raises(Exception, match="simple folder name"):
            workspace_from_args(argparse.Namespace(workdir="a/b"))
