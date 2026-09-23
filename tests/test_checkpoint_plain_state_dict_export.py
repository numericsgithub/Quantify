"""CheckpointManager also saves a plain `torch.save(model.state_dict(), path)`
companion file alongside every checkpoint -- no training_harness wrapper
dict, and no Brevitas/Quantify quantizer bookkeeping either -- so the file
is exactly what a non-quantized equivalent model's `state_dict()` would
have produced, loadable with just `model.load_state_dict(torch.load(path))`
in any vanilla PyTorch script that knows nothing about this repo.
"""
import brevitas.nn as qnn
import torch
import torch.nn as nn

from quantizers import FixedPointPerTensorWeightQuant
from training_harness.checkpointing import (
    CheckpointManager, export_plain_state_dict, strip_quantizer_state,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _new_manager(tmp_path, **kwargs):
    return CheckpointManager(save_dir=str(tmp_path / "checkpoints"), experiment_name="exp", **kwargs)


def _assert_plain_and_matches(path, model):
    loaded = torch.load(path, map_location="cpu")
    # a plain state_dict, NOT this repo's {"model_state_dict": ..., "epoch": ...} wrapper
    assert "model_state_dict" not in loaded
    assert set(loaded.keys()) == set(model.state_dict().keys())
    for k, v in model.state_dict().items():
        torch.testing.assert_close(loaded[k], v)

    # loadable by a completely fresh, vanilla model with no knowledge of this repo
    fresh = TinyModel()
    fresh.load_state_dict(torch.load(path))


def test_last_checkpoint_gets_plain_companion(tmp_path):
    model = TinyModel()
    mgr = _new_manager(tmp_path, save_last=True, top_k=0)
    mgr.save(epoch=0, metric_value=1.0, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))

    last_pt = tmp_path / "checkpoints" / "last.pt"
    plain_pt = tmp_path / "checkpoints" / "last_state_dict.pt"
    assert last_pt.exists()
    assert plain_pt.exists()

    # the harness's own file DOES have the wrapper
    wrapped = torch.load(last_pt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in wrapped

    _assert_plain_and_matches(plain_pt, model)


def test_periodic_checkpoint_gets_plain_companion(tmp_path):
    model = TinyModel()
    mgr = _new_manager(tmp_path, save_last=False, top_k=0, save_every_n_epochs=1)
    mgr.save(epoch=0, metric_value=1.0, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))

    periodic_pt = tmp_path / "checkpoints" / "exp_epoch0000.pt"
    plain_pt = tmp_path / "checkpoints" / "exp_epoch0000_state_dict.pt"
    assert periodic_pt.exists()
    _assert_plain_and_matches(plain_pt, model)


def test_topk_checkpoint_gets_plain_companion_and_is_evicted_together(tmp_path):
    model = TinyModel()
    mgr = _new_manager(tmp_path, save_last=False, top_k=1, monitor_mode="min")

    mgr.save(epoch=0, metric_value=2.0, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    first_pt = tmp_path / "checkpoints" / "exp_epoch0000_metric2.000000.pt"
    first_plain = tmp_path / "checkpoints" / "exp_epoch0000_metric2.000000_state_dict.pt"
    assert first_pt.exists() and first_plain.exists()

    # a better (lower) metric evicts the first checkpoint since top_k=1
    mgr.save(epoch=1, metric_value=1.0, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    second_pt = tmp_path / "checkpoints" / "exp_epoch0001_metric1.000000.pt"
    second_plain = tmp_path / "checkpoints" / "exp_epoch0001_metric1.000000_state_dict.pt"
    assert second_pt.exists() and second_plain.exists()

    # the evicted checkpoint's plain companion must be cleaned up too, not orphaned
    assert not first_pt.exists()
    assert not first_plain.exists()


def test_export_plain_state_dict_converts_an_existing_checkpoint(tmp_path):
    model = TinyModel()
    mgr = _new_manager(tmp_path, save_last=True, top_k=0)
    mgr.save(epoch=0, metric_value=1.0, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))

    last_pt = str(tmp_path / "checkpoints" / "last.pt")
    out = str(tmp_path / "my_plain_export.pt")
    returned_path = export_plain_state_dict(last_pt, out)

    assert returned_path == out
    _assert_plain_and_matches(out, model)


def test_top_k_zero_disables_pool_without_crashing(tmp_path):
    """Regression test: top_k=0 used to crash with IndexError in
    _should_save (0 < 0 is False, so it always fell through to
    self._records[-1] on an empty list)."""
    model = TinyModel()
    mgr = _new_manager(tmp_path, save_last=False, top_k=0)
    mgr.save(epoch=0, metric_value=1.0, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    mgr.save(epoch=1, metric_value=0.5, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    assert mgr._records == []


def test_export_plain_state_dict_default_output_path(tmp_path):
    model = TinyModel()
    mgr = _new_manager(tmp_path, save_last=True, top_k=0)
    mgr.save(epoch=0, metric_value=1.0, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))

    last_pt = str(tmp_path / "checkpoints" / "last.pt")
    returned_path = export_plain_state_dict(last_pt)

    assert returned_path == str(tmp_path / "checkpoints" / "last_state_dict.pt")
    _assert_plain_and_matches(returned_path, model)


class TinyQuantModel(nn.Module):
    """Mirrors TinyModel's shape (Linear(4, 2)) but as a Brevitas
    QuantLinear with a Quantify fixed-point weight quantizer, so its
    state_dict has extra quantizer bookkeeping a plain nn.Linear doesn't."""

    def __init__(self):
        super().__init__()
        self.fc = qnn.QuantLinear(4, 2, weight_quant=FixedPointPerTensorWeightQuant)

    def forward(self, x):
        return self.fc(x)


def _calibrated_quant_model():
    model = TinyQuantModel()
    model.train()
    with torch.no_grad():
        model(torch.randn(4, 4))
    model.eval()
    return model


def test_strip_quantizer_state_removes_only_quantizer_proxy_keys():
    import collections

    model = _calibrated_quant_model()
    full = model.state_dict()
    stripped = strip_quantizer_state(full)

    assert type(stripped) is collections.OrderedDict, (
        "must match the exact container type nn.Module.state_dict() returns"
    )
    assert "fc.weight" in stripped
    assert "fc.bias" in stripped
    quant_keys = [k for k in full if k not in stripped]
    assert quant_keys, "expected some quantizer keys to be stripped"
    assert all("_quant" in k for k in quant_keys)
    assert not any("_quant" in k for k in stripped)

    # equivalent to what a plain (non-quantized) TinyModel would have --
    # same tensors, no quantizer bookkeeping
    plain_equivalent = TinyModel().state_dict()
    assert set(stripped.keys()) == set(plain_equivalent.keys())


def test_checkpoint_manager_auto_export_strips_quantizer_state(tmp_path):
    model = _calibrated_quant_model()
    mgr = _new_manager(tmp_path, save_last=True, top_k=0)
    mgr.save(epoch=0, metric_value=1.0, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))

    plain_pt = tmp_path / "checkpoints" / "last_state_dict.pt"
    loaded = torch.load(plain_pt, map_location="cpu")
    assert not any("_quant" in k for k in loaded.keys())
    assert set(loaded.keys()) == {"fc.weight", "fc.bias"}

    # loadable into a genuinely plain (non-quantized) equivalent model
    TinyModel().load_state_dict(loaded)


def test_export_plain_state_dict_function_strips_quantizer_state(tmp_path):
    model = _calibrated_quant_model()
    mgr = _new_manager(tmp_path, save_last=True, top_k=0)
    mgr.save(epoch=0, metric_value=1.0, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))

    last_pt = str(tmp_path / "checkpoints" / "last.pt")
    out = export_plain_state_dict(last_pt, str(tmp_path / "converted.pt"))
    loaded = torch.load(out, map_location="cpu")
    assert set(loaded.keys()) == {"fc.weight", "fc.bias"}


class _PlainConvBNNet(nn.Module):
    """A genuinely plain (no Brevitas/Quantify at all) equivalent of
    _QuantConvBNNet below -- same layer names/shapes, ordinary nn.Conv2d."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.fc(self.flatten(self.pool(x)))


class _QuantConvBNNet(nn.Module):
    """Same architecture as _PlainConvBNNet, but conv1/conv2 are Brevitas
    QuantConv2d with Quantify's fixed-point weight quantizer -- mirrors
    examples/basics/importance_mnist.py::ImportanceMNISTNet."""

    def __init__(self):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(1, 8, 3, padding=1, bias=False, weight_quant=FixedPointPerTensorWeightQuant)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = qnn.QuantConv2d(8, 16, 3, padding=1, stride=2, bias=False, weight_quant=FixedPointPerTensorWeightQuant)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.fc(self.flatten(self.pool(x)))


def test_qat_export_is_structurally_identical_to_plain_pytorch_save(tmp_path):
    """End-to-end regression test for the exact scenario walked through
    manually: train/save a QAT (Brevitas+Quantify) model through
    CheckpointManager, save an architecturally-equivalent plain model the
    textbook way (`torch.save(model.state_dict(), path)`), and assert the
    two on-disk files are indistinguishable in structure -- same container
    type, same key set, same shapes, and the QAT export loads straight into
    the plain model with `strict=True`.
    """
    torch.manual_seed(0)
    quant_model = _QuantConvBNNet()
    quant_model.train()
    with torch.no_grad():
        quant_model(torch.randn(4, 1, 28, 28))  # calibrate
    quant_model.eval()

    mgr = _new_manager(tmp_path, save_last=True, top_k=0)
    mgr.save(epoch=0, metric_value=1.0, model=quant_model,
              optimizer=torch.optim.SGD(quant_model.parameters(), lr=0.1))
    qat_export_path = tmp_path / "checkpoints" / "last_state_dict.pt"

    plain_model = _PlainConvBNNet()
    plain_export_path = tmp_path / "plain_model_state_dict.pt"
    torch.save(plain_model.state_dict(), plain_export_path)  # the textbook way

    qat_loaded = torch.load(qat_export_path, map_location="cpu")
    plain_loaded = torch.load(plain_export_path, map_location="cpu")

    assert type(qat_loaded) is type(plain_loaded)
    assert set(qat_loaded.keys()) == set(plain_loaded.keys())
    for key in plain_loaded:
        assert qat_loaded[key].shape == plain_loaded[key].shape, key
    assert not any("_quant" in k for k in qat_loaded.keys())

    # the ultimate proof: load the QAT export straight into an independently
    # constructed plain model with strict=True (no missing/unexpected keys)
    fresh_plain = _PlainConvBNNet()
    fresh_plain.load_state_dict(qat_loaded, strict=True)
