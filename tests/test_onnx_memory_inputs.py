import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

from tests.onnx_run import run_onnx
from utils.onnx_export import export_onnx_with_io, load_memory_inputs


class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.fx = nn.Linear(4, 4)
        self.fh = nn.Linear(4, 4)

    def forward(self, x, h):
        return torch.tanh(self.fx(x) + self.fh(h))


def _export(tmp_path, **kw):
    path = str(tmp_path / "m.onnx")
    m = export_onnx_with_io(
        Cell(), (torch.randn(1, 4), torch.zeros(1, 4)), path,
        input_names=["x", "h_in"], output_names=["h_out"],
        canonical_checks="off", **kw,
    )
    return path, m


def test_memory_marking_roundtrip_and_execution_unchanged(tmp_path):
    path, m = _export(tmp_path, memory_inputs={"h_in": "cell.h_out"})
    assert load_memory_inputs(path) == {"h_in": "cell.h_out"}
    assert [i.name for i in m.graph.input] == ["x", "h_in"]
    out = run_onnx(path, {"x": np.ones((1, 4)), "h_in": np.zeros((1, 4))})
    assert out[0].shape == (1, 4)


def test_no_marking_is_empty(tmp_path):
    path, _ = _export(tmp_path)
    assert load_memory_inputs(path) == {}


def test_unknown_memory_input_rejected(tmp_path):
    with pytest.raises(ValueError, match="not found in input_names"):
        _export(tmp_path, memory_inputs={"nope": "x"})
