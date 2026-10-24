"""
Tests for `quantizers/layers.py::QuantAvgPool{1,2}d` / `QuantGlobalAvgPool{1,2}d`:
average pooling implemented as a depthwise QuantConv with frozen weights
(1/prod(kernel)) and a frozen zero bias; weights 12 bit and output 10 bit by
default; exported as a plain depthwise Conv.
"""

import copy
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper

from quantizers import FixedPointPerTensorWeightQuant, QuantReLU
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer
from quantizers.layers import (
    QuantAvgPool1d,
    QuantAvgPool2d,
    QuantConv1d,
    QuantGlobalAvgPool1d,
    QuantGlobalAvgPool2d,
)
from quantizers.manager import QuantizerManager
from tests.onnx_run import run_onnx
from utils.onnx_export import export_onnx_with_io

torch.manual_seed(0)


def _fresh():
    QuantizerManager().reset()


def _quants(m):
    """(weight, bias, output) quantizers of a pool layer, by role in conv."""
    c = m.conv
    pick = lambda root: next(q for q in root.modules() if isinstance(q, FixedPointPerTensorQuantizer))
    return pick(c.weight_quant), pick(c.bias_quant), pick(c.output_quant)


def _calibrated(m, x):
    m.train()
    m(x)
    m.eval()
    return m


def _export(model, x):
    f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    f.close()
    try:
        onnx_model = export_onnx_with_io(model, x, f.name, input_names=["x"])
        out = run_onnx(f.name, {"x": x.numpy()})[0]
    finally:
        os.remove(f.name)
    return onnx_model, out


class TestConstruction:
    def test_is_a_frozen_depthwise_conv(self):
        _fresh()
        m = QuantAvgPool2d(5, 3)
        c = m.conv
        assert c.groups == 5 and c.in_channels == c.out_channels == 5
        assert tuple(c.kernel_size) == (3, 3)
        assert torch.all(c.weight == 1.0 / 9) and torch.all(c.bias == 0)
        assert not c.weight.requires_grad and not c.bias.requires_grad

    def test_stride_defaults_to_kernel_like_nn_avgpool(self):
        _fresh()
        assert tuple(QuantAvgPool1d(2, 4).conv.stride) == (4,)
        assert tuple(QuantAvgPool2d(2, (2, 3)).conv.stride) == (2, 3)
        assert tuple(QuantAvgPool1d(2, 4, stride=1).conv.stride) == (1,)

    def test_default_bit_widths_are_12_weights_10_output(self):
        _fresh()
        w, b, o = _quants(QuantAvgPool2d(3, 2))
        assert (w.bit_width, o.bit_width) == (12, 10)

    def test_bit_widths_are_overridable(self):
        _fresh()
        w, _, o = _quants(QuantAvgPool2d(3, 2, weight_bit_width=8, output_bit_width=6))
        assert (w.bit_width, o.bit_width) == (8, 6)

    def test_parameters_are_not_trainable(self):
        _fresh()
        m = QuantAvgPool2d(3, 2)
        assert [p for p in m.parameters() if p.requires_grad] == []


class TestNumerics:
    @pytest.mark.parametrize("kernel", [2, 3, 4])
    def test_matches_float_avg_pool_2d(self, kernel):
        _fresh()
        x = torch.rand(4, 3, 12, 12) * 4
        m = _calibrated(QuantAvgPool2d(3, kernel), x)
        ref = F.avg_pool2d(x, kernel)
        with torch.no_grad():
            out = m(x)
        # 12-bit weight rounding (<0.1%) + 10-bit output grid
        assert out.shape == ref.shape
        assert (out - ref).abs().max() < 0.02 * ref.abs().max() + 0.02

    @pytest.mark.parametrize("kernel,stride", [(2, 2), (3, 1), (4, 2)])
    def test_matches_float_avg_pool_1d(self, kernel, stride):
        _fresh()
        x = torch.rand(4, 3, 40) * 4
        m = _calibrated(QuantAvgPool1d(3, kernel, stride), x)
        ref = F.avg_pool1d(x, kernel, stride)
        with torch.no_grad():
            out = m(x)
        assert out.shape == ref.shape
        assert (out - ref).abs().max() < 0.02 * ref.abs().max() + 0.02

    def test_signed_inputs(self):
        _fresh()
        x = torch.randn(4, 3, 16)
        m = _calibrated(QuantAvgPool1d(3, 4), x)
        with torch.no_grad():
            assert (m(x) - F.avg_pool1d(x, 4)).abs().max() < 0.05

    def test_non_square_kernel(self):
        _fresh()
        x = torch.rand(2, 3, 8, 12)
        m = _calibrated(QuantAvgPool2d(3, (2, 3)), x)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 3, 4, 4)
        assert (out - F.avg_pool2d(x, (2, 3))).abs().max() < 0.05

    def test_channels_are_independent(self):
        _fresh()
        x = torch.zeros(1, 3, 8)
        x[:, 1] = 5.0
        m = _calibrated(QuantAvgPool1d(3, 4), x)
        with torch.no_grad():
            out = m(x)
        assert out[:, 0].abs().max() < 1e-3 and out[:, 2].abs().max() < 1e-3
        assert (out[:, 1] - 5.0).abs().max() < 0.1

    def test_padding_same_stride_2_keeps_ceil_shape(self):
        _fresh()
        x = torch.rand(2, 3, 9)
        m = _calibrated(QuantAvgPool1d(3, 3, stride=2, padding="same"), x)
        with torch.no_grad():
            assert m(x).shape == (2, 3, 5)


class TestGlobalAvgPool:
    def test_1d_shape_and_value(self):
        _fresh()
        x = torch.relu(torch.randn(6, 4, 25)) * 3
        m = _calibrated(QuantGlobalAvgPool1d(4, 25), x)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (6, 4, 1)
        assert (out.squeeze(-1) - x.mean(-1)).abs().max() < 0.05

    def test_2d_shape_and_value(self):
        _fresh()
        x = torch.rand(3, 5, 6, 4)
        m = _calibrated(QuantGlobalAvgPool2d(5, (6, 4)), x)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (3, 5, 1, 1)
        assert (out.flatten(1) - x.mean((2, 3))).abs().max() < 0.05

    def test_size_mismatch_raises_clear_error(self):
        _fresh()
        m = QuantGlobalAvgPool1d(4, 25)
        with pytest.raises(ValueError, match="spatial size"):
            m(torch.rand(2, 4, 24))


class TestFrozen:
    def test_optimizer_step_leaves_weights_alone_but_input_gets_grad(self):
        _fresh()
        m = QuantAvgPool1d(3, 4).train()
        x = torch.rand(2, 3, 16, requires_grad=True)
        opt = torch.optim.SGD(m.parameters(), lr=1.0, weight_decay=0.5)
        m(x)  # calibrate
        before = [p.clone() for p in m.parameters()]
        m(x).sum().backward()
        opt.step()
        assert x.grad is not None and x.grad.abs().sum() > 0
        assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))

    def test_check_frozen_detects_tampering_and_reset_repairs(self):
        _fresh()
        m = QuantAvgPool2d(2, 3)
        m.check_frozen()
        with torch.no_grad():
            m.conv.weight.normal_()
        with pytest.raises(RuntimeError, match="modified"):
            m.check_frozen()
        m.reset_parameters()
        m.check_frozen()
        m.conv.weight.requires_grad_(True)
        with pytest.raises(RuntimeError, match="frozen"):
            m.check_frozen()

    def test_reset_parameters_after_external_reinit(self):
        _fresh()
        m = QuantAvgPool2d(2, 3)
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight)
        m.reset_parameters()
        assert torch.all(m.conv.weight == 1 / 9)


class TestCalibration:
    def test_constant_weights_and_zero_bias_calibrate_once(self):
        """A one-valued tensor must count as calibrated; otherwise eval raises
        'has not been calibrated' (or recalibrates every forward)."""
        _fresh()
        m = QuantAvgPool2d(3, 2)
        m.train()
        m(torch.rand(2, 3, 8, 8))
        assert all(q.search_done_value for q in _quants(m))

    def test_eval_after_training_forward_does_not_raise(self):
        _fresh()
        x = torch.rand(2, 3, 8, 8)
        m = _calibrated(QuantAvgPool2d(3, 2), x)
        with torch.no_grad():
            m(x)

    def test_weight_grid_represents_one_over_n(self):
        _fresh()
        m = _calibrated(QuantAvgPool2d(3, 3), torch.rand(2, 3, 9, 9))
        w, _, _ = _quants(m)
        step = 2.0 ** w._lsb_value
        q = torch.round(torch.tensor(1 / 9) / step) * step
        assert abs(q.item() - 1 / 9) / (1 / 9) < 1e-3  # <0.1% gain error

    @pytest.mark.parametrize("n", [2, 3, 5, 7, 25, 64, 200])
    def test_weight_gain_error_small_for_many_kernel_sizes(self, n):
        _fresh()
        x = torch.rand(1, 2, n)
        m = _calibrated(QuantGlobalAvgPool1d(2, n), x)
        with torch.no_grad():
            out = m(x)
        # gain error of the 12-bit 1/n weight stays under 0.1% of the mean
        assert (out.squeeze(-1) - x.mean(-1)).abs().max() < 0.01 + 1e-3 * x.mean()

    def test_state_dict_roundtrip(self):
        _fresh()
        x = torch.rand(2, 3, 8, 8)
        a = _calibrated(QuantAvgPool2d(3, 2), x)
        b = QuantAvgPool2d(3, 2)
        b.load_state_dict(a.state_dict())
        b.eval()
        with torch.no_grad():
            assert torch.equal(a(x), b(x))
        b.check_frozen()

    def test_deepcopy(self):
        _fresh()
        x = torch.rand(2, 3, 8, 8)
        a = _calibrated(QuantAvgPool2d(3, 2), x)
        c = copy.deepcopy(a)
        with torch.no_grad():
            assert torch.equal(a(x), c(x))


class TestOnnxExport:
    def test_exports_as_one_depthwise_conv_no_averagepool(self):
        _fresh()
        x = torch.rand(1, 4, 25)
        m = nn.Sequential(QuantGlobalAvgPool1d(4, 25))
        with torch.no_grad():
            _calibrated(m, x)
            eager = m(x).numpy()
        onnx_model, out = _export(m, x)
        ops = [(n.domain, n.op_type) for n in onnx_model.graph.node]
        assert ("", "Conv") in ops
        assert not any("Pool" in op for _, op in ops)
        assert [op for d, op in ops if d == ""] == ["Conv"]
        conv = next(n for n in onnx_model.graph.node if n.op_type == "Conv")
        attrs = {a.name: a for a in conv.attribute}
        assert attrs["group"].i == 4
        assert list(attrs["kernel_shape"].ints) == [25]
        np.testing.assert_allclose(out, eager, atol=1e-5)

    def test_weight_initializer_is_all_one_value_and_bias_zero(self):
        _fresh()
        x = torch.rand(1, 3, 6, 6)
        m = nn.Sequential(QuantAvgPool2d(3, 2))
        with torch.no_grad():
            _calibrated(m, x)
        onnx_model, _ = _export(m, x)
        inits = {i.name: numpy_helper.to_array(i) for i in onnx_model.graph.initializer}
        w = next(v for v in inits.values() if v.shape == (3, 1, 2, 2))
        b = next(v for v in inits.values() if v.shape == (3,))
        assert np.unique(w).tolist() == [np.float32(0.25)]
        assert not b.any()

    def test_conv_pads_are_valid_so_canonical_checks_pass(self):
        _fresh()
        x = torch.rand(1, 3, 8)
        m = nn.Sequential(QuantAvgPool1d(3, 2))
        with torch.no_grad():
            _calibrated(m, x)
        onnx_model, _ = _export(m, x)  # canonical_checks="raise" by default
        conv = next(n for n in onnx_model.graph.node if n.op_type == "Conv")
        pads = next(a for a in conv.attribute if a.name == "pads")
        assert list(pads.ints) == [0, 0]

    def test_same_padding_exports_as_single_conv(self):
        _fresh()
        x = torch.rand(1, 3, 9)
        m = nn.Sequential(QuantAvgPool1d(3, 3, stride=2, padding="same"))
        with torch.no_grad():
            _calibrated(m, x)
            eager = m(x).numpy()
        onnx_model, out = _export(m, x)
        assert [n.op_type for n in onnx_model.graph.node if n.domain == ""] == ["Conv"]
        np.testing.assert_allclose(out, eager, atol=1e-5)

    def test_in_a_network_with_a_conv_relu_and_dense(self):
        _fresh()
        net = nn.Sequential(
            QuantConv1d(4, 8, 3, padding="same", weight_quant=FixedPointPerTensorWeightQuant),
            QuantReLU(),
            QuantGlobalAvgPool1d(8, 25),
            nn.Flatten(),
            torch.nn.Linear(8, 5),
        )
        x = torch.randn(2, 4, 25)
        with torch.no_grad():
            net.train()
            net(x)
            net.eval()
            eager = net(x).numpy()
        onnx_model, out = _export(net, x)
        np.testing.assert_allclose(out, eager, atol=1e-4)
        assert not any("Pool" in n.op_type for n in onnx_model.graph.node)

    def test_live_model_still_works_after_export(self):
        _fresh()
        x = torch.rand(1, 3, 8)
        m = nn.Sequential(QuantAvgPool1d(3, 2))
        with torch.no_grad():
            _calibrated(m, x)
            before = m(x)
        _export(m, x)
        with torch.no_grad():
            assert torch.equal(m(x), before)
        m[0].check_frozen()
