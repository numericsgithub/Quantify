"""
Canonical-graph rules enforced by `utils/onnx_export.py::export_onnx_with_io`
(see `utils/onnx_canonical.py`):

1. every Linear -> exactly one `Quantify::Dense(x, W[out,in], b)`, any input
   rank, with or without bias, no runtime Transpose, live model untouched;
2. Conv/ConvTranspose/MaxPool/AveragePool pads are VALID or exactly TF-SAME
   (`quantizers/layers.py` provides `padding="same"` at any stride);
3. the graph input feeds straight into the first compute layer (only a
   FixedPointQuant may sit in between);
4. activations are Quantify custom nodes; a plain activation with a Quantify
   equivalent is rejected unless it produces a graph output.
"""

import os
import tempfile
import warnings

import brevitas.nn as qnn
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper

from quantizers import (
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorWeightQuant,
    QuantGELU,
    QuantHardSigmoid,
    QuantHardSwish,
    QuantLeakyReLU,
    QuantReLU,
    QuantReLU6,
    QuantSigmoid,
    QuantSiLU,
    QuantSoftmax,
    QuantTanh,
)
from quantizers.layers import (
    MaxPool1d,
    MaxPool2d,
    QuantConv1d,
    QuantConv2d,
    same_pads,
)
from tests.onnx_run import run_onnx
from utils.onnx_canonical import NonCanonicalONNXError, check_onnx_graph
from utils.onnx_export import export_onnx_with_io

torch.manual_seed(0)

WQ = dict(weight_quant=FixedPointPerTensorWeightQuant)


def _calibrate(model, x):
    model.train()
    model(x)
    model.eval()
    with torch.no_grad():
        return model(x)


def _export(model, x, **kwargs):
    f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    f.close()
    try:
        onnx_model = export_onnx_with_io(model, x, f.name, input_names=["x"], **kwargs)
        out = run_onnx(f.name, {"x": x.numpy()})[0]
    finally:
        os.remove(f.name)
    return onnx_model, out


def _ops(onnx_model):
    return [(n.domain, n.op_type) for n in onnx_model.graph.node]


def _messages(excinfo):
    return str(excinfo.value)


# =========================================================================
# 1. Dense
# =========================================================================


class TestDense:
    @pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
    @pytest.mark.parametrize("shape", [(3, 8), (2, 5, 8)], ids=["2d", "3d"])
    def test_quant_linear_exports_as_one_dense_node(self, shape, bias):
        lin = qnn.QuantLinear(8, 4, bias=bias, **WQ)
        model = nn.Sequential(lin)
        x = torch.randn(*shape)
        eager = _calibrate(model, x)

        onnx_model, out = _export(model, x)

        assert _ops(onnx_model) == [("Quantify", "FixedPointQuant"), ("Quantify", "Dense")]
        g = onnx_model.graph
        dense = g.node[1]
        inits = {i.name: i for i in g.initializer}
        quant_w = g.node[0]
        assert dense.input[1] == quant_w.output[0]
        assert list(inits[quant_w.input[0]].dims) == [4, 8], "weight must stay in [out, in] layout"
        assert dense.input[2] in inits and list(inits[dense.input[2]].dims) == [4]
        if not bias:
            assert not onnx.numpy_helper.to_array(inits[dense.input[2]]).any()
        assert np.allclose(out, eager.numpy(), atol=1e-5)

    @pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
    @pytest.mark.parametrize("shape", [(3, 8), (2, 5, 8)], ids=["2d", "3d"])
    def test_plain_linear_exports_as_one_dense_node(self, shape, bias):
        model = nn.Sequential(nn.Linear(8, 4, bias=bias))
        x = torch.randn(*shape)
        eager = _calibrate(model, x)
        onnx_model, out = _export(model, x)
        assert _ops(onnx_model) == [("Quantify", "Dense")]
        assert np.allclose(out, eager.numpy(), atol=1e-5)

    def test_live_model_is_not_modified(self):
        lin = qnn.QuantLinear(8, 4, bias=False, **WQ)
        plain = nn.Linear(4, 2, bias=False)
        model = nn.Sequential(lin, plain)
        x = torch.randn(3, 8)
        _calibrate(model, x)
        sd_keys = set(model.state_dict())
        _export(model, x)
        assert lin.bias is None and plain.bias is None
        assert set(model.state_dict()) == sd_keys
        assert "inner_forward_impl" not in lin.__dict__ and "forward" not in plain.__dict__

    def test_linear_not_lowered_through_dense_is_flagged(self):
        """nn.MultiheadAttention calls F.linear on its own weights, bypassing
        any Linear module -- its projections must be reported."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = qnn.QuantLinear(8, 8, **WQ)
                self.attn = nn.MultiheadAttention(8, 2, batch_first=True)

            def forward(self, x):
                x = self.inp(x)
                return self.attn(x, x, x, need_weights=False)[0]

        model = M()
        x = torch.randn(1, 4, 8)
        _calibrate(model, x)
        with pytest.raises(NonCanonicalONNXError, match=r"\[dense\]"):
            _export(model, x)

    def test_graph_check_rejects_gemm_and_weight_transpose(self):
        w = helper.make_tensor("W", TensorProto.FLOAT, [4, 8], np.zeros(32, np.float32))
        graph = helper.make_graph(
            [helper.make_node("Transpose", ["W"], ["Wt"], perm=[1, 0]),
             helper.make_node("MatMul", ["x", "Wt"], ["h"]),
             helper.make_node("Gemm", ["h", "W2"], ["y"], transB=1)],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
            [w, helper.make_tensor("W2", TensorProto.FLOAT, [4, 4], np.zeros(16, np.float32))],
        )
        violations, _ = check_onnx_graph(helper.make_model(graph))
        text = "\n".join(violations)
        assert "runtime Transpose" in text
        assert "Gemm in the graph" in text


# =========================================================================
# 2. Padding
# =========================================================================


def _ort_same_upper_conv(x, w, b, stride, dilation):
    """Independent oracle: onnxruntime's own auto_pad=SAME_UPPER Conv. ORT
    doesn't support dilation with SAME_UPPER, so a dilated kernel is first
    expanded to the equivalent dense one (zeros between taps, dilation 1)."""
    if any(d > 1 for d in dilation):
        k = w.shape[2:]
        dense = torch.zeros(*w.shape[:2], *[(kk - 1) * d + 1 for kk, d in zip(k, dilation)])
        dense[(slice(None), slice(None)) + tuple(slice(None, None, d) for d in dilation)] = w
        w, dilation = dense, (1,) * len(dilation)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w", "b"], ["y"], auto_pad="SAME_UPPER",
                          strides=list(stride), dilations=list(dilation),
                          kernel_shape=list(w.shape[2:]))],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [onnx.numpy_helper.from_array(w.numpy(), "w"), onnx.numpy_helper.from_array(b.numpy(), "b")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return ort.InferenceSession(m.SerializeToString()).run(None, {"x": x.numpy()})[0]


def _ort_same_upper_pool(op, x, kernel, stride):
    attrs = dict(auto_pad="SAME_UPPER", kernel_shape=list(kernel), strides=list(stride))
    if op == "AveragePool":
        attrs["count_include_pad"] = 0
    graph = helper.make_graph(
        [helper.make_node(op, ["x"], ["y"], **attrs)], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return ort.InferenceSession(m.SerializeToString()).run(None, {"x": x.numpy()})[0]


CONV2D_CASES = [
    # (in_hw, kernel, stride, dilation)
    ((32, 32), 3, 2, 1),
    ((33, 33), 3, 2, 1),
    ((32, 32), 1, 2, 1),
    ((31, 30), 5, 2, 1),
    ((28, 28), 7, 2, 1),
    ((16, 16), 3, 1, 1),
    ((15, 16), 5, 1, 1),
    ((16, 16), 4, 1, 1),   # even kernel, stride 1 -> asymmetric SAME
    ((49, 10), (10, 4), 2, 1),  # the 0057/0058 case
    ((20, 21), (3, 5), (2, 1), 1),
    ((32, 32), 3, 2, 2),
    ((17, 17), 3, 1, 2),
]


def _ids(cases):
    return ["-".join(str(v).replace(" ", "") for v in c) for c in cases]


class TestSamePadding:
    @pytest.mark.parametrize("in_hw,k,s,d", CONV2D_CASES, ids=_ids(CONV2D_CASES))
    def test_conv2d_same_is_one_conv_matching_tf_same(self, in_hw, k, s, d):
        conv = QuantConv2d(2, 3, k, stride=s, padding="same", dilation=d, **WQ)
        model = nn.Sequential(conv)
        x = torch.randn(1, 2, *in_hw)
        eager = _calibrate(model, x)

        onnx_model, out = _export(model, x)
        assert _ops(onnx_model) == [("Quantify", "FixedPointQuant"), ("", "Conv")]
        assert np.allclose(out, eager.numpy(), atol=1e-5)

        # Independent oracle: ORT's SAME_UPPER with the same quantized weight.
        with torch.no_grad():
            w = conv.quant_weight().value
        oracle = _ort_same_upper_conv(x, w, conv.bias.detach(), conv.stride, conv.dilation)
        assert oracle.shape == out.shape
        assert np.allclose(out, oracle, atol=1e-5)

    @pytest.mark.parametrize("length,k,s,d", [
        (78, 3, 1, 1), (78, 3, 2, 1), (77, 5, 2, 1), (64, 7, 2, 1), (64, 1, 2, 1), (50, 4, 3, 2),
    ])
    def test_conv1d_same_matches_tf_same(self, length, k, s, d):
        conv = QuantConv1d(2, 3, k, stride=s, padding="same", dilation=d, **WQ)
        model = nn.Sequential(conv)
        x = torch.randn(1, 2, length)
        eager = _calibrate(model, x)
        onnx_model, out = _export(model, x)
        assert np.allclose(out, eager.numpy(), atol=1e-5)
        with torch.no_grad():
            w = conv.quant_weight().value
        oracle = _ort_same_upper_conv(x, w, conv.bias.detach(), conv.stride, conv.dilation)
        assert np.allclose(out, oracle, atol=1e-5)

    @pytest.mark.parametrize("pool_cls,op", [(MaxPool2d, "MaxPool")])
    @pytest.mark.parametrize("in_hw,k,s", [((32, 32), 3, 2), ((33, 31), 3, 2), ((16, 16), 2, 2),
                                           ((15, 15), (3, 5), (2, 1)), ((12, 12), 5, 1)])
    def test_pool2d_same_matches_tf_same(self, pool_cls, op, in_hw, k, s):
        conv = QuantConv2d(1, 2, 1, **WQ)
        model = nn.Sequential(conv, pool_cls(k, s, padding="same"))
        x = torch.randn(1, 1, *in_hw)
        eager = _calibrate(model, x)
        onnx_model, out = _export(model, x)
        assert [o for _, o in _ops(onnx_model)].count(op) == 1
        assert np.allclose(out, eager.numpy(), atol=1e-5)
        with torch.no_grad():
            conv_out = conv(x)
        kk = k if isinstance(k, tuple) else (k, k)
        ss = s if isinstance(s, tuple) else (s, s)
        oracle = _ort_same_upper_pool(op, conv_out, kk, ss)
        assert np.allclose(out, oracle, atol=1e-5)

    @pytest.mark.parametrize("pool_cls", [MaxPool1d])
    @pytest.mark.parametrize("length,k,s", [(64, 3, 2), (63, 2, 2), (20, 5, 1)])
    def test_pool1d_same_exports_and_matches(self, pool_cls, length, k, s):
        model = nn.Sequential(QuantConv1d(1, 2, 1, **WQ), pool_cls(k, s, padding="same"))
        x = torch.randn(1, 1, length)
        eager = _calibrate(model, x)
        _, out = _export(model, x)
        assert out.shape[-1] == -(-length // s)
        assert np.allclose(out, eager.numpy(), atol=1e-5)

    def test_same_pads_formula(self):
        assert same_pads([32], [3], [2], [1]) == [(0, 1)]
        assert same_pads([33], [3], [2], [1]) == [(1, 1)]
        assert same_pads([49, 10], [10, 4], [2, 2], [1, 1]) == [(4, 5), (1, 1)]
        assert same_pads([8], [1], [2], [1]) == [(0, 0)]

    def test_same_conv_trains(self):
        conv = QuantConv2d(2, 3, 3, stride=2, padding="same", **WQ)
        x = torch.randn(2, 2, 8, 8, requires_grad=True)
        conv.train()
        conv(x).sum().backward()
        assert x.grad is not None and conv.weight.grad is not None

    def test_numeric_padding_still_behaves_like_parent(self):
        ours = QuantConv2d(2, 3, 3, stride=1, padding=1, **WQ)
        assert ours.padding == (1, 1) and not ours.same_padding


class TestPaddingViolations:
    def test_pytorch_symmetric_pad_stride2_even_input_raises(self):
        model = nn.Sequential(qnn.QuantConv2d(3, 4, 3, stride=2, padding=1, **WQ))
        x = torch.randn(1, 3, 32, 32)
        _calibrate(model, x)
        with pytest.raises(NonCanonicalONNXError) as e:
            _export(model, x)
        msg = _messages(e)
        assert "[1, 3, 32, 32]" in msg and "[1, 1, 1, 1]" in msg and "[0, 0, 1, 1]" in msg

    def test_symmetric_pad_that_equals_same_is_accepted(self):
        # stride 2 on an ODD input: TF-SAME is symmetric (1, 1) as well.
        model = nn.Sequential(qnn.QuantConv2d(3, 4, 3, stride=2, padding=1, **WQ))
        x = torch.randn(1, 3, 33, 33)
        _calibrate(model, x)
        _export(model, x)

    def test_valid_padding_is_accepted(self):
        model = nn.Sequential(qnn.QuantConv2d(3, 4, 3, stride=2, **WQ), nn.MaxPool2d(2))
        x = torch.randn(1, 3, 32, 32)
        _calibrate(model, x)
        _export(model, x)

    def test_ceil_mode_raises(self):
        model = nn.Sequential(qnn.QuantConv2d(3, 4, 3, **WQ), nn.MaxPool2d(2, ceil_mode=True))
        x = torch.randn(1, 3, 9, 9)
        _calibrate(model, x)
        with pytest.raises(NonCanonicalONNXError, match="ceil_mode"):
            _export(model, x)

    def test_auto_pad_same_lower_raises(self):
        graph = helper.make_graph(
            [helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[3, 3], strides=[2, 2],
                              auto_pad="SAME_LOWER")],
            "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
        violations, _ = check_onnx_graph(helper.make_model(graph))
        assert any("auto_pad=SAME_LOWER" in v for v in violations)

    @pytest.mark.parametrize("k,s,p,op,ok", [
        (2, 2, 0, 0, True),    # VALID
        (3, 2, 1, 1, True),    # SAME: out = 2*in, pads (1, 1)
        (4, 2, 1, 0, True),    # SAME: pads (1, 1)
        (3, 2, 1, 0, False),   # SAME would be (0, 1)
    ])
    def test_conv_transpose(self, k, s, p, op, ok):
        model = nn.Sequential(qnn.QuantConvTranspose2d(2, 2, k, stride=s, padding=p,
                                                       output_padding=op, **WQ))
        x = torch.randn(1, 2, 8, 8)
        _calibrate(model, x)
        if ok:
            _export(model, x)
        else:
            with pytest.raises(NonCanonicalONNXError, match=r"\[padding\] ConvTranspose"):
                _export(model, x)


# =========================================================================
# 3. Graph input
# =========================================================================


class _Glue(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(10, 4)
        self.conv = QuantConv1d(1, 2, 3, padding="same", **WQ)
        self.conv4 = QuantConv1d(4, 2, 3, padding="same", **WQ)

    def forward(self, x):
        if self.kind == "unsqueeze":
            return self.conv(x.unsqueeze(1))
        if self.kind == "reshape":
            return self.conv(x.reshape(1, 1, -1))
        if self.kind == "transpose":
            return self.conv4(x.transpose(1, 2))
        if self.kind == "gather":
            return self.conv4(self.emb(x).transpose(1, 2))
        raise AssertionError


class TestGraphInput:
    def test_clean_model_passes(self):
        model = nn.Sequential(
            qnn.QuantIdentity(act_quant=FixedPointPerTensorActivationQuant),
            QuantConv2d(1, 2, 3, padding="same", **WQ),
        )
        x = torch.randn(1, 1, 8, 8)
        _calibrate(model, x)
        _export(model, x)

    @pytest.mark.parametrize("kind,x,op", [
        ("unsqueeze", torch.randn(1, 16), "Unsqueeze"),
        ("reshape", torch.randn(1, 16), "Reshape"),
        ("transpose", torch.randn(1, 16, 4), "Transpose"),
        ("gather", torch.randint(0, 10, (1, 16)), "Gather"),
    ])
    def test_glue_before_first_layer_is_flagged(self, kind, x, op):
        model = _Glue(kind)
        _calibrate(model, x)
        with pytest.raises(NonCanonicalONNXError) as e:
            _export(model, x)
        lines = [l for l in _messages(e).splitlines() if "[input]" in l]
        assert lines and op in lines[0], _messages(e)


# =========================================================================
# 4. Activations
# =========================================================================


ACTS = [
    (QuantReLU, "Relu"), (QuantReLU6, "Relu6"), (QuantSigmoid, "Sigmoid"), (QuantTanh, "Tanh"),
    (QuantSiLU, "SiLU"), (QuantGELU, "GELU"), (QuantLeakyReLU, "LeakyRelu"),
    (QuantSoftmax, "Softmax"), (QuantHardSwish, "HardSwish"), (QuantHardSigmoid, "HardSigmoid"),
]


class TestActivations:
    @pytest.mark.parametrize("cls,op", ACTS, ids=[op for _, op in ACTS])
    def test_quantify_activation_exports_as_custom_node_and_runs(self, cls, op):
        model = nn.Sequential(qnn.QuantLinear(6, 5, **WQ), cls(bit_width=8), qnn.QuantLinear(5, 3, **WQ))
        x = torch.randn(4, 6) * 3
        eager = _calibrate(model, x)
        onnx_model, out = _export(model, x)
        assert ("Quantify", op) in _ops(onnx_model)
        assert not any(d == "" and o == op for d, o in _ops(onnx_model))
        assert np.allclose(out, eager.numpy(), atol=1e-4)

    @pytest.mark.parametrize("act,suggest", [
        (nn.ReLU(), "QuantReLU"), (nn.Softmax(dim=-1), "QuantSoftmax"), (nn.Sigmoid(), "QuantSigmoid"),
        (nn.Hardswish(), "QuantHardSwish"), (nn.ReLU6(), "QuantReLU6"),
        (qnn.QuantReLU(act_quant=FixedPointPerTensorActivationQuant), "QuantReLU"),
    ], ids=["relu", "softmax", "sigmoid", "hardswish", "relu6", "brevitas_quantrelu"])
    def test_plain_activation_inside_raises_with_replacement(self, act, suggest):
        model = nn.Sequential(qnn.QuantLinear(6, 5, **WQ), act, qnn.QuantLinear(5, 3, **WQ))
        x = torch.randn(4, 6)
        _calibrate(model, x)
        with pytest.raises(NonCanonicalONNXError) as e:
            _export(model, x)
        assert "'1'" in _messages(e) and f"quantizers.{suggest}" in _messages(e)

    def test_functional_activation_inside_raises(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = qnn.QuantLinear(6, 5, **WQ)
                self.b = qnn.QuantLinear(5, 3, **WQ)

            def forward(self, x):
                return self.b(F.relu(self.a(x)))

        model = M()
        x = torch.randn(4, 6)
        _calibrate(model, x)
        with pytest.raises(NonCanonicalONNXError, match=r"\[activation\] Relu"):
            _export(model, x)

    @pytest.mark.parametrize("act", [nn.Softmax(dim=-1), nn.Sigmoid()], ids=["softmax", "sigmoid"])
    def test_plain_activation_at_graph_output_is_allowed(self, act):
        model = nn.Sequential(qnn.QuantLinear(6, 3, **WQ), act)
        x = torch.randn(4, 6)
        eager = _calibrate(model, x)
        onnx_model, out = _export(model, x)
        assert np.allclose(out, eager.numpy(), atol=1e-5)

    def test_layernorm_warns_but_exports(self):
        model = nn.Sequential(qnn.QuantLinear(6, 5, **WQ), nn.LayerNorm(5), qnn.QuantLinear(5, 3, **WQ))
        x = torch.randn(4, 6)
        _calibrate(model, x)
        with pytest.warns(UserWarning, match="LayerNorm"):
            _export(model, x)

    def test_warn_mode_warns_instead_of_raising(self):
        model = nn.Sequential(qnn.QuantLinear(6, 5, **WQ), nn.ReLU(), qnn.QuantLinear(5, 3, **WQ))
        x = torch.randn(4, 6)
        _calibrate(model, x)
        with pytest.warns(UserWarning, match="not canonical"):
            _export(model, x, canonical_checks="warn")

    def test_off_mode_skips_checks(self):
        model = nn.Sequential(qnn.QuantLinear(6, 5, **WQ), nn.ReLU(), qnn.QuantLinear(5, 3, **WQ))
        x = torch.randn(4, 6)
        _calibrate(model, x)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=r".*\[onnx_export\].*")
            _export(model, x, canonical_checks="off")

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="canonical_checks"):
            export_onnx_with_io(nn.Linear(2, 2), torch.randn(1, 2), "unused.onnx", canonical_checks="nope")
