"""
Layers with canonical ONNX lowerings.

**TF-style SAME padding.** PyTorch's own `padding=k//2` is symmetric, which
for stride > 1 on an even input is NOT what TF/ONNX "SAME" means (SAME puts
the odd extra pixel at the end: e.g. 32x32, k3 s2 -> pads `(0, 1)`, PyTorch
-> `(1, 1)` -- same output shape, values shifted by one pixel).
`utils/onnx_export.py::export_onnx_with_io` rejects any Conv/Pool whose pads
are neither VALID (all zero) nor exactly TF SAME (see
`utils/onnx_canonical.py`), so models need a way to express real SAME
padding at any stride. These layers are drop-in replacements accepting
`padding="same"` for ANY stride (and any kernel/dilation), computing the
asymmetric pads from the actual input shape:

- `QuantConv1d` / `QuantConv2d` -- Brevitas `qnn.QuantConv1d/2d` subclasses
  (same parameters/state_dict/quantizers). Brevitas itself accepts
  `padding='same'` at stride > 1, but exports it as `Pad -> Conv`; these
  export one `Conv` node with the explicit TF-SAME `pads`.
- `MaxPool1d` / `MaxPool2d` -- exported as one `MaxPool` with SAME pads
  (padded positions never win the max, as in TF/ONNX).
- Average pooling is NOT a float pool here: `QuantAvgPool1d/2d` and
  `QuantGlobalAvgPool1d/2d` (below) replace the former `AvgPool1d/2d`.

Any other `padding` value just behaves like the parent class.

**Quantized average pooling.** `QuantAvgPool1d/2d` and
`QuantGlobalAvgPool1d/2d` (which replaced the former float `AvgPool1d/2d`) implement average pooling as a depthwise
`QuantConv1d/2d` whose weights are frozen to `1/prod(kernel)` and whose bias
is frozen to zero -- so they quantize (weights 12 bit, output 10 bit by
default) and export exactly like any other depthwise Conv, no `AveragePool`
node. See the section below.

Like the activation shims in `quantizers/activations.py`, the single-node
symbolic Functions are only used while `torch.onnx.is_in_onnx_export()`;
training runs the plain `F.pad` + op path (same numerics, normal autograd).

**`Quantify::Dense`.** `DenseFn` emits one `Quantify::Dense(x, W, b)` node
for a Linear regardless of input rank or bias (`W` in PyTorch's
`[out, in]` layout, `b` always present). `export_onnx_with_io` routes every
`nn.Linear` (incl. Brevitas `QuantLinear`) through it for the duration of an
export -- see `utils/onnx_canonical.py::dense_export`.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from torch.autograd import Function

from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorBiasQuant,
    FixedPointPerTensorWeightQuant,
)


def _value(t):
    """Unwrap a Brevitas QuantTensor to its plain tensor."""
    return getattr(t, "value", t)


def same_pads(in_sizes: Sequence[int], kernel: Sequence[int], stride: Sequence[int],
              dilation: Sequence[int]) -> List[Tuple[int, int]]:
    """TF-style SAME `(begin, end)` pads per spatial axis: `out = ceil(in /
    stride)`, `total = max((out-1)*stride + (k-1)*dilation + 1 - in, 0)`,
    `begin = total // 2`, `end = total - begin`."""
    pads = []
    for i, k, s, d in zip(in_sizes, kernel, stride, dilation):
        out = math.ceil(i / s)
        total = max((out - 1) * s + (k - 1) * d + 1 - i, 0)
        pads.append((total // 2, total - total // 2))
    return pads


def _onnx_pads(pads: List[Tuple[int, int]]) -> List[int]:
    """ONNX order: all begins, then all ends."""
    return [b for b, _ in pads] + [e for _, e in pads]


def _torch_pad(pads: List[Tuple[int, int]]) -> List[int]:
    """F.pad order: last axis first, (begin, end) each."""
    out = []
    for b, e in reversed(pads):
        out += [b, e]
    return out


def _out_type(x, out_sizes):
    try:
        return x.type().with_sizes(out_sizes)
    except Exception:
        return None


def _set_type(node, x, out_sizes):
    t = _out_type(x, out_sizes)
    return node.setType(t) if t is not None else node


# ---------------------------------------------------------------------------
# Export-only symbolic Functions
# ---------------------------------------------------------------------------

class SameConvFn(Function):
    """One ONNX `Conv` node with explicit (possibly asymmetric) pads."""

    @staticmethod
    def symbolic(g, x, w, b, stride, dilation, groups, pads, out_sizes):
        inputs = [x, w] if b is None else [x, w, b]
        kernel = w.type().sizes()[2:]
        node = g.op("Conv", *inputs, kernel_shape_i=list(kernel), strides_i=list(stride),
                    dilations_i=list(dilation), group_i=int(groups), pads_i=_onnx_pads(pads))
        return _set_type(node, x, out_sizes)

    @staticmethod
    def forward(ctx, x, w, b, stride, dilation, groups, pads, out_sizes):
        conv = F.conv1d if x.dim() == 3 else F.conv2d
        return conv(F.pad(x, _torch_pad(pads)), w, b, stride, 0, dilation, groups)


class SameMaxPoolFn(Function):
    """One ONNX `MaxPool` node with explicit (possibly asymmetric) pads."""

    @staticmethod
    def symbolic(g, x, kernel, stride, dilation, pads, out_sizes):
        node = g.op("MaxPool", x, kernel_shape_i=list(kernel), strides_i=list(stride),
                    dilations_i=list(dilation), pads_i=_onnx_pads(pads), ceil_mode_i=0)
        return _set_type(node, x, out_sizes)

    @staticmethod
    def forward(ctx, x, kernel, stride, dilation, pads, out_sizes):
        return _same_max_pool(x, kernel, stride, dilation, pads)


class DenseFn(Function):
    """One `Quantify::Dense(x, W, b)` node: `y = x @ W^T + b`, any input rank,
    `W` in `[out, in]` layout, `b` always present."""

    @staticmethod
    def symbolic(g, x, w, b):
        out_features, in_features = w.type().sizes()
        node = g.op("Quantify::Dense", x, w, b,
                    in_features_i=int(in_features), out_features_i=int(out_features))
        try:
            out_sizes = list(x.type().sizes()[:-1]) + [out_features]
        except Exception:
            return node
        return _set_type(node, x, out_sizes)

    @staticmethod
    def forward(ctx, x, w, b):
        return F.linear(x, w, b)


# ---------------------------------------------------------------------------
# Eager SAME implementations
# ---------------------------------------------------------------------------

def _same_max_pool(x, kernel, stride, dilation, pads):
    pool = F.max_pool1d if x.dim() == 3 else F.max_pool2d
    x = F.pad(x, _torch_pad(pads), value=float("-inf"))
    return pool(x, kernel, stride, 0, dilation)


def _spatial(x, n):
    return [int(s) for s in x.shape[-n:]]


def _out_sizes(x, n, spatial_out, channels=None):
    lead = list(x.shape[:-n])
    if channels is not None:
        lead[-1] = channels
    return [int(s) for s in lead] + spatial_out


def _tuple(v, n):
    return tuple(v) if isinstance(v, (tuple, list)) else (v,) * n


# ---------------------------------------------------------------------------
# Conv layers
# ---------------------------------------------------------------------------

class _SameConvMixin:
    _spatial_dims: int

    def _init_same(self, padding):
        is_same = isinstance(padding, str) and padding.lower() == "same"
        return (0 if is_same else padding), is_same

    def inner_forward_impl(self, x, quant_weight, quant_bias):
        if not self.same_padding:
            return super().inner_forward_impl(x, quant_weight, quant_bias)
        x, w = _value(x), _value(quant_weight)
        b = None if quant_bias is None else _value(quant_bias)
        n = self._spatial_dims
        in_sizes = _spatial(x, n)
        pads = same_pads(in_sizes, self.kernel_size, self.stride, self.dilation)
        if torch.onnx.is_in_onnx_export():
            out = [math.ceil(i / s) for i, s in zip(in_sizes, self.stride)]
            return SameConvFn.apply(x, w, b, tuple(self.stride), tuple(self.dilation),
                                    self.groups, pads, _out_sizes(x, n, out, self.out_channels))
        conv = F.conv1d if n == 1 else F.conv2d
        return conv(F.pad(x, _torch_pad(pads)), w, b, self.stride, 0, self.dilation, self.groups)


class QuantConv1d(_SameConvMixin, qnn.QuantConv1d):
    """`qnn.QuantConv1d` that also accepts `padding="same"` at any stride,
    exported as one `Conv` with TF-SAME pads (see module docstring)."""

    _spatial_dims = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, *args, **kwargs):
        padding, same = self._init_same(padding)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, *args, **kwargs)
        self.same_padding = same


class QuantConv2d(_SameConvMixin, qnn.QuantConv2d):
    """`qnn.QuantConv2d` that also accepts `padding="same"` at any stride,
    exported as one `Conv` with TF-SAME pads (see module docstring)."""

    _spatial_dims = 2

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, *args, **kwargs):
        padding, same = self._init_same(padding)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, *args, **kwargs)
        self.same_padding = same


# ---------------------------------------------------------------------------
# Pool layers
# ---------------------------------------------------------------------------

class _SameMaxPoolMixin:
    _spatial_dims: int

    def forward(self, x):
        if not self.same_padding:
            return super().forward(x)
        n = self._spatial_dims
        kernel, stride = _tuple(self.kernel_size, n), _tuple(self.stride, n)
        dilation = _tuple(self.dilation, n)
        in_sizes = _spatial(x, n)
        pads = same_pads(in_sizes, kernel, stride, dilation)
        if torch.onnx.is_in_onnx_export():
            out = [math.ceil(i / s) for i, s in zip(in_sizes, stride)]
            return SameMaxPoolFn.apply(x, kernel, stride, dilation, pads, _out_sizes(x, n, out))
        return _same_max_pool(x, kernel, stride, dilation, pads)


def _pool_init(self, parent, kernel_size, stride, padding, kwargs):
    same = isinstance(padding, str) and padding.lower() == "same"
    if same and kwargs.get("ceil_mode"):
        raise ValueError("padding='same' cannot be combined with ceil_mode=True")
    parent.__init__(self, kernel_size, stride=stride, padding=0 if same else padding, **kwargs)
    self.same_padding = same


class MaxPool1d(_SameMaxPoolMixin, nn.MaxPool1d):
    """`nn.MaxPool1d` that also accepts `padding="same"` (TF-SAME, any stride)."""
    _spatial_dims = 1

    def __init__(self, kernel_size, stride=None, padding=0, **kwargs):
        _pool_init(self, nn.MaxPool1d, kernel_size, stride, padding, kwargs)


class MaxPool2d(_SameMaxPoolMixin, nn.MaxPool2d):
    """`nn.MaxPool2d` that also accepts `padding="same"` (TF-SAME, any stride)."""
    _spatial_dims = 2

    def __init__(self, kernel_size, stride=None, padding=0, **kwargs):
        _pool_init(self, nn.MaxPool2d, kernel_size, stride, padding, kwargs)


# ---------------------------------------------------------------------------
# Quantized average pooling == depthwise conv with frozen weights/bias
# ---------------------------------------------------------------------------

class AvgPoolWeightQuant(FixedPointPerTensorWeightQuant):
    """Weight quantizer for `QuantAvgPool*d`: 12 bit; the all-equal kernel is
    a legitimate constant (`constant_ok`), not a failed calibration."""
    bit_width = 12
    constant_ok = True


class AvgPoolBiasQuant(FixedPointPerTensorBiasQuant):
    """Bias quantizer for `QuantAvgPool*d`: 16 bit; the bias is always zero."""
    bit_width = 16
    constant_ok = True


class _QuantAvgPoolNd(nn.Module):
    """Average pooling as a frozen depthwise convolution.

    `self.conv` is a `QuantConv{1,2}d` with `groups == channels`, every
    kernel weight `1/prod(kernel_size)` and a zero bias, both
    `requires_grad=False`. Its weights are fake-quantized (default 12 bit),
    and so is its output (default 10 bit), like any other quantized layer.
    ONNX export therefore yields a plain depthwise `Conv` (+ the usual
    `FixedPointQuant` nodes) whose weights are all one value.

    Notes
    -----
    * `channels` is required (a depthwise conv needs it at construction).
    * `padding` is an int/tuple or `"same"` (TF-SAME, any stride). Padded
      positions count as zeros in the average (`count_include_pad=True`
      semantics), so border windows are darker than a TF-SAME
      `AveragePool` (`count_include_pad=0`) would give. With `padding=0`
      (VALID) there is no difference.
    * The weight 1/N is rounded to the 12-bit grid (e.g. 1/9 -> 1820/2**14,
      a gain of 0.99976); raise `weight_bit_width` for more accuracy.
    * Anything that re-initialises every Conv's weights after construction
      (`model.apply(init_fn)`) would overwrite the constant: call
      `reset_parameters()` afterwards, or `check_frozen()` to verify.
    """

    _conv_cls = None
    _n = 0

    def __init__(
        self,
        channels: int,
        kernel_size,
        stride=None,
        padding=0,
        weight_bit_width: int = 12,
        output_bit_width: int = 10,
        weight_quant=AvgPoolWeightQuant,
        bias_quant=AvgPoolBiasQuant,
        output_quant=FixedPointPerTensorActivationQuant,
        return_quant_tensor: bool = False,
    ):
        super().__init__()
        n = self._n
        kernel = _tuple(kernel_size, n)
        stride = kernel if stride is None else _tuple(stride, n)
        self.channels = channels
        self.kernel_size = kernel
        self.stride = stride
        self.conv = self._conv_cls(
            channels, channels, kernel, stride, padding, groups=channels, bias=True,
            weight_quant=weight_quant, bias_quant=bias_quant,
            weight_bit_width=weight_bit_width,
            output_quant=output_quant, output_bit_width=output_bit_width,
            return_quant_tensor=return_quant_tensor,
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """(Re)set the frozen values: weights 1/prod(kernel), bias 0."""
        self.conv.weight.fill_(1.0 / math.prod(self.kernel_size))
        self.conv.bias.zero_()
        self.conv.weight.requires_grad_(False)
        self.conv.bias.requires_grad_(False)

    def check_frozen(self) -> None:
        """Raise if weights/bias were modified or became trainable."""
        w, b = self.conv.weight, self.conv.bias
        if w.requires_grad or b.requires_grad:
            raise RuntimeError("average-pool weights/bias must stay frozen (requires_grad=False)")
        if not torch.all(w == 1.0 / math.prod(self.kernel_size)) or torch.any(b != 0):
            raise RuntimeError("average-pool weights/bias were modified; call reset_parameters()")

    def forward(self, x):
        return self.conv(x)


class QuantAvgPool1d(_QuantAvgPoolNd):
    """Quantized 1-D average pooling (frozen depthwise `QuantConv1d`)."""
    _conv_cls = QuantConv1d
    _n = 1


class QuantAvgPool2d(_QuantAvgPoolNd):
    """Quantized 2-D average pooling (frozen depthwise `QuantConv2d`)."""
    _conv_cls = QuantConv2d
    _n = 2


class _QuantGlobalAvgPoolNd(_QuantAvgPoolNd):
    """Global average pooling: kernel == the (static) feature-map size, so the
    spatial dims collapse to 1 -- the same output shape as
    `nn.AdaptiveAvgPoolNd(1)`. `size` must match the input exactly (a
    depthwise conv's kernel is fixed at construction); a mismatch raises."""

    def __init__(self, channels: int, size, **kwargs):
        super().__init__(channels, size, stride=1, padding=0, **kwargs)

    def forward(self, x):
        got = tuple(x.shape[-self._n:])
        if got != self.kernel_size:
            raise ValueError(
                f"{type(self).__name__} was built for spatial size {self.kernel_size}, got {got}")
        return self.conv(x)


class QuantGlobalAvgPool1d(_QuantGlobalAvgPoolNd):
    """`(B, C, L) -> (B, C, 1)`; `size` is `L`."""
    _conv_cls = QuantConv1d
    _n = 1


class QuantGlobalAvgPool2d(_QuantGlobalAvgPoolNd):
    """`(B, C, H, W) -> (B, C, 1, 1)`; `size` is `(H, W)` or an int."""
    _conv_cls = QuantConv2d
    _n = 2
