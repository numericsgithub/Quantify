"""
Custom Brevitas export manager that emits `Quantify::FixedPointQuant` ONNX nodes
for the fixed-point quantizers in this project.

Plain `torch.onnx.export` does not install Brevitas export handlers, so the
custom symbolic on `FixedPointQuantFn` never fires reliably through Brevitas
proxy tracing. This manager subclasses `StdONNXBaseManager` and registers
per-proxy handlers that explicitly invoke `FixedPointQuantFn.apply` inside
`symbolic_execution`, guaranteeing the custom op lands in the exported graph.

Drive an export via::

    QuantifyONNXManager.export(model, args=dummy_input, export_path="model.onnx",
                               opset_version=13, custom_opsets={"Quantify": 1})
"""

from __future__ import annotations

import torch
from torch.nn import Module

from brevitas.export.manager import (
    _set_proxy_export_handler,
    _set_proxy_export_mode,
)
from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.export.onnx.manager import ONNXBaseManager

from quantizers.fixedpoint_per_tensor import FixedPointQuantFn
from quantizers.fixedpoint_proxy import (
    FixedPointActQuantProxy,
    FixedPointBiasQuantProxy,
    FixedPointWeightQuantProxy,
)


# ---------------------------------------------------------------------------
# Handler base — shared cache + symbolic_execution
# ---------------------------------------------------------------------------

class _FixedPointHandlerBase(ONNXBaseHandler):
    """
    Shared logic. Subclasses set `handled_layer` and override
    `_tensor_quant_of(module)` to point at the FixedPointPerTensorQuantizer
    instance inside the proxy.
    """

    handled_layer = None  # set by subclass

    def _tensor_quant_of(self, module):
        raise NotImplementedError

    def prepare_for_export(self, module):
        tq = self._tensor_quant_of(module)
        if tq is None:
            raise RuntimeError(
                f"{type(self).__name__}: proxy has no FixedPointPerTensorQuantizer to export"
            )
        # Calibrated buffers (populated after the first forward pass)
        self._lsb = int(tq.search_result_lsb.item())
        self._signed = bool(tq.search_result_is_signed.item())
        # Static config from the quantizer
        self._bit_width = int(tq.bit_width)
        self._narrow_range = bool(tq.narrow_range)
        self._rounding_mode = tq.rounding_mode

    def symbolic_execution(self, x: torch.Tensor):
        scale = torch.tensor(2.0 ** self._lsb, dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        quantized, scale_out, zp_out, bw_out = FixedPointQuantFn.apply(
            x,
            scale,
            zero_point,
            self._lsb,
            self._bit_width,
            self._signed,
            self._narrow_range,
            self._rounding_mode,
        )
        return quantized, scale_out, zp_out, bw_out


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------

class FixedPointWeightHandler(_FixedPointHandlerBase):
    handled_layer = FixedPointWeightQuantProxy

    def _tensor_quant_of(self, module):
        return module.tensor_quant


class FixedPointBiasHandler(_FixedPointHandlerBase):
    handled_layer = FixedPointBiasQuantProxy

    def _tensor_quant_of(self, module):
        return module.tensor_quant


class FixedPointActHandler(_FixedPointHandlerBase):
    handled_layer = FixedPointActQuantProxy

    def _tensor_quant_of(self, module):
        # Activation proxies wrap tensor_quant inside a FusedActivationQuantProxy
        return module.fused_activation_quant_proxy.tensor_quant


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class QuantifyONNXManager(ONNXBaseManager):
    """
    Drives `torch.onnx.export` with our handlers installed on every
    FixedPoint{Weight,Act,Bias}QuantProxy in the model.
    """

    target_name = "Quantify"
    dequantize_tracing_input = False
    run_onnx_passes = False  # we don't need onnxoptimizer passes
    onnx_passes = []
    handlers = [
        FixedPointWeightHandler,
        FixedPointActHandler,
        FixedPointBiasHandler,
    ]
    # FixedPointQuantFn's symbolic is defined directly on the autograd.Function
    # and invoked via .apply() — no need to register via register_custom_op_symbolic.
    custom_fns = []

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool) -> None:
        _set_proxy_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module) -> None:
        _set_proxy_export_handler(cls, module)
