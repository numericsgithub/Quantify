"""Centralized ONNX export utilities for Brevitas QAT models.

Provides a unified export function that handles:
- Legacy exporter requirement (`dynamo=False`)
- Custom quantizer state reset (FIFO deque cleanup)
- Dummy input/output embedding (metadata or initializer)
- Zero-bias injection for QuantLinear layers
- QCDQ vs QONNX routing helpers
"""

from __future__ import annotations

import ast
import base64
from typing import Optional

import numpy as np
import onnx
import torch
from onnx import numpy_helper


def reset_quantizer_states() -> None:
    """
    Reset capture states (FIFO deques) for all custom quantizer autograd.Functions.

    This must be called before every ``torch.onnx.export()`` to prevent state
    leakage between runs or notebooks. Uses lazy imports to avoid circular
    dependencies.
    """
    try:
        from quantizers.fixedpoint_per_tensor import FixedPointQuantFn
        FixedPointQuantFn.reset_capture_state()
    except (ImportError, AttributeError):
        pass

    try:
        from quantizers.silu_quant import SiLUQuantFn
        SiLUQuantFn.reset_capture_state()
    except (ImportError, AttributeError):
        pass

    try:
        from quantizers.coefficient_per_tensor_weights import CoefficientQuantFn
        CoefficientQuantFn.reset_capture_state()
    except (ImportError, AttributeError):
        pass


def export_onnx_with_io(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    filepath: str,
    embed_mode: str = "metadata",
    opset_version: int = 17,
    custom_opsets: Optional[dict] = None,
    dynamo: bool = False,
    reset_states: bool = True,
    **export_kwargs,
) -> onnx.ModelProto:
    """
    Export a Brevitas QAT model to ONNX with embedded dummy I/O tensors.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model (will be set to eval mode automatically).
    dummy_input : torch.Tensor
        Example input tensor used for tracing and stored as reference input.
    filepath : str
        Destination path for the exported .onnx file.
    embed_mode : {"metadata", "initializer"}
        How to embed the tensors:
        - "metadata"    : stored as base64 strings in model.metadata_props
                          (no effect on graph execution, safe for any model).
        - "initializer" : stored as named TensorProto initializers inside the
                          graph (visible in Netron, but keep names unique).
    opset_version : int
        ONNX opset version. Defaults to 17.
    custom_opsets : dict, optional
        Custom opset domains and versions (e.g., ``{"Quantify": 1}``).
    dynamo : bool
        Whether to use the modern dynamo exporter. Defaults to False.
        **Must be False** when using custom ``torch.autograd.Function.symbolic``
        nodes (e.g., FixedPointQuant, SiLUQuant).
    reset_states : bool
        If True, calls ``reset_quantizer_states()`` before export to clear
        FIFO deque buffers used by custom quantizer functions.
    **export_kwargs
        Extra keyword arguments forwarded verbatim to ``torch.onnx.export``.

    Returns
    -----
    onnx.ModelProto
        The loaded-and-augmented ONNX model (also saved to filepath).
    """
    if custom_opsets is None:
        custom_opsets = {"Quantify": 1}

    # 0. Reset quantizer capture states to avoid FIFO deque collisions
    if reset_states:
        reset_quantizer_states()

    model.eval()

    def inject_zero_biases(model: torch.nn.Module) -> None:
        """Inject zero biases into all bias=False QuantLinear layers in-place."""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.bias is None:
                module.bias = torch.nn.Parameter(
                    torch.zeros(module.out_features, device=module.weight.device)
                )

    inject_zero_biases(model)

    # 1. Force-enable quantization on every Brevitas proxy for the duration of
    #    the export so the graph captures the quantized model even when the
    #    trainer is still in float-warmup (proxies otherwise pass `x` through
    #    and no Quantify::FixedPointQuant nodes get emitted).
    saved_disable_quant: dict[int, bool] = {}
    for m in model.modules():
        if hasattr(m, "disable_quant"):
            saved_disable_quant[id(m)] = bool(m.disable_quant)
            m.disable_quant = False

    # 2. Export via QuantifyONNXManager so our FixedPoint handlers get installed
    #    and `Quantify::FixedPointQuant` nodes land in the graph. Imported lazily
    #    to avoid an import cycle with quantizers/*.
    try:
        from utils.quantify_export_manager import QuantifyONNXManager
        QuantifyONNXManager.export(
            model,
            args=dummy_input,
            export_path=filepath,
            opset_version=opset_version,
            do_constant_folding=True,
            custom_opsets=custom_opsets,
            dynamo=dynamo,
            **export_kwargs,
        )
    finally:
        for m in model.modules():
            if id(m) in saved_disable_quant:
                m.disable_quant = saved_disable_quant[id(m)]

    # 2. Compute reference output
    with torch.no_grad():
        dummy_output = model(dummy_input)

    # Unwrap QuantTensor (Brevitas) if needed
    if hasattr(dummy_output, "value"):
        dummy_output = dummy_output.value

    dummy_input_np = dummy_input.detach().cpu().numpy()
    dummy_output_np = dummy_output.detach().cpu().numpy()

    # 3. Embed into the ONNX model
    onnx_model = onnx.load(filepath)

    if embed_mode == "metadata":
        _embed_as_metadata(onnx_model, dummy_input_np, dummy_output_np)
    elif embed_mode == "initializer":
        _embed_as_initializer(onnx_model, dummy_input_np, dummy_output_np)
    else:
        raise ValueError(
            f"embed_mode must be 'metadata' or 'initializer', got '{embed_mode}'"
        )

    onnx.save(onnx_model, filepath)
    return onnx_model


def _embed_as_metadata(
    onnx_model: onnx.ModelProto,
    inp: np.ndarray,
    out: np.ndarray,
) -> None:
    """Store tensors as base64 strings in metadata_props."""
    def _add(key: str, value: str) -> None:
        prop = onnx_model.metadata_props.add()
        prop.key = key
        prop.value = value

    def _encode(arr: np.ndarray) -> str:
        return base64.b64encode(arr.tobytes()).decode("utf-8")

    _add("dummy_input", _encode(inp))
    _add("dummy_input_shape", str(list(inp.shape)))
    _add("dummy_input_dtype", str(inp.dtype))
    _add("dummy_output", _encode(out))
    _add("dummy_output_shape", str(list(out.shape)))
    _add("dummy_output_dtype", str(out.dtype))


def _embed_as_initializer(
    onnx_model: onnx.ModelProto,
    inp: np.ndarray,
    out: np.ndarray,
) -> None:
    """Store tensors as named TensorProto initializers in the graph."""
    onnx_model.graph.initializer.append(
        numpy_helper.from_array(inp, name="dummy_input_ref")
    )
    onnx_model.graph.initializer.append(
        numpy_helper.from_array(out, name="dummy_output_ref")
    )


def load_embedded_io(
    onnx_model_or_path: str | onnx.ModelProto,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the dummy input and output previously embedded via
    ``export_onnx_with_io(..., embed_mode='metadata')``.

    Parameters
    ----------
    onnx_model_or_path : str or onnx.ModelProto
        Path to the .onnx file or an already-loaded ModelProto.

    Returns
    -----
    tuple[np.ndarray, np.ndarray]
        (dummy_input, dummy_output)
    """
    if isinstance(onnx_model_or_path, str):
        onnx_model = onnx.load(onnx_model_or_path)
    else:
        onnx_model = onnx_model_or_path

    props = {p.key: p.value for p in onnx_model.metadata_props}

    required = {
        "dummy_input", "dummy_input_shape", "dummy_input_dtype",
        "dummy_output", "dummy_output_shape", "dummy_output_dtype",
    }
    missing = required - props.keys()
    if missing:
        raise KeyError(
            f"The ONNX model is missing metadata keys: {missing}. "
            "Was it exported with embed_mode='metadata'?"
        )

    def _decode(key_data: str, key_shape: str, key_dtype: str) -> np.ndarray:
        raw = base64.b64decode(props[key_data])
        shape = ast.literal_eval(props[key_shape])
        dtype = np.dtype(props[key_dtype])
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    inp = _decode("dummy_input", "dummy_input_shape", "dummy_input_dtype")
    out = _decode("dummy_output", "dummy_output_shape", "dummy_output_dtype")
    return inp, out


def export_onnx_qcdq(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    filepath: str,
    **kwargs,
) -> None:
    """
    Export a Brevitas model to ONNX using the standard QCDQ format.

    Wraps ``brevitas.export.onnx.standard.qcdq.export_onnx_qcdq``.
    QCDQ models use standard ``QuantizeLinear``/``DequantizeLinear`` nodes
    and are compatible with ONNX Runtime, TensorRT, and most INT8 inference stacks.

    Parameters
    ----------
    model : torch.nn.Module
        Trained Brevitas model.
    dummy_input : torch.Tensor
        Example input tensor for tracing.
    filepath : str
        Destination path for the exported .onnx file.
    **kwargs
        Additional arguments forwarded to Brevitas' QCDQ exporter.
    """
    from brevitas.export.onnx.standard.qcdq import export_onnx_qcdq as brevitas_export

    model.eval()
    with torch.no_grad():
        brevitas_export(model, args=dummy_input, export_path=filepath, **kwargs)
