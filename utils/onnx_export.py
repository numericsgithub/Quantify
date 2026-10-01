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


def _freeze_annealing(model: torch.nn.Module) -> dict:
    """Force every quantizer's annealing_alpha to 1.0 (fully quantized, no
    blend) and return a {quantizer: original_alpha_tensor} snapshot so it can
    be restored with `_restore_annealing`.

    Why this is needed: `BaseQuantizer.forward()` only emits a clean
    quantize-only graph when `annealing_alpha == 1.0`. Any alpha < 1.0 (mid
    QAT-anneal) traces through `AnnealingBlendFn.forward`'s
    `(1-alpha)*x + alpha*quantized` arithmetic instead (it has no `symbolic`
    method), which shows up in the exported ONNX graph as a `Mul` (by alpha)
    then an `Add` (of `(1-alpha)*original_float_weight`) right after the
    `FixedPointQuant` node -- i.e. the exported "quantized" weight is
    actually a float/quantized blend, not the real quantized value. See
    docs/llm/pitfalls/brevitas_pitfalls.md.
    """
    from quantizers.base_quantizer import BaseQuantizer

    saved = {}
    for module in model.modules():
        if isinstance(module, BaseQuantizer):
            saved[module] = module.annealing_alpha.clone()
            module.annealing_alpha.fill_(1.0)
    return saved


def _restore_annealing(saved: dict) -> None:
    for module, alpha in saved.items():
        module.annealing_alpha.copy_(alpha)


def _freeze_uncalibrated(model: torch.nn.Module) -> list:
    """Force every NOT-YET-CALIBRATED quantizer (`search_done=False`) to a
    plain float passthrough for the duration of the export, and return the
    list of quantizers touched so `_unfreeze_uncalibrated` can restore them.

    Why this is needed: `export_onnx_with_io` runs the model through two real
    forward passes outside training -- the `torch.onnx.export()` trace, and a
    `model(dummy_input)` call afterward to embed a reference input/output pair
    (see below). Without this guard, either of those forward passes is
    indistinguishable from a real one to `BaseQuantizer.forward()`: a
    quantizer that hasn't been reached by training yet would run its very
    first calibration search against `dummy_input` -- a meaningless,
    arbitrary tensor (a hardcoded `torch.randn(1, 3, 32, 32)` when
    `training_harness/checkpointing.py::CheckpointManager` doesn't have a
    real batch handy) -- and would also have its gating/annealing state
    advanced as if real training had happened. Since checkpoints are commonly
    saved (with an ONNX export alongside) every epoch, this can trigger a
    quantizer's actual production calibration from export noise the very
    first time an export happens to run after that quantizer's gate would
    otherwise have opened -- see pitfall #19 in
    docs/llm/pitfalls/brevitas_pitfalls.md.

    A plain float passthrough is also the MOST ACCURATE representation of
    such a quantizer in the exported graph: it hasn't started quantizing in
    the real (live) model either, so showing a `Quantify::FixedPointQuant`
    node with a bogus, uncalibrated LSB would be actively misleading.
    """
    from quantizers.base_quantizer import BaseQuantizer

    frozen = []
    for module in model.modules():
        if isinstance(module, BaseQuantizer) and not module.search_done_value:
            module.force_passthrough_for_export = True
            frozen.append(module)
    return frozen


def _unfreeze_uncalibrated(frozen: list) -> None:
    for module in frozen:
        module.force_passthrough_for_export = False


def _suppress_lifecycle_logging(model: torch.nn.Module) -> list:
    """Set `suppress_lifecycle_logging=True` on every quantizer for the
    duration of the export, and return the list touched so
    `_unsuppress_lifecycle_logging` can restore them.

    Why this is needed: even an ALREADY-calibrated quantizer that's still
    mid-anneal gets its `annealing_alpha` temporarily forced to 1.0 by
    `_freeze_annealing` above (needed so the exported graph shows a clean
    quantized value instead of a float/quantized blend). Without this guard,
    that forced alpha=1.0 would trip `BaseQuantizer`'s one-shot
    `annealing complete` log flag during the export's forward pass(es) --
    permanently suppressing the real log message once annealing genuinely
    finishes later in training. See pitfall #19 in
    docs/llm/pitfalls/brevitas_pitfalls.md.
    """
    from quantizers.base_quantizer import BaseQuantizer

    touched = []
    for module in model.modules():
        if isinstance(module, BaseQuantizer):
            module.suppress_lifecycle_logging = True
            touched.append(module)
    return touched


def _unsuppress_lifecycle_logging(touched: list) -> None:
    for module in touched:
        module.suppress_lifecycle_logging = False


def export_onnx_with_io(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    filepath: str,
    embed_mode: str = "metadata",
    opset_version: int = 17,
    custom_opsets: Optional[dict] = None,
    dynamo: bool = False,
    reset_states: bool = True,
    freeze_annealing: bool = True,
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
    freeze_annealing : bool
        If True (default), every quantizer's ``annealing_alpha`` is forced to
        1.0 for the duration of the export (and the reference forward pass
        used for the embedded dummy I/O), then restored to its original
        value afterward -- even if export raises. Without this, a quantizer
        mid-QAT-anneal (``0.0 < annealing_alpha < 1.0``) exports a
        `FixedPointQuant -> Mul -> Add` blend of the float and quantized
        weight instead of a clean quantized value; see `_freeze_annealing`'s
        docstring and docs/llm/pitfalls/brevitas_pitfalls.md.

        Regardless of this flag, every export also always (a) exports a
        not-yet-calibrated quantizer as a plain float passthrough instead of
        calibrating it against `dummy_input`, and (b) isolates the one-shot
        lifecycle log flags (gate opened / calibration / annealing
        started-complete) from the export's own forward passes -- see
        `_freeze_uncalibrated`/`_suppress_lifecycle_logging` and pitfall #19
        in docs/llm/pitfalls/brevitas_pitfalls.md.
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

    # Export requires eval mode (stable BN stats, no annealing-step advance --
    # see the `if self.training:` guard in BaseQuantizer.forward()), but the
    # caller's train/eval mode is restored afterward (even if export raises)
    # so a manual `export_onnx_with_io(...)` call mid-training doesn't leave
    # the model silently running in eval mode for whatever forward pass the
    # caller makes next. Harmless when called from the training harness
    # (every epoch starts with `model.train(is_train)` regardless), but a
    # real footgun for any other caller -- see pitfall #19 in
    # docs/llm/pitfalls/brevitas_pitfalls.md.
    was_training = model.training
    model.eval()

    def inject_zero_biases(model: torch.nn.Module) -> None:
        """Inject zero biases into all bias=False QuantLinear layers in-place."""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.bias is None:
                module.bias = torch.nn.Parameter(
                    torch.zeros(module.out_features, device=module.weight.device)
                )

    inject_zero_biases(model)

    # 1+2. Export via torch.onnx.export, then compute the reference output
    # used for the embedded dummy I/O -- both done with annealing frozen off
    # (see freeze_annealing docs above) so the graph and the reference output
    # agree and neither leaks a Mul/Add blend of the float weight.
    #
    # Both forward passes below (the trace inside torch.onnx.export, and the
    # dummy_output call) are isolated from training: _freeze_uncalibrated
    # stops either one from calibrating a not-yet-reached quantizer against
    # meaningless dummy data, and _suppress_lifecycle_logging stops the
    # forced annealing_alpha=1.0 above from tripping a premature "annealing
    # complete" log for a quantizer that's really still mid-anneal. See
    # pitfall #19 in docs/llm/pitfalls/brevitas_pitfalls.md.
    saved_annealing = _freeze_annealing(model) if freeze_annealing else {}
    frozen_uncalibrated = _freeze_uncalibrated(model)
    suppressed_logging = _suppress_lifecycle_logging(model)
    if frozen_uncalibrated:
        names = [
            getattr(q, "display_name", getattr(q, "quant_id", repr(id(q))))
            for q in frozen_uncalibrated
        ]
        print(
            f"[onnx_export] {len(frozen_uncalibrated)} quantizer(s) not yet calibrated in "
            f"the live model -- exported as a plain float passthrough (no quantize node) "
            f"rather than calibrating them against dummy_input: {names}"
        )
    try:
        torch.onnx.export(
            model,
            dummy_input,
            filepath,
            opset_version=opset_version,
            do_constant_folding=True,
            custom_opsets=custom_opsets,
            dynamo=dynamo,
            **export_kwargs,
        )

        with torch.no_grad():
            dummy_output = model(dummy_input)
    finally:
        _restore_annealing(saved_annealing)
        _unfreeze_uncalibrated(frozen_uncalibrated)
        _unsuppress_lifecycle_logging(suppressed_logging)
        model.train(was_training)

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
