import ast
import base64

import numpy as np
import onnx
import torch
from onnx import numpy_helper
import onnxoptimizer
import brevitas.nn as qnn

def export_onnx_with_io(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    filepath: str,
    embed_mode: str = "metadata",
    opset_version=17,
    custom_opsets={"Quantify": 1},
    dynamo=False,
    **export_kwargs,
) -> onnx.ModelProto:
    """
    Export a model to ONNX via torch.onnx.export and embed the dummy input
    and its corresponding model output into the saved file.

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
    **export_kwargs
        Extra keyword arguments forwarded verbatim to torch.onnx.export,
        e.g. opset_version=17, custom_opsets={'Quantify': 1}, dynamo=False.

    Returns
    -------
    onnx.ModelProto
        The loaded-and-augmented ONNX model (also saved to filepath).

    Reloading the embedded tensors
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    model.eval()

    def inject_zero_biases(model: torch.nn.Module) -> None:
        """Inject zero biases into all bias=False QuantLinear layers in-place."""
        for name, module in model.named_modules():
            if isinstance(module, qnn.QuantLinear) and module.bias is None:
                module.bias = torch.nn.Parameter(
                    torch.zeros(module.out_features, device=module.weight.device)
                )
                print(f"Injected zero bias into: {name}")
    
    inject_zero_biases(model)

    # ------------------------------------------------------------------ #
    # 1. Export via torch.onnx.export                                     #
    # ------------------------------------------------------------------ #
    torch.onnx.export(model, dummy_input, filepath, opset_version=opset_version, do_constant_folding=True, custom_opsets=custom_opsets, dynamo=dynamo, **export_kwargs)

    # ------------------------------------------------------------------ #
    # 2. Compute reference output                                         #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        dummy_output = model(dummy_input)

    # Unwrap QuantTensor (Brevitas) if needed
    if hasattr(dummy_output, "value"):
        dummy_output = dummy_output.value

    dummy_input_np  = dummy_input.detach().cpu().numpy()
    dummy_output_np = dummy_output.detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    # 3. Embed into the ONNX model                                        #
    # ------------------------------------------------------------------ #
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


# --------------------------------------------------------------------------- #
# Helpers – embedding                                                          #
# --------------------------------------------------------------------------- #

def _embed_as_metadata(
    onnx_model: onnx.ModelProto,
    inp: np.ndarray,
    out: np.ndarray,
) -> None:
    """Store tensors as base64 strings in metadata_props."""
    def _add(key, value):
        prop = onnx_model.metadata_props.add()
        prop.key   = key
        prop.value = value

    def _encode(arr: np.ndarray) -> str:
        return base64.b64encode(arr.tobytes()).decode("utf-8")

    _add("dummy_input",        _encode(inp))
    _add("dummy_input_shape",  str(list(inp.shape)))
    _add("dummy_input_dtype",  str(inp.dtype))
    _add("dummy_output",       _encode(out))
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


# --------------------------------------------------------------------------- #
# Helpers – reloading                                                          #
# --------------------------------------------------------------------------- #

def load_embedded_io(
    onnx_model_or_path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the dummy input and output previously embedded via
    export_onnx_with_io(..., embed_mode='metadata').

    Parameters
    ----------
    onnx_model_or_path : str or onnx.ModelProto
        Path to the .onnx file or an already-loaded ModelProto.

    Returns
    -------
    (dummy_input, dummy_output) : tuple of np.ndarray
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

    def _decode(key_data, key_shape, key_dtype) -> np.ndarray:
        raw   = base64.b64decode(props[key_data])
        shape = ast.literal_eval(props[key_shape])
        dtype = np.dtype(props[key_dtype])
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    inp = _decode("dummy_input",  "dummy_input_shape",  "dummy_input_dtype")
    out = _decode("dummy_output", "dummy_output_shape", "dummy_output_dtype")
    return inp, out