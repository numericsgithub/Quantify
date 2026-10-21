"""Run an exported (float64) ONNX model in tests.

onnxruntime's CPU provider lacks float64 kernels for some ops (Conv,
ConvTranspose, Erf, HardSigmoid, ...), while `onnx`'s reference evaluator
supports those but has gaps of its own (e.g. attributes forwarded into standard
ops inside a function body). So: try onnxruntime first, fall back to the
reference evaluator for what it can't run.
"""
import numpy as np
import onnx
import onnx.numpy_helper
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator


def _to_float32(model):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    for i, init in enumerate(g.initializer):
        if init.data_type == onnx.TensorProto.DOUBLE:
            g.initializer[i].CopyFrom(onnx.numpy_helper.from_array(
                onnx.numpy_helper.to_array(init).astype(np.float32), init.name))
    for vi in list(g.input) + list(g.output):
        if vi.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    del g.value_info[:]
    return m


def run_onnx(model_or_path, feeds, output_names=None):
    """Return the list of outputs; floating inputs are cast to float64."""
    feeds = {k: (np.asarray(v, np.float64) if np.issubdtype(np.asarray(v).dtype, np.floating)
                 else np.asarray(v)) for k, v in feeds.items()}
    try:
        data = (model_or_path if isinstance(model_or_path, (str, bytes))
                else model_or_path.SerializeToString())
        return ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(output_names, feeds)
    except Exception as ort_err:  # missing float64 kernel -> reference evaluator
        if isinstance(model_or_path, (str, bytes)):
            model_or_path = onnx.load(model_or_path) if isinstance(model_or_path, str) \
                else onnx.load_from_string(model_or_path)
        try:
            return ReferenceEvaluator(model_or_path).run(output_names, feeds)
        except Exception as ref_err:
            # Last resort: run a float32 copy (loses float64 exactness, but still
            # checks the graph's structure/geometry).
            try:
                m32 = _to_float32(model_or_path)
                f32 = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in feeds.items()}
                return ort.InferenceSession(m32.SerializeToString(),
                                            providers=["CPUExecutionProvider"]).run(output_names, f32)
            except Exception as f32_err:
                raise RuntimeError(f"onnxruntime: {ort_err}\nreference evaluator: {ref_err}\n"
                                   f"float32 fallback: {f32_err}")
