# Canonical ONNX export

`utils/onnx_export.py::export_onnx_with_io` rejects (by default) any graph
that is not canonical — see `utils/onnx_canonical.py` and pitfall #24 in
`docs/llm/pitfalls/brevitas_pitfalls.md`. Build models so they pass:

| Instead of | Use |
|---|---|
| `qnn.QuantConv2d(..., stride=2, padding=1)` | `quantizers.layers.QuantConv2d(..., stride=2, padding="same")` |
| `nn.MaxPool2d(3, 2, padding=1)` / `ceil_mode=True` | `quantizers.layers.MaxPool2d(3, 2, padding="same")` |
| `nn.AvgPool2d(k, s, padding=p)` | `quantizers.layers.AvgPool2d(k, s, padding="same")` |
| `nn.ReLU` / `F.relu` / `qnn.QuantReLU` between layers | `quantizers.QuantReLU` (likewise `QuantSigmoid`, `QuantSoftmax`, `QuantHardSwish`, ...) — `act_quant=None, input_quant=None` for an unquantized one |
| `x.unsqueeze(1)` / `.transpose()` / embedding lookup before the first layer | feed the input already in the first layer's shape |
| `nn.MultiheadAttention`, functional `F.linear` | `nn.Linear` / `qnn.QuantLinear` modules (each exports as `Quantify::Dense`) |

VALID padding (`padding=0`) is always fine. A plain activation is allowed
only when its output is a graph output (e.g. a final `nn.Softmax`).
LayerNorm and activations without a Quantify equivalent only warn.

Escape hatch for toy/debug graphs: `export_onnx_with_io(..., canonical_checks="warn")`
or `"off"`. Run the checks on an existing file with
`utils.onnx_canonical.check_onnx_graph(onnx.load(path))` -> `(violations, warnings)`.

## Adding a new custom-node lowering

Pattern used by `quantizers/activations.py` (`ReluFn` ...) and
`quantizers/layers.py` (`DenseFn`, `SameConvFn` ...):

1. A `torch.autograd.Function` whose `forward` computes the eager op and
   whose `symbolic` emits the single node (`g.op("Quantify::<Name>", ...)` or
   a standard op with explicit attributes); `.setType(...)` the output.
2. Call it only `if torch.onnx.is_in_onnx_export()`; otherwise run the plain
   op, so training keeps its exact gradient and no `backward()` is needed.
3. For a `Quantify::*` node, add a `FunctionProto` builder to
   `utils/onnx_self_contained.py::_BUILDERS` (attribute refs use the
   suffix-stripped names — pitfall #22) and a case to
   `tests/test_onnx_self_contained.py`.
