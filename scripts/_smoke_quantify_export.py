"""Smoke test: build SimpleMNISTNet, warm calibration with one forward, export, inspect."""
import collections
import sys
import torch

sys.path.insert(0, ".")
from examples.simple_mnist_qat import SimpleMNISTNet
from utils.onnx_export import export_onnx_with_io


def main():
    torch.manual_seed(0)
    model = SimpleMNISTNet()
    dummy = torch.randn(1, 1, 28, 28)
    model.eval()
    with torch.no_grad():
        _ = model(dummy)  # populate calibration buffers
    out_path = "_smoke_quantify.onnx"
    export_onnx_with_io(model, dummy, out_path)

    import onnx
    m = onnx.load(out_path)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    doms = collections.Counter(n.domain for n in m.graph.node)
    print("ops:", dict(ops))
    print("domains:", dict(doms))
    fp = ops.get("FixedPointQuant", 0)
    print(f"FixedPointQuant nodes: {fp}")
    if fp == 0:
        sys.exit("FAIL: no Quantify::FixedPointQuant nodes in graph")
    print("OK: Quantify nodes present")


if __name__ == "__main__":
    main()
