import onnx
from onnx import numpy_helper

def print_attribute(attr):
    """Pretty-print ONNX attribute depending on type."""
    if attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif attr.type == onnx.AttributeProto.INT:
        return attr.i
    elif attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8")
    elif attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    else:
        return f"<unsupported type {attr.type}>"

def inspect_onnx_model(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    print("\n===== ONNX MODEL INSPECTION =====\n")

    print("Nodes:\n")

    for idx, node in enumerate(graph.node):
        print(f"Node {idx}")
        print(f"  Name  : {node.name if node.name else '<unnamed>'}")
        print(f"  OpType: {node.op_type}")

        print(f"  Inputs : {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")

        if node.attribute:
            print("  Attributes:")
            for attr in node.attribute:
                value = print_attribute(attr)
                print(f"    - {attr.name}: {value}")
        else:
            print("  Attributes: None")

        print("-" * 50)

    print("\n===== DONE =====\n")

if __name__ == "__main__":
    model_path = "simple_mnist_fixedpoint.onnx"  # <-- change this
    inspect_onnx_model(model_path)