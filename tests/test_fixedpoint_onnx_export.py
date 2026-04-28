import os
import tempfile
import torch
import torch.nn as nn
import brevitas.nn as qnn
import onnx
import numpy as np
import pytest

from quantizers.fixedpoint_per_tensor_weights import FixedPointPerTensorWeightQuant, FixedPointPerTensorWeightQuantizer, RoundingMode


class SimpleFixedPointCNN(nn.Module):
    """A minimal CNN for CIFAR-10 using fixed-point weight quantization."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            qnn.QuantConv2d(3, 16, kernel_size=3, padding=1, weight_quant=FixedPointPerTensorWeightQuant),
            qnn.QuantReLU(),
            nn.MaxPool2d(2),
            qnn.QuantConv2d(16, 32, kernel_size=3, padding=1, weight_quant=FixedPointPerTensorWeightQuant),
            qnn.QuantReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(32 * 8 * 8, 64, weight_quant=FixedPointPerTensorWeightQuant),
            qnn.QuantReLU(),
            qnn.QuantLinear(64, num_classes, weight_quant=FixedPointPerTensorWeightQuant),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_onnx_model(onnx_path: str):
    return onnx.load(onnx_path)


def count_custom_nodes(onnx_model):
    return sum(1 for node in onnx_model.graph.node if node.op_type == "FixedPointQuant" and node.domain == "mydomain")


def get_custom_node_attributes(onnx_model):
    attrs = []
    for node in onnx_model.graph.node:
        if node.op_type == "FixedPointQuant" and node.domain == "mydomain":
            attr_dict = {}
            for a in node.attribute:
                if a.HasField("i"):
                    attr_dict[a.name] = a.i
                elif a.HasField("f"):
                    attr_dict[a.name] = a.f
                elif a.HasField("s"):
                    val = a.s
                    attr_dict[a.name] = val.decode('utf-8') if isinstance(val, bytes) else val
                elif a.HasField("t"):
                    attr_dict[a.name] = a.t
                else:
                    attr_dict[a.name] = None
            attrs.append(attr_dict)
    return attrs


class TestFixedPointOnnxExport:
    @pytest.fixture
    def model(self):
        return SimpleFixedPointCNN(num_classes=10).eval()

    @pytest.fixture
    def dummy_input(self):
        return torch.randn(1, 3, 32, 32)

    def test_export_creates_onnx_file(self, model, dummy_input, tmp_path):
        onnx_path = tmp_path / "test_model.onnx"
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            dynamo=False, opset_version=13,
            input_names=["input"], output_names=["output"]
        )
        assert onnx_path.exists()

    def test_export_contains_custom_quantizer_nodes(self, model, dummy_input, tmp_path):
        onnx_path = tmp_path / "test_model.onnx"
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            dynamo=False, opset_version=13,
            input_names=["input"], output_names=["output"]
        )
        onnx_model = get_onnx_model(str(onnx_path))
        assert count_custom_nodes(onnx_model) > 0, "Expected at least one custom FixedPointQuant node"

    def test_custom_node_attributes_are_correct(self, model, dummy_input, tmp_path):
        onnx_path = tmp_path / "test_model.onnx"
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            dynamo=False, opset_version=13,
            input_names=["input"], output_names=["output"]
        )
        onnx_model = get_onnx_model(str(onnx_path))
        attrs_list = get_custom_node_attributes(onnx_model)
        
        assert len(attrs_list) > 0
        for attrs in attrs_list:
            # ONNX attribute names do not include type suffixes (_i, _f, _s)
            assert "lsb" in attrs
            assert "bit_width" in attrs
            assert "signed" in attrs
            assert "narrow_range" in attrs
            assert "rounding_mode" in attrs
            assert "scale" in attrs
            assert "zero_point" in attrs

    def test_quantizer_parameters_roundtrip(self, tmp_path):
        """Verify that quantizer parameters (lsb, bit_width, signed, etc.) are correctly exported and match expected values."""
        # Create a quantizer instance
        quantizer = FixedPointPerTensorWeightQuantizer(
            bit_width=8,
            rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN,
            narrow_range=True
        )
        
        # Perform a dummy inference to initialize search buffers (simulating real usage)
        dummy_weights = torch.randn(10, 10)
        _ = quantizer(dummy_weights)
        
        # Force a specific lsb to test round-trip explicitly
        quantizer.search_result_lsb.fill_(-3)
        quantizer.search_result_is_signed.fill_(True)
        quantizer.search_done.fill_(True)
        
        # Create a dummy model that uses this quantizer
        class DummyModel(nn.Module):
            def __init__(self, quantizer):
                super().__init__()
                self.quant = quantizer
            
            def forward(self, x):
                return self.quant(x)
                
        model = DummyModel(quantizer).eval()
        dummy_input = torch.randn(1, 10)
        
        onnx_path = tmp_path / "test_quantizer.onnx"
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            dynamo=False, opset_version=13,
            input_names=["input"], output_names=["output"]
        )
        
        onnx_model = get_onnx_model(str(onnx_path))
        attrs_list = get_custom_node_attributes(onnx_model)
        
        assert len(attrs_list) == 1
        attrs = attrs_list[0]
        
        # Verify attributes match the quantizer state
        assert attrs["lsb"] == -3, f"Expected lsb=-3, got {attrs['lsb']}"
        assert attrs["bit_width"] == 8, f"Expected bit_width=8, got {attrs['bit_width']}"
        assert attrs["signed"] == 1, f"Expected signed=1, got {attrs['signed']}"
        assert attrs["narrow_range"] == 1, f"Expected narrow_range=1, got {attrs['narrow_range']}"
        assert attrs["rounding_mode"] == "round_to_nearest_even", f"Expected rounding_mode='round_to_nearest_even', got {attrs['rounding_mode']}"

    def test_onnx_model_validates(self, model, dummy_input, tmp_path):
        onnx_path = tmp_path / "test_model.onnx"
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            dynamo=False, opset_version=13,
            input_names=["input"], output_names=["output"]
        )
        onnx_model = get_onnx_model(str(onnx_path))
        onnx.checker.check_model(onnx_model)  # Should not raise

    def test_onnx_loads_and_parses_correctly(self, model, dummy_input, tmp_path):
        """Ensure the ONNX file can be loaded and parsed without errors."""
        onnx_path = tmp_path / "test_model.onnx"
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            dynamo=False, opset_version=13,
            input_names=["input"], output_names=["output"]
        )
        # Load and verify structure
        onnx_model = get_onnx_model(str(onnx_path))
        assert onnx_model.graph.input[0].name == "input"
        assert onnx_model.graph.output[0].name == "output"
        assert len(onnx_model.graph.node) > 0
