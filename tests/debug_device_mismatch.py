"""
Debug script to identify device mismatches in QuantMobileNetV2.
Run with: python tests/debug_device_mismatch.py
"""
import torch
import torch.nn as nn
from models.mobilenetv2_quant import QuantMobileNetV2


def print_device_locations(model, device_name="cuda"):
    print(f"\n=== Device Locations for model on {device_name} ===")
    
    # 1. Parameters
    print("\n[Parameters]")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.device}")
        
    # 2. Buffers
    print("\n[Buffers]")
    for name, buf in model.named_buffers():
        print(f"  {name}: {buf.device}")
        
    # 3. Custom Quantizer Internal Buffers
    print("\n[Custom Quantizer Buffers]")
    for name, module in model.named_modules():
        if hasattr(module, 'search_done'):
            print(f"  Module: {name}")
            print(f"    search_done: {module.search_done.device}")
            print(f"    search_result_is_signed: {module.search_result_is_signed.device}")
            print(f"    search_result_lsb: {module.search_result_lsb.device}")
            
    # 4. Check for any nn.Module that might have been missed
    print("\n[All Modules with 'quant' in name]")
    for name, module in model.named_modules():
        if 'quant' in name.lower() or 'Quant' in name:
            print(f"  {name}: {type(module).__name__}")
            for sub_name, sub_mod in module.named_modules():
                if hasattr(sub_mod, 'search_done'):
                    print(f"    -> {sub_name}: search_done on {sub_mod.search_done.device}")


def test_device_mismatch():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA test.")
        return

    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    model = QuantMobileNetV2(num_classes=1000, weight_bit_width=8, act_bit_width=8).to(device)
    print_device_locations(model, device)
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    print(f"\nInput device: {dummy_input.device}")
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"\nOutput device: {output.device}")
        print("Forward pass succeeded.")
    except RuntimeError as e:
        print(f"\nForward pass failed with error: {e}")
        print("\nRe-printing locations after failure:")
        print_device_locations(model, device)


if __name__ == "__main__":
    test_device_mismatch()
