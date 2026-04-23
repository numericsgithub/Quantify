import torch
import torch.nn as nn
import logging

def load_pretrained_weights(quant_model: nn.Module, float_model: nn.Module):
    """
    Maps weights from a floating-point model to a quantized Brevitas model.
    
    This function assumes that the architecture of the quant_model mirrors 
    the float_model. It loads the state dict with strict=False because 
    Brevitas layers contain additional buffers and parameters for 
    quantization that are not present in standard PyTorch layers.
    """
    logging.info("Mapping pretrained floating-point weights to quantized model...")
    float_state_dict = float_model.state_dict()
    quant_state_dict = quant_model.state_dict()

    # Filter out keys that don't exist in the quantized model or are not parameters/buffers
    # In most cases, if the architecture is mirrored, the keys will match.
    filtered_dict = {k: v for k, v in float_state_dict.items() if k in quant_state_dict}
    
    missing_keys, unexpected_keys = quant_model.load_state_dict(filtered_dict, strict=False)
    
    logging.info(f"Successfully mapped {len(filtered_dict)} tensors.")
    if missing_keys:
        logging.debug(f"Missing keys (expected for Brevitas): {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected keys: {unexpected_keys}")

    return quant_model
