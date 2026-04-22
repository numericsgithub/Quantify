"""
Coefficient Per-Tensor Weight Quantizer for Brevitas.

This quantizer rounds weights to the nearest value from a set of predefined 
coefficients provided in a text file. It searches for the optimal coefficient 
set and a power-of-two scaling factor (2^n) that minimizes the Sum of 
Absolute Differences (SAD) between the original and quantized weights.

The text file format:
    Each line contains one set of coefficients.
    Coefficients within a set are separated by spaces.

Example:
    -1.0 0.0 1.0
    -0.5 -0.25 0.0 0.25 0.5
"""

import torch
import torch.nn as nn
from typing import Tuple

from brevitas.inject import BaseInjector as Injector
from brevitas.inject.enum import QuantType
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector


class CoefficientPerTensorWeightQuantizer(nn.Module):
    """
    A self-contained coefficient-based per-tensor weight quantizer.

    Usage::
        quantizer = CoefficientPerTensorWeightQuantizer(filepath="coeffs.txt")
        q_weights, scale, zero_point, bw = quantizer(linear.weight)

    Parameters
    ----------
    filepath : str
        Path to the text file containing the coefficient sets.
    """

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
        
        # Read coefficient sets from the text file during initialization
        self.coefficient_sets = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Convert space-separated values in each line to a float tensor
                    coeffs = torch.tensor([float(x) for x in line.split()], dtype=torch.float32)
                    self.coefficient_sets.append(coeffs)

        if not self.coefficient_sets:
            raise ValueError(f"No valid coefficient sets found in file: {filepath}")

        # Register search results as buffers to ensure they are serialized in state_dict
        self.register_buffer('search_done', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('best_set_idx', torch.tensor(0, dtype=torch.long))
        self.register_buffer('best_n', torch.tensor(0, dtype=torch.long))

    def forward(
        self, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize weights by finding the best coefficient set and scaling factor.

        Returns
        -------
        quantized : torch.Tensor
            Weights snapped to the best scaled coefficient grid.
        scale : torch.Tensor
            The chosen scaling factor 2^n.
        zero_point : torch.Tensor
            Always 0.0.
        bit_width : torch.Tensor
            The number of coefficients in the chosen set.
        """
        device = weights.device
        dtype = weights.dtype

        if not self.search_done.item():
            best_sad = float("inf")
            best_set_idx = 0
            best_n = 0

            # Search through all coefficient sets and scalings 2^n for n in [-12, 12]
            for idx, coeffs in enumerate(self.coefficient_sets):
                # Move coeffs to device for computation
                coeffs_dev = coeffs.to(device)
                
                for n in range(-12, 13):
                    s = 2.0 ** n
                    scaled_coeffs = coeffs_dev * s
                    
                    # Find nearest coefficient for each weight
                    # weights: (W,), scaled_coeffs: (C,) -> diffs: (W, C)
                    diffs = torch.abs(weights.unsqueeze(-1) - scaled_coeffs)
                    min_indices = torch.argmin(diffs, dim=-1)
                    quantized_temp = scaled_coeffs[min_indices]
                    
                    # Calculate Sum of Absolute Differences (SAD)
                    sad = torch.sum(torch.abs(weights - quantized_temp)).item()
                    
                    if sad < best_sad:
                        best_sad = sad
                        best_set_idx = idx
                        best_n = n

            self.best_set_idx.fill_(best_set_idx)
            self.best_n.fill_(best_n)
            self.search_done.fill_(True)
        else:
            best_set_idx = self.best_set_idx.item()
            best_n = self.best_n.item()

        # Apply the optimal quantization
        chosen_coeffs = self.coefficient_sets[best_set_idx].to(device)
        s = 2.0 ** best_n
        scaled_coeffs = chosen_coeffs * s
        
        diffs = torch.abs(weights.unsqueeze(-1) - scaled_coeffs)
        min_indices = torch.argmin(diffs, dim=-1)
        quantized = scaled_coeffs[min_indices]

        scale = torch.tensor(s, dtype=dtype, device=device)
        zero_point = torch.tensor(0.0, dtype=dtype, device=device)
        bw = torch.tensor(float(len(chosen_coeffs)), device=device)

        return quantized, scale, zero_point, bw


class CoefficientPerTensorWeightQuant(Injector):
    """
    Brevitas-compatible Injector for the coefficient-based per-tensor weight
    quantizer.

    Usage::
        from brevitas.nn import QuantLinear
        layer = QuantLinear(
            in_features=64,
            out_features=32,
            bias=True,
            weight_quant=CoefficientPerTensorWeightQuant,
        )

    Override class attributes to customise::
        class MyCoeffQuant(CoefficientPerTensorWeightQuant):
            filepath = "my_custom_coeffs.txt"
    """

    quant_type = QuantType.INT
    proxy_class = WeightQuantProxyFromInjector
    tensor_quant = CoefficientPerTensorWeightQuantizer
    
    # Path to the text file containing coefficient sets. 
    # This will be passed to the CoefficientPerTensorWeightQuantizer constructor.
    filepath = "coefficients.txt"
    
    signed = True
