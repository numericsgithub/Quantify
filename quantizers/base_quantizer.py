"""
Base Quantizer Infrastructure for Brevitas.

Provides shared boilerplate for per-tensor quantizers, including:
- Calibration state management
- ONNX export guards
- Brevitas 4-tuple return contract
- Configurable inference gating (decoupled from global state)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

from quantizers.manager import QuantizerManager


class BaseQuantizer(nn.Module, ABC):
    """
    Abstract base class for per-tensor quantizers.
    
    Handles calibration state, ONNX export guards, and Brevitas 4-tuple return contract.
    Subclasses implement domain-specific calibration and quantization math.
    Gating is now configurable per-instance to avoid global state coupling.
    """

    def __init__(
        self,
        bit_width: int = 8,
        quantizer_manager: Optional[QuantizerManager] = None,
        **kwargs
    ):
        super().__init__()
        self.bit_width = bit_width
        self.inference_counter = 0
        self.inference_sequence_id = -1
        self.annealing_alpha_step = 0.1

        # Register annealing state buffers for checkpoint persistence
        self.register_buffer('annealing_alpha', torch.tensor(1.0))

        # Calibration state buffers
        self.register_buffer('search_done', torch.tensor(False, dtype=torch.bool))
        
        # Use provided manager or create a local instance to avoid global state
        self.quantizer_manager = quantizer_manager if quantizer_manager is not None else QuantizerManager()
        
        # Register with manager for coordination
        self.quantizer_manager.register_quantizer(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.inference_sequence_id == -1:
            self.inference_sequence_id = self.quantizer_manager.get_inference_sequence_id()

        # 1. Inference gating
        perform_quantization = True
        if self.inference_counter < self.inference_sequence_id * self.quantizer_manager.quantization_start_gap:
            if self.training:
                self.inference_counter += 1
            perform_quantization = False
            
        if not perform_quantization:
            return x, torch.tensor(1.0, dtype=x.dtype, device=x.device), \
                   torch.tensor(0.0, dtype=x.dtype, device=x.device), \
                   torch.tensor(float(self.bit_width), dtype=x.dtype, device=x.device)

        # 2. Calibration check
        is_exporting = torch.onnx.is_in_onnx_export()
        should_calibrate = not self.search_done.item() or self.quantizer_manager.force_recalibration
        
        if not is_exporting and should_calibrate:
            params = self._calibrate(x)
            self._save_calibration(params)
            # Reset global flag after triggering recalibration to avoid forcing it on every forward
            self.quantizer_manager.reset_global_flag()
        else:
            params = self._load_calibration()
            
        # 3. Quantize & format output
        quantized = self._quantize(x, params)
        scale, zero_point, bit_width = self._get_metadata(params, x)

        if self.annealing_alpha < 1.0:
            result = (1 - self.annealing_alpha) * x + self.annealing_alpha * quantized
            if self.training:
                new_alpha = min(self.annealing_alpha.item() + self.annealing_alpha_step, 1.0)
                self.annealing_alpha.data.fill_(new_alpha)
        else:
            result = quantized

        return result, scale, zero_point, bit_width

    # Abstract methods for subclasses
    @abstractmethod
    def _calibrate(self, x: torch.Tensor) -> Any:
        """Run calibration/search logic and return a params dict."""
        raise NotImplementedError

    @abstractmethod
    def _save_calibration(self, params: Any) -> None:
        """Save calibration results to buffers."""
        raise NotImplementedError

    @abstractmethod
    def _load_calibration(self) -> Any:
        """Load calibration results from buffers."""
        raise NotImplementedError

    @abstractmethod
    def _quantize(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        """Apply quantization using the provided parameters."""
        raise NotImplementedError

    @abstractmethod
    def _get_metadata(self, params: Any, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scale, zero_point, and bit_width tensors matching x's dtype/device."""
        raise NotImplementedError
