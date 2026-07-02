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


class AnnealingBlendFn(torch.autograd.Function):
    """Compute (1-alpha)*x + alpha*quantized in the forward pass, but present
    a single straight-through node in the backward graph instead of the
    Add/Mul chain that tensor arithmetic would produce.

    alpha is passed as a plain Python float so it never appears as a graph
    input.  Gradient goes entirely through the `x` input (slope = 1); `None`
    is returned for `quantized` to avoid double-counting — both x and
    quantized are derived from the same upstream leaf.
    """

    @staticmethod
    def forward(ctx, x, quantized, alpha):
        return (1.0 - alpha) * x + alpha * quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


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

        # Diagnostics state (not buffers — ephemeral, not needed in checkpoints)
        self._calibration_count: int = 0
        self._was_annealing: bool = False
        self._post_annealing_fired: bool = False
        self._last_snapshot_seen: int = 0

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
        _calibration_triggered = should_calibrate and not is_exporting

        if not is_exporting and should_calibrate:
            if not self.training and self.annealing_alpha.item() > 0.0:
                qid = getattr(self, "quant_id", repr(id(self)))
                raise RuntimeError(
                    f"Quantizer {qid!r} has not been calibrated (search_done=False) "
                    f"but is active (annealing_alpha={self.annealing_alpha.item():.2f}) "
                    f"while the model is in eval mode. "
                    f"Quantizing with uncalibrated parameters produces garbage output. "
                    f"Call QuantizerManager().disable_quantization() before evaluating "
                    f"an uncalibrated model, or run a calibration forward pass in "
                    f"training mode first."
                )
            params = self._calibrate(x)
            self._save_calibration(params)
            # Reset global flag after triggering recalibration to avoid forcing it on every forward
            self.quantizer_manager.reset_global_flag()
        else:
            params = self._load_calibration()

        # 3. Quantize & format output
        quantized = self._quantize(x, params)
        scale, zero_point, bit_width = self._get_metadata(params, x)

        alpha_before = self.annealing_alpha.item()
        if alpha_before < 1.0:
            result = AnnealingBlendFn.apply(x, quantized, alpha_before)
            if self.training:
                new_alpha = min(alpha_before + self.annealing_alpha_step, 1.0)
                self.annealing_alpha.data.fill_(new_alpha)
        else:
            result = quantized

        # 4. Diagnostics (runs only when diagnostics_dir is set; never in ONNX export)
        if not is_exporting and self.quantizer_manager.diagnostics_dir is not None:
            self._maybe_run_diagnostics(x, quantized, params, _calibration_triggered, alpha_before)

        return result, scale, zero_point, bit_width

    def backward(ctx, grad_quantized, grad_scale, grad_zero_point, grad_bw):
        print("grad_quantizedgrad_quantized", grad_quantized)
        # Straight-Through Estimator: pass gradient through for the first input
        return grad_quantized, None, None, None, None, None, None, None

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

    def _get_diagnostics_params(self, params: Any) -> Optional[dict]:
        """
        Return {lsb, bit_width, signed} for diagnostics, or None to skip.
        Override in subclasses that have a well-defined LSB / step size.
        """
        return None

    def _maybe_run_diagnostics(
        self,
        x: torch.Tensor,
        quantized: torch.Tensor,
        params: Any,
        calibration_triggered: bool,
        alpha_before: float,
    ) -> None:
        diag_params = self._get_diagnostics_params(params)
        if diag_params is None:
            return

        from pathlib import Path
        from utils.quantizer_diagnostics import run_diagnostics

        out_dir = Path(self.quantizer_manager.diagnostics_dir)
        qid = getattr(self, "quant_id", "unknown")

        def _emit(trigger: str) -> None:
            run_diagnostics(
                quant_id=qid,
                x=x,
                quantized=quantized,
                trigger=trigger,
                out_dir=out_dir,
                **diag_params,
            )

        # Track whether annealing was ever active on this quantizer
        if alpha_before < 1.0:
            self._was_annealing = True

        # Trigger 1: calibration just ran successfully
        if calibration_triggered and self.search_done.item():
            self._calibration_count += 1
            _emit(f"calibration_{self._calibration_count}")

        # Trigger 2: annealing just finished (alpha crossed 1.0 this pass)
        if (
            self.annealing_alpha.item() >= 1.0
            and self._was_annealing
            and not self._post_annealing_fired
        ):
            self._post_annealing_fired = True
            _emit("post_annealing")

        # Trigger 3: snapshot requested by manager
        mgr_snap = self.quantizer_manager._snapshot_count
        if mgr_snap > self._last_snapshot_seen:
            self._last_snapshot_seen = mgr_snap
            _emit(f"snapshot_{mgr_snap:04d}")
