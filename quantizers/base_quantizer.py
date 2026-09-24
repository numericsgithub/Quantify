"""
Base Quantizer Infrastructure for Brevitas.

Provides shared boilerplate for per-tensor quantizers, including:
- Calibration state management
- ONNX export guards
- Brevitas 4-tuple return contract
- Configurable inference gating (decoupled from global state)
"""

import logging
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

from quantizers.manager import QuantizerManager

logger = logging.getLogger("quantizers")


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


class ClippedSTEFn(torch.autograd.Function):
    """Clipped Straight-Through Estimator, shared by all quantizers.

    Forward is a value-preserving identity on `quantized` (so the numeric
    output of the quantizer is completely unchanged). Backward multiplies the
    incoming gradient by a precomputed in-range mask, so the local slope is 1
    for inputs that landed INSIDE the quantizer's representable range and 0 for
    inputs the forward clamp saturated. This is the state-of-the-art refinement
    over plain STE, which uses slope 1 everywhere.

    The mask is a 0/1 tensor (in the quantized tensor's dtype) computed once in
    BaseQuantizer.forward from the ORIGINAL float input and the quantizer's
    range bounds; `None` is returned for it in backward since it is not a
    differentiable input.
    """

    @staticmethod
    def forward(ctx, quantized, in_range_mask):
        ctx.save_for_backward(in_range_mask)
        # Return a fresh tensor (not the input alias) so autograd treats this as
        # a distinct node; the value is identical to `quantized`.
        return quantized.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (in_range_mask,) = ctx.saved_tensors
        return grad_output * in_range_mask, None


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
        clipped_ste: bool = False,
        **kwargs
    ):
        super().__init__()
        self.bit_width = bit_width
        # Clipped-STE toggle lives here (shared by every quantizer) so any
        # subclass can honor it just by overriding _in_range_mask(). Plain STE
        # (slope 1 everywhere) remains the default. See ClippedSTEFn and
        # _in_range_mask below.
        self.clipped_ste = clipped_ste
        self.inference_counter = 0
        self.inference_sequence_id = -1
        self.annealing_alpha_step = 0.1

        # Register annealing state buffers for checkpoint persistence
        self.register_buffer('annealing_alpha', torch.tensor(1.0))

        # Calibration state buffers
        self.register_buffer('search_done', torch.tensor(False, dtype=torch.bool))

        # Python-side mirror of search_done, invalidated via the buffer's
        # autograd `._version` counter rather than trusted blindly. Every
        # forward call reads search_done to make a control-flow decision;
        # reading it via `.item()` forces a full GPU sync (queue drain +
        # device->host copy) every single time, even though it's almost
        # always unchanged between calls. `._version` is bumped by any
        # in-place write made directly to the tensor -- our own
        # set_search_done() below, load_state_dict(), or external code that
        # pokes the buffer directly (PTQ LSB-override scripts/tests
        # deliberately do this to inject a calibration result without going
        # through _calibrate) -- and reading `._version` needs no sync (it's
        # plain CPU-side autograd bookkeeping), so `search_done_value` below
        # can detect staleness and only pay the `.item()` cost when the
        # buffer actually changed since the last read, no matter who changed
        # it. Subclasses with their own extra buffer mirrors (e.g.
        # search_result_lsb) should use `_cached_scalar()` the same way.
        #
        # `annealing_alpha` does NOT get this treatment -- see
        # `annealing_alpha_value`'s docstring for why a `.data.fill_()` write
        # (the convention used for this buffer throughout the codebase)
        # can't be detected this way.
        self._search_done_cached: bool = False
        self._search_done_version: int = -1

        # Use provided manager or create a local instance to avoid global state
        self.quantizer_manager = quantizer_manager if quantizer_manager is not None else QuantizerManager()
        
        # Register with manager for coordination
        self.quantizer_manager.register_quantizer(self)

        # Diagnostics state (not buffers — ephemeral, not needed in checkpoints)
        self._calibration_count: int = 0
        self._was_annealing: bool = False
        self._post_annealing_fired: bool = False
        self._last_snapshot_seen: int = 0

        # Lifecycle-event logging state (not buffers — ephemeral, one-shot per
        # quantizer *object*; a `load_state_dict()` call recreates the proxy
        # and therefore this object, see pitfall #12 in
        # docs/llm/pitfalls/brevitas_pitfalls.md, so events will be logged
        # again -- once -- for the new object even if the "real" milestone
        # happened earlier, before the checkpoint was saved).
        self._log_gate_opened: bool = False
        self._log_calibration_count: int = 0
        self._log_annealing_started: bool = False
        self._log_annealing_complete: bool = False

    def _cached_scalar(self, buffer: torch.Tensor, cache_attr: str, version_attr: str, cast):
        """Read a 0-dim buffer as a Python scalar, paying the `.item()` GPU
        sync only if `buffer` was mutated (its `._version` differs from the
        version recorded at the last read) since the last call -- regardless
        of whether the mutation went through one of this class's own setters
        or came from external code poking the buffer directly. See the note
        in `__init__` above `_search_done_cached`.
        """
        current_version = buffer._version
        if current_version != getattr(self, version_attr):
            setattr(self, cache_attr, cast(buffer.item()))
            setattr(self, version_attr, current_version)
        return getattr(self, cache_attr)

    @property
    def search_done_value(self) -> bool:
        """Cached, sync-free (when unchanged) read of `self.search_done`."""
        return self._cached_scalar(self.search_done, "_search_done_cached", "_search_done_version", bool)

    @property
    def annealing_alpha_value(self) -> float:
        """Read of `self.annealing_alpha` -- deliberately *not* cached across
        calls, unlike `search_done_value`/the subclass LSB caches.

        `._version` (the staleness signal `_cached_scalar` relies on) is only
        bumped by in-place writes to the tensor *itself*, not through a
        `.data` view of it -- `t.data.fill_(x)` leaves `t._version` unchanged
        (verified empirically; `.data` carries its own, separate autograd
        metadata). `annealing_alpha` is conventionally written as
        `q.annealing_alpha.data.fill_(...)` throughout this codebase --
        `manager.py` used to, several tests still do, and so do
        `examples/find_perfect_lsbs_imagenet_ptq.py` /
        `examples/train_imagenet_qat.py` -- so a version-checked cache would
        silently go stale against any of those. `search_done` and the
        subclass search-result buffers are, empirically, never written via
        `.data.fill_()` anywhere in this codebase, so their caches are safe.
        Forward() still only reads this once per call (into `alpha_before`,
        reusing that local for the rest of the call) rather than the 2-3
        `.item()` calls it used to make, so this is still a real reduction --
        just not a cross-call one.
        """
        return float(self.annealing_alpha.item())

    def set_search_done(self, value: bool) -> None:
        """Set `search_done`. Equivalent to `self.search_done.fill_(value)`
        plus immediately refreshing the cache (so the very next read doesn't
        even need the cheap `._version` check to know it's current) -- purely
        a convenience, `self.search_done.fill_(value)` directly is just as
        correct since `search_done_value` self-invalidates either way."""
        self.search_done.fill_(value)
        self._search_done_cached = bool(value)
        self._search_done_version = self.search_done._version

    def set_annealing_alpha(self, value: float) -> None:
        """Set `annealing_alpha`. Uses plain `.fill_()` (not `.data.fill_()`)
        -- see `annealing_alpha_value` for why that distinction matters."""
        self.annealing_alpha.fill_(value)

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

        if not self._log_gate_opened:
            self._log_gate_opened = True
            qid = getattr(self, "quant_id", repr(id(self)))
            gap = self.inference_sequence_id * self.quantizer_manager.quantization_start_gap
            logger.info(
                "Quantizer %r: gate opened, starting to quantize (inference_sequence_id=%d, "
                "quantization_start_gap=%d, waited %d gated-off forward call(s)).",
                qid, self.inference_sequence_id, self.quantizer_manager.quantization_start_gap, gap,
            )

        # 2. Calibration check
        is_exporting = torch.onnx.is_in_onnx_export()
        should_calibrate = not self.search_done_value or self.quantizer_manager.force_recalibration
        _calibration_triggered = should_calibrate and not is_exporting

        if not is_exporting and should_calibrate:
            if not self.training and self.annealing_alpha_value > 0.0:
                qid = getattr(self, "quant_id", repr(id(self)))
                raise RuntimeError(
                    f"Quantizer {qid!r} has not been calibrated (search_done=False) "
                    f"but is active (annealing_alpha={self.annealing_alpha_value:.2f}) "
                    f"while the model is in eval mode. "
                    f"Quantizing with uncalibrated parameters produces garbage output. "
                    f"Call QuantizerManager().disable_quantization() before evaluating "
                    f"an uncalibrated model, or run a calibration forward pass in "
                    f"training mode first."
                )
            params = self._calibrate(x)
            self._save_calibration(params)
            self._log_calibration_count += 1
            qid = getattr(self, "quant_id", repr(id(self)))
            logger.info(
                "Quantizer %r: calibration %s (search_done -> True).",
                qid, "completed" if self._log_calibration_count == 1 else
                f"re-run (#{self._log_calibration_count}, force_recalibration)",
            )
            # Reset global flag after triggering recalibration to avoid forcing it on every forward
            self.quantizer_manager.reset_global_flag()
        else:
            params = self._load_calibration()

        # 3. Quantize & format output
        quantized = self._quantize(x, params)
        scale, zero_point, bit_width = self._get_metadata(params, x)

        # 3b. Clipped STE (optional, shared by all quantizers): zero the
        # gradient for inputs the forward clamp saturated. The mask is derived
        # from the ORIGINAL float input `x` and the quantizer's range bounds.
        # A subclass that does not define its range returns None here, in which
        # case clipped_ste is a no-op (plain STE). Value is unchanged; only the
        # backward slope is masked. Skipped during ONNX export to keep that
        # path's graph unchanged.
        if self.clipped_ste and not is_exporting:
            in_range_mask = self._in_range_mask(x, params)
            if in_range_mask is not None:
                quantized = ClippedSTEFn.apply(quantized, in_range_mask.to(dtype=x.dtype))

        # No .item() anywhere in this block: alpha_before comes from the
        # cache, and current_alpha is tracked in Python since we're the ones
        # setting it via set_annealing_alpha() (which updates the cache too).
        alpha_before = self.annealing_alpha_value
        current_alpha = alpha_before
        if alpha_before < 1.0:
            if not self._log_annealing_started:
                self._log_annealing_started = True
                qid = getattr(self, "quant_id", repr(id(self)))
                logger.info(
                    "Quantizer %r: annealing started (annealing_alpha=%.3f, step=%.4f) -- "
                    "output is a (1-alpha)*float + alpha*quantized blend until alpha reaches 1.0.",
                    qid, alpha_before, self.annealing_alpha_step,
                )
            result = AnnealingBlendFn.apply(x, quantized, alpha_before)
            if self.training:
                current_alpha = min(alpha_before + self.annealing_alpha_step, 1.0)
                self.set_annealing_alpha(current_alpha)
        else:
            result = quantized

        if current_alpha >= 1.0 and not self._log_annealing_complete:
            self._log_annealing_complete = True
            qid = getattr(self, "quant_id", repr(id(self)))
            logger.info(
                "Quantizer %r: annealing complete (annealing_alpha=1.0) -- output is now fully quantized.",
                qid,
            )

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

    def _in_range_mask(self, x: torch.Tensor, params: Any) -> Optional[torch.Tensor]:
        """
        Return a boolean/0-1 tensor (same shape as x) that is True where the
        float input x lies INSIDE the quantizer's representable range and False
        where the forward clamp saturated it. Used only when clipped_ste=True.

        Boundary convention is left to the subclass but should be inclusive
        (the exact min/max grid values count as in-range).

        The base implementation returns None, meaning "this quantizer does not
        define a range" -> clipped STE degrades to plain STE (no masking).
        Subclasses with a well-defined grid (e.g. fixed-point) override this.
        """
        return None

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
        if calibration_triggered and self.search_done_value:
            self._calibration_count += 1
            _emit(f"calibration_{self._calibration_count}")

        # Trigger 2: annealing just finished (alpha crossed 1.0 this pass)
        if (
            self.annealing_alpha_value >= 1.0
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


def reset_calibration_state(model: nn.Module, preserve_calibrated: bool = False) -> int:
    """Reset `search_done` to False for every `BaseQuantizer` submodule of
    `model`, forcing recalibration on the next forward pass.

    Used by `training_harness` (`checkpointing.py`, `trainer_v2.py`) instead
    of the previous pattern of iterating `model.named_buffers()` and
    string-matching on `"search_done"`/`"calibration_done"`: that was fragile
    (name matching) and, for the preserve-calibrated variant, called
    `buf.item()` (a GPU sync) per quantizer where `search_done_value` (cached,
    self-invalidating) does just as well.

    Args:
        model: Model to walk.
        preserve_calibrated: If True, quantizers that are already calibrated
            (`search_done=True`) are left untouched instead of being reset.

    Returns:
        How many quantizers were actually reset.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, BaseQuantizer):
            if preserve_calibrated and module.search_done_value:
                continue
            module.set_search_done(False)
            count += 1
    return count
