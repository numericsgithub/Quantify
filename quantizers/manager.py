class QuantizerManager:
    """
    Singleton manager object for coordinating quantizer instances across the entire project.
    Ensures a single shared reference exists for global coordination, such as forcing 
    re-calibration or tracking global quantization statistics.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization on subsequent calls to __new__
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        # Global flag to force all quantizers to re-run their search/calibration
        self.force_recalibration = False
        self.quantization_start_gap = 0
        # Registry to keep track of all active quantizer instances {id: quantizer}
        self.quantizers = {}
        # Counter to generate unique identifiers
        self._id_counter = 0
        self._inference_sequence_id_counter = 0
        # Diagnostics
        self.diagnostics_dir = None   # set by trainer; None disables diagnostics
        self._snapshot_count = 0      # incremented by request_snapshot()

    def reset(self):
        """
        Reset the manager's internal state. Useful for testing or restarting experiments.
        """
        self.force_recalibration = False
        self.quantization_start_gap = 0
        self.quantizers.clear()
        self._id_counter = 0
        self._inference_sequence_id_counter = 0
        self.diagnostics_dir = None
        self._snapshot_count = 0

    @property
    def is_quantizing_everything_fully(self):
        for quant in self.quantizers.values():
            if quant.annealing_alpha != 1.0:
                return False
        return True

    @property
    def is_not_quantizing_at_all(self):
        for quant in self.quantizers.values():
            if quant.annealing_alpha != 0.0:
                return False
        return True

    def stop_quantization_for_n_inferences(self, n):
        for quant in self.quantizers.values():
            quant.inference_counter = -n

    def set_annealing_for_n_inferences(self, n, skip_calibrated=False):
        """
        Args:
            n: Number of forward passes over which annealing_alpha ramps 0 -> 1.
            skip_calibrated: If True, quantizers that already have
                search_done=True (e.g. loaded from a PTQ checkpoint) are set
                straight to annealing_alpha=1.0 with no further annealing,
                instead of being reset to annealing_alpha=0.0 and made to
                ramp up again like a freshly-calibrated quantizer.
        """
        if n < 1:
            n = 1
        alpha_step = 1.0/n
        for quant in self.quantizers.values():
            if skip_calibrated and quant.search_done.item():
                quant.annealing_alpha.data.fill_(1.0)
                quant.annealing_alpha_step = 0.0
                continue
            quant.annealing_alpha.data.fill_(0)
            quant.annealing_alpha_step = alpha_step

    def skip_gating_for_calibrated_quantizers(self) -> None:
        """
        For quantizers already calibrated (search_done=True), set
        inference_counter so the staggered quantization_start_gap gate in
        BaseQuantizer.forward() is immediately satisfied — they become active
        on their very next forward pass instead of individually waiting
        inference_sequence_id * quantization_start_gap forward calls.

        Gating and annealing are independent: set_annealing_for_n_inferences(
        skip_calibrated=True) only bypasses the annealing ramp (alpha -> 1.0
        immediately). Gating is checked FIRST in BaseQuantizer.forward(),
        before alpha/search_done are even read, so without this a "preserved"
        quantizer silently keeps running as float passthrough for its own
        sequence_id * quantization_start_gap calls despite alpha already
        being 1.0.

        Quantizers still at inference_sequence_id == -1 (never reached by a
        forward pass yet) are left untouched — they'll get a real sequence_id
        and the normal gating treatment on their first forward call, same as
        before.
        """
        for q in self.quantizers.values():
            if q.search_done.item() and q.inference_sequence_id != -1:
                q.inference_counter = q.inference_sequence_id * self.quantization_start_gap

    def disable_quantization(self):
        """Disable quantization by setting annealing_alpha and annealing_alpha_step to zero for all registered quantizers."""
        for quant in self.quantizers.values():
            quant.annealing_alpha.data.fill_(0.0)
            quant.annealing_alpha_step = 0.0

    def enable_quantization(self):
        """Enable quantization by setting annealing_alpha to one and annealing_alpha_step to 0.1 for all registered quantizers."""
        for quant in self.quantizers.values():
            quant.annealing_alpha.data.fill_(1.0)
            quant.annealing_alpha_step = 0.1

    def register_quantizer(self, quantizer):
        """
        Registers a quantizer instance with the manager and assigns it a unique ID.
        """
        if quantizer in self.quantizers.values():
            return

        qid = f"quant_{self._id_counter}"
        self.quantizers[qid] = quantizer
        
        # Assign the unique ID back to the quantizer object for easy reference
        quantizer.quant_id = qid
        quantizer.inference_sequence_id = -1

        self._id_counter += 1

    def get_inference_sequence_id(self):
        self._inference_sequence_id_counter += 1
        return self._inference_sequence_id_counter - 1

    def quantizers_in_execution_order(self, *, include_unreached: bool = False):
        """
        Return registered quantizers sorted by true forward-execution order.

        self.quantizers iterates in __init__ attribute-declaration order (it's
        built from named_modules()/registration order), which does NOT match
        forward() call order in general — e.g. QuantResNet18 declares
        input_quant after layer1..4, but forward() calls input_quant first.
        inference_sequence_id is assigned per-quantizer via a monotonic counter
        (get_inference_sequence_id) on each quantizer's FIRST forward() call,
        so after at least one forward pass it IS the true execution order. Use
        this whenever processing order matters (e.g. a greedy per-quantizer PTQ
        search) instead of iterating self.quantizers directly.

        Architectural note: this manager is a flat, model-agnostic singleton
        registry with no notion of "the model" it belongs to. This method only
        ever sorts whatever is CURRENTLY registered; it is meaningless across
        multiple models unless callers follow the existing
        QuantizerManager().reset() + re-register convention before building
        each new model. This method does not fix that assumption — it inherits it.

        Args:
            include_unreached: If False (default), drop quantizers whose
                inference_sequence_id is still -1 — registered with the manager
                but never reached by a forward() call. Brevitas's
                injector-resolution machinery creates throwaway quantizer
                objects like this that register but are never wired into the
                model's real module tree. Set True to keep them (e.g. for
                diagnostics that want to see what never fired).

        Raises:
            RuntimeError: if at least one quantizer is registered but NONE has
                been reached by a forward pass yet. Execution order is
                undefined at that point, not just incomplete — silently
                returning a misleading order would hide the same kind of bug
                this method exists to prevent. An empty registry (e.g. right
                after reset()) returns [] without raising — that's a normal
                state, not a misuse.
        """
        if not self.quantizers:
            return []

        reached = [q for q in self.quantizers.values() if q.inference_sequence_id != -1]
        if not reached:
            raise RuntimeError(
                "quantizers_in_execution_order() called but no quantizer has "
                "been reached by a forward() pass yet (every "
                "inference_sequence_id is -1). Execution order is undefined "
                "until at least one forward pass has run. Run a forward pass "
                "over the model first, then call this method."
            )

        pool = self.quantizers.values() if include_unreached else reached
        return sorted(pool, key=lambda q: q.inference_sequence_id)

    def trigger_global_recalibration(self):
        """Sets the flag to force all quantizers to re-calibrate on next forward."""
        self.force_recalibration = True

    def reset_global_flag(self):
        """Resets the global recalibration flag."""
        self.force_recalibration = False

    def request_snapshot(self) -> None:
        """Ask every quantizer to emit a diagnostics snapshot on its next forward pass."""
        self._snapshot_count += 1
