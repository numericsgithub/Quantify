# Canonical, SINGULAR per-quantizer roles used for group-targeted control.
# These are the values stored on a quantizer's `.role` (see stamp_roles) and
# match the quantizer_role attribute set by the fixed-point injectors.
_CANONICAL_ROLES = ("weight", "bias", "activation")

# The plural group argument accepted by select_quantizers() -> canonical role.
_GROUP_TO_ROLE = {
    "weights": "weight",
    "biases": "bias",
    "activations": "activation",
}

# quant_id suffixes assigned by trainer_v2._make_quant_id, mapped to a canonical
# role. This is the PRIMARY role signal because it is derived from the model's
# module tree, so it is correct for EVERY quantizer type — including the
# SiLU/Coefficient quantizers whose tensor_quant __init__ never receives
# quantizer_role and would otherwise report "unknown". Ordered most-specific
# first so "_act_in"/"_act_out" are matched before "_act".
_QUANT_ID_SUFFIX_TO_ROLE = [
    ("_act_in", "activation"),
    ("_act_out", "activation"),
    ("_act", "activation"),
    ("_weight", "weight"),
    ("_bias", "bias"),
]


def _role_from_quant_id(quant_id):
    """Return the canonical role encoded in a descriptive quant_id, or None.

    V1-style ids ("quant_0") and any id without a known role suffix return None
    so the caller can fall back to the quantizer_role attribute.
    """
    if not isinstance(quant_id, str):
        return None
    for suffix, role in _QUANT_ID_SUFFIX_TO_ROLE:
        if quant_id.endswith(suffix):
            return role
    return None


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

    # ------------------------------------------------------------------
    # Role-based selection (keystone for group-targeted control)
    # ------------------------------------------------------------------

    def resolve_role(self, quantizer) -> str:
        """
        Determine a quantizer's canonical role: "weight" | "bias" |
        "activation" | "unknown".

        Resolution order (decided in the migration plan):
          1. The quant_id suffix (structure-derived; reliable for ALL quantizer
             types, including SiLU/Coefficient which never receive
             quantizer_role).
          2. The quantizer_role attribute (only the fixed-point family sets it).
          3. "unknown" — the quantizer cannot be group-addressed.
        """
        role = _role_from_quant_id(getattr(quantizer, "quant_id", None))
        if role is not None:
            return role
        attr = getattr(quantizer, "quantizer_role", None)
        if attr in _CANONICAL_ROLES:
            return attr
        return "unknown"

    def stamp_roles(self) -> None:
        """
        Cache each registered quantizer's canonical role on `.role`.

        Called by the trainer once quant_ids are finalized
        (trainer_v2._reset_and_register). Makes the resolved role explicit and
        cheap to read for the quantizer inspector and group operations.
        """
        for q in self.quantizers.values():
            q.role = self.resolve_role(q)

    def _role_of(self, quantizer) -> str:
        """Prefer the stamped `.role`; fall back to a fresh resolve."""
        stamped = getattr(quantizer, "role", None)
        if stamped is not None:
            return stamped
        return self.resolve_role(quantizer)

    def select_quantizers(self, group: str) -> list:
        """
        Return the registered quantizers belonging to ``group``.

        Args:
            group: one of "weights", "biases", "activations" (plural), or "all".

        Never silently drops quantizers: unknown-role quantizers simply do not
        belong to any of the three named groups. A caller performing a group
        MUTATION must consult unknown_role_count()/role_histogram() and warn
        loudly, so a quantizer with an unresolved role is never quietly left in
        the wrong quantization state.
        """
        if group == "all":
            return list(self.quantizers.values())
        if group not in _GROUP_TO_ROLE:
            raise ValueError(
                f"unknown quantizer group {group!r}; expected one of "
                f"'weights', 'biases', 'activations', 'all'"
            )
        target = _GROUP_TO_ROLE[group]
        return [q for q in self.quantizers.values() if self._role_of(q) == target]

    def role_histogram(self) -> dict:
        """
        Count registered quantizers per canonical role, plus total.

        The standing safety net for the "unknown role" hazard: a nonzero
        ``unknown`` count means some quantizers cannot be group-addressed and
        every group operation built on select_quantizers would skip them.
        """
        counts = {"weight": 0, "bias": 0, "activation": 0, "unknown": 0}
        for q in self.quantizers.values():
            counts[self._role_of(q)] += 1
        counts["total"] = len(self.quantizers)
        return counts

    def unknown_role_count(self) -> int:
        """Number of registered quantizers whose role resolves to "unknown"."""
        return sum(1 for q in self.quantizers.values() if self._role_of(q) == "unknown")

    # ------------------------------------------------------------------
    # Group-targeted QAT control (Phase 4)
    # ------------------------------------------------------------------
    #
    # Each mutator selects a group via select_quantizers() and applies a
    # per-quantizer change, returning an accounting dict. The dict ALWAYS
    # includes unknown_role (the count of quantizers that could not be
    # classified) so callers can warn loudly — a group op silently skipping an
    # unclassifiable quantizer is exactly the failure the role histogram guards.

    def _apply_to_group(self, group: str, fn) -> dict:
        qs = self.select_quantizers(group)
        for q in qs:
            fn(q)
        return {
            "group": group,
            "count": len(qs),
            "affected": [getattr(q, "quant_id", "?") for q in qs],
            "unknown_role": self.unknown_role_count(),
        }

    def set_group_annealing_alpha(self, group: str, alpha: float) -> dict:
        """Set annealing_alpha to an ABSOLUTE value for a group (leaves the
        per-step increment untouched — pass step separately or use disable to
        freeze at 0)."""
        a = float(alpha)
        return self._apply_to_group(group, lambda q: q.annealing_alpha.data.fill_(a))

    def set_group_annealing_step(self, group: str, step: float) -> dict:
        """Set the per-forward annealing increment (annealing_alpha_step)."""
        s = float(step)

        def _fn(q):
            q.annealing_alpha_step = s
        return self._apply_to_group(group, _fn)

    def set_group_annealing_ramp(self, group: str, n: int) -> dict:
        """Ramp annealing_alpha 0 -> 1 over n forward passes for a group
        (per-group form of set_annealing_for_n_inferences)."""
        step = 1.0 / max(1, int(n))

        def _fn(q):
            q.annealing_alpha.data.fill_(0.0)
            q.annealing_alpha_step = step
        return self._apply_to_group(group, _fn)

    def disable_group(self, group: str) -> dict:
        """Deactivate quantization for a group: annealing_alpha=0 AND step=0 so
        it stays float (does not re-anneal). This is the annealing-alpha
        mechanism, chosen over Brevitas's per-module disable_quant because it is
        addressable per registered quantizer by role and is reversible via
        set_group_annealing_alpha / _ramp. Do not mix the two mechanisms."""
        def _fn(q):
            q.annealing_alpha.data.fill_(0.0)
            q.annealing_alpha_step = 0.0
        return self._apply_to_group(group, _fn)

    def recalibrate_group(self, group: str) -> dict:
        """Clear search_done for a group so each quantizer re-runs its LSB search
        on its NEXT forward pass (lazy — it calibrates against whatever batch
        comes next). Uniform mechanism for both global ("all") and per-group
        recalibration; more robust than the force_recalibration flag, which
        resets after the first quantizer calibrates."""
        def _fn(q):
            if hasattr(q, "search_done"):
                q.search_done.fill_(False)
        return self._apply_to_group(group, _fn)

    def set_lsb(self, quant_id: str, lsb: int) -> dict:
        """Set the fixed-point LSB for ONE quantizer, addressed by quant_id.
        Fixed-point-specific (writes the search_result_lsb buffer); does NOT
        trigger recalibration and takes effect on the next forward."""
        q = self.quantizers.get(quant_id)
        if q is None:
            raise KeyError(f"no quantizer with id {quant_id!r}")
        if not hasattr(q, "search_result_lsb"):
            raise TypeError(
                f"quantizer {quant_id!r} ({self._role_of(q)}) has no LSB — it is "
                f"not a fixed-point quantizer")
        q.search_result_lsb.fill_(int(lsb))
        return {"quant_id": quant_id, "role": self._role_of(q), "lsb": int(lsb)}

    def describe_quantizers(self) -> list:
        """Per-quantizer snapshot for the inspector / LSB addressing UI."""
        out = []
        for qid, q in self.quantizers.items():
            entry = {"quant_id": qid, "role": self._role_of(q)}
            if hasattr(q, "annealing_alpha"):
                entry["alpha"] = float(q.annealing_alpha)
            entry["alpha_step"] = float(getattr(q, "annealing_alpha_step", 0.0))
            if hasattr(q, "search_done"):
                entry["search_done"] = bool(q.search_done)
            if hasattr(q, "search_result_lsb"):
                entry["lsb"] = int(q.search_result_lsb)
            if hasattr(q, "bit_width"):
                entry["bit_width"] = int(q.bit_width)
            out.append(entry)
        return out

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
