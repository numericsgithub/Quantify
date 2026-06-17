"""
Tests for QuantizerManager.is_quantizing_everything_fully and the QAT cascade.

The early-stopping gate in trainer_v2 fires only after
`QuantizerManager().is_quantizing_everything_fully` returns True.  These tests
verify that the flag:
  - stays False while any quantizer is still gated or still annealing
  - becomes True only after the LAST quantizer has finished annealing (alpha==1.0)
  - is not fooled by empty registries, eval-mode passes, or float-warmup state
"""

import pytest
import torch

from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer
from quantizers.manager import QuantizerManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_quantizer(bit_width: int = 8) -> FixedPointPerTensorQuantizer:
    """Return a quantizer registered with the singleton manager."""
    return FixedPointPerTensorQuantizer(bit_width=bit_width)


def _run_forward(quantizer: FixedPointPerTensorQuantizer,
                 n: int,
                 training: bool = True) -> None:
    """Run n forward passes through one quantizer in the requested mode."""
    quantizer.train(training)
    x = torch.randn(16)
    for _ in range(n):
        quantizer(x)


def _run_model_forward(quantizers: list,
                       n: int,
                       training: bool = True) -> None:
    """
    Simulate one model forward pass (all quantizers see the same batch) n times.
    The order matters: quantizer[0] runs first and gets sequence_id=0, etc.
    """
    x = torch.randn(16)
    for q in quantizers:
        q.train(training)
    for _ in range(n):
        for q in quantizers:
            q(x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_manager():
    """Guarantee a clean QuantizerManager for every test."""
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


# ---------------------------------------------------------------------------
# 1. Basic property tests on manager state
# ---------------------------------------------------------------------------

class TestIsQuantizingEverythingFullyBasic:

    def test_empty_registry_returns_true(self):
        # Vacuously true — no quantizers means nothing is unfinished.
        # Important: the trainer guards on _qat_active AND is_quantizing_everything_fully,
        # so an empty registry only causes premature early-stopping if the model
        # genuinely has no custom quantizers (i.e., BaseQuantizer instances).
        mgr = QuantizerManager()
        assert mgr.is_quantizing_everything_fully is True

    def test_all_alpha_zero_returns_false(self):
        q1 = _fresh_quantizer()
        q2 = _fresh_quantizer()
        mgr = QuantizerManager()
        mgr.disable_quantization()  # sets all alpha=0
        assert mgr.is_quantizing_everything_fully is False

    def test_all_alpha_one_returns_true(self):
        q1 = _fresh_quantizer()
        q2 = _fresh_quantizer()
        mgr = QuantizerManager()
        mgr.enable_quantization()  # sets all alpha=1
        assert mgr.is_quantizing_everything_fully is True

    def test_mixed_alpha_returns_false(self):
        q1 = _fresh_quantizer()
        q2 = _fresh_quantizer()
        mgr = QuantizerManager()
        q1.annealing_alpha.data.fill_(1.0)
        q2.annealing_alpha.data.fill_(0.0)
        assert mgr.is_quantizing_everything_fully is False

    def test_partial_annealing_returns_false(self):
        q = _fresh_quantizer()
        q.annealing_alpha.data.fill_(0.5)
        assert QuantizerManager().is_quantizing_everything_fully is False

    def test_single_quantizer_at_one_returns_true(self):
        q = _fresh_quantizer()
        q.annealing_alpha.data.fill_(1.0)
        assert QuantizerManager().is_quantizing_everything_fully is True

    def test_is_not_quantizing_at_all_all_zero(self):
        q1 = _fresh_quantizer()
        q2 = _fresh_quantizer()
        QuantizerManager().disable_quantization()
        assert QuantizerManager().is_not_quantizing_at_all is True

    def test_is_not_quantizing_at_all_mixed_returns_false(self):
        q1 = _fresh_quantizer()
        q2 = _fresh_quantizer()
        q1.annealing_alpha.data.fill_(0.0)
        q2.annealing_alpha.data.fill_(0.5)
        assert QuantizerManager().is_not_quantizing_at_all is False


# ---------------------------------------------------------------------------
# 2. Annealing mechanics
# ---------------------------------------------------------------------------

class TestAnnealingMechanics:

    def test_alpha_increments_each_forward_in_train_mode(self):
        q = _fresh_quantizer()
        mgr = QuantizerManager()
        mgr.set_annealing_for_n_inferences(10)  # step = 0.1
        assert q.annealing_alpha.item() == pytest.approx(0.0)

        _run_forward(q, 1, training=True)
        assert q.annealing_alpha.item() == pytest.approx(0.1)

        _run_forward(q, 4, training=True)
        assert q.annealing_alpha.item() == pytest.approx(0.5)

    def test_alpha_clamps_at_one(self):
        q = _fresh_quantizer()
        QuantizerManager().set_annealing_for_n_inferences(5)  # step = 0.2
        _run_forward(q, 10, training=True)  # more than enough
        assert q.annealing_alpha.item() == pytest.approx(1.0)

    def test_alpha_increments_in_eval_mode_too(self):
        # Annealing advances regardless of train/eval — only gating is
        # training-mode-only.
        q = _fresh_quantizer()
        QuantizerManager().set_annealing_for_n_inferences(10)
        _run_forward(q, 5, training=False)
        # alpha should have advanced during eval passes
        assert q.annealing_alpha.item() > 0.0

    def test_fully_quantized_flag_flips_after_n_inferences(self):
        q = _fresh_quantizer()
        mgr = QuantizerManager()
        mgr.set_annealing_for_n_inferences(10)

        assert mgr.is_quantizing_everything_fully is False
        _run_forward(q, 9, training=True)
        assert mgr.is_quantizing_everything_fully is False

        _run_forward(q, 1, training=True)   # pass 10 → alpha=1.0
        assert mgr.is_quantizing_everything_fully is True


# ---------------------------------------------------------------------------
# 3. Gating mechanics
# ---------------------------------------------------------------------------

class TestGatingMechanics:

    def test_gated_quantizer_does_not_advance_alpha(self):
        """Quantizer with id=1 and gap=100 should stay at alpha=0 while gated."""
        # Create quantizers BEFORE configuring the manager so set_annealing
        # can iterate over the already-registered instances.
        q1 = _fresh_quantizer()
        q2 = _fresh_quantizer()
        mgr = QuantizerManager()
        mgr.set_annealing_for_n_inferences(10)  # step=0.1, alpha=0
        mgr.quantization_start_gap = 100

        x = torch.randn(16)
        q1.train(True)
        q2.train(True)

        # First forward assigns IDs: q1→0 (gate=0, active), q2→1 (gate=100)
        q1(x)
        q2(x)  # counter becomes 1 (gated)

        # 99 more passes: counter goes from 1 → 100 (still < 100 is False on the
        # 100th pass, so q2 stays gated for all 99 of these extra passes)
        for _ in range(99):
            q1(x)
            q2(x)

        # counter == 100, gate == 100: q2 reaches the threshold on the NEXT call
        assert q2.inference_counter == 100
        assert q2.annealing_alpha.item() == pytest.approx(0.0)  # never annealed yet
        assert mgr.is_quantizing_everything_fully is False

    def test_gated_counter_does_not_advance_in_eval_mode(self):
        """Eval-mode passes must NOT count toward the gating threshold."""
        # Create quantizers first so set_annealing iterates the populated registry.
        q1 = _fresh_quantizer()
        q2 = _fresh_quantizer()
        mgr = QuantizerManager()
        mgr.set_annealing_for_n_inferences(10)   # alpha=0, step=0.1
        mgr.quantization_start_gap = 50

        x = torch.randn(16)
        # One training pass to assign IDs
        q1.train(True); q1(x)   # id=0, gate=0, active immediately
        q2.train(True); q2(x)   # id=1, gate=50, gated → counter=1

        counter_after_train = q2.inference_counter  # 1

        # 100 eval passes: q2 is still gated in eval, so counter must not move
        q1.eval(); q2.eval()
        for _ in range(100):
            q1(x); q2(x)

        assert q2.inference_counter == counter_after_train  # counter frozen

        # NOTE: q1 (non-gated) DOES advance alpha during eval passes because
        # the annealing block runs regardless of training mode.  This is the
        # intended design — only the gating countdown is training-mode-only.
        assert q2.annealing_alpha.item() == pytest.approx(0.0)  # gated, never annealed

    def test_quantizer_starts_annealing_after_gate_passes(self):
        """Once inference_counter >= id * gap the quantizer should start annealing."""
        q1 = _fresh_quantizer()
        q2 = _fresh_quantizer()
        mgr = QuantizerManager()
        mgr.set_annealing_for_n_inferences(10)
        mgr.quantization_start_gap = 10

        x = torch.randn(16)
        q1.train(True); q2.train(True)

        # Assign IDs (q1→0, q2→1)
        q1(x); q2(x)

        # Run exactly 10 more passes (gate for q2 is 1*10=10; counter starts at 1)
        for _ in range(9):
            q1(x); q2(x)

        # counter is now 10 ≥ gate=10 → q2 should start on next pass
        assert q2.inference_counter == 10

        # One more pass: q2 should now annealing
        q1(x); q2(x)
        assert q2.annealing_alpha.item() > 0.0


# ---------------------------------------------------------------------------
# 4. Full cascade simulation
# ---------------------------------------------------------------------------

class TestFullCascadeSimulation:
    """
    Simulate the complete QAT cascade for a small model:
    3 quantizers, gap=5 training passes, annealing=5 passes each.

    Timeline (training passes after QAT activation):
      Pass 1:   q0 starts (id=0, gate=0),  q1&q2 gated
      Pass 5:   q0 annealing done (alpha=1.0)
      Pass 6:   q1 starts (counter=5 >= 1*5),  q2 still gated
      Pass 10:  q1 done,  q2 starts (counter=10 >= 2*5)
      Pass 15:  q2 done → ALL fully quantized
    """

    @pytest.fixture
    def cascade_setup(self):
        mgr = QuantizerManager()
        q0 = _fresh_quantizer()
        q1 = _fresh_quantizer()
        q2 = _fresh_quantizer()
        mgr.set_annealing_for_n_inferences(5)   # step=0.2
        mgr.quantization_start_gap = 5
        return mgr, [q0, q1, q2]

    def test_not_fully_quantized_at_start(self, cascade_setup):
        mgr, qs = cascade_setup
        assert mgr.is_quantizing_everything_fully is False

    def test_not_fully_quantized_mid_cascade(self, cascade_setup):
        mgr, qs = cascade_setup
        # Run 7 passes: q0 done, q1 started but not done, q2 still gated
        _run_model_forward(qs, 7, training=True)
        assert mgr.is_quantizing_everything_fully is False

    def test_not_fully_quantized_just_before_completion(self, cascade_setup):
        mgr, qs = cascade_setup
        # 14 passes: q0 done, q1 done, q2 has annealed 4/5 steps → alpha≈0.8
        _run_model_forward(qs, 14, training=True)
        assert mgr.is_quantizing_everything_fully is False

    def test_fully_quantized_after_cascade_completes(self, cascade_setup):
        mgr, qs = cascade_setup
        # 15+ passes: all three quantizers at alpha=1.0
        _run_model_forward(qs, 20, training=True)
        assert mgr.is_quantizing_everything_fully is True

    def test_individual_quantizer_alphas_at_completion(self, cascade_setup):
        """All three quantizers must be at exactly 1.0 when the flag turns True."""
        mgr, qs = cascade_setup
        _run_model_forward(qs, 20, training=True)
        for i, q in enumerate(qs):
            assert q.annealing_alpha.item() == pytest.approx(1.0), (
                f"quantizer {i} alpha={q.annealing_alpha.item():.4f} (expected 1.0)"
            )

    def test_flag_is_false_before_last_quantizer_finishes(self, cascade_setup):
        """Flag must remain False until the very last quantizer reaches alpha=1.0."""
        mgr, qs = cascade_setup
        # Pass 14: q2 is one step away from finishing
        _run_model_forward(qs, 14, training=True)
        alpha_q2_before = qs[2].annealing_alpha.item()
        assert alpha_q2_before < 1.0
        assert mgr.is_quantizing_everything_fully is False

        # Pass 15: q2 finishes
        _run_model_forward(qs, 1, training=True)
        assert mgr.is_quantizing_everything_fully is True


# ---------------------------------------------------------------------------
# 5. Float-warmup state does not prematurely set flag
# ---------------------------------------------------------------------------

class TestFloatWarmupDoesNotSetFlag:

    def test_flag_false_after_disable_quantization(self):
        """
        After float-warmup setup (disable_quantization), the flag must be False
        even though quantizers are registered and have been called.
        """
        mgr = QuantizerManager()
        q1 = _fresh_quantizer()
        q2 = _fresh_quantizer()
        mgr.disable_quantization()       # alpha=0, step=0

        # Simulate float-warmup forward passes
        _run_model_forward([q1, q2], 50, training=True)

        assert mgr.is_quantizing_everything_fully is False

    def test_inference_counter_zero_after_warmup_with_gap_zero(self):
        """
        With quantization_start_gap=0 during warmup, the gating condition
        counter < id * 0 = 0 is never True, so inference_counter stays at 0.
        This is required for the QAT cascade to work correctly afterwards.
        """
        mgr = QuantizerManager()
        mgr.quantization_start_gap = 0
        q1 = _fresh_quantizer()
        q2 = _fresh_quantizer()
        mgr.disable_quantization()

        _run_model_forward([q1, q2], 100, training=True)

        assert q1.inference_counter == 0
        assert q2.inference_counter == 0

    def test_qat_activation_starts_cascade_correctly_after_warmup(self):
        """
        Full warmup → activate_qat pattern: after warmup, activating QAT
        must start the staggered cascade from scratch (not skip it because
        inference_counter was left at some non-zero value).
        """
        mgr = QuantizerManager()
        q1 = _fresh_quantizer()
        q2 = _fresh_quantizer()

        # --- Float warmup ---
        mgr.disable_quantization()
        mgr.quantization_start_gap = 0
        _run_model_forward([q1, q2], 50, training=True)  # warmup

        # Counters must be 0 so the cascade can work
        assert q1.inference_counter == 0
        assert q2.inference_counter == 0

        # --- QAT activation ---
        mgr.set_annealing_for_n_inferences(5)   # step=0.2
        mgr.quantization_start_gap = 5

        # One forward: q1 (id=0) starts, q2 (id=1) gated
        _run_model_forward([q1, q2], 1, training=True)
        assert q1.annealing_alpha.item() > 0.0   # q1 annealing
        assert q2.annealing_alpha.item() == pytest.approx(0.0)   # q2 still gated
        assert mgr.is_quantizing_everything_fully is False
