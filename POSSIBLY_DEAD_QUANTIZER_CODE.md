# Possibly Dead Quantizer Code

This file documents code segments in `quantizers/` that appear to be dead,
leftover, or superseded. Each entry describes the location, why it looks dead,
how to confirm it really is dead, and what to do once confirmed.

When confirmed dead: remove the code, then add a new entry to
`docs/llm/pitfalls/brevitas_pitfalls.md` explaining why it was removed and
what the correct pattern is. This prevents the same code from being
reintroduced.

---

## 1. `FixedPointQuantFnTestingThings` class
**File**: `quantizers/fixedpoint_per_tensor.py`, line 243

**What it is**: A `torch.autograd.Function` that runs `quantize_fixed_point_with_integers`
in the forward pass and returns a 4-tuple `(quantized, 0.0, 0.0, bw)`. The name
"TestingThings" strongly suggests it is a development leftover.

**Why it looks dead**:
- Has no `symbolic()` method — it cannot be used for ONNX export.
- It is only called by `quantize_fixed_point()` (line 136), which ignores outputs
  1–3 of the 4-tuple (`quantized, _, _, _ = FixedPointQuantFnTestingThings.apply(...)`).
  The only effect is the forward computation, which could be done directly by calling
  `quantize_fixed_point_with_integers()`.
- There is a commented-out alternative directly below it (line 137):
  `# quantized, _ = quantize_fixed_point_with_integers(inputs, lsb, ...)`
  This is the simpler, direct call that was presumably replaced by the Testing class.
- The class introduces unnecessary graph overhead (extra Function application)
  during the LSB calibration search, which calls `quantize_fixed_point()` in a tight loop.

**How to confirm**:
1. Replace the call in `quantize_fixed_point()` (line 136) with the commented-out
   direct call to `quantize_fixed_point_with_integers()`.
2. Un-comment line 137 and delete line 136.
3. Run `pytest tests/ -v` — all tests involving weight/activation quantization
   should pass identically.
4. If passing: `FixedPointQuantFnTestingThings` and the commented-out line 137 can both
   be deleted.

**Pitfall to document**: "Use `quantize_fixed_point_with_integers()` directly for
calibration math; `FixedPointQuantFn` is only for the real forward/ONNX path."

---

## 2. `quantize_fixed_point()` wrapper function
**File**: `quantizers/fixedpoint_per_tensor.py`, line 111

**What it is**: A public function that wraps `FixedPointQuantFnTestingThings.apply()`.

**Why it looks suspicious**:
- It exists only as a thin wrapper around the Testing class (see item 1).
- It is called in two places:
  - `find_optimal_lsb()` (line 197) — calibration loop.
  - `SiLUTensorQuant._quantize()` (line 100) — non-ONNX inference path.
- If the Testing class is removed (item 1), this function should be refactored
  to call `quantize_fixed_point_with_integers()[0]` directly, or the callers
  should be updated to call `quantize_fixed_point_with_integers()` directly.

**How to confirm**: See item 1. After fixing item 1, also update the
`SiLUTensorQuant._quantize()` call site if necessary.

---

## 3. `BaseQuantizer.backward()` instance method
**File**: `quantizers/base_quantizer.py`, line 137

**What it is**:
```python
def backward(ctx, grad_quantized, grad_scale, grad_zero_point, grad_bw):
    print("grad_quantizedgrad_quantized", grad_quantized)
    return grad_quantized, None, None, None, None, None, None, None
```

**Why it looks dead**:
- This is defined as a regular instance method on `BaseQuantizer` (an `nn.Module`),
  not as a `@staticmethod` on a `torch.autograd.Function`.
- `nn.Module` subclasses do not use a `backward()` method for autograd; gradients
  are handled by the `Function.backward()` of the custom ops called inside `forward()`.
- The debug print `"grad_quantizedgrad_quantized"` would fire on every backward pass
  of every sample if this were ever called, which would produce enormous output.
- There is no call site anywhere in the codebase.

**How to confirm**:
1. Add a `raise RuntimeError("BaseQuantizer.backward was called!")` at the top of
   the method.
2. Run a short training loop (e.g., `pytest tests/test_training_harness.py -v`).
3. If no error is raised: the method is never called. Delete it.

**Pitfall to document**: "Do not add a `backward()` method to `nn.Module` subclasses —
it has no effect on autograd. Backward logic for custom quantization ops belongs in
the `Function.backward()` of the corresponding `torch.autograd.Function`."

---

## 4. `FixedPointQuantFn.backward()` debug print
**File**: `quantizers/fixedpoint_per_tensor.py`, line 312

**What it is**:
```python
@staticmethod
def backward(ctx, grad_quantized, grad_scale, grad_zero_point, grad_bw):
    print("grad_quantizedgrad_quantized", grad_quantized)
    return grad_quantized, None, None, None, None, None, None, None
```

**Why it is a bug**:
- This is the real STE backward for `FixedPointQuantFn`, which IS called during
  training. The `print()` fires on every backward pass through every quantized
  layer — in a ResNet-18 that is O(100) prints per batch.
- Output never appeared in user's terminal (or was lost in training noise), which
  means it was introduced as a debug aid and never cleaned up.

**How to confirm**: Just remove the print. It is obviously debug output and its
removal requires no test — it is not testing any invariant.

**Pitfall to document**: "Never leave print statements in `Function.backward()` —
they fire once per quantized tensor per backward pass and produce unbounded output."

---

## 5. `find_optimal_lsb()` debug print
**File**: `quantizers/fixedpoint_per_tensor.py`, line 172

**What it is**: `print("find_optimal_lsb was called!")`

**Why it is dead / wrong**:
- `find_optimal_lsb()` is called inside the calibration loop and potentially inside
  `quantize_fixed_point()` (via the Testing class path — see item 1). A ResNet-18
  has ~20 quantizers × search over ~25 LSB candidates = ~500 calls per calibration
  pass. Each call prints this message.
- This is clearly debug output from development and should have been removed.

**How to confirm**: Remove the line. No test needed — it is unambiguously debug output.

---

## 6. `FixedPointPerTensorQuantizer.detect_signed()` method
**File**: `quantizers/fixedpoint_per_tensor.py`, line 440

**What it is**:
```python
def detect_signed(self, inputs: torch.Tensor) -> bool:
    """Return True if any input is negative. (Kept for backward compatibility / manual checks)"""
    return bool((inputs < 0).any().item())
```

**Why it looks dead**:
- The docstring itself says "Kept for backward compatibility / manual checks" — an
  admission it is not used in the main path.
- `_calibrate()` determines signedness inline (`torch.all(x >= 0.0)`) and does not
  call this method.
- No call site exists in the codebase (confirmed by grep).

**How to confirm**:
1. `grep -r "detect_signed" .` — if only the definition appears, it is dead.
2. Delete it and run `pytest tests/ -v`.

**Pitfall to document**: "Signedness detection is done inline in `_calibrate()`; do not
expose a separate `detect_signed()` method on the quantizer."

---

## 7. Duplicate `return` in `CoefficientPerTensorWeightQuantizer._quantize()`
**File**: `quantizers/coefficient_per_tensor_weights.py`, lines 161–165

**What it is**:
```python
        quantized, _, _ = apply_non_uniform_quantization(x, chosen_coeffs, bit_shift_scale)
        return quantized

        quantized, _, _ = apply_non_uniform_quantization(x, chosen_coeffs, bit_shift_scale)
        return quantized
```

Lines 164–165 are **unreachable** — they follow a `return` on line 162.

**Why it looks dead**: This is an accidental duplication introduced during editing.
`apply_non_uniform_quantization` is called and its result returned on line 162;
lines 164–165 can never execute.

**How to confirm**: Static analysis / common sense — code after `return` is unreachable
in Python. Linters (e.g. `ruff`) will flag this as a warning.

**Action**: Simply delete lines 164–165. Run `pytest tests/ -v` to confirm no regression.

**Pitfall to document**: "In `_quantize()` implementations, ensure there is only one
return per branch. Duplicate `apply_` + `return` lines after an earlier `return` are
silently ignored by Python but confuse readers."

---

## 8. `QuantizerManager.stop_quantization_for_n_inferences()` — possibly unused
**File**: `quantizers/manager.py`, line 58

**What it is**: Sets `quant.inference_counter = -n` for all registered quantizers,
which makes each quantizer skip quantization for the next `n` forward passes.

**Why it looks possibly dead**:
- `grep -r "stop_quantization_for_n_inferences" .` — check whether any call site
  exists outside of tests.
- If no call site exists, this method is dead API surface.

**How to confirm**:
1. `grep -rn "stop_quantization_for_n_inferences" .`
2. If only the definition in `manager.py` appears: delete it, run tests.

---

## 9. `QuantizerManager.enable_quantization()` — possibly unused
**File**: `quantizers/manager.py`, line 114

**What it is**: Sets `annealing_alpha=1.0` and `annealing_alpha_step=0.1` for all
quantizers. This is the "turn full quantization on immediately" path.

**Why it looks possibly dead**:
- The normal QAT path uses `set_annealing_for_n_inferences()` to ramp alpha 0→1
  gradually; `enable_quantization()` bypasses the ramp.
- `grep -r "enable_quantization" .` — if no call site exists in training code
  (only in tests), this is dead.

**How to confirm**:
1. `grep -rn "enable_quantization" .`
2. If unused in production code: delete, run tests.

---

## 10. `RoundingMode.ROUND_TO_NEAREST_EVEN` vs `RoundingMode.ROUND` — possible consolidation
**File**: `quantizers/fixedpoint_per_tensor.py`, lines 222–235

**What it is**: Three rounding modes exist:
- `ROUND_TO_NEAREST_EVEN` → `torch.round()` (banker's rounding)
- `FLOOR` → `torch.floor()`
- `ROUND` → `torch.floor(x + 0.5)` (arithmetic round-half-up)

**Why suspicious**:
- `ROUND_TO_NEAREST_EVEN` and `ROUND` are subtly different (banker's vs. half-up)
  and it is unclear whether this distinction is intentional.
- `FixedPointPerTensorWeightQuant` uses `ROUND`, `FixedPointPerTensorActivationQuant`
  uses `FLOOR`, `QuantSiLUActivationQuant` uses `ROUND_TO_NEAREST_EVEN`. The
  inconsistency may be intentional (different error properties for each role) or
  may be an oversight.
- This is not dead code, but one of the two `ROUND*` modes may be redundant.

**How to confirm**:
- Audit whether the distinction between `ROUND` and `ROUND_TO_NEAREST_EVEN` is
  intentional in any accuracy-critical test. If both produce the same results
  within test tolerances, consolidate to one mode.
- This is a lower priority than items 1–9.

---

## General instructions for handling confirmed dead code

1. **Remove the code** and run the full test suite (`pytest tests/ -v`).
2. **If tests fail**: the code was not dead — document why and mark the entry
   here as "NOT dead, kept because ...".
3. **If tests pass**: commit the deletion with a message like
   `remove dead code: <description>`.
4. **Add a pitfall entry** to `docs/llm/pitfalls/brevitas_pitfalls.md` or
   `docs/llm/pitfalls/training_harness_pitfalls.md` explaining:
   - What the removed code was
   - Why it looked alive but wasn't (to prevent reintroduction)
   - What the correct pattern is going forward
5. **Update this file**: mark the item as confirmed and link to the pitfall entry.
