# Feature ideas (backlog)

Proposed, not-yet-built features. Each entry is a design sketch + initial
research, deliberately kept out of the code until we choose to implement. Newest
first.

Conventions reminder (`docs/llm/CONVENTIONS.md`): reusable patterns → `skills/`,
pitfalls → `pitfalls/brevitas_pitfalls.md`. Feature *proposals* live here.

---

## 1. Per-layer quantization sensitivity analysis → mixed-precision bit allocation

**Status:** proposed, not implemented. Do not build yet.
**Requested:** 2026-07-20, while the 10-run MobileNetV2 QAT chain was running
(≈71% weight+bias-only top-1, up from a 43.6% PTQ start).

### The question

Two dual questions, both the same underlying quantity — a per-layer
*sensitivity to quantization*:

- **"Which layer would profit the most from more word width?"** → the layers
  that break the most under the current grid. Give them more bits (or a gentler
  LSB / larger `k`).
- **"Which layer still works fine with harsher quantization?"** → the layers
  that barely move when quantized. Take bits away (fewer bits / smaller `k`)
  and spend them where they matter.

Answering this is exactly the **mixed-precision bit-allocation** problem: under a
budget (target average bit-width, or a hardware constraint), assign each layer a
bit-width so total accuracy loss is minimized.

### Why this is the natural next step for *this* project

Everything measured so far says sensitivity is **strongly per-layer**, and a
single global knob is leaving accuracy on the table:

- `examples/sweep_weight_k.py` established that weight-only accuracy is a **step
  function of the global `k`** — it moves *only* where a layer's integer LSB
  flips, and **each of the 53 weight layers flips at its own `k`**. Four layers
  flipping between k=17 and k=18 swung the model ~24 points. A global `k`
  therefore can't be simultaneously right for all layers.
- The old greedy LSB search was already trying to make per-layer decisions —
  before the `_set_search_states` annealing leak corrupted its signal (see
  `tests/test_search_states_stay_disabled.py`, fix still pending).
- The parked per-channel-scale note (+7.5 dB mean / +21.9 dB max, tracking the
  max/median filter ratio) is the same story one level down: heterogeneity
  across the network that a uniform format ignores.

So the payoff: per-layer bit/LSB allocation is the most plausible path from the
current high-50s/low-70s uniform-8-bit result toward float's ~70–72%.

### Methods surveyed (initial research)

Ordered by how much I trust them **in this project specifically**, because we
have hard evidence that cheap proxy metrics lie here.

**A. Leave-one-out (one-at-a-time) accuracy drop — RECOMMENDED as the ground
truth.**
Hold every layer at the good/float config, quantize (or coarsen) exactly ONE
layer, measure val-accuracy drop on a fixed subset. The drop *is* that layer's
sensitivity. Two directions, both useful:
  - *quantize-one* (float baseline, one layer quantized): isolates each layer's
    individual damage.
  - *keep-one-float* (all quantized, one layer restored): isolates each layer's
    individual *recovery* — captures interaction effects the first misses.
Cost: `O(L)` evaluations (L≈53 weight layers). At ~40 val batches this is
minutes, not hours. Directly measures the only arbiter that has been reliable
here (see the SQNR caveat below).

**B. Per-layer bit-width / `k` sweep — RECOMMENDED as the actionable output.**
For each layer, sweep bit-width ∈ {2,3,4,6,8} (or `k`) with all others fixed,
record accuracy. Produces a per-layer "bits vs accuracy" curve, i.e. exactly
*how few* bits each layer tolerates. Cost `O(L · B)`; still tractable on a val
subset. This is the table a bit-allocator consumes directly. Natural extension
of the existing global-`k` sweep in `examples/sweep_weight_k.py` (reuse its
`build_at_k` / `_evaluate` scaffolding, just vary one layer at a time).

**C. Hessian / curvature-based (HAWQ family) — proxy, validate before trusting.**
Sensitivity ∝ (top eigenvalue or trace of the Hessian of the loss w.r.t. a
layer's weights) × (quantization perturbation magnitude). Estimated without
forming the Hessian via Hutchinson trace estimation or power iteration on
Hessian-vector products. Reference line: HAWQ (top eigenvalue), HAWQ-V2 (average
Hessian trace, + Pareto bit allocation), HAWQ-V3 (integer-only). Pro: one
backprop-based pass, no per-layer eval loop, principled second-order account of
why a flat-loss layer tolerates coarse quant. Con: Hessian estimation is finicky
(stochastic trace variance, needs a representative batch); it is still a *proxy*
for accuracy.

**D. Gradient / first-order Taylor — cheaper proxy.**
Approximate the loss change from quantizing a layer as `|g · Δw|` (gradient dot
quantization error), one backward pass. Cheaper than Hessian, coarser. A
reasonable pre-filter to rank layers before spending A/B evals on the top
candidates.

**E. Local signal metrics (SQNR / MSE / cosine per tensor) — DO NOT trust as the
primary signal here.**
Cheapest of all, and the trap this project already fell into. Documented
finding: **SQNR/MSE have been anti-predictive of accuracy** — SQNR rated a
0.000% grid ~10 dB *better* than a 56.7% one, and the unique-value-count
objective (a utilization proxy) is biased toward finer, more-clipping grids. Any
proxy (C, D, or E) must be **validated against the method-A accuracy ranking on a
handful of layers before it is trusted** for the rest. If it doesn't correlate,
throw it out.

### Recommended path when we build it

1. Implement method **A** first (leave-one-out accuracy) as the trusted
   sensitivity oracle. Reuse the weight-only eval harness
   (`examples/sweep_weight_k.py`, `find_perfect_lsbs_imagenet_ptq._evaluate`).
2. Add method **B** (per-layer bit sweep) for the actionable curves.
3. *Optionally* add **C/D** as cheap proxies, but gate their use on a measured
   correlation against A. Never ship a proxy unvalidated — that is the exact
   mistake that produced the broken grid.
4. Feed the result into a bit-allocator: given per-layer sensitivity + a target
   average bit-width, solve the (integer) allocation (greedy-by-sensitivity, or
   the Pareto/ILP formulation from HAWQ-V2/V3). Output a per-layer `bit_width`
   (and/or per-layer `k`) map.

### Integration points (for later — not now)

- Config already carries a mixed-precision switch (`--no-mixed-precision` in
  `examples/train_imagenet_qat.py`); a per-layer bit map would plug in there.
- Quantizer bit-width is per-quantizer already (`FixedPointPerTensor*Quant`), so
  a per-layer allocation is a config/wiring change, not a new quantizer.
- The same harness that picks per-layer bits can pick **per-layer `k`/LSB** —
  the same knob the corrupted greedy search was groping for. Consider unifying:
  "choose per-layer (bit_width, LSB) to minimize accuracy loss under a budget".
- ONNX: mixed per-layer *bit-width* exports fine (each quantizer already emits
  its own width). Per-**channel scales** would additionally need `Quantify`
  custom-node support — that is the separate parked item, don't conflate.

### Open caveats to carry in

- **Accuracy is the only trustworthy arbiter in this codebase.** SQNR/MSE/unique-
  count have all misled. Every proxy must earn trust against measured accuracy.
- **Interaction effects.** One-at-a-time misses that two coarse layers together
  can hurt more (or less) than the sum. keep-one-float (variant of A) and a
  final joint eval of the chosen allocation catch this.
- **Fixed val subset, fixed seed.** The step-function nature means noise can flip
  a layer's apparent rank; hold the eval subset constant across all layers.

### Related, already-noted follow-ups (not part of this feature)

- Land the `_set_search_states` alpha-leak fix + `tests/test_search_states_stay_disabled.py`.
- Bump a `LSB_SEARCH_VERSION` into the pretrained-qat cache key (stale caches).
- Per-channel weight scales (needs `Quantify` per-channel support).

### LR observation logged alongside (2026-07-20)

From the running chain: lr=1e-7 took the model 43.6% → ~71.1% cleanly. User's
read is that **1e-6 / 1e-7 is the sweet spot**; worth a single 1e-5/40-epoch
probe to confirm it's too hot (run 3 of the current chain already does exactly
this — check its result before designing future schedules). Keep future LR
schedules low and let the seed-best floor guard against an overshoot.
