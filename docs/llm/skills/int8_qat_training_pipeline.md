# Full-INT8 QAT training pipeline (how we reached W8/A8/B8 ≈ float)

**Result it produced:** MobileNetV2, weights + biases + activations all quantized
to 8-bit fixed-point per-tensor, **~72.1% ImageNet top-1** (float reference
72.71%). This is the exact, ordered procedure — every phase, every knob, and the
reason for each — so it can be reproduced for other models.

The single most important principle learned the hard way: **accuracy measured on
the val set is the only trustworthy signal.** SQNR/MSE/unique-code metrics were
repeatedly anti-predictive here (SQNR once rated a 0.0% grid 10 dB *better* than a
56.7% one). Every decision below is grounded in measured accuracy, not a proxy.

---

## The shape of it: four phases

| Phase | What | Activations | Result (MNv2) |
|------:|------|-------------|---------------|
| 0 | PTQ checkpoint (single forward) | float | 43.6% |
| 1 | weights+bias QAT chain | float | 72.3% |
| 2 | **progressive** activation introduction | ramped in one-by-one | 71.7% |
| 3 | post-activation QAT chain (polish) | quantized | 72.1% |

Why this order and not "quantize everything, then train"? Two reasons, both
measured: (a) weight quantization is nearly lossless and converges fast, so
getting a strong weights+bias model first gives activation-QAT a good starting
point; (b) activation quantization is the hard part — dropping all 35 in at once
collapses the model, so they are introduced **gradually while training** so the
network can adapt to each one before the next arrives.

---

## The calibration recipe (LSB selection)

All three roles use fixed-point per-tensor quantizers
(`quantizers/fixedpoint_per_tensor.py`) with a symmetric grid about zero (no
zero-point). "Calibration" = choosing the integer **LSB position** (grid step =
`2^lsb`). The rule differs per role because the tensor distributions differ.

Constants (top of `fixedpoint_per_tensor.py`):
```
ROBUST_SIGMA_K_WEIGHT     = 12.0
ROBUST_SIGMA_K_ACTIVATION = 16.0
BIAS_COVERAGE_PCT         = 99.9
```

- **Weights — robust-σ, k=12.** `threshold = |median| + 12 · (1.4826·MAD)`; pick
  the *finest* LSB whose representable range covers `threshold`. Robust σ (median
  absolute deviation) is immune to outlier *magnitude*, so the grid is set by the
  bulk of the weights, deliberately clipping a few outliers. k=12 was chosen
  empirically: k=4 was too harsh (0.0% — clips into oblivion), k=16 slightly
  coarser; k=12 sits in a stable band. **Accuracy vs k is a STEP function** — each
  layer's LSB flips at its own k — so pick a k in a stable plateau, not a lucky
  spike.
- **Biases — coverage 99.9%.** `threshold = 99.9th-percentile of |bias|`, finest
  covering LSB. Folded biases (created by BN fusion) carry a DC offset
  `β − γμ/√(var+ε)` and are often **not centered on zero** (e.g. features.20 sat
  at −5.43), so the rule keys off `|value|`, not spread-about-median. 80% coverage
  killed accuracy (0.0%); 99.9% keeps essentially all bias mass.
- **Activations — robust-σ k=16, `ignore_zeros`, `prefer_high_lsb`.** Same robust-σ
  machinery but with a **wider** grid: larger k AND "prefer the highest LSB among
  the max-unique candidates" = the widest range, so activation quantization
  **covers more outliers** rather than clipping them (activations after ReLU are
  one-sided and spiky; clipping them hurts more than clipping weights).

The role is dispatched in `FixedPointPerTensorQuantizer._calibrate` via
`self.quantizer_role`.

---

## Phase 0 — PTQ checkpoint  (`examples/create_ptq_checkpoint.py`)

```bash
python -m examples.create_ptq_checkpoint --model <M> --data-dir $DATA
# -> output/ptq/<M>_W8_Anone_B8_singlepass.pt
```

1. Build the model **weights + biases only** (`act_quant=None`).
2. Load timm pretrained float weights.
3. **Fuse BatchNorm** (`utils/bn_fusion.fuse_bn_into_conv`). This must happen
   before calibration: folding rewrites the weight distribution the weight
   quantizers calibrate against, *and* it is what creates `conv.bias`
   (`nn.Parameter` assignment triggers Brevitas to wire up the bias quantizer at
   all).
4. **One** train-mode forward with `quantization_start_gap=0` → every weight and
   bias quantizer calibrates simultaneously. This is sound *only* for weights and
   biases: each reads its own parameter tensor, never activations, so no
   quantizer's calibration depends on another's. (Activations are the opposite —
   handled in Phase 2.) This replaced a 4h38m greedy per-quantizer search that
   took *seconds*.
5. Verify every *reached* quantizer calibrated; ignore Brevitas "ghost"
   quantizers (registered but never reached by forward — dropped by
   `QuantizerManager.quantizers_in_execution_order()`). Do **not** force
   `search_done`/`alpha` on a quantizer that did not calibrate — that exact
   mistake once left LSB=0 on 52 quantizers.
6. Expect **poor** accuracy (un-adapted PTQ). MNv2: 43.6%. That is fine.

---

## Phase 1 — weights+bias QAT chain  (`scripts/run_mnv2_qat_chain.sh`)

A **chain** of independent runs (fresh process each) that hand the best
checkpoint forward. Fresh process per run so a leak / wedged DALI iterator / CUDA
fragmentation in one run cannot bleed into the next, and any run can be
relaunched alone.

```bash
MODEL=<M> PTQ_CKPT=<phase0.pt> OUT_ROOT=output/<M>_wchain \
LRS="1e-7 1e-6 1e-5 1e-6 1e-7 1e-8 1e-7 1e-6 1e-7 1e-8" \
  scripts/run_mnv2_qat_chain.sh
```

Key ingredients:
- **10 runs × 40 epochs.** Each run its own base LR (see LRS), held 30 epochs then
  ×0.8 for the last 10 (`--step-lr --step-lr-milestones 0.75 --step-lr-gamma 0.8`).
  Empirically **1e-5 was the best-performing run**, not too hot as feared — but
  keep the low LRs too; the chain explores a range. (We ran two chains, the second
  at 2× the LRs; gains were marginal past 72.3% — the model plateaus near float.)
- **`--no-act-quant`** — activations stay float this whole phase.
- **`--no-mixed-precision`** — Brevitas fake-quant + AMP produced NaNs; keep AMP off.
- **`--float-warmup-epochs 0`** — start QAT immediately (the model is pretrained).
- **`preserve_calibrated_quantizers`** (auto-set by `--init-from-ptq`): the loaded
  weight/bias quantizers are kept active from step 0 and their staggered gate is
  **skipped** (`QuantizerManager.skip_gating_for_calibrated_quantizers`). See
  pitfall #11 — gating and annealing are independent; preserving needs both handled.
- **One-time clamp** (`train_imagenet_qat._clip_params_to_quant_range`): right after
  loading, clamp weights/biases **into** their quantizer's representable range,
  once, **clip-only (no rounding)**. A loaded checkpoint routinely holds parameters
  ~5× outside the grid; those pin at the clip bound (quantized value frozen) while
  plain STE keeps pushing them further out. Clamping moves the latent float value
  back inside. Disable with `--no-clip-to-quant-range` only to A/B it.
- **Seed-best trick** (`CheckpointManager.seed_best`, called from `trainer_v2.fit`):
  `best.pt` is seeded from the *incoming* checkpoint with its score as the
  best-threshold. So a run that only ever gets worse leaves the seed in place and
  the chain **cannot regress** — run N+1 picks up run N's model, never a worse one.
  The whole chain is monotonic.
- Weight decay `1e-8`.

MNv2: 43.6% → **72.3%**.

---

## Phase 2 — progressive activation introduction  (the crux)

This is what makes A8 work. Activations are quantized **one quantizer at a time,
input-side first, while training**, each annealed in gradually, so the network
adapts to each before the next arrives.

Driver: `scripts/run_mnv2_act_then_chain.sh` (Phase 1 of that script). It runs
`train_imagenet_qat` **with** activation quantization:

```bash
python -m examples.train_imagenet_qat --model <M> \
  --init-from-ptq <phase1_best.pt> \
  --qat-gap <G> --annealing-steps 100 \
  --no-seed-best --require-full-quant-for-best \
  --lr 2e-8 --weight-decay 0 --epochs 130 \
  --float-warmup-epochs 0 --no-mixed-precision \
  --batch-size <B> --output-dir output/<M>_act_intro
```

**The gating mechanism (built in, `quantizers/base_quantizer.py:129`):** a
quantizer with execution-order id `K` stays OFF (float passthrough) for its first
`K × qat_gap` forward passes. When its gate opens it calibrates itself (first
quantizing forward, in train mode) and its `annealing_alpha` ramps 0→1 over
`annealing_steps` passes (`AnnealingBlendFn` blends float↔quantized). Because
weights+biases are preserved (gate skipped, α=1 from step 0), **only the new
activation quantizers ramp in.** Input-side activations have low `K`, so they come
on first — exactly "start at the input, then the next layer, ..."

**Settings that worked (MNv2, batch 1024):** `qat_gap=800`, `annealing_steps=100`,
`lr=2e-8`, `weight_decay=0` (don't need it — accuracy is *expected* to dip),
`epochs=130`.

**Critical: the gap is in GLOBAL execution-order units and must be tuned per model
AND per batch size.** Activation quantizers are interleaved with weight/bias ones,
so consecutive activations are several seq-ids (≈3500 steps at gap 800) apart —
each anneals in 100 steps then holds ~3 epochs before the next. The last
activation opens at `last_act_seqid × gap` steps; divide by `steps/epoch`
(= `1.28M / batch`) to get the epoch it finishes. Aim for the ramp to finish
around **2/3 of the run** so there are ~40+ fully-quantized recovery epochs.

Measured per model (probe: build with act quant, one forward, read
`inference_sequence_id` of activation quantizers):

| Model | batch | act Q | last act seq-id | steps/epoch | **gap for ramp-end ≈ ep88** |
|-------|------:|------:|----------------:|------------:|----------------------------:|
| mobilenetv2 | 1024 | 35 | 138 | 1251 | **800** |
| mobilenetv1 | 1024 | 27 |  80 | 1251 | **1376** |
| resnet18 | 512 | 30 |  69 | 2502 | **3191** |
| resnet50 | 512 | 71 | 176 | 2502 | **1251** |

(Naively reusing gap=800 would finish resnet18's ramp far too early.) **The
last-act seq-id — and therefore the gap — depends on bias quantizers being
threaded through the model** (they interleave into the execution order). Every
model here quantizes all folded biases (bias-quantizer count == weight count);
re-probe the seq-id if that ever changes.

**best.pt guards — essential for this phase:**
- `--no-seed-best`: do **not** seed best.pt from the starting model. The start is
  weights+bias-only at 72.3%; seeding it would set an unbeatable threshold and
  freeze best.pt at a non-fully-quantized model. best.pt starts empty.
- `--require-full-quant-for-best`: only epochs where **every** quantizer is fully
  quantized (`quant_pct == 1.0`, all gates open, all annealing done) can win
  best.pt. During the ramp a partial epoch scores *higher* (fewer activations
  quantized = closer to float) — this stops it winning. So best.pt only ever holds
  a genuinely fully-quantized model. (Both flags added in `config_v2.py` →
  `seed_best_from_start`, `require_full_quant_for_best`.)

Accuracy **dips during the ramp and may not fully return** — that is expected and
fine. MNv2 finished the 130 epochs fully quantized at **71.7%**.

**Memory (pitfall #14):** at batch 1024 the first activation tensor is
1024×32×112×112 = 411M elements. The quantizer *diagnostics* ran `torch.unique`
over the full tensor (~9 GiB single alloc) and OOM'd. Fixed:
`utils/quantizer_diagnostics.py` now bounds every reduction to a 4M random
subsample (`MAX_METRIC_SAMPLES`, via `torch.take`) — 9190 MB → 116 MB, metrics
unchanged. Tests: `tests/test_diagnostics_memory.py`. Also set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for fragmentation headroom.

---

## Phase 3 — post-activation QAT chain (polish)

Same chain script, now **with** activation quantization
(`ACT_QUANT=1` → drops `--no-act-quant`, passes `--qat-gap 0 --annealing-steps 1`).
Seeded from Phase 2's fully-quantized best. Every quantizer is already calibrated,
so `preserve_calibrated_quantizers` makes the model **fully quantized from step 0**
— no re-annealing. Same LRs as the previous chain; seed-best keeps it monotonic
(floor = Phase 2's best).

```bash
ACT_QUANT=1 MODEL=<M> PTQ_CKPT=<phase2_best.pt> OUT_ROOT=output/<M>_actchain \
LRS="2e-7 2e-6 1e-5 2e-6 2e-7 2e-8 2e-7 2e-6 2e-7 2e-8" \
  scripts/run_mnv2_qat_chain.sh
```

MNv2: 71.7% → **72.1%**.

---

## Per-model knobs summary

- **Batch size:** MobileNetV2 / MobileNetV1 → 1024; ResNet18 / ResNet50 → **512**
  (bigger/more activation tensors, more memory). Batch size changes steps/epoch,
  which changes the Phase-2 gap (see the table). It does NOT affect Phase 0/1
  (weights calibrate from parameter tensors, gap=0).
- **Phase-2 gap:** per the table above; recompute if you change batch or epochs.
- Everything else (LSB recipe, LRs, seed-best, one-time clamp, `--no-mixed-precision`,
  `--float-warmup-epochs 0`) is model-independent.

## Scripts
- `scripts/run_mnv2_qat_chain.sh` — the chain engine. Model-agnostic despite the
  name: `MODEL`, `PTQ_CKPT`, `OUT_ROOT`, `LRS`, `ACT_QUANT`, `BATCH` env vars.
- `scripts/run_mnv2_act_then_chain.sh` — MNv2 Phase 2 + Phase 3, hands-off.
- `scripts/run_int8_pipeline.sh` — generic all-phases pipeline for one model.
- `scripts/schedule_other_models.sh` — runs the pipeline for the remaining models
  sequentially (waits for the GPU first).

## Known caveats carried forward
- The `_set_search_states` **alpha leak** (annealing re-enables "disabled"
  quantizers) is latent — only bites on a greedy-search cache MISS, which this
  pipeline never triggers (it uses `--init-from-ptq`, not the greedy search). Fix +
  `tests/test_search_states_stay_disabled.py` still unlanded (3 tests red on main).
- Peak training memory is high at batch 1024 (~77 GiB / 95). If a fresh OOM appears
  at higher quant-%, batch size is the lever — but lowering it shifts steps/epoch
  and therefore the Phase-2 gap.
- Per-layer / mixed-precision sensitivity (the likely next gain) is written up
  separately in `docs/llm/FEATURE_IDEAS.md`.
```
