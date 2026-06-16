# Quantization-Aware Training: A Complete Guide
## With Brevitas Coverage per Phase

---

## Overview

Training a model for quantization is a pipeline, not a single step. The correct mental model is:

```
FP32 Baseline → PTQ Diagnostic → Quantization Design → QAT Fine-tuning → Export
```

Each phase feeds information into the next. Skipping the early phases is the most common source of poor results.

---

## Phase 0: Start With a Strong FP32 Baseline

This sounds obvious but is frequently underestimated.

**The ceiling rule**: QAT can recover accuracy lost *due to quantization*, but it cannot recover accuracy that was never there. Every FP32 accuracy point you leave on the table costs disproportionately more in quantized form.

**What "well-converged" actually means here**:
- Not just "training loss has plateaued" -- the model's weight distributions should also have stabilized. Newly initialized or partially trained weights have erratic distributions that interact poorly with quantization.
- Validation accuracy should be stable across recent checkpoints, not still climbing.
- There should be no obvious training artifacts: dead ReLU channels, near-zero-norm weight tensors, degenerate batch norm statistics.

**What to measure before you touch quantization**:
- Full validation metrics on your deployment distribution (not just training distribution).
- Per-layer weight statistics: mean, std, min, max, kurtosis. High kurtosis (heavy tails, many outliers) is a warning sign.
- If applicable: per-channel weight norms. Channels with very different norms are harder to quantize.

**If you are using a pretrained checkpoint** (e.g. from a model zoo): verify it was fully trained and not just a "good enough" checkpoint. Also verify the BN statistics are meaningful (the checkpoint was finalized after training, not mid-epoch).

> **Brevitas**: Nothing specific here. Phase 0 is pure PyTorch. Brevitas only enters when you start defining quantized layers or running calibration.

---

## Phase 1: Post-Training Quantization (PTQ) as a Diagnostic

Before setting up QAT, run PTQ first. Most practitioners skip this. Don't.

**PTQ here is not the goal -- it is a measurement tool.**

### What to measure

**Global PTQ accuracy drop**: Apply uniform PTQ across the whole model and measure validation accuracy. This tells you whether the drop is negligible (PTQ may be sufficient), moderate (standard QAT should recover it), or large (a structural problem that more training won't fix).

**Layer-wise sensitivity analysis**: Quantize one layer at a time (all others stay FP32), measure accuracy drop per layer. This is the single most useful diagnostic you can run. It reveals which layers are most sensitive and where to spend your mixed-precision budget.

**Inspect weight and activation distributions**: For each layer, plot the weight histogram and the activation histogram. Look for outliers, asymmetry, bimodal distributions, and very wide dynamic range across channels. High per-channel magnitude variance is the primary argument for per-channel quantization of weights.

> **Brevitas**:
>
> - **`calibration_mode` context manager** (`brevitas.graph.calibrate`): disables fake quantization and instead collects running activation statistics (min, max, percentile, MSE) during forward passes over calibration data. This is the core tool for PTQ-style range initialization. After exiting the context, scale factors are set from the collected statistics.
> - **`calibrate_bn`**: a separate utility for recalibrating BN running statistics on the quantized model, useful after BN folding or when running stats have drifted.
> - **`bias_correction_mode`** context manager: corrects systematic bias shift introduced by weight quantization by analytically adjusting biases. Standard PTQ technique, available out of the box.
> - **Graph equalization** (`brevitas.graph.equalize`): cross-layer weight rescaling to equalize channel magnitudes across adjacent layers, reducing per-channel dynamic range before calibration. Enabled as a flag in the PTQ example flow.
> - **Activation equalization** (`activation_equalization_mode`, also `apply_act_equalization`): SmoothQuant-style technique that migrates quantization difficulty from activations to weights via per-channel scaling. Built in and usable both in PTQ and as a preprocessing step before QAT.
> - **Channel splitting**: a PTQ technique to split weight channels with large dynamic range into two narrower-range channels, reducing per-channel quantization error. Available in the PTQ example flow.
> - **Learned Round** (AdaRound-style, `brevitas.graph.gpxq`): instead of rounding weights to nearest, learns the optimal rounding direction per weight during PTQ. Available as part of the PTQ flow.
> - **Clipping/range methods**: MinMax, percentile (configurable), MSE minimization, and KL divergence are all supported for activation calibration range setting via the `StatsOp` enum.
> - **Layer-wise sensitivity analysis**: **not built in.** There is no automated tool in Brevitas that quantizes one layer at a time and reports accuracy drop. You implement this manually by toggling quantizers per layer.

---

## Phase 2: Design Your Quantization Scheme

This is where most of the structural decisions are made. Do this *before* inserting fake quantizers. Changing the scheme mid-QAT is expensive.

### 2.1 What to Quantize

| Parameter | Typical Treatment |
|-----------|-------------------|
| Weights | Always quantized |
| Activations | Quantized for inference efficiency (especially on integer-only hardware) |
| Biases | Usually kept FP32 or INT32 -- they contribute negligibly to memory/compute but quantization errors here are impactful |
| BN parameters | Folded into weights before deployment; irrelevant post-folding |

### 2.2 Symmetric vs Asymmetric

**Symmetric quantization** maps the range [-α, α] with zero-point = 0. Hardware-friendly, works well for weights. **Asymmetric quantization** uses a zero-point offset to represent non-zero-centered distributions (e.g. ReLU outputs). More expressive for activations, slightly more hardware cost.

Typical choice: symmetric for weights, asymmetric for activations. If your hardware does not support asymmetric, use symmetric everywhere and accept a small accuracy penalty on activations.

### 2.3 Granularity

**Per-tensor**: one scale for the entire tensor. Simplest, worst accuracy. **Per-channel**: one scale per output channel. Near-mandatory for weights at INT8 and below -- large accuracy benefit, low hardware cost. **Per-group**: a block of N weights shares a scale. Critical for INT4 and below. Group size is a hyperparameter: smaller = more accurate, more overhead.

### 2.4 Clipping Range / Scale Factor Initialization

Calibrate from representative data (100--1000 samples). Methods ranked by typical quality: Learned (LSQ) > MSE > KL divergence > Percentile > MinMax. For QAT, initialize from calibration and then let scales evolve if using learnable scales.

### 2.5 Static vs Dynamic (for Activations)

**Static**: scale factors fixed before inference from calibration. Required for deterministic latency hardware (FPGAs, ASICs). **Dynamic**: computed on-the-fly per input. Avoids calibration but adds runtime overhead.

> **Brevitas**:
>
> - **Symmetric / asymmetric**: both supported. Controlled via `signed`/`narrow_range` flags on quantizers and via built-in quantizer presets. Zero-point is explicitly tracked in `QuantTensor`.
> - **Granularity**: per-tensor, per-channel, and per-group all natively supported for weights and activations. Row-wise scaling at the input of `QuantLinear` also available.
> - **Clipping methods**: MinMax, percentile (configurable), MSE, and KL divergence are all available as `StatsOp` options. The `weights_param_method` / `activations_param_method` flags in the programmatic flow accept `'stats'` or `'mse'` explicitly.
> - **Static vs dynamic**: static is the default and primary mode. Dynamic activation quantization is experimentally supported (`is_static=False` in programmatic quantization config).
> - **Bias**: `Int32Bias` and `Int16Bias` are built-in quantizer presets. The bias scale is automatically derived as `input_scale * weight_scale`, which is the standard hardware convention. Unquantized (FP) bias is the default.
> - **Scale initialization**: handled through `calibration_mode` + forward passes as described in Phase 1. Scales can be either frozen (PTQ mode) or set as learnable parameters (QAT mode, see Phase 6).

---

## Phase 3: Batch Normalization Handling

BN interacts with quantization in several non-obvious ways. This is one of the most common sources of QAT bugs.

### The BN-quantization discrepancy problem

During training, BN normalizes using the current batch's mean and variance. During inference, it uses accumulated running statistics. This train/inference mismatch is amplified by quantization: fake quantizers calibrated against training-mode BN statistics will see a different distribution at inference when running statistics are used instead.

### BN folding

At deployment, BN is always folded into the preceding linear layer. The merged weights have a different distribution than the original weights, which means scale factors calibrated before folding will be mismatched after folding.

**The key decision: fold before QAT or after?** Fold-before gives QAT direct training on deployment weights but loses the ability to update BN stats. Fold-after is simpler but requires post-QAT calibration. Simulated folding (fold-during-QAT in the forward pass) is the most accurate but most complex approach.

### BN statistics freezing

Freeze BN running statistics early in QAT. If running stats keep updating, scale factors chase a moving target. Best practice: start QAT with BN in eval mode (statistics frozen), or freeze after a short warmup.

> **Brevitas**:
>
> - **BN folding (graph-based)**: Brevitas supports BN folding via torch.fx graph transformations as part of the programmatic quantization flow. This is the fold-before-QAT approach.
> - **`calibrate_bn`**: re-runs BN calibration on a quantized model by running a few forward passes. Useful after graph transformations that may shift BN statistics.
> - **Simulated BN folding during QAT**: **not explicitly provided** as a built-in mode. Fold-before-QAT via graph transformations is the recommended path.
> - **BN statistics freezing**: standard PyTorch (`model.eval()` on BN submodules), not a Brevitas-specific feature.

---

## Phase 4: Fake Quantization Setup

"Fake quantization" inserts quantize-dequantize nodes in the computation graph. The model runs in floating point, but values are rounded to quantization levels and back, simulating quantization error during training for gradient-based optimization.

### The Straight-Through Estimator (STE)

Quantization (rounding) has zero gradient almost everywhere. The STE passes the gradient through the quantizer as if it were the identity. This is biased but works in practice. Key failure modes: **oscillation** at INT4 and below (weights flip between adjacent levels without converging), and **dead clipping regions** (values outside the range receive zero gradient, freezing those weights if scales are fixed).

> **Brevitas**:
>
> - **This is Brevitas's core purpose.** All `brevitas.nn` layers (`QuantConv1d`, `QuantConv2d`, `QuantLinear`, `QuantReLU`, `QuantIdentity`, `QuantLSTM`, `QuantRNN`, `QuantMultiheadAttention`, etc.) are fake quantizers. They perform quantize-dequantize in forward and use the STE in backward by default.
> - **`QuantTensor`**: Brevitas's data structure carrying quantization metadata (scale, zero-point, bit-width, signedness) alongside the dequantized tensor. Propagates automatically through supported operations when `return_quant_tensor=True`.
> - **Programmatic quantization** (`brevitas.graph.quantize`, `layerwise_quantize`): takes a floating-point model and inserts Brevitas quantized layers automatically via torch.fx. Requires the model to be symbolically traceable.
> - **Input quantization**: `QuantIdentity` placed at the start of the forward pass is the standard tool for quantizing the model input.
> - **Residual/skip connections**: `QuantEltwiseAdd` quantizes the output of elementwise additions. Alternatively, `QuantIdentity` after `torch.add` achieves the same.
> - **STE**: used by default. No alternative backward estimators (DSQ, quantized backpropagation) are built in.
> - **Oscillation mitigation**: **not built in.** EMA on shadow weights, per-parameter gradient clipping for quantized training, etc. must be implemented in the training loop.
> - **A2Q (Accumulator-Aware Quantization)**: a Brevitas-specific research contribution (`brevitas.graph.gpxq`). A2Q constrains weight quantization so that the MAC accumulator is provably guaranteed not to overflow for a given accumulator bit-width. Highly relevant for FPGA targets with fixed accumulator widths. This is unique to Brevitas among major QAT frameworks.

---

## Phase 5: QAT Fine-Tuning Protocol

### 5.1 Learning Rate

Use significantly lower LR than original training: typically 1--10% of the original. Too high causes oscillation; too low causes underfitting. AdamW generally outperforms SGD for QAT because per-parameter adaptation handles the heterogeneous gradient scales that fake quantization introduces.

### 5.2 Duration

For INT8 on a well-trained model: 5--20% of original training epochs. For INT4: 20--50%, sometimes more with distillation. Do not over-train.

### 5.3 Data and Augmentation

Use the same training distribution. If you used heavy augmentation originally, consider reducing it slightly -- strong augmentation adds gradient noise that compounds with STE approximation error.

### 5.4 What to Track

Best checkpoint by validation accuracy, not last checkpoint. QAT loss curves are noisy. Monitor scale factors for collapse or explosion.

> **Brevitas**:
>
> - **No built-in QAT training loop or trainer.** The QAT training loop is standard PyTorch -- you write it yourself. Brevitas does not provide a `Trainer`, optimizer wrapper, or LR scheduler for QAT. This is intentional: Brevitas is a modeling library, not a training framework.
> - **`BREVITAS_JIT=1`** environment variable: enables TorchScript JIT compilation of quantization operators, which meaningfully speeds up QAT training. Worth enabling once the setup is validated. Note: currently unsupported when export is also toggled, or with MSE-based scales.
> - **Scale factor inspection**: since scales are PyTorch parameters or buffers, standard monitoring tools work. `return_quant_tensor=True` lets you inspect scale/zero-point at any point in the graph.

---

## Phase 6: Modern QAT Techniques

### 6.1 Learned Step Size Quantization (LSQ)

Makes the quantization step size a learnable parameter with an analytically computed gradient (not just STE). More accurate than fixed-scale QAT at INT4 and below. Now effectively the standard for serious QAT.

### 6.2 PACT (Parameterized Clipping Activation)

Specifically for activations: the clipping bound is a learned parameter with L2 regularization. Mostly superseded by LSQ for unified weight+activation training.

### 6.3 Knowledge Distillation during QAT

Keep the FP32 model frozen as a teacher. During QAT, add a distillation loss (KL divergence or MSE between teacher and student outputs). The soft labels from the FP32 teacher provide a much richer gradient signal than hard labels, especially at INT4 and below. Intermediate feature distillation (matching activations at specific layers) is a stronger but more complex variant.

### 6.4 Mixed-Precision Assignment

Based on Phase 1 sensitivity analysis, assign different bit-widths per layer. Sensitive layers at INT8, insensitive at INT4. First and last layers almost always at INT8. Only useful if your hardware executes different precisions.

### 6.5 Gradual Quantization

For very low bit-widths: quantize to INT8 first, fine-tune briefly, then lower to INT4. Or quantize weights before activations. Or add layers progressively from least to most sensitive.

> **Brevitas**:
>
> - **LSQ / learnable scales**: supported natively via `ScalingImplType.PARAMETER_FROM_STATS`. Scale factors are initialized from calibration data (same as PTQ), then treated as learnable parameters updated by gradient descent during QAT. This is the LSQ mechanism. It is a core Brevitas feature.
> - **PACT**: covered by the same `PARAMETER_FROM_STATS` mechanism for activation quantizers. No separate PACT module.
> - **Knowledge distillation**: **not built into Brevitas.** KD is entirely in the training loop (compute soft targets from FP32 model, add KL/MSE term to loss). No KD utilities, loss functions, or teacher-student hooks are provided.
> - **Mixed-precision assignment**: **manual only.** You specify `bit_width` per layer when defining the model. The programmatic quantization API allows per-layer overrides. There is no automatic mixed-precision search algorithm in Brevitas for QAT (or for PTQ). You assign based on your own Phase 1 sensitivity results.
> - **Gradual quantization**: **not built in.** No staged training protocol or bit-width schedule is provided. Managed manually: train at INT8, checkpoint, lower to INT4, continue.
> - **Activation equalization (SmoothQuant)**: as noted in Phase 1, can be applied before QAT as a preprocessing step, reducing the activation distribution difficulty before training begins.
> - **A2Q (accumulator-aware)**: see Phase 4. Unique Brevitas contribution. Particularly relevant if your hardware has a fixed accumulator width and you want a provable non-overflow guarantee.

---

## Phase 7: Key Pitfalls and Caveats

### 7.1 First and Last Layers

Always keep the first and last layer at INT8, even in an INT4 model. These layers are structurally most sensitive.

### 7.2 Residual Connections

Quantizing both the main path and the shortcut requires handling the output range after addition carefully. The sum of two quantized tensors can overflow.

### 7.3 Activation Outliers

Specific channels with values 10--100x larger than typical force wide quantization ranges, destroying resolution for the majority of values. SmoothQuant migrates this difficulty from activations to weights.

### 7.4 Oscillation

At INT4 and below, weights can oscillate between adjacent quantization levels without converging. Signs: loss oscillating, weight histogram spikes at level boundaries. Mitigations: lower LR, gradient clipping, LSQ-style learned scales.

### 7.5 Calibration Data Distribution

Calibrate from deployment-distribution data, not just training data, when the two differ.

### 7.6 Post-QAT Calibration

After QAT and BN folding, run a fresh activation calibration pass on the exported model. Often recovers 0.1--0.5% accuracy for free.

### 7.7 Evaluating During QAT

Always switch to eval mode for validation. The train/eval accuracy gap during QAT is often larger than in FP32 training and can mislead you.

> **Brevitas**:
>
> - **First/last layer protection**: manual. Assign higher `bit_width` or leave unquantized. The programmatic quantization API exposes `quantize_first_layer` / `quantize_last_layer` boolean flags to easily skip them.
> - **Residual connections**: `QuantEltwiseAdd` handles this explicitly. `QuantIdentity` placed after `torch.add` is the lighter-weight alternative.
> - **Activation outliers**: `activation_equalization_mode` (SmoothQuant) addresses this. No other built-in outlier suppression.
> - **Oscillation**: no built-in mitigation. Implement in training loop.
> - **Post-QAT calibration**: `calibration_mode` can be re-run on the folded final model. Fully supported.
> - **Eval mode during QAT**: standard PyTorch `model.eval()` / `model.train()`. Brevitas respects these; stateful fake quantizers collecting running statistics are active only in training mode.

---

## Phase 8: Export

After QAT: fold BN, freeze scale factors, export integer weights and scale factors to the inference toolchain.

> **Brevitas**:
>
> - **QONNX export** (`brevitas.export.export_qonnx`): the primary export path for FINN and FPGA deployment. Carries full quantization metadata (scales, zero-points, bit-widths) in a FINN-compatible format.
> - **ONNX QDQ/QCDQ export** (`export_onnx_qcdq`): exports to standard ONNX with Quantize-Dequantize nodes, plus a Clip extension for sub-8-bit. Compatible with ONNX Runtime and other ONNX-based inference chains.
> - **BN folding at export**: supported via graph transformations. Run `calibrate_bn` afterward if needed.
> - **Scale factor freezing**: when using `PARAMETER_FROM_STATS` (learnable QAT scales), they freeze automatically upon switching to eval mode. No explicit freeze step required.
> - **FINN integration**: QONNX is the handoff format to FINN. A2Q accumulator constraints are preserved through this export path.

---

## Summary Table: Brevitas Coverage

| Feature | Brevitas Status |
|---------|----------------|
| Fake quantization layers (core QAT) | Full -- all standard layers covered |
| QuantTensor metadata propagation | Full |
| Programmatic model quantization (torch.fx) | Full |
| Calibration: MinMax, percentile, MSE, KL | Full -- `calibration_mode` |
| Bias correction (PTQ) | Full -- `bias_correction_mode` |
| Graph equalization (cross-layer weight rescaling) | Full |
| Activation equalization (SmoothQuant-style) | Full -- `activation_equalization_mode` |
| Channel splitting | Full (in PTQ example flow) |
| Learned rounding (AdaRound-style) | Full -- `learned_round` / GPxQ |
| BN folding (graph-based, before QAT) | Full via torch.fx transforms |
| BN re-calibration after folding | Full -- `calibrate_bn` |
| Simulated BN folding during QAT | Not provided |
| BN statistics freezing | Manual (standard PyTorch) |
| Symmetric / asymmetric quantization | Full |
| Per-tensor / per-channel / per-group granularity | Full |
| Static activation quantization | Full |
| Dynamic activation quantization | Experimental |
| Learnable scales (LSQ-style) | Full -- `PARAMETER_FROM_STATS` |
| A2Q (accumulator-aware, overflow guarantee) | Full -- unique Brevitas feature |
| Knowledge distillation | Not provided -- implement in training loop |
| Automatic mixed-precision search | Not provided -- manual assignment only |
| Gradual / staged quantization protocol | Not provided -- manual |
| STE oscillation suppression | Not provided |
| Built-in QAT training loop / trainer | Not provided -- standard PyTorch loop |
| QONNX export (FINN / FPGA) | Full |
| ONNX QDQ/QCDQ export | Full |
| JIT speedup for QAT training | Full -- `BREVITAS_JIT=1` flag |
