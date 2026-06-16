# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_fixedpoint_per_tensor.py -v

# Set required env var for tests
export QUANT_WORKDIR=/tmp/quanttests
```

The `scripts/test.sh` wrapper sets `QUANT_WORKDIR` and reinstalls before running pytest — use it when the environment may be stale.

## Architecture

### Package Layout

- `quantizers/` — Custom Brevitas quantizers with ONNX export support
- `training_harness/` — End-to-end QAT pipeline (Trainer, config, schedulers, checkpointing, logging)
- `models/` — Quantized model definitions (CIFAR-10, MobileNetV2, YOLOv8 variants)
- `utils/` — ONNX export helpers, workspace management, model introspection
- `examples/` — Full training scripts using the harness
- `docs/llm/` — Agent-facing docs: conventions, pitfalls, skills (read these before touching unfamiliar subsystems)

### Quantizer Hierarchy

All custom quantizers inherit from `quantizers/base_quantizer.py:BaseQuantizer`. It handles:
- Calibration state (`search_done` buffer, `force_recalibration` flag)
- Brevitas 4-tuple return contract `(tensor, scale, zero_point, bit_width)`
- ONNX export guards (`torch.onnx.is_in_onnx_export()`)
- Registration with the singleton `QuantizerManager`

`QuantizerManager` (`quantizers/manager.py`) is a **singleton** that coordinates all quantizer instances: global recalibration, quantization annealing, and inference gating. Subclasses plug in via `base_injector.py`, which wires `QuantConv2d`/`QuantLinear` layers to a quantizer at construction time.

Public quantizers (re-exported from `quantizers/__init__.py`):
- `FixedPointPerTensorWeightQuant` / `FixedPointPerTensorActivationQuant` / `FixedPointPerTensorBiasQuant`
- `CoefficientPerTensorWeightQuant`
- `QuantSiLUActivationQuant` (used in YOLO necks)

### Training Harness Flow

`Trainer` (`training_harness/trainer.py`) runs three phases in sequence:

1. **Float warmup** (`float_warmup_epochs`) — full-precision training; quantization gated off
2. **Calibration** (`calibration_batches`) — PTQ-style pass; `QuantizerManager` sets ranges
3. **QAT** — fake-quantization enabled; scales learned via STE gradients

Use `TrainerConfig` + `QuantScheduleConfig` to control timing. Do **not** write manual training loops for QAT — the harness manages `QuantizerManager` state transitions that are easy to get wrong (see pitfall #1 in `docs/llm/pitfalls/training_harness_pitfalls.md`).

### ONNX Export

`utils/onnx_export.py:export_onnx_with_io` is the canonical export path. Key constraints:
- Always pass `dynamo=False` — custom quantizers use `torch.autograd.Function.symbolic`, which the dynamo exporter does not support (pitfall #3 in `docs/llm/pitfalls/brevitas_pitfalls.md`).
- Call `reset_quantizer_states()` before every export to flush FIFO deque capture state.
- Custom ONNX nodes land in the `Quantify` domain (e.g., `Quantify::FixedPointQuant`). They are for graph inspection, not ORT inference.

## Key Conventions (`docs/llm/CONVENTIONS.md`)

- Add new packages to `requirements.txt` before importing them.
- New reusable patterns go in `docs/llm/skills/<pattern-name>.md`.
- New pitfalls go in `docs/llm/pitfalls/brevitas_pitfalls.md`.

## Critical Brevitas Pitfalls

Full list in `docs/llm/pitfalls/brevitas_pitfalls.md`. The most surprising ones:

- **No `QuantGlobalAvgPool2d` or `QuantLayerNorm`** — Brevitas only wraps Conv, Linear, BN, and basic activations. Use standard `nn.AdaptiveAvgPool2d` and wrap outputs with `QuantIdentity` if needed.
- **`QuantLinear` requires explicit flatten** — it does not reshape `(B, C, 1, 1)` automatically; always call `nn.Flatten()` before it.
- **Bias quantization requires `input_quant`** — `Int8Bias` derives scale from `input_scale * weight_scale`; omitting `input_quant` raises a runtime error.
- **`load_state_dict` with FP checkpoint needs `strict=False`** — quantized models have extra scale/zero-point parameters not present in FP checkpoints.
