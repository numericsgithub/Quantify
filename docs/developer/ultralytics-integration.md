# Ultralytics Integration Guide

This document covers the Ultralytics side of integrating Brevitas-based QAT into the YOLO training infrastructure.

## Core Concepts
- **`YOLO` high-level API**: Unified wrapper resolving task-specific `Trainer`, `Validator`, `Predictor`, and `Model`.
- **`DetectionModel`**: A `BaseModel`/`nn.Module` whose `self.model` is a `nn.Sequential` produced by `parse_model()`.
- **`BaseTrainer` → `DetectionTrainer`**: Owns the training loop, validation, EMA, checkpointing, AMP, DDP setup.
- **Callback system**: Dict of event-name → list-of-callables, dispatched at fixed lifecycle points.

## Three Levels of Customization
| Level | When to use | Mechanism |
|---|---|---|
| **Callbacks** | Inject logic at fixed lifecycle points | `model.add_callback("on_train_epoch_start", fn)` |
| **Custom Trainer subclass** | Replace model factory, optimizer, validator | Subclass `DetectionTrainer`, pass `trainer=MyTrainer` |
| **Custom YAML + module registration** | Swap `Conv`/`Linear` for `qnn.QuantConv2d`/`qnn.QuantLinear` | YAML referencing custom module names; modules registered into `ultralytics.nn.tasks` globals |

## Quantization Strategy: PTQ vs QAT in Ultralytics
- **Built-in PTQ**: `model.export(format="engine", int8=True, data="coco.yaml")`
- **QAT with Brevitas**: 
  1. Build FP32 model normally.
  2. Swap layers for Brevitas equivalents.
  3. Calibrate activation ranges.
  4. Train with `amp=False`, lower LR.
  5. Export to ONNX QCDQ via Brevitas' `export_onnx_qcdq()`.

## AMP & Brevitas
Recommendation: Pass `amp=False` for QAT runs. Brevitas fake-quant ops interact badly with `autocast`.

## EMA & Quantization
`ModelEMA` copies all `state_dict` floats, including Brevitas scales. Consider disabling EMA or excluding quantizer parameters to prevent scale drift.

## DDP & Quantization
Brevitas fake-quant scales are synced by DDP's gradient all-reduce. Statistics-collection scales are buffers and require `broadcast_buffers=True`.

## Export & Deployment
Prefer Brevitas' native exporters (`export_onnx_qcdq`, `export_qonnx`) over Ultralytics' `model.export()`.
