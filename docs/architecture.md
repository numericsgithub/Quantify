# Architecture Overview

This document outlines the high-level architecture of the Brevitas QAT Framework.

## Core Components
- **Models**: PyTorch implementations of quantized architectures (CIFAR-10, MobileNet, YOLOv8).
- **Quantizers**: Custom Brevitas quantizers (Fixed-Point, Coefficient, SiLU) with ONNX export support.
- **Training Harness**: End-to-end QAT pipeline including float warmup, calibration, QAT, checkpointing, and logging.
- **Utils**: Shared utilities for ONNX export, workspace management, and model inspection.

## Data Flow
1. **Data Loading**: Datasets are wrapped and fed into PyTorch DataLoaders.
2. **Model Forward Pass**: Inputs pass through quantized layers (`QuantConv2d`, `QuantLinear`, etc.).
3. **Loss Computation**: Standard losses (CrossEntropy, v8DetectionLoss) are applied.
4. **Backward Pass & Optimization**: Gradients flow through Straight-Through Estimators (STE) in quantizers.
5. **Checkpointing & Export**: Models are saved periodically and exported to ONNX QCDQ or custom nodes.

For detailed component documentation, see the respective sections below.
