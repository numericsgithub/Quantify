# Quantify

A modular framework for Quantization-Aware Training (QAT) using [Brevitas](https://github.com/Xilinx/Brevitas), featuring custom fixed-point quantizers, a comprehensive training harness, and seamless ONNX export with custom nodes.

## 📦 Installation

```bash
conda create -n quantify python=3.12
conda activate quantify
pip install -e ".[dev]"
```

## 📂 Project Structure

```
├── models/              # PyTorch model definitions (YOLOv8, CIFAR-10, MobileNet)
├── quantizers/          # Custom Brevitas quantizers (Fixed-Point, Coefficient, SiLU)
├── training_harness/    # QAT training loop, calibration, checkpointing, logging
├── examples/            # Training scripts (MNIST, ImageNet, YOLOv8)
│   ├── basics/          # Simple, self-contained scripts
│   ├── training/        # Full-scale harness pipelines
│   └── yolo/            # YOLOv8 integration examples
├── scripts/             # Utility/debug scripts
├── tests/               # Pytest suite
├── utils/               # Shared utilities (ONNX export, workspace management)
└── docs/                # Project documentation
    └── llm/             # LLM agent guides, third-party package refs, pitfalls, conventions
```

## 🚀 Quick Start

### Train a Quantized Model (MNIST Example)
```bash
# Default: alpha-mix annealing (mixes float and quantized outputs as alpha 0→1)
PYTHONPATH=. python examples/simple_mnist_qat.py

# Bit-width annealing (steps effective bit-width down from start_bit_width to target).
# Recommended: on MNIST reaches ~97% val_acc at 8-bit with no collapse.
PYTHONPATH=. python examples/simple_mnist_qat_bitwidth.py
```

### Train YOLOv8n (PAN-Only Variant) on COCO
```bash
python examples/yolo/train_custom_yolo.py \
    --data /path/to/coco.yaml \
    --epochs 300 \
    --batch 64 \
    --device cuda
```

## 🛠 Training Harness

The `training_harness` provides an end-to-end QAT pipeline:
1. **Smooth annealing**: gradually transition from float-equivalent precision to the target quantized grid (two modes — see below).
2. **Lazy calibration**: each quantizer auto-calibrates its LSB on its first forward; recalibration is automatic at every bit-width transition in bit-width mode.
3. **QAT**: once annealing finishes, the model trains at the target bit-width with a Straight-Through Estimator so gradients flow through the round/clamp.
4. **Checkpointing & Logging**: Automatic top-K checkpointing, CSV/TensorBoard/W&B logging, and training curve plotting.

### Annealing modes

| Mode | What it does | When to use |
|---|---|---|
| `"alpha"` (default) | Per-batch ramp of `α` from 0 → 1 over the warmup window. Each quantizer's output is `(1−α)·x + α·quantize(x)`. | Drop-in compatibility with the original recipe. |
| `"bit_width"` | Per-epoch step of the *effective* bit-width from `start_bit_width` (e.g. 16) down to the quantizer's target (e.g. 8). `α` pinned at 1.0. Recalibration runs at every step. | Recommended. Cleaner transition, no fictional convex midpoints, model trains continuously through the schedule. |

### Basic Usage
```python
from training_harness.trainer import Trainer
from training_harness.config import TrainerConfig, QuantScheduleConfig

config = TrainerConfig(
    experiment_name="my_qat_run",
    epochs=50,
    learning_rate=1e-3,
    quant_schedule=QuantScheduleConfig(
        float_warmup_epochs=5,         # length of the warmup/annealing window
        annealing_mode="bit_width",    # "alpha" (default) or "bit_width"
        start_bit_width=16,            # only used in "bit_width" mode
    ),
)

trainer = Trainer(
    config=config,
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    loss_fn=nn.CrossEntropyLoss(),
)

tracker = trainer.fit()
```

## 🔢 Custom Quantizers & ONNX Export

This framework includes custom quantizers (`FixedPointPerTensorQuantizer`, `CoefficientPerTensorWeightQuant`, etc.) that export to ONNX as custom nodes in the `Quantify` domain (`Quantify::FixedPointQuant`, `Quantify::CoefficientQuant`, `Quantify::QuantSiLU`).

`export_onnx_with_io` routes through `QuantifyONNXManager` (Brevitas-style export handlers in `utils/quantify_export_manager.py`) so the custom symbolic actually fires. It also force-enables quant on every proxy for the duration of the export and restores afterwards, so per-checkpoint exports during float warmup also contain the quantized graph.

### Export to ONNX
```python
from utils import export_onnx_with_io

export_onnx_with_io(
    model=model.eval(),
    dummy_input=torch.randn(1, 3, 640, 640),
    filepath="model_qat.onnx",
    opset_version=13,
    custom_opsets={"Quantify": 1},
    dynamo=False,  # Required for custom autograd.Function nodes
)
```

To verify a trained model is actually quantized end-to-end (quantizer state, weights/activations on the fixed-point grid, ONNX node integrity), run:
```bash
PYTHONPATH=. python scripts/_verify_bw_quantized.py
```

## 📖 Documentation & Skills

- **Conventions**: See `docs/llm/CONVENTIONS.md` for dependency and skill management rules.
- **Pitfalls**: Check `docs/llm/pitfalls/brevitas_pitfalls.md` for common Brevitas/ONNX export gotchas.
- **Ultralytics Integration**: `docs/llm/ULTRALYTICS_DOCUMENTATION.md` covers YOLOv8/QAT integration patterns.

## 🧪 Testing

Run the full test suite:
```bash
pytest tests/ -v
```

## 🤝 Contributing

1. Follow the established conventions in `docs/llm/CONVENTIONS.md`.
2. Add new reusable patterns to the `docs/llm/` folder.
3. Document any new pitfalls in `docs/llm/pitfalls/brevitas_pitfalls.md`.
4. Ensure all tests pass before submitting changes.
