# Brevitas QAT Framework

A modular framework for Quantization-Aware Training (QAT) using [Brevitas](https://github.com/Xilinx/Brevitas), featuring custom fixed-point quantizers, a comprehensive training harness, and seamless ONNX export with custom nodes.

## 📦 Installation

```bash
conda create -n brevitas-qat python=3.12
conda activate brevitas-qat
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
└── docs/                # Framework conventions, pitfalls, and skill guides
```

## 🚀 Quick Start

### Train a Quantized Model (MNIST Example)
```bash
python examples/basics/simple_mnist_qat.py
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
1. **Float Warmup**: Train in full precision to learn robust weights.
2. **Calibration**: Run a PTQ-style pass to initialize quantization ranges.
3. **QAT**: Enable fake-quantization and fine-tune with learned scales.
4. **Checkpointing & Logging**: Automatic top-K checkpointing, CSV/TensorBoard/W&B logging, and training curve plotting.

### Basic Usage
```python
from training_harness.trainer import Trainer
from training_harness.config import TrainerConfig, QuantScheduleConfig

config = TrainerConfig(
    experiment_name="my_qat_run",
    epochs=50,
    learning_rate=1e-3,
    quant_schedule=QuantScheduleConfig(float_warmup_epochs=5, calibration_batches=100),
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

This framework includes custom quantizers (`FixedPointPerTensorQuantizer`, `CoefficientPerTensorWeightQuant`, etc.) that export to ONNX as custom nodes (`Quantify::FixedPointQuant`).

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

## 📖 Documentation & Skills

- **Conventions**: See `docs/developer/conventions.md` for dependency and skill management rules.
- **Pitfalls**: Check `docs/developer/brevitas-pitfalls.md` for common Brevitas/ONNX export gotchas.
- **Ultralytics Integration**: `docs/developer/ultralytics-integration.md` covers YOLOv8/QAT integration patterns.

## 🧪 Testing

Run the full test suite:
```bash
pytest tests/ -v
```

## 🤝 Contributing

1. Follow the established conventions in `docs/developer/conventions.md`.
2. Add new reusable patterns to the `docs/developer/` folder.
3. Document any new pitfalls in `docs/developer/brevitas-pitfalls.md`.
4. Ensure all tests pass before submitting changes.
