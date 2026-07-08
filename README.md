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

## 📊 Live Training Dashboard (read-only)

Every training run can expose a **read-only HTTP monitoring API** from inside the
training process, and a separate lightweight web UI visualizes it live. Works with
both `Trainer` (V1) and `QATTrainerV2`. The API is versioned under `/api/v1/` so
control endpoints can be added later without redesign.

### 1. Enable the API on a run

Set `api_port` on the config — that's the only change; leaving it unset (default)
keeps training behavior exactly as before:

```python
config = TrainerConfig(            # or TrainerConfigV2
    experiment_name="my_qat_run",
    api_port=8765,                 # 0 = let the OS pick a free port
    # api_host="0.0.0.0",          # allow remote access (default: localhost only)
)
```

The server runs in a daemon thread and only reads state — it never blocks or
mutates the training loop. Step/epoch metrics are also appended to
`<log_dir>/<experiment>/<run_id>/api_metrics.jsonl` so history survives crashes.

### 2. Start the web UI (separate process)

```bash
python dashboard/serve.py --port 8080 --api http://127.0.0.1:8765
```

Open `http://127.0.0.1:8080/`. The UI polls the API incrementally (`?since_step=`),
shows train loss (with smoothing), validation accuracy vs. a configurable target,
LR over time, the current QAT phase with quantizer progress, ETA, and the top-K
checkpoint list. If the API becomes unreachable (run finished or crashed), the UI
keeps the last known state and shows a *disconnected* indicator. The API base URL
can be switched in the page header to watch multiple concurrent runs.

For a remote GPU box, either set `api_host="0.0.0.0"` or tunnel the port:
`ssh -L 8765:localhost:8765 user@gpu-box`.

### 3. API endpoints

```bash
curl http://127.0.0.1:8765/api/v1/health           # {"ok": true}
curl http://127.0.0.1:8765/api/v1/status           # run state, phase (float_warmup/qat),
                                                   # epoch, step, ETA, LR, best metric, pid
curl http://127.0.0.1:8765/api/v1/config           # effective TrainerConfig as JSON
curl http://127.0.0.1:8765/api/v1/metrics          # full step + epoch history
curl "http://127.0.0.1:8765/api/v1/metrics?since_step=500&since_epoch=10"   # increments only
curl http://127.0.0.1:8765/api/v1/metrics/latest   # newest values, cheap to poll
curl http://127.0.0.1:8765/api/v1/checkpoints      # top-K checkpoints (epoch, metric, path)
```

> **Note:** `train_acc` in the metrics responses is flagged unreliable (nonstandard
> computation; approximate under mixup). Use `train_loss` and `val_acc` as the
> meaningful signals — the bundled UI deliberately does not plot train accuracy.

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
