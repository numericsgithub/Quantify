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
├── models/              # PyTorch model definitions (MobileNet, ResNet)
├── quantizers/          # Custom Brevitas quantizers (Fixed-Point, Coefficient, SiLU)
├── training_harness/    # QAT training loop, calibration, checkpointing, logging
├── importance/          # Importance analysis + interactive graph viewer
├── scripts/             # Utility/debug/demo scripts
├── tests/               # Pytest suite
├── utils/               # Shared utilities (ONNX export, workspace management)
└── docs/                # Project documentation
    └── llm/             # LLM agent guides, third-party package refs, pitfalls, conventions
```

## 🚀 Quick Start

### Train a small model and explore its importance
```bash
python scripts/train_residual_importance_demo.py --kernel-lasso 0.3 --filter-lasso 0.4
python -m importance.serve runs/imp_residual_mnist
```

For QAT, use the training harness below. The old example scripts (MNIST,
CIFAR-10, ImageNet, YOLOv8 -- plus the YOLOv8 models and their Ultralytics
trainer) are archived on the `old_examples` branch.

## 🛠 Training Harness

The `training_harness` provides an end-to-end QAT pipeline (`QATTrainerV2`):
1. **Float warmup**: all quantization is fully disabled (`QuantizerManager().disable_quantization()`) and the model trains in full precision.
2. **QAT activation**: after `float_warmup_epochs`, or earlier when `val_loss` plateaus, calibration buffers are reset, BatchNorm is frozen, and the quantizers switch on one by one (`quantization_start_gap`), each annealing from float to fully quantized over `annealing_steps` forward passes.
3. **QAT**: fine-tune with fake quantization; gradients flow through the STE.
4. **Checkpointing & Logging**: automatic top-K checkpointing, CSV/TensorBoard/W&B logging, and training curve plotting.

### Basic Usage
```python
from training_harness import QATTrainerV2, TrainerConfigV2, QATScheduleConfigV2

config = TrainerConfigV2(
    experiment_name="my_qat_run",
    epochs=50,
    learning_rate=1e-3,
    qat=QATScheduleConfigV2(
        float_warmup_epochs=10,    # QAT starts after this, or earlier on a val_loss plateau
        plateau_patience=5,
        annealing_steps=100,       # forward passes for each quantizer to ramp 0 -> 1
        quantization_start_gap=20, # quantizer N waits N x gap passes before switching on
    ),
)

trainer = QATTrainerV2(
    config=config,
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=nn.CrossEntropyLoss(),
)

tracker = trainer.fit()
```

## 🔢 Custom Quantizers & ONNX Export

This framework includes custom quantizers (`FixedPointPerTensorQuantizer`, `CoefficientPerTensorWeightQuant`, etc.) that export to ONNX as custom nodes (`Quantify::FixedPointQuant`).

### Coefficient files

`coefficients/` holds ready-made coefficient sets for `CoefficientPerTensorWeightQuant` (plain text, one set per line, powers of two and their sums for n = 2..12; `coeffs_single_*` and `coeffs_all_*` have one set, `coeffs_comb{1..4}_*` ten). See `coefficients/README.md` for the naming scheme and usage.

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

The export is checked to be a canonical graph and raises `NonCanonicalONNXError`
otherwise (`canonical_checks="warn"`/`"off"` to relax): every Linear is one
`Quantify::Dense` node, Conv/Pool pads are VALID or TF-SAME (use
`padding="same"` layers from `quantizers/layers.py`), the graph input feeds
the first layer directly, and activations are Quantify modules. See
`docs/llm/skills/canonical-onnx-export.md`.

## 🔍 Importance Analysis & Viewer

`importance/` computes, for every Conv/Linear layer (plain PyTorch,
Brevitas, or Quantify quantized), how important each weight/kernel/filter
is for each output feature, averaged over a dataset, and ships an
interactive local web viewer to explore the result:

```python
from importance import analyze, view

result = analyze(model, dataloader)   # everything auto-detected
result.save("runs/imp_mymodel")
view("runs/imp_mymodel")              # starts the offline viewer, opens the browser
```

or from the CLI: `python -m importance.serve runs/imp_mymodel [--port 8000]`.

The viewer draws the model as a Netron-style dataflow graph (residuals,
concats and multi-output heads included); every Conv/Linear node carries a
mini histogram of its per-filter importance, and clicking a node drills
down filter → kernel → weight in a sidebar. Zero filters, kernels and
weights are marked throughout (exact zeros for quantized weights, a
size-aware tolerance for float weights).

See `docs/llm/importance_analysis.md` for the metrics (including the
BatchNorm scale-invariance caveat and how `act_filter` fixes it), the
on-disk format, and known v1 limitations.

## 📖 Documentation & Skills

- **Conventions**: See `docs/llm/CONVENTIONS.md` for dependency and skill management rules.
- **Pitfalls**: Check `docs/llm/pitfalls/brevitas_pitfalls.md` for common Brevitas/ONNX export gotchas.
- **Index**: `docs/index.md` links every guide, including the skills under `docs/llm/skills/`.

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
