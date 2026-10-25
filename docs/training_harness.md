# Training Harness

The `training_harness` provides an end-to-end QAT pipeline built around
`QATTrainer` (`training_harness/trainer.py`).

## Pipeline Phases
1. **Float warmup**: all quantization is fully disabled (`QuantizerManager().disable_quantization()`) and the model trains in full precision.
2. **QAT activation**: after `float_warmup_epochs`, or earlier when `val_loss` plateaus, calibration buffers are reset, BatchNorm is frozen, and the quantizers switch on one by one (`quantization_start_gap`), each annealing from float to fully quantized over `annealing_steps` forward passes.
3. **QAT**: fine-tune with fake quantization; gradients flow through the STE.
4. **Checkpointing & Logging**: automatic top-K checkpointing, CSV/TensorBoard/W&B logging, and training curve plotting.

## Configuration and usage
`TrainerConfig` holds the run settings, and its `qat` field
(`QATScheduleConfig`) controls when and how quantization switches on:
```python
from training_harness import QATTrainer, TrainerConfig, QATScheduleConfig

config = TrainerConfig(
    experiment_name="my_qat_run",
    epochs=50,
    learning_rate=1e-3,
    qat=QATScheduleConfig(
        float_warmup_epochs=10,    # QAT starts after this, or earlier on a val_loss plateau
        plateau_patience=5,
        annealing_steps=100,       # forward passes for each quantizer to ramp 0 -> 1
        quantization_start_gap=20, # quantizer N waits N x gap passes before switching on
    ),
)

trainer = QATTrainer(
    config=config,
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=nn.CrossEntropyLoss(),
)

tracker = trainer.fit()
```

## Key Classes
- `QATTrainer`: orchestrates the training loop, QAT activation, validation, checkpointing and logging.
- `TrainerConfig` / `QATScheduleConfig` / `CheckpointConfig` / `LoggingConfig`: configuration (`config.py`).
- `CheckpointManager`: saving/loading model state and ONNX export.
- `ExperimentLogger`: routes metrics to CSV, TensorBoard and W&B.
- `find_lr`: learning-rate range test using the same model setup as the trainer.
