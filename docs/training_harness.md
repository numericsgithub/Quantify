# Training Harness

The `training_harness` provides an end-to-end QAT pipeline built around
`QATTrainerV2` (`training_harness/trainer_v2.py`).

## Pipeline Phases
1. **Float warmup**: all quantization is fully disabled (`QuantizerManager().disable_quantization()`) and the model trains in full precision.
2. **QAT activation**: after `float_warmup_epochs`, or earlier when `val_loss` plateaus, calibration buffers are reset, BatchNorm is frozen, and the quantizers switch on one by one (`quantization_start_gap`), each annealing from float to fully quantized over `annealing_steps` forward passes.
3. **QAT**: fine-tune with fake quantization; gradients flow through the STE.
4. **Checkpointing & Logging**: automatic top-K checkpointing, CSV/TensorBoard/W&B logging, and training curve plotting.

## Configuration and usage
`TrainerConfigV2` holds the run settings, and its `qat` field
(`QATScheduleConfigV2`) controls when and how quantization switches on:
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

## Key Classes
- `QATTrainerV2`: orchestrates the training loop, QAT activation, validation, checkpointing and logging.
- `TrainerConfigV2` / `QATScheduleConfigV2` / `CheckpointConfig` / `LoggingConfig`: configuration (`config_v2.py`).
- `CheckpointManager`: saving/loading model state and ONNX export.
- `ExperimentLogger`: routes metrics to CSV, TensorBoard and W&B.
- `find_lr`: learning-rate range test using the same model setup as the trainer.
