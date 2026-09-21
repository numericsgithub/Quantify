"""Train ImportanceMNISTNet with proper QAT via training_harness.Trainer
(float warmup -> calibration -> QAT, not a manual loop -- see
docs/llm/pitfalls/training_harness_pitfalls.md #1), then export every saved
checkpoint to ONNX with the correct input shape via
utils.onnx_export.export_onnx_with_io.

    python scripts/train_qat_and_export_onnx.py
"""
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training_harness.trainer import Trainer
from training_harness.config import TrainerConfig, QuantScheduleConfig, CheckpointConfig
from utils.onnx_export import export_onnx_with_io

from examples.basics.importance_mnist import ImportanceMNISTNet, _load_mnist_or_synthetic


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImportanceMNISTNet().to(device)

    train_ds = _load_mnist_or_synthetic(train=True)
    val_ds = _load_mnist_or_synthetic(train=False, n_synthetic=256)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    config = TrainerConfig(
        experiment_name="qat_mnist_demo",
        output_dir="runs/qat_mnist",
        epochs=4,
        batch_size=128,
        learning_rate=1e-3,
        quant_schedule=QuantScheduleConfig(float_warmup_epochs=1, calibration_batches=50),
        checkpoint=CheckpointConfig(
            save_dir="checkpoints", top_k=2, save_last=True, save_every_n_epochs=1,
            monitor_metric="val_loss", monitor_mode="min",
        ),
    )

    trainer = Trainer(
        config=config, model=model, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=nn.CrossEntropyLoss(),
    )
    tracker = trainer.fit()
    print(f"training done -- checkpoints in {config.checkpoint_dir}")

    # --- Export every saved checkpoint to ONNX -----------------------------
    # Note: Trainer's own per-epoch auto-export of last.pt -> last.onnx uses
    # a hardcoded (1, 3, 32, 32) dummy input (training_harness/checkpointing.py
    # does not thread a dummy_input through from Trainer.fit()) which is the
    # wrong shape for this (1, 1, 28, 28) MNIST model, so it silently prints
    # "ONNX export skipped" every epoch -- harmless, but the exports below
    # (with the correct dummy input) are the ones to actually use.
    onnx_dir = os.path.join(config.output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    dummy_input, _ = train_ds[0]
    dummy_input = dummy_input.unsqueeze(0).to(device)

    ckpt_paths = sorted(glob.glob(os.path.join(config.checkpoint_dir, "*.pt")))
    print(f"found {len(ckpt_paths)} checkpoint(s): {[os.path.basename(p) for p in ckpt_paths]}")

    exported = []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Reuse the already-trained `model` object (not a fresh instance) so
        # each quantizer's QuantizerManager registration/inference_sequence_id
        # stays the one actually exercised during training -- see pitfall #12
        # in docs/llm/pitfalls/brevitas_pitfalls.md (load_state_dict recreates
        # quantizer proxies; a *fresh* model would also register a *new*
        # inference_sequence_id from the shared singleton QuantizerManager).
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        onnx_name = os.path.splitext(os.path.basename(ckpt_path))[0] + ".onnx"
        onnx_path = os.path.join(onnx_dir, onnx_name)
        try:
            export_onnx_with_io(
                model=model,
                dummy_input=dummy_input,
                filepath=onnx_path,
                opset_version=13,
                custom_opsets={"Quantify": 1},
                dynamo=False,
            )
            print(f"  exported epoch {ckpt.get('epoch')}: {ckpt_path} -> {onnx_path}")
            exported.append(onnx_path)
        except Exception as e:  # noqa: BLE001
            print(f"  ONNX export FAILED for {ckpt_path}: {type(e).__name__}: {e}")

    print(f"\n{len(exported)}/{len(ckpt_paths)} checkpoints exported to ONNX under {onnx_dir}")
    for p in exported:
        print(f"  {p}")


if __name__ == "__main__":
    main()
