"""
mnist_qat_v2.py — MNIST QAT example using the V2 training harness.

Pipeline:
  1. Float warmup: model trains in FP32, quantizers disabled entirely.
  2. Plateau detected: calibration buffers reset, gradual QAT cascade begins.
     - Each quantizer self-calibrates on its first live forward pass (finds optimal LSB).
     - Annealing ramps 0 → 1 over `annealing_steps` forward passes (model adapts smoothly).
     - Staggered gating: quantizer N waits N × `quantization_start_gap` passes before activating.
  3. Fully quantized: all quantizers at annealing_alpha=1.0, training continues to epoch budget.

Run:
    python examples/mnist_qat_v2.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import brevitas.nn as qnn

from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorWeightQuant,
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorBiasQuant,
)
from training_harness.trainer_v2 import QATTrainerV2
from training_harness.config_v2 import TrainerConfigV2, QATScheduleConfigV2
from training_harness.config import CheckpointConfig
from utils.onnx_export import export_onnx_with_io


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MNISTQuantNet(nn.Module):
    """
    Small fixed-point CNN for MNIST.

    Architecture:
        QuantIdentity → Conv(1→16, 3×3, s=2) → ReLU
                      → Conv(16→32, 3×3, s=2) → ReLU
                      → Flatten → Linear(32·6·6→ 10)

    Spatial trace (28×28 input, stride=2, no padding):
        28 → (28-3)//2 + 1 = 13  (conv1)
        13 → (13-3)//2 + 1 = 6   (conv2)
        Flattened: 32 × 6 × 6 = 1152
    """

    def __init__(self):
        super().__init__()

        self.input_quant = qnn.QuantIdentity(
            act_quant=FixedPointPerTensorActivationQuant,
            return_quant_tensor=False,
        )

        self.conv1 = qnn.QuantConv2d(
            1, 16, kernel_size=3, stride=2, bias=True,
            weight_quant=FixedPointPerTensorWeightQuant,
            bias_quant=FixedPointPerTensorBiasQuant,
            output_quant=FixedPointPerTensorActivationQuant,
            return_quant_tensor=False,
        )
        self.relu1 = nn.ReLU()

        self.conv2 = qnn.QuantConv2d(
            16, 32, kernel_size=3, stride=2, bias=True,
            weight_quant=FixedPointPerTensorWeightQuant,
            bias_quant=FixedPointPerTensorBiasQuant,
            output_quant=FixedPointPerTensorActivationQuant,
            return_quant_tensor=False,
        )
        self.relu2 = nn.ReLU()

        self.flatten = nn.Flatten()

        self.fc = qnn.QuantLinear(
            32 * 6 * 6, 10, bias=True,
            weight_quant=FixedPointPerTensorWeightQuant,
            output_quant=FixedPointPerTensorActivationQuant,
            return_quant_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    BATCH_SIZE = 256
    EPOCHS     = 60   # float warmup + QAT fine-tuning combined

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True,  download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model + optimizer
    # ------------------------------------------------------------------
    model     = MNISTQuantNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ------------------------------------------------------------------
    # V2 config
    # ------------------------------------------------------------------
    config = TrainerConfigV2(
        experiment_name = "mnist_qat_v2",
        output_dir      = "output/mnist_qat_v2",
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        learning_rate   = 1e-3,
        grad_clip_norm  = 1.0,

        qat = QATScheduleConfigV2(
            # Plateau detector watches val_loss (must be a decreasing metric).
            # QAT starts when val_loss hasn't improved by more than min_delta
            # for plateau_patience consecutive epochs, or after float_warmup_epochs
            # at the latest.
            float_warmup_epochs    = 30,
            plateau_metric         = "val_loss",
            plateau_patience       = 10,
            plateau_min_delta      = 1e-4,

            # Each quantizer anneals 0→1 over ~2 epochs worth of batches,
            # then the next one activates 1 epoch later.
            annealing_steps        = 400,
            quantization_start_gap = 200,

            freeze_bn_at_qat       = True,
            track_scale_factors    = True,
        ),

        checkpoint = CheckpointConfig(
            monitor_metric = "val_acc",
            monitor_mode   = "max",
            top_k          = 3,
            save_last      = True,
        ),

        # Early stopping only activates once all quantizers are at alpha=1.0
        # (no stopping during the annealing cascade).
        early_stopping_patience = 15,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = QATTrainerV2(
        config            = config,
        model             = model,
        optimizer         = optimizer,
        train_loader      = train_loader,
        val_loader        = val_loader,
        loss_fn           = nn.CrossEntropyLoss(),
        onnx_dummy_input  = torch.zeros(1, 1, 28, 28),
    )

    tracker = trainer.fit()

    # ------------------------------------------------------------------
    # Export to ONNX
    # ------------------------------------------------------------------
    print("\nExporting to ONNX …")
    model.cpu().eval()
    dummy = torch.zeros(1, 1, 28, 28)
    export_onnx_with_io(model, dummy, "mnist_fixedpoint_v2.onnx")
    print("Exported → mnist_fixedpoint_v2.onnx")


if __name__ == "__main__":
    main()
