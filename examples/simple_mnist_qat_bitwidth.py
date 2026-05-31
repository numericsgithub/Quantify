"""
MNIST QAT example using BIT-WIDTH annealing.

Differs from `simple_mnist_qat.py` only in the QuantScheduleConfig:

    annealing_mode = 'bit_width'
    start_bit_width = 16

The model trains at effective bit-width 16, then steps down once per epoch
through {14, 13, 11, 10, 8} over `float_warmup_epochs` epochs. At every step
the model operates on a real quantized grid (no convex midpoints between
float and quantized), so there is no cliff when the target bit-width is
reached and accuracy keeps improving continuously into the QAT phase.

Empirically on MNIST: ~97.5 % val_acc by epoch 20 with no collapse,
compared to the alpha-mix mode which peaks ~89 % and recovers to ~73 %.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import brevitas.nn as qnn
from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorBiasQuant,
    FixedPointPerTensorWeightQuant,
)
from utils.onnx_export import export_onnx_with_io

# Training Harness Imports
from training_harness.trainer import Trainer
from training_harness.config import TrainerConfig, QuantScheduleConfig
from training_harness.engine_utils import set_seed


class SimpleMNISTNet(nn.Module):
    """
    A small CNN for MNIST using Fixed-Point quantization for both
    weights and activations.
    Architecture mirrors SimpleMNISTFloatNet to allow loading float checkpoints.
    """
    def __init__(self):
        super().__init__()

        # Quantize the input image
        self.input_quant = qnn.QuantIdentity(
            act_quant=FixedPointPerTensorActivationQuant
        )

        # Layer 1: Conv -> ReLU -> Pool
        self.conv1 = qnn.QuantConv2d(
            1, 16, kernel_size=3, stride=2, bias=True,
            bias_quant=FixedPointPerTensorBiasQuant,
            weight_quant=FixedPointPerTensorWeightQuant,
            output_quant=FixedPointPerTensorActivationQuant
        )
        self.relu1 = nn.ReLU()

        # Layer 2: Conv -> ReLU -> Pool
        self.conv2 = qnn.QuantConv2d(
            16, 8, kernel_size=3, stride=2,
            bias_quant=FixedPointPerTensorBiasQuant,
            weight_quant=FixedPointPerTensorWeightQuant,
            output_quant=FixedPointPerTensorActivationQuant
        )
        self.relu2 = nn.ReLU()

        self.conv3 = qnn.QuantConv2d(
            4, 6, kernel_size=3, stride=2,
            bias_quant=FixedPointPerTensorBiasQuant,
            weight_quant=FixedPointPerTensorWeightQuant,
            output_quant=FixedPointPerTensorActivationQuant
        )

        self.conv4 = qnn.QuantConv2d(
            4, 6, kernel_size=3, stride=2,
            bias_quant=FixedPointPerTensorBiasQuant,
            weight_quant=FixedPointPerTensorWeightQuant,
            output_quant=FixedPointPerTensorActivationQuant
        )

        self.flatten = nn.Flatten()

        # Final Linear Layer
        # Input size: 12 channels * 2x2 spatial (after convs with stride=2)
        self.fc = qnn.QuantLinear(
            12 * 2 * 2, 10,
            weight_quant=FixedPointPerTensorWeightQuant,
            output_quant=FixedPointPerTensorActivationQuant
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        xa, xb = torch.split(x, 4, dim=1)
        x1 = self.conv3(xa)
        x2 = self.conv4(xb)
        x = torch.cat((x1, x2), 1)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def train():
    # --- Configuration ---
    SEED = 42
    BATCH_SIZE = 256
    EPOCHS = 50
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG_DIR = "logs/simple_mnist_qat_bitwidth"
    EXPERIMENT_NAME = "simple_mnist_qat_bitwidth"

    # 1. Reproducibility
    set_seed(SEED, deterministic=True)

    # 2. Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model, Loss, Optimizer
    model = SimpleMNISTNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training Harness Configuration — bit-width annealing instead of alpha-mix
    config = TrainerConfig(
        experiment_name=EXPERIMENT_NAME,
        output_dir=LOG_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        quant_schedule=QuantScheduleConfig(
            float_warmup_epochs=5,        # 5 epoch-grained bit-width steps
            calibration_batches=10,
            track_scale_factors=True,
            annealing_mode="bit_width",   # the only line that matters
            start_bit_width=16,           # 16 → ... → target (8) over warmup
        ),
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=criterion,
    )

    print(f"Training on {DEVICE}...")

    # 6. Run Training
    tracker = trainer.fit()

    # 7. ONNX Export
    print("Exporting model to ONNX...")
    model.eval()
    dummy_input, _ = train_dataset[0]
    dummy_input = dummy_input.unsqueeze(0).to(DEVICE)
    onnx_path = "simple_mnist_fixedpoint_bitwidth.onnx"

    export_onnx_with_io(model, dummy_input, onnx_path)
    print(f"Model successfully exported to {onnx_path}")


if __name__ == "__main__":
    train()
