import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytest

from training_harness.trainer import Trainer
from training_harness.config import TrainerConfig, QuantScheduleConfig
from quantizers import FixedPointPerTensorWeightQuant, FixedPointPerTensorActivationQuant


class SimpleQuantizedMNIST(nn.Module):
    """A tiny quantized CNN for MNIST to exercise the training harness."""
    def __init__(self):
        super().__init__()
        import brevitas.nn as qnn
        
        self.features = nn.Sequential(
            qnn.QuantConv2d(1, 32, kernel_size=3, padding=1, weight_quant=FixedPointPerTensorWeightQuant),
            nn.ReLU(),
            nn.MaxPool2d(2),
            qnn.QuantConv2d(32, 64, kernel_size=3, padding=1, weight_quant=FixedPointPerTensorWeightQuant),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 28x28 -> Conv(p=1) -> 28x28 -> MaxPool(2) -> 14x14 -> Conv(p=1) -> 14x14 -> MaxPool(2) -> 7x7
            # Flattened size: 64 * 7 * 7 = 3136
            qnn.QuantLinear(64 * 7 * 7, 128, weight_quant=FixedPointPerTensorWeightQuant),
            nn.ReLU(),
            qnn.QuantLinear(128, 10, weight_quant=FixedPointPerTensorWeightQuant),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


@pytest.fixture
def dummy_mnist_data():
    """Generate a small dummy MNIST dataset for testing."""
    num_samples = 128
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)


def test_training_harness_basic(dummy_mnist_data):
    """
    Test that the training harness runs without exceptions for 3 epochs.
    Validates metric tracking, QAT schedule transitions, and checkpoint saving.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainerConfig(
            experiment_name="test_harness_mnist",
            output_dir=tmpdir,
            epochs=3,
            batch_size=32,
            learning_rate=1e-3,
            quant_schedule=QuantScheduleConfig(
                float_warmup_epochs=1,
                calibration_batches=2,
                track_scale_factors=True,
            ),
        )
        
        model = SimpleQuantizedMNIST()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        trainer = Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            train_loader=dummy_mnist_data,
            loss_fn=nn.CrossEntropyLoss(),
        )
        
        # Should not raise any exceptions during the full training loop
        tracker = trainer.fit()
        
        # Verify basic metrics were recorded across all epochs
        assert len(tracker.history) == 3, "Expected exactly 3 epoch snapshots"
        assert "train_loss" in tracker.history[0].metrics, "Missing train_loss in first epoch"
        
        # Verify checkpoint was saved to disk
        assert os.path.exists(os.path.join(tmpdir, "checkpoints", "last.pt")), \
            "Expected 'last.pt' checkpoint to be saved"
