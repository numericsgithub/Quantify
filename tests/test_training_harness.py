import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from quantizers import FixedPointPerTensorWeightQuant
from quantizers.base_quantizer import BaseQuantizer
from quantizers.manager import QuantizerManager
from training_harness import QATScheduleConfigV2, QATTrainerV2, TrainerConfigV2


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
    QATTrainerV2 end to end on a tiny quantized CNN: float warmup with
    quantization fully disabled, then QAT activates after the warmup budget,
    the quantizers calibrate and start annealing, metrics are tracked and a
    checkpoint is written.
    """
    QuantizerManager().reset()
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainerConfigV2(
            experiment_name="test_harness_mnist",
            output_dir=tmpdir,
            epochs=3,
            batch_size=32,
            learning_rate=1e-3,
            device="cpu",
            num_classes=10,
            qat=QATScheduleConfigV2(
                float_warmup_epochs=1,
                annealing_steps=4,
                quantization_start_gap=1,
                track_scale_factors=True,
            ),
        )

        model = SimpleQuantizedMNIST()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        quantizers = [m for m in model.modules() if isinstance(m, BaseQuantizer)]
        assert quantizers, "test model has no Quantify quantizers"

        trainer = QATTrainerV2(
            config=config,
            model=model,
            optimizer=optimizer,
            train_loader=dummy_mnist_data,
            loss_fn=nn.CrossEntropyLoss(),
        )
        tracker = trainer.fit()

        assert len(tracker.history) == 3, "Expected exactly 3 epoch snapshots"
        assert "train_loss" in tracker.history[0].metrics, "Missing train_loss in first epoch"
        assert trainer._qat_active, "QAT never activated after the float warmup"
        assert all(q.search_done_value for q in quantizers), "quantizers did not calibrate during QAT"
        assert any(q.annealing_alpha_value > 0 for q in quantizers), "no quantizer started annealing"
        assert os.path.exists(os.path.join(config.checkpoint_dir, "last.pt")), \
            "Expected 'last.pt' checkpoint to be saved"
    QuantizerManager().reset()
