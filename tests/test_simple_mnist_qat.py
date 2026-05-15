import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pytest

import brevitas.nn as qnn
from quantizers.manager import QuantizerManager
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorWeightQuant, FixedPointPerTensorActivationQuant, FixedPointPerTensorBiasQuant
from utils.onnx_export import export_onnx_with_io

# Training Harness Imports
from training_harness.engine_utils import set_seed, EarlyStopping, EpochTimer
from training_harness.logger import ExperimentLogger
from training_harness.metrics import AverageMeter, MetricsTracker
from training_harness.schedulers import WarmupCosineScheduler


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
        x = torch.cat((x1,x2),1)
        x = self.flatten(x)
        x = self.fc(x)
        return x


@pytest.fixture
def dataloader(device, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # Use a small subset for fast testing
    train_subset = Subset(train_dataset, range(0, 256))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_subset = Subset(test_dataset, range(0, 128))
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, device


def test_training_harness_integration(dataloader, device, tmp_path):
    """Test that the training harness components work correctly with the MNIST model."""
    train_loader, test_loader, device = dataloader
    
    # Reset quantizer manager singleton for a clean test run
    QuantizerManager._instance = None
    QuantizerManager()
    
    # 1. Reproducibility
    set_seed(42, deterministic=True)

    # 2. Model, Loss, Optimizer
    model = SimpleMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Harness Components
    log_dir = str(tmp_path / "logs")
    logger = ExperimentLogger(
        experiment_name="test_mnist_qat",
        run_id="test_run",
        log_dir=log_dir,
        use_tensorboard=False,
        use_wandb=False,
    )
    metrics_tracker = MetricsTracker()
    timer = EpochTimer(total_epochs=2)
    early_stopper = EarlyStopping(patience=3, mode="min", restore_best_weights=True)
    
    total_steps = len(train_loader) * 2
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=10,
        total_steps=total_steps,
        eta_min=1e-5
    )

    # Quantization Manager Setup
    quantizer_manager = QuantizerManager()
    quantizer_manager.quantization_start_gap = 20
    quantizer_manager.set_annealing_for_n_inferences(6)

    EPOCHS = 2
    BATCH_SIZE = 64

    for epoch in range(EPOCHS):
        model.train()
        timer.start()
        
        train_loss_meter = AverageMeter("train_loss")
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss_meter.update(loss.item(), data.size(0))
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

        # Epoch metrics
        train_acc = 100. * correct / total
        train_loss_avg = train_loss_meter.avg
        metrics_tracker.log("train_loss", train_loss_avg)
        metrics_tracker.log("train_acc", train_acc)
        metrics_tracker.commit_epoch(epoch, phase="train")
        
        elapsed, eta = timer.stop(epoch)

        # Validation
        model.eval()
        val_loss_meter = AverageMeter("val_loss")
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss_meter.update(loss.item(), data.size(0))
                
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += data.size(0)
        
        val_acc = 100. * val_correct / val_total
        val_loss_avg = val_loss_meter.avg
        metrics_tracker.log("val_loss", val_loss_avg)
        metrics_tracker.log("val_acc", val_acc)
        metrics_tracker.commit_epoch(epoch, phase="val")

        # Early stopping check
        if early_stopper.step(val_loss_avg, model, epoch):
            early_stopper.restore(model)
            break

    # Restore best weights if early stopping triggered
    if early_stopper.stopped_epoch is not None:
        pass

    # 4. ONNX Export
    model.eval()
    dummy_input, _ = next(iter(test_loader))
    dummy_input = dummy_input[0].unsqueeze(0).to(device)
    onnx_path = str(tmp_path / "test_mnist.onnx")

    export_onnx_with_io(model, dummy_input, onnx_path)
    
    logger.close()
    
    # Assertions
    train_steps, train_values = metrics_tracker.get_metric_series("train_loss")
    assert len(train_values) > 0, "Train loss should be recorded"
    
    val_steps, val_values = metrics_tracker.get_metric_series("val_acc")
    assert len(val_values) > 0, "Val accuracy should be recorded"
    
    assert early_stopper.best is not None, "Early stopper should have a best value"
    assert timer.total_epochs == 2
    assert os.path.exists(onnx_path), "ONNX file should be created"
