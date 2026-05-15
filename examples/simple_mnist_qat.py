import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
            weight_quant=FixedPointPerTensorWeightQuant, #CoefficientPerTensorWeightQuant,
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
        # x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def train():
    # --- Configuration ---
    SEED = 42
    BATCH_SIZE = 256
    EPOCHS = 50
    LR = 0.001
    WARMUP_STEPS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG_DIR = "logs/simple_mnist_qat"
    EXPERIMENT_NAME = "simple_mnist_qat"

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

    # 4. Training Harness Components
    logger = ExperimentLogger(
        experiment_name=EXPERIMENT_NAME,
        run_id="default",
        log_dir=LOG_DIR,
        use_tensorboard=True,
        use_wandb=False,
    )
    metrics_tracker = MetricsTracker()
    timer = EpochTimer(total_epochs=EPOCHS)
    early_stopper = EarlyStopping(patience=3, mode="min", restore_best_weights=True)
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * EPOCHS
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=WARMUP_STEPS,
        total_steps=total_steps,
        eta_min=1e-5
    )

    # Quantization Manager Setup
    quantizer_manager = QuantizerManager()
    quantizer_manager.quantization_start_gap = 20
    quantizer_manager.set_annealing_for_n_inferences(6)

    print(f"Training on {DEVICE}...")
    logger.log_text("config", f"Seed={SEED}, Batch={BATCH_SIZE}, Epochs={EPOCHS}, LR={LR}")

    for epoch in range(EPOCHS):
        model.train()
        timer.start()
        
        train_loss_meter = AverageMeter("train_loss")
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
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
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} [{batch_idx*BATCH_SIZE}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")

        # Epoch metrics
        train_acc = 100. * correct / total
        train_loss_avg = train_loss_meter.avg
        metrics_tracker.log("train_loss", train_loss_avg)
        metrics_tracker.log("train_acc", train_acc)
        
        elapsed, eta = timer.stop(epoch)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | Acc: {train_acc:.2f}% | Time: {elapsed:.1f}s | ETA: {eta}")
        
        logger.log_text(f"epoch_{epoch}", f"Loss={train_loss_avg:.4f}, Acc={train_acc:.2f}%")

        # Validation
        model.eval()
        val_loss_meter = AverageMeter("val_loss")
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
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
        print(f"Epoch {epoch+1} Test Loss: {val_loss_avg:.4f} | Acc: {val_acc:.2f}%")
        logger.log_text(f"val_epoch_{epoch}", f"Loss={val_loss_avg:.4f}, Acc={val_acc:.2f}%")

        # Early stopping check
        if early_stopper.step(val_loss_avg, model, epoch):
            print(f"Early stopping triggered at epoch {epoch+1}")
            early_stopper.restore(model)
            break

    # Restore best weights if early stopping triggered
    if early_stopper.stopped_epoch is not None:
        print("Restored best weights.")

    # 5. ONNX Export
    print("Exporting model to ONNX...")
    model.eval()
    dummy_input, _ = train_dataset[0]
    dummy_input = dummy_input.unsqueeze(0).to(DEVICE)
    onnx_path = "simple_mnist_fixedpoint.onnx"

    export_onnx_with_io(model, dummy_input, onnx_path)
    print(f"Model successfully exported to {onnx_path}")

    logger.close()

if __name__ == "__main__":
    train()
