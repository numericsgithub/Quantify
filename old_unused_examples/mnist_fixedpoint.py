import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU

from quantizers.fixedpoint_per_tensor_weights import FixedPointPerTensorWeightQuant

# -----------------------------------------------------------------------------
# Model Definition
# -----------------------------------------------------------------------------

class FixedPointCNN(nn.Module):
    """
    A simple CNN for MNIST using Brevitas layers and the custom 
    FixedPointPerTensorWeightQuant quantizer.
    """
    def __init__(self):
        super(FixedPointCNN, self).__init__()
        
        # We use the custom FixedPointPerTensorWeightQuant for all weight quantization
        weight_quant = FixedPointPerTensorWeightQuant
        
        self.features = nn.Sequential(
            # Layer 1: Conv -> ReLU -> MaxPool
            QuantConv2d(1, 16, kernel_size=3, stride=2, padding=1, weight_quant=weight_quant),
            QuantReLU(),
            QuantConv2d(16, 32, kernel_size=3, stride=2, padding=1, weight_quant=weight_quant),
            QuantReLU(),
            #nn.MaxPool2d(kernel_size=2),
            
            # Layer 2: Conv -> ReLU -> MaxPool
            QuantConv2d(32, 64, kernel_size=3, stride=2, padding=1, weight_quant=weight_quant),
            QuantReLU(),
            #nn.MaxPool2d(kernel_size=2),
            #QuantConv2d(64, 64, kernel_size=3, stride=2, padding=0, weight_quant=weight_quant),
            #QuantReLU(),
            #QuantConv2d(64, 64, kernel_size=3, stride=2, padding=0, weight_quant=weight_quant),
        )
        
        self.classifier = nn.Sequential(
            QuantLinear(64*4*4, 10, weight_quant=weight_quant),
            #QuantReLU(),
            #QuantLinear(128, 10, weight_quant=weight_quant),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -----------------------------------------------------------------------------
# Training Utilities
# -----------------------------------------------------------------------------

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
    acc = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch: {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    acc = 100.0 * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    # Hyperparameters
    batch_size = 512
    epochs = 5
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Optimizer, Loss
    model = FixedPointCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

if __name__ == "__main__":
    main()
