import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SimpleMNISTFloatNet(nn.Module):
    """
    A small CNN for MNIST using standard floating-point layers.
    This architecture mirrors SimpleMNISTNet in simple_mnist_qat.py.
    """
    def __init__(self):
        super().__init__()
        
        # Mirroring the input_quant from the QAT version
        self.input_quant = nn.Identity()
        
        # Layer 1: Conv -> ReLU -> Pool
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        # Layer 2: Conv -> ReLU -> Pool
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        
        # Final Linear Layer
        # Input size: 32 channels * 5x5 spatial (after two 2x2 pools and two 3x3 convs)
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = self.input_quant(x)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

def train():
    # Hyperparameters
    batch_size = 64
    epochs = 2
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = SimpleMNISTFloatNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training floating-point model on {device}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx*batch_size}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1} Test Accuracy: {accuracy:.2f}%")

    # --- Save Checkpoint ---
    checkpoint_path = "simple_mnist_float.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    # --- ONNX Export ---
    print("Exporting model to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    onnx_path = "simple_mnist_float.onnx"
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        opset_version=13, 
        dynamo=False 
    )
    print(f"Model successfully exported to {onnx_path}")

if __name__ == "__main__":
    train()
