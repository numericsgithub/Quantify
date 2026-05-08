import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import brevitas.nn as qnn

# Import the custom fixed-point quantizers
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorWeightQuant, FixedPointPerTensorActivationQuant

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
            1, 16, kernel_size=3, stride=1,
            weight_quant=FixedPointPerTensorWeightQuant,
            output_quant=FixedPointPerTensorActivationQuant
        )
        self.relu1 = qnn.QuantReLU(
            act_quant=FixedPointPerTensorActivationQuant
        )
        self.pool1 = nn.MaxPool2d(2)

        # Layer 2: Conv -> ReLU -> Pool
        self.conv2 = qnn.QuantConv2d(
            16, 32, kernel_size=3, stride=1,
            weight_quant=FixedPointPerTensorWeightQuant,
            output_quant=FixedPointPerTensorActivationQuant
        )
        self.relu2 = qnn.QuantReLU(
            act_quant=FixedPointPerTensorActivationQuant
        )
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        
        # Final Linear Layer
        # Input size: 32 channels * 5x5 spatial (after two 3x3 convs and two 2x2 pools)
        self.fc = qnn.QuantLinear(
            32 * 5 * 5, 10,
            weight_quant=FixedPointPerTensorWeightQuant,
            output_quant=FixedPointPerTensorActivationQuant
        )

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
    model = SimpleMNISTNet().to(device)
    
    # --- Load Floating-Point Checkpoint ---
    checkpoint_path = "simple_mnist_float.pt"
    try:
        # strict=False is required because the float state_dict lacks Brevitas quantization parameters
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        print(f"Successfully loaded floating-point checkpoint from {checkpoint_path} for fine-tuning.")
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_path} not found. Training from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {device}...")

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

    # --- ONNX Export ---
    print("Exporting model to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    onnx_path = "simple_mnist_fixedpoint.onnx"
    
    # We MUST use dynamo=False because FixedPointPerTensorQuantizer 
    # uses torch.autograd.Function.symbolic for custom ONNX nodes.
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        opset_version=13, 
        custom_opsets={'mydomain': 1}, 
        dynamo=False 
    )
    print(f"Model successfully exported to {onnx_path}")

if __name__ == "__main__":
    train()
