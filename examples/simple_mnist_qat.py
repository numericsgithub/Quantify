import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import brevitas.nn as qnn
from quantizers.manager import QuantizerManager
quantizer_manager = QuantizerManager()

# Import the custom fixed-point quantizers
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorWeightQuant, FixedPointPerTensorActivationQuant, FixedPointPerTensorBiasQuant
from quantizers.coefficient_per_tensor_weights import CoefficientPerTensorWeightQuant
from utils.onnx_export import export_onnx_with_io

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
    # Hyperparameters
    batch_size = 256
    epochs = 5
    lr = 0.001
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
    # checkpoint_path = "simple_mnist_float.pt"
    # try:
    #     # strict=False is required because the float state_dict lacks Brevitas quantization parameters
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    #     print(f"Successfully loaded floating-point checkpoint from {checkpoint_path} for fine-tuning.")
    # except FileNotFoundError:
    #     print(f"Checkpoint {checkpoint_path} not found. Training from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Quantization Manager Setup ---
    # Note: QuantizerManager is no longer a global singleton. 
    # Instantiate it explicitly per training run to avoid state leakage across experiments or DDP ranks.
    quantizer_manager.quantization_start_gap = 20
    quantizer_manager.set_annealing_for_n_inferences(6)

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
    # dummy_input = torch.ones(1, 1, 28, 28) * (2.0 ** -6.0)
    # dummy_input = dummy_input.to(device)
    dummy_input, _ = train_dataset[0]  # shape: [1, 28, 28]
    dummy_input = dummy_input.unsqueeze(0).to(device)  # add batch dimension -> [1, 1, 28, 28]
    onnx_path = "simple_mnist_fixedpoint.onnx"

    # We MUST use dynamo=False because FixedPointPerTensorQuantizer 
    # uses torch.autograd.Function.symbolic for custom ONNX nodes.
    # torch.onnx.export(
    #     model,
    #     dummy_input,
    #     onnx_path,
    #     opset_version=17,
    #     custom_opsets={'Quantify': 1},
    #     dynamo=False
    # )

    # def export_with_test_vector(model, dummy_input, path):
    #     model.eval()
    #
    #     with torch.no_grad():
    #         expected_output = model(dummy_input)
    #
    #     torch.onnx.export(
    #         model,
    #         dummy_input,
    #         path,
    #         opset_version=17,
    #         dynamo=False
    #     )
    #
    #     proto = onnx.load(path)
    #
    #     entry1 = onnx.StringStringEntryProto()
    #     entry1.key = "dummy_input"
    #     entry1.value = json.dumps(dummy_input.cpu().numpy().tolist())
    #
    #     entry2 = onnx.StringStringEntryProto()
    #     entry2.key = "expected_output"
    #     entry2.value = json.dumps(expected_output.cpu().numpy().tolist())
    #
    #     proto.metadata_props.extend([entry1, entry2])
    #
    #     torch.onnx.save(proto, path)

    export_onnx_with_io(model, dummy_input, "simple_mnist_fixedpoint2.onnx")

    print(f"Model successfully exported to {onnx_path}")

if __name__ == "__main__":
    train()
