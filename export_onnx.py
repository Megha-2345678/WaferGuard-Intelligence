import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 9  # your number of classes

# Create model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# Load trained weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))

model.to(device)
model.eval()

print("Model loaded successfully!")

dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("ONNX model exported successfully!")
