import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================
# 1. Config
# ==========================
data_dir = "dataset"
batch_size = 32
num_epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ==========================
# 2. Transforms
# ==========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ==========================
# 3. Load Data
# ==========================
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))

# ==========================
# 4. Model (Fine-Tuning)
# ==========================
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 2 feature blocks
for param in model.features[-2:].parameters():
    param.requires_grad = True

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Only optimize trainable parameters
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# ==========================
# 5. Training Loop with Validation
# ==========================
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels)

    train_acc = train_correct.double() / len(train_dataset)

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels)

    val_acc = val_correct.double() / len(val_dataset)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    print("-" * 40)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

print("Training complete!")
print("Best Validation Accuracy:", best_val_acc)

# ==========================
# 6. Load Best Model
# ==========================
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# ==========================
# 7. Test Evaluation
# ==========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print("Test Accuracy:", accuracy)

print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ==========================
# 8. Export ONNX
# ==========================
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
