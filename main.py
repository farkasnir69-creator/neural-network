
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import random_split,DataLoader
import torch.nn.functional as F



# Load the training dataset and convert images to tensors
dataset = MNIST(
    root="data/",
    train=True,
    download=True,
    transform=ToTensor()
)

print("Number of images:", len(dataset))

# Get the 11th image (index 10)
image_tensor, label = dataset[10]

print("Label:", label)
print("Tensor shape:", image_tensor.shape)

# Display the entire image
plt.imshow(image_tensor.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()

# Inspect a 5x5 region of pixels
region = image_tensor[:, 10:15, 10:15]
print("Pixel region:\n", region)

# Inspect pixel intensity range
print("Maximum:", torch.max(image_tensor))
print("Minimum:", torch.min(image_tensor))

# Display the selected region
plt.imshow(image_tensor.squeeze()[10:15, 10:15], cmap="gray")
plt.title("5x5 pixel region")
plt.axis("off")
plt.show()
train_data, validation_data = random_split(dataset, [50000, 10000])
## Print the length of train and validation datasets
print("length of Train Datasets: ", len(train_data))
print("length of Validation Datasets: ", len(validation_data))
batch_size = 128
train_loader = DataLoader(train_data, batch_size, shuffle = True)
val_loader = DataLoader(validation_data, batch_size, shuffle = False)
import torch.nn as nn

import torch.nn as nn

input_size = 28 * 28
num_classes = 10

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, input_size)
        return self.linear(xb)

model = MnistModel()

# Get one batch
images, labels = next(iter(train_loader))

# Make predictions
outputs = model(images)

print("Input batch shape:", images.shape)
print("Flattened input shape:", images.reshape(-1, 784).shape)
print("Output shape:", outputs.shape)

# Convert scores into probabilities
probs = F.softmax(outputs, dim=1)

print("Sample probabilities:\n", probs[:2].detach())
print("Probability sum:", probs[0].sum().item())

# Get predicted classes
max_probs, preds = torch.max(probs, dim=1)

print("Predictions:", preds[:10])
print("Actual labels:", labels[:10])

# Calculate accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return (preds == labels).float().mean().item()

print("Initial accuracy:", accuracy(outputs, labels))

# Calculate cross-entropy loss
loss = F.cross_entropy(outputs, labels)
print("Initial loss:", loss.item())
