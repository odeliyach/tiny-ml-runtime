from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_data = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=1000)

# Define network: 784 → 128 → 10
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)   # flatten 28x28 image to 784 numbers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MNISTNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train for 5 epochs
for epoch in range(5):
    model.train()
    for batch_x, batch_y in train_loader:
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check accuracy on test set
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            pred = model(batch_x)
            correct += (pred.argmax(dim=1) == batch_y).sum().item()
    print(f"Epoch {epoch+1}: accuracy = {correct/100:.1f}%")

# Export weights in generic format
print("\nExporting weights...")
np.save(ROOT / 'mnist_weights.npy', {
    'architecture': [784, 128, 10],
    'w1': model.fc1.weight.detach().numpy().astype(np.float32),
    'b1': model.fc1.bias.detach().numpy().astype(np.float32),
    'w2': model.fc2.weight.detach().numpy().astype(np.float32),
    'b2': model.fc2.bias.detach().numpy().astype(np.float32),
})
print(f"Saved to {ROOT / 'mnist_weights.npy'}")
