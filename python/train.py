from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# טעינת הדאטה
iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target

# נרמול
scaler = StandardScaler()
X = scaler.fit_transform(X)

# חלוקה לאימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# המרה ל-tensors
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# הגדרת הרשת
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = IrisNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# אימון
for epoch in range(500):
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# בדיקת דיוק
model.eval()
with torch.no_grad():
    out = model(X_test)
    predicted = out.argmax(dim=1)
    accuracy = (predicted == y_test).float().mean()
    print(f"\nAccuracy: {accuracy.item()*100:.1f}%")

ROOT = Path(__file__).resolve().parent.parent

# שמירת המשקלות
weights = {
    'w1': model.fc1.weight.detach().numpy(),
    'b1': model.fc1.bias.detach().numpy(),
    'w2': model.fc2.weight.detach().numpy(),
    'b2': model.fc2.bias.detach().numpy(),
    'scaler_mean': scaler.mean_.astype(np.float32),
    'scaler_std': scaler.scale_.astype(np.float32)
}
np.save(ROOT / 'weights.npy', weights)
print(f"\nWeights saved to {ROOT / 'weights.npy'}")
