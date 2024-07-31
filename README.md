# Lion
Torch implementation of [Lion](https://arxiv.org/abs/2302.06675) optimizer.

Currently supports sparse gradients and foreach operations.

# Installation
```bash
git clone https://github.com/diffraction-zebra/Lion.git
```

# Usage
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from lion import Lion

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Generate some dummy data
np.random.seed(42)
torch.manual_seed(42)
X = np.random.rand(100, 2).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 1).astype(np.float32).reshape(-1, 1)

# Create DataLoader
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Initialize the network, loss function and optimizer
net = SimpleNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = Lion(net.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation
with torch.no_grad():
    outputs = net(torch.tensor(X))
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    accuracy = (predicted == torch.tensor(y)).sum().item() / y.shape[0]
    print(f'Accuracy: {accuracy:.4f}')
```
