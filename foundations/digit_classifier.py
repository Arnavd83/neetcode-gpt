import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.l1 = nn.Linear(784, 512)
        self.r1 = nn.ReLU()
        self.d1 = nn.Dropout(.2)
        self.l2 = nn.Linear(512,10)
        self.s = nn.Sigmoid()
        pass

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x = self.l1(images)
        x = self.r1(x)
        x = self.d1(x)
        x = self.l2(x)
        y_hat = self.s(x)
        return torch.round(y_hat, decimals=4)

    '''import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))    # (batch, 784) → (batch, 256)
            x = self.relu(self.fc2(x))    # (batch, 256) → (batch, 256)
            x = self.fc3(x)               # (batch, 256) → (batch, 10)
            return x                       # raw logits — no activation on output'''
