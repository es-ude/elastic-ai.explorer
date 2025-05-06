from torch import Tensor, nn
import torch
from torch.nn import functional


class SampleMLP(nn.Module):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        super().__init__()
        self.fc1 = nn.Linear(self.input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.view(-1, self.input_dim)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
