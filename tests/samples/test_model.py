from torch import nn
from torch.nn import functional

class test_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x