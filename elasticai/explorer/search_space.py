import torch.nn as nn
import torch.nn.functional as F
import nni
from nni.nas.nn.pytorch import  ModelSpace
from mutable_types import  MutableLinear


class MLP(ModelSpace):
    def __init__(self):
        super().__init__()

        h1 = nni.choice("h1", [256, 512])
        h2 = nni.choice("h2", [128, 512])
        self.fc1 = MutableLinear(28 * 28, h1)
        self.fc2 = MutableLinear(h1, h2)
        self.fc3 = MutableLinear(h2, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
