import nni
import torch.nn.functional as F
from nni.nas.nn.pytorch import ModelSpace
from nni.nas.nn.pytorch.layers import MutableDropout, MutableLinear


class MLP(ModelSpace):
    def __init__(self):
        super().__init__()
        h1 = nni.choice("layer_1", [4, 32])
        h2 = nni.choice("layer_2", [4, 32])
        h3 = nni.choice("dropout", [0.2, 0.25, 0.9])
        self.fc1 = MutableLinear(28 * 28, h1)
        self.fc2 = MutableLinear(h1, h2)
        self.fc3 = MutableLinear(h2, 10)
        self.dropout = MutableDropout(h3)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
