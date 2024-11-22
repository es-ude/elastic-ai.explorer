import torch.nn as nn
import torch.nn.functional as F
import nni
from nni.nas.nn.pytorch import  ModelSpace
from nni.nas.nn.pytorch import layers
#from elasticai.explorer.mutable_types import  MutableLinear
from nni.nas.profiler.pytorch.utils.shape_formula import linear_formula, register_shape_inference_formula, keep_shape_formula

#from elasticai.explorer.mutable_types import MutableLinear, MutableDropout



class MLP(ModelSpace):
    def __init__(self):
        super().__init__()

        h1 = nni.choice("h1", [256, 512])
        h2 = nni.choice("h2", [128, 512])
        self.fc1 = layers.MutableLinear(28 * 28, h1)
        self.fc2 = layers.MutableLinear(h1, h2)
        self.fc3 = layers.MutableLinear(h2, 10)
        self.dropout = layers.MutableDropout(0.2)

        

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
