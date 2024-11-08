import torch.nn as nn
from nni.nas.nn.pytorch import ParametrizedModule
class MutableLinear(ParametrizedModule):
    def __init__(self, input, output):
        super().__init__()
        self.linear = nn.Linear(input, output)

    def forward(self, x):
        return self.linear(x)


class MutableDropout(ParametrizedModule):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
