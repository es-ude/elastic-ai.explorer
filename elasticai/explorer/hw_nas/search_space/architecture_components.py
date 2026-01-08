import torch
from torch import nn


# We don't use the cell states that are returned by the pytorch module. To make it easier to connect the layers,
# this class throws the cell states away and only returns the output.
class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        batch_first: bool,
        bias: bool,
        dropout: float,
    ):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            bidirectional=bidirectional,
            batch_first=batch_first,
            dropout=dropout,
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out


class RepeatVector(nn.Module):
    def __init__(self, times):
        super(RepeatVector, self).__init__()
        self.times = times

    def forward(self, x):
        print(x.shape)
        print(x)
        x = torch.Tensor.repeat(x, self.times)
        print(x.shape)
        print(x)


class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        if p < 0 or p >= 1:
            raise Exception("p value should accomplish 0 <= p < 1")
        self.p = p

    def forward(self, x):
        if self.training:
            stddev = (self.p / (1.0 - self.p)) ** 0.5
            epsilon = torch.randn_like(x) * stddev
            return x * epsilon
        else:
            return x


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        y = y.contiguous().view(
            x.size(0), -1, y.size(-1)
        )  # (samples, timesteps, output_size)

        if not self.batch_first:
            y = y.transpose(
                0, 1
            ).contiguous()  # transpose to (timesteps, samples, output_size)

        return y
