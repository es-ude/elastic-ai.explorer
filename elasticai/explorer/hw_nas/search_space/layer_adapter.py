from torch import nn


class Conv2dToLSTMAdapter(nn.Module):
    """(B, C, H, W) → (B, H*W, C)"""

    def forward(self, x):
        b, c, h, w = x.size()
        return x.view(b, c, -1).permute(0, 2, 1)

    @staticmethod
    def infer_output_shape(input_shape):
        # input_shape = (C, H, W)
        c, h, w = input_shape
        return (h * w, c)  # sequence_length, feature_dim


class Conv2dToLinearAdapter(nn.Module):
    """(B, C, H, W) → (B, C*H*W)"""

    def forward(self, x):
        return x.view(x.size(0), -1)

    @staticmethod
    def infer_output_shape(input_shape):
        c, h, w = input_shape
        return c * h * w  # Flattened vector


class LSTMNoSequenceAdapter(nn.Module):
    """(B, T, H) → (B, H)"""

    def forward(self, x):
        return x[:, -1, :]

    @staticmethod
    def infer_output_shape(input_shape):
        _, h = input_shape
        return h


class LSTMToConv2dAdapter(nn.Module):
    def __init__(self):
        raise AdapterNotImplementedError(
            "LSTM to to Conv2d adapter is not implemented. "
        )


class LinearToConv2dAdapter(nn.Module):
    def __init__(self):
        raise AdapterNotImplementedError(
            "Linear layer to Conv2d adapter is not implemented. "
        )


class LinearToLSTMAdapter(nn.Module):
    def __init__(self):
        raise AdapterNotImplementedError(
            "Linear layer to LSTM adapter is not implemented. "
        )


class AdapterNotImplementedError(NotImplementedError):
    def __init__(self, msg):
        super(AdapterNotImplementedError, self).__init__(msg)
