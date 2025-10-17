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


class LSTMToLinearAdapter(nn.Module):
    """(B, T, H) → (B, H)"""

    def forward(self, x):
        out, c = x
        return out[:, -1, :]

    @staticmethod
    def infer_output_shape(input_shape):
        # input_shape = (T, H)
        _, h = input_shape
        return h
