from torch import nn

from elasticai.explorer.hw_nas.search_space.layer_adapter import (
    Conv2dToLSTMAdapter,
    LinearToConv2dAdapter,
    LSTMNoSequenceAdapter,
    LSTMToConv2dAdapter,
    Conv2dToLinearAdapter,
    LinearToLstmAdapter,
    ToLinearAdapter,
)

ADAPTER_REGISTRY = {
    ("conv2d", "lstm"): Conv2dToLSTMAdapter,
    ("linear", "conv2d"): LinearToConv2dAdapter,
    ("linear", "lstm"): LinearToLstmAdapter,
    ("conv2d", "linear"): Conv2dToLinearAdapter,
    ("lstm", "linear"): LSTMNoSequenceAdapter,
    ("lstm", "conv2d"): LSTMToConv2dAdapter,
    ("lstm", None): LSTMNoSequenceAdapter,
    (None, "linear"): ToLinearAdapter,
    ("conv1d", "linear"): ToLinearAdapter,
    ("*", "linear"): ToLinearAdapter,
}

activation_mapping = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "identity": nn.Identity(),
    "tanh": nn.Tanh(),
}
