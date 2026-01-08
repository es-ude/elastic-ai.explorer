from torch import nn

from search_space.layer_adapter import LinearToConv2dAdapter, LinearToLstmAdapter, \
    Conv2dToLinearAdapter, LSTMNoSequenceAdapter, LSTMToConv2dAdapter, ToLinearAdapter,  \
    Conv2dToLSTMAdapter

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

