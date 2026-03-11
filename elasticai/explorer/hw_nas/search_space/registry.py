from torch import nn

from elasticai.explorer.hw_nas.search_space.layer_adapter import (
    Conv2dToLSTMAdapter,
    LinearToConv2dAdapter,
    LinearToLstmAdapter,
    Conv2dToLinearAdapter,
    LSTMNoSequenceAdapter,
    LSTMToConv2dAdapter,
    ToLinearAdapter,
)

ADAPTER_REGISTRY = {}
LAYER_REGISTRY = {}
ACTIVATION_REGISTRY = {}
DEFAULT_ADAPTER = {
    ("conv2d", "lstm"): Conv2dToLSTMAdapter,
    ("linear", "conv2d"): LinearToConv2dAdapter,
    ("linear", "lstm"): LinearToLstmAdapter,
    ("conv2d", "linear"): Conv2dToLinearAdapter,
    ("lstm", "linear"): LSTMNoSequenceAdapter,
    ("lstm", "conv2d"): LSTMToConv2dAdapter,
    ("lstm", None): LSTMNoSequenceAdapter,
    (None, "linear"): ToLinearAdapter,
    ("*", "linear"): ToLinearAdapter,
    ("conv1d", "linear"): ToLinearAdapter,
}


COMPOSITE_REGISTRY = {}

DEFAULT_ACTIVATION = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "identity": nn.Identity(),
    "tanh": nn.Tanh(),
}
