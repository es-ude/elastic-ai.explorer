from torch import nn

from elasticai.explorer.hw_nas.search_space.layer_adapter import (
    Conv2dToLSTMAdapter,
    Conv2dToLinearAdapter,
    LSTMToLinearAdapter,
)

ADAPTER_REGISTRY = {
    #  ("conv2d", "lstm"): Conv2dToLSTMAdapter,
    ("conv2d", "linear"): Conv2dToLinearAdapter,
    ("lstm", "linear"): LSTMToLinearAdapter,
}

activation_mapping = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "identity": nn.Identity(),
    "tanh": nn.Tanh(),
}
