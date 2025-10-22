import math

from torch import nn

from elasticai.explorer.hw_nas.search_space.architecture_components import SimpleLSTM
from elasticai.explorer.hw_nas.search_space.layer_adapter import LSTMNoSequenceAdapter

from elasticai.explorer.hw_nas.search_space.registry import (
    activation_mapping,
)
from elasticai.explorer.hw_nas.search_space.utils import calculate_conv_output_shape

LAYER_REGISTRY = {}


def parse_search_param(trial, name, param):
    if isinstance(param, list):
        return trial.suggest_categorical(name, param)
    elif isinstance(param, dict) and "start" in param and "end" in param:
        if isinstance(param["start"], int):
            return trial.suggest_int(name, param["start"], param["end"])
    else:
        return param
    raise ValueError(f"Invalid search space parameter '{name}'")


def register_layer(name):
    """Decorator to register new layer types."""

    def wrapper(cls):
        LAYER_REGISTRY[name] = cls
        return cls

    return wrapper


class LayerBuilder:

    def __init__(
        self, trial, block, search_params, block_id, input_shape, output_shape
    ):
        self.trial = trial
        self.block = block
        self.search_params = search_params
        self.block_id = block_id
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = []

    def add_activation(self, activation_name):
        self.layers.append(activation_mapping[activation_name])

    def get_layers(self):
        return self.layers, self.input_shape


@register_layer("linear")
class LinearBuilder(LayerBuilder):
    def build(self, num_layers, is_last_block):
        if isinstance(self.input_shape, list):
            self.layers.append(nn.Flatten())
            self.input_shape = math.prod(self.input_shape)

        for i in range(num_layers):
            width = parse_search_param(
                self.trial,
                f"layer_width_b{self.block_id}_l{i}",
                self.search_params["width"],
            )
            activation = parse_search_param(
                self.trial,
                f"activation_func_b{self.block_id}_l{i}",
                self.block["activation"],
            )
            if is_last_block and i == num_layers - 1:
                self.layers.append(nn.Linear(self.input_shape, self.output_shape))
            else:
                self.layers.append(nn.Linear(self.input_shape, width))
                self.input_shape = width
            self.add_activation(activation)
        return self.get_layers()


@register_layer("conv2d")
class Conv2dBuilder(LayerBuilder):
    def build(self, num_layers, is_last_block):
        for i in range(num_layers):
            out_channels = parse_search_param(
                self.trial,
                f"out_channels_b{self.block_id}_l{i}",
                self.search_params["out_channels"],
            )
            kernel_size = parse_search_param(
                self.trial,
                f"kernel_size_b{self.block_id}_l{i}",
                self.search_params["kernel_size"],
            )
            stride = parse_search_param(
                self.trial,
                f"stride_b{self.block_id}_l{i}",
                self.search_params["stride"],
            )
            activation = parse_search_param(
                self.trial,
                f"activation_func_b{self.block_id}_l{i}",
                self.block["activation"],
            )

            self.layers.append(
                nn.Conv2d(self.input_shape[0], out_channels, kernel_size, stride)
            )
            self.add_activation(activation)

            self.input_shape = calculate_conv_output_shape(
                self.input_shape, out_channels, kernel_size, stride
            )

        return self.get_layers()


@register_layer("lstm")
class LSTMBuilder(LayerBuilder):
    def build(self, num_layers, is_last_block):
        return_sequence = parse_search_param(
            self.trial,
            f"return_sequence_b_{self.block_id}",
            self.search_params["return_sequence"],
        )
        for i in range(num_layers):
            hidden_size = parse_search_param(
                self.trial,
                f"hidden_size_b{self.block_id}_l{i}",
                self.search_params["hidden_size"],
            )
            num_lstm_layers = parse_search_param(
                self.trial,
                f"num_lstm_layers_b{self.block_id}_l{i}",
                self.search_params["num_lstm_layers"],
            )
            bidirectional = parse_search_param(
                self.trial,
                f"bidirectional_b{self.block_id}_l{i}",
                self.search_params["bidirectional"],
            )

            nn.LSTM(hidden_size, num_lstm_layers, bidirectional=bidirectional)
            self.layers.append(
                SimpleLSTM(
                    self.input_shape[-1],
                    hidden_size,
                    num_layers=num_lstm_layers,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
            )
            # 32,50,1 , batch, sequence, d*hiddensize
            # input shape (seq length, feature size)
            self.input_shape = [
                self.input_shape[0],
                hidden_size * 2 if bidirectional else hidden_size,
            ]

        if return_sequence == False:
            no_sequence_layer = LSTMNoSequenceAdapter()
            self.input_shape = no_sequence_layer.infer_output_shape(self.input_shape)
            self.layers.append(no_sequence_layer)

        return self.get_layers()


# sequenzlänge von dataset-> sollte auch suchparam sein? ja aber erstmal nicht
# output : final cell state for each element in sequence -> braucht man den? kann weg
# h: final hidden state for each element in sequence -> braucht man nur den letzten?, Suchparameter?
# KLassifikator der mehrere hiddenstates braucht
# tcn
