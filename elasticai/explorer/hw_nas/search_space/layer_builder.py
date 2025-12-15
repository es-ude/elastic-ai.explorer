from abc import abstractmethod, ABC
from typing import Any
from torch import nn
from yaml import YAMLError

from elasticai.explorer.hw_nas.search_space.architecture_components import SimpleLSTM

from elasticai.explorer.hw_nas.search_space.registry import (
    activation_mapping,
)
from elasticai.explorer.hw_nas.search_space.utils import calculate_conv_output_shape

LAYER_REGISTRY = {}


def parse_search_param(
    trial, name: str, params: dict, key: str, default_value: Any = None
) -> Any:
    if key in params:
        param = params[key]
    else:
        if default_value is not None:
            return default_value
        else:
            raise YAMLError(
                "Parameter '{}' is not optional and missing in configuration.".format(
                    name
                )
            )
    if isinstance(param, list):
        return trial.suggest_categorical(name, param)
    elif isinstance(param, dict) and "start" in param and "end" in param:
        if isinstance(param["start"], int):
            return trial.suggest_int(name, param["start"], param["end"])
    else:
        return param
    raise ValueError(f"Invalid search space parameter '{name}'")


def register_layer(name: str):
    """Decorator to register new layer types."""

    def wrapper(cls):
        LAYER_REGISTRY[name] = cls
        return cls

    return wrapper


class LayerBuilder(ABC):

    def __init__(
        self,
        trial,
        block: dict,
        search_params: dict,
        block_id,
        input_shape: list | int,
        output_shape: list | int,
    ):
        self.trial = trial
        self.block = block
        self.search_params = search_params
        self.block_id = block_id
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = []

    @abstractmethod
    def build(self, num_layers: int, is_last_block: bool):
        pass

    def add_activation(self, activation_name: str):
        self.layers.append(activation_mapping[activation_name])

    def get_layers(self):
        return self.layers, self.input_shape


@register_layer("linear")
class LinearBuilder(LayerBuilder):

    def build(self, num_layers: int, is_last_block: bool):
        for i in range(num_layers):
            width = parse_search_param(
                self.trial,
                f"layer_width_b{self.block_id}_l{i}",
                self.search_params,
                "width",
            )
            activation = parse_search_param(
                self.trial,
                f"activation_func_b{self.block_id}_l{i}",
                self.block,
                "activation",
                "identity",
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
    def build(self, num_layers: int, is_last_block: bool):
        for i in range(num_layers):
            out_channels = parse_search_param(
                self.trial,
                f"out_channels_b{self.block_id}_l{i}",
                self.search_params,
                "out_channels",
                default_value=None,
            )
            kernel_size = parse_search_param(
                self.trial,
                f"kernel_size_b{self.block_id}_l{i}",
                self.search_params,
                "kernel_size",
                default_value=None,
            )
            stride = parse_search_param(
                self.trial,
                f"stride_b{self.block_id}_l{i}",
                self.search_params,
                "stride",
                default_value=1,
            )
            activation = parse_search_param(
                self.trial,
                f"activation_func_b{self.block_id}_l{i}",
                self.block,
                "activation",
                default_value="identity",
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
    def build(self, num_layers: int, is_last_block: bool):
        for i in range(num_layers):
            hidden_size = parse_search_param(
                self.trial,
                f"hidden_size_b{self.block_id}_l{i}",
                self.search_params,
                "hidden_size",
                default_value=None,
            )
            num_lstm_layers = parse_search_param(
                self.trial,
                f"num_lstm_layers_b{self.block_id}_l{i}",
                self.search_params,
                "num_lstm_layers",
                default_value=None,
            )
            bidirectional = parse_search_param(
                self.trial,
                f"bidirectional_b{self.block_id}_l{i}",
                self.search_params,
                "bidirectional",
                default_value=False,
            )
            dropout = parse_search_param(
                self.trial,
                f"dropout_b{self.block_id}_l{i}",
                self.search_params,
                key="dropout",
                default_value=0.0,
            )
            if is_last_block and i == num_layers - 1:
                hidden_size = self.output_shape
                if bidirectional & ((self.output_shape % 2) != 0):
                    raise NotImplementedError
                elif bidirectional:
                    hidden_size = self.output_shape / 2

            self.layers.append(
                SimpleLSTM(
                    self.input_shape[-1],
                    hidden_size,
                    num_layers=num_lstm_layers,
                    bidirectional=bidirectional,
                    batch_first=True,
                    dropout=dropout,
                )
            )
            self.input_shape = [
                self.input_shape[0],
                hidden_size * 2 if bidirectional else hidden_size,
            ]
        return self.get_layers()
