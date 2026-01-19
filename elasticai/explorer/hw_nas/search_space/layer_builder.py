from abc import abstractmethod, ABC

from torch import nn as nn

from elasticai.explorer.hw_nas.search_space.architecture_components import (
    GaussianDropout,
    RepeatVector,
    TimeDistributed,
    SimpleLSTM,
)
from elasticai.explorer.hw_nas.search_space.registry import activation_mapping
from elasticai.explorer.hw_nas.search_space.utils import calculate_output_shape

LAYER_REGISTRY = {}


def register_layer(name: str):
    """Decorator to register new layer types."""

    def wrapper(cls):
        LAYER_REGISTRY[name] = cls
        return cls

    return wrapper


class LayerBuilder(ABC):

    @abstractmethod
    def build(self, input_shape, search_parameters: dict, output_shape=None):
        pass


@register_layer("linear")
class LinearLayer(LayerBuilder):

    def build(self, input_shape, search_parameters: dict, output_shape=None):
        activation = search_parameters.get("activation", None)

        if output_shape is not None:
            linear = nn.Linear(input_shape, output_shape)
            next_in_shape = output_shape
        else:
            linear = nn.Linear(input_shape, search_parameters["width"])
            next_in_shape = search_parameters["width"]
        if activation is not None:
            return nn.Sequential(linear, activation_mapping[activation]), next_in_shape
        else:
            return linear, next_in_shape


class ConvLayer(LayerBuilder):
    conv_class: type[nn.Module] = None
    layer_type: str = None

    def build(self, input_shape, search_parameters: dict, output_shape=None):
        activation = search_parameters.get("activation", None)
        output_shape = calculate_output_shape(
            input_shape,
            search_parameters["kernel_size"],
            search_parameters.get("stride", 1),
            search_parameters.get("padding", 0),
            out_channels=search_parameters["out_channels"],
            layer_type=self.layer_type,
        )
        conv = self.conv_class(
                input_shape[0],
                search_parameters["out_channels"],
                search_parameters["kernel_size"],
                search_parameters.get("stride", 1),
                search_parameters.get("padding", 0),
            )

        if activation is not None:
            return nn.Sequential(conv, activation_mapping[activation]), output_shape
        else:
            return (
                conv,
                output_shape,
            )


@register_layer("conv2d")
class Conv2dLayer(ConvLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_class = nn.Conv2d
        self.layer_type = "conv2d"


@register_layer("conv1d")
class Conv1dLayer(ConvLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_class = nn.Conv1d
        self.layer_type = "conv1d"


@register_layer("lstm")
class LSTMLayer(LayerBuilder):
    def build(self, input_shape, search_parameters: dict, output_shape=None):
        bidirectional: bool = search_parameters.get("bidirectional", False)
        hidden_size = search_parameters["hidden_size"]
        if output_shape is not None:
            hidden_size = output_shape
            if bidirectional:
                if (output_shape % 2) != 0:
                    raise NotImplementedError
                else:
                    hidden_size = output_shape / 2

        lstm = SimpleLSTM(
            input_shape[-1],
            hidden_size=hidden_size,
            num_layers=search_parameters.get("num_layers", 1),
            bidirectional=bidirectional,
            batch_first=search_parameters.get("batch_first", True),
            bias=search_parameters.get("bias", True),
            dropout=search_parameters.get("dropout", 0),
        )

        input_shape = [
            input_shape[0],
            hidden_size * 2 if bidirectional else hidden_size,
        ]

        return lstm, input_shape


class PoolLayer(LayerBuilder):

    layer_map = {}
    param_keys = {"kernel_size": None, "stride": 1, "padding": 0}

    def build(self, input_shape, search_parameters: dict, output_shape=None):
        if isinstance(input_shape, int):
            return nn.Identity(), input_shape
        ndim = 2 if len(input_shape) == 3 else 1
        layer_cls = self.layer_map.get(f"{ndim}d", None)
        if layer_cls is None:
            raise ValueError(
                f"No matching class for {ndim}D in {self.__class__.__name__}"
            )

        pool = layer_cls(
            **{k: search_parameters.get(k, v) for k, v in self.param_keys.items()}
        )

        shape = calculate_output_shape(
            input_shape,
            search_parameters["kernel_size"],
            search_parameters.get("stride", self.param_keys["stride"]),
            search_parameters.get("padding", self.param_keys["padding"]),
            layer_type=f"conv{ndim}d",
        )
        return pool, shape


@register_layer("maxpool")
class MaxPoolLayer(PoolLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer_map = {"1d": nn.MaxPool1d, "2d": nn.MaxPool2d}


@register_layer("avgpool")
class AvgPoolLayer(PoolLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_map = {"1d": nn.AvgPool1d, "2d": nn.AvgPool2d}


@register_layer("batch_norm")
class BatchNormLayer(LayerBuilder):
    def build(self, input_shape, search_parameters: dict, output_shape=None):
        num_features = (
            input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
        )
        if isinstance(input_shape, int):
            layer_cls = nn.BatchNorm1d
        elif len(input_shape) == 3:
            layer_cls = nn.BatchNorm2d
        else:
            layer_cls = nn.BatchNorm1d

        return layer_cls(num_features=num_features), input_shape


@register_layer("dropout")
class DropoutLayer(LayerBuilder):
    param_keys = ["p"]
    layer_map = {"1d": nn.Dropout, "2d": nn.Dropout2d}

    def build(self, input_shape, search_parameters: dict, output_shape=None):
        return nn.Dropout(search_parameters.get("p", 0.5)), input_shape


@register_layer("activation")
class ActivationLayer(LayerBuilder):
    def build(self, input_shape, search_parameters: dict, output_shape=None):
        return activation_mapping[search_parameters.get("op", "identity")], input_shape


@register_layer("layer_norm")
class LayerNorm(LayerBuilder):
    def build(self, input_shape, search_parameters: dict, output_shape=None):
        return nn.LayerNorm(input_shape), input_shape


@register_layer("gaussian_dropout")
class GaussianDropoutLayer(LayerBuilder):
    def build(self, input_shape, search_parameters: dict, output_shape=None):
        return GaussianDropout(search_parameters.get("p", 0.5)), input_shape


@register_layer("repeat_vector")
class RepeatVectorLayer(LayerBuilder):
    def build(self, input_shape, search_parameters: dict, output_shape=None):
        times = search_parameters["times"]
        input_shape.append(times)
        return RepeatVector(times), input_shape


@register_layer("time_distributed_linear")
class TimeDistributedLinear(LayerBuilder):
    def build(self, input_shape, search_parameters: dict, output_shape=None):
        batch_first = search_parameters.get("batch_first", True)
        if output_shape is None:
            output_sample_shape = [input_shape[0], search_parameters["width"]]
        else:
            output_sample_shape = output_shape
        print(
            f"shape before time_distributed_linear input: {input_shape} Output: {output_shape}"
        )
        module = nn.Sequential(
            nn.Linear(input_shape[-1], output_sample_shape[-1]),
            activation_mapping[search_parameters.get("activation", "identity")],
        )

        return TimeDistributed(module, batch_first=batch_first), output_sample_shape
