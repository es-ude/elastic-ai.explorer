from collections import OrderedDict
import logging

from torch import nn
from typing import Any, Sequence
import torch

from elasticai.explorer.generator.model_builder import ModelBuilder
from elasticai.explorer.hw_nas.search_space.layer_builder import layer_registry
from elasticai.explorer.hw_nas.search_space.quantization import (
    QuantizationScheme,
)

from elasticai.explorer.hw_nas.search_space.registry import (
    adapter_registry,
    DEFAULT_ACTIVATION_REGISTRY,
    DEFAULT_ADAPTER_REGISTRY,
)
from elasticai.explorer.hw_nas.search_space.sample_blocks import Sampler


def insert_needed_adapters(input_shape, op, prev_operation, layers):
    adapter_cls = adapter_registry.get((prev_operation, op))
    if adapter_cls is None:
        adapter_cls = adapter_registry.get(("*", op))
    if adapter_cls is not None:
        adapter = adapter_cls()
        layers.append(adapter)
        next_input_shape = adapter_cls.infer_output_shape(input_shape)
        return layers, next_input_shape
    return layers, input_shape


def is_last_layer(block_index, layer_index, sample):
    block_id, layers = next(reversed(sample.items()))
    layer_id, layer = next(reversed(layers.items()))
    return block_index == block_id and layer_id == layer_index


def is_negative(value):
    if isinstance(value, Sequence):
        for val in value:
            if val <= 0:
                return True
    else:
        if value <= 0:
            return True
    return False


class ShapeValueError(ValueError):
    pass


class DefaultModelBuilder(ModelBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(
            "explorer.generator.model_builder.TorchModelBuilder"
        )

    def get_activation_mappings(self) -> dict[str, Any]:
        return DEFAULT_ACTIVATION_REGISTRY

    def get_adapter_mappings(self) -> dict[tuple[str | None, str | None], type | None]:
        return DEFAULT_ADAPTER_REGISTRY

    def get_supported_quantization(self) -> dict[str, Any]:
        return {}

    def construct_layers(
        self, sample: OrderedDict, in_dim, out_dim, quant_scheme=None
    ):
        layers = []
        next_in_shape = in_dim
        prev_op = None
        for i, block in sample.items():
            for layer_index, layer_params in block.items():
                layers, next_in_shape = insert_needed_adapters(
                    next_in_shape, layer_params["operation"], prev_op, layers
                )
                layer = layer_registry[layer_params["operation"]]()
                if is_last_layer(i, layer_index, sample):
                    build_layer, next_in_shape = layer.build(
                        input_shape=next_in_shape,
                        search_parameters=layer_params["params"],
                        output_shape=out_dim,
                        quantization_scheme=quant_scheme,
                    )
                else:
                    build_layer, next_in_shape = layer.build(
                        input_shape=next_in_shape,
                        search_parameters=layer_params["params"],
                        quantization_scheme=quant_scheme,
                    )
                if type(build_layer) == list:
                    for layer in build_layer:
                        layers.append(layer)
                else:
                    layers.append(build_layer)
                prev_op = layer_params["operation"]
                if is_negative(next_in_shape):
                    raise ShapeValueError("Shape must not be negative")

        return layers

    def build_from_trial(
        self, trial, search_space: dict
    ) -> tuple[torch.nn.Module, QuantizationScheme | None]:
        sampler = Sampler(trial)
        sample = sampler.construct_sample(search_space)
        quant_scheme = sampler.get_quantization_scheme(search_space)
        return (
            nn.Sequential(
                *self.construct_layers(
                    sample,
                    search_space["input"],
                    search_space["output"],
                    quant_scheme=quant_scheme,
                )
            ),
            quant_scheme,
        )
