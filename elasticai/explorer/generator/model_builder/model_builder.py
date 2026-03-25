from abc import ABC, abstractmethod
from collections import OrderedDict
import logging

from torch import nn
from typing import Any, Sequence
import torch
from elasticai.explorer.generator.reflection import Reflective

from elasticai.explorer.hw_nas.search_space.layer_builder import layer_registry
from elasticai.explorer.hw_nas.search_space.quantization import (
    FullPrecisionScheme,
    PTQFullyQuantizedInt8Scheme,
    QuantizationScheme,
)

from elasticai.explorer.hw_nas.search_space.quantization_builder import (
    FullPrecisionBuilder,
    PTQFullyQuantizedInt8Builder,
    quantization_registry,
)
from elasticai.explorer.hw_nas.search_space.registry import (
    activation_registry,
    adapter_registry,
    DEFAULT_ACTIVATION,
    DEFAULT_ADAPTER,
    DEFAULT_QUANTIZATION,
    layer_registry,
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


class ModelBuilder(Reflective, ABC):
    @abstractmethod
    def build_from_trial(
        self, trial, search_space: dict
    ) -> tuple[Any, QuantizationScheme]:
        pass

    def setup_registries(self, replace=False):
        if replace:
            activation_registry.clear()
            adapter_registry.clear()
            layer_registry.clear()
            quantization_registry.clear()

        activation_registry.update(self.get_activation_mappings())
        adapter_registry.update(self.get_adapter_mappings())
        layer_registry.update(self.get_layer_mappings())
        quantization_registry.update(self.get_supported_quantization_schemes())


class DefaultModelBuilder(ModelBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(
            "explorer.generator.model_builder.TorchModelBuilder"
        )
        self.setup_registries()

    def get_activation_mappings(self) -> dict[str, Any]:
        return DEFAULT_ACTIVATION

    def get_adapter_mappings(self) -> dict[tuple[str | None, str | None], type | None]:
        return DEFAULT_ADAPTER

    def get_supported_quantization_schemes(self) -> dict[str, Any]:
        return DEFAULT_QUANTIZATION

    def construct_model(self, sample: OrderedDict, in_dim, out_dim):
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
                    )
                else:
                    build_layer, next_in_shape = layer.build(
                        input_shape=next_in_shape,
                        search_parameters=layer_params["params"],
                    )
                layers.append(build_layer)
                prev_op = layer_params["operation"]
                if is_negative(next_in_shape):
                    raise ShapeValueError("Shape must not be negative")

        return nn.Sequential(*layers)

    def build_from_trial(
        self, trial, search_space: dict
    ) -> tuple[torch.nn.Module, QuantizationScheme]:
        sampler = Sampler(trial)
        sample = sampler.construct_sample(search_space)
        quant_scheme = sampler.get_quantization_scheme(search_space)
        return (
            nn.Sequential(
                *self.construct_model(
                    sample, search_space["input"], search_space["output"]
                )
            ),
            quant_scheme,
        )


class PicoModelBuilder(DefaultModelBuilder):
    def get_supported_quantization_schemes(self) -> dict[str, Any]:
        return {
            PTQFullyQuantizedInt8Scheme.name(): PTQFullyQuantizedInt8Builder,
            FullPrecisionScheme.name(): FullPrecisionBuilder,
        }

