from abc import ABC, abstractmethod
import logging
import math

from torch import nn
from typing import Any
from elasticai.creator import nn as creator_nn
from elasticai.creator.nn import fixed_point
import torch
from elasticai.explorer.generator.reflection import Reflective
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace
from elasticai.explorer.hw_nas.search_space.layer_builder import (
    LayerBuilder,
    parse_search_param,
    register_layer,
)
from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
    QuantizationScheme,
)
from elasticai.explorer.hw_nas.search_space.registry import (
    ADAPTER_REGISTRY,
    activation_mapping,
)


class ModelBuilder(ABC, Reflective):
    @abstractmethod
    def build_from_trial(self, trial, searchspace: SearchSpace) -> Any:
        pass


class DefaultModelBuilder(ModelBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(
            "explorer.generator.model_builder.TorchModelBuilder"
        )

    def build_from_trial(self, trial, searchspace: SearchSpace) -> torch.nn.Module:
        return nn.Sequential(*searchspace.create_model_layers(trial))


class CreatorLinearBuilder(LayerBuilder):
    def build(self, num_layers: int, is_last_block: bool) -> Any:
        for i in range(num_layers):
            if isinstance(self.input_shape, int):
                self.input_shape = self.input_shape
            else:
                self.input_shape = math.prod(self.input_shape)
            if isinstance(self.output_shape, int):
                self.output_shape = self.output_shape
            else:
                self.output_shape = math.prod(self.output_shape)
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
                self.layers.append(
                    fixed_point.Linear(
                        self.input_shape,
                        self.output_shape,
                        total_bits=self.quantization_scheme.total_bits,
                        frac_bits=self.quantization_scheme.frac_bits,
                    )
                )
            else:
                self.layers.append(
                    fixed_point.Linear(
                        self.input_shape,
                        width,
                        total_bits=self.quantization_scheme.total_bits,
                        frac_bits=self.quantization_scheme.frac_bits,
                    )
                )
                self.input_shape = width
            self.add_activation(activation)

        return self.get_layers()


class CreatorModelBuilder(ModelBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.quantization_scheme = FixedPointInt8Scheme()
        self.logger = logging.getLogger(
            "explorer.generator.model_builder.CreatorModelBuilder"
        )

        # TODO streamline this process via Reflection api
        activation_mapping["relu"] = fixed_point.ReLU(
            total_bits=self.quantization_scheme.total_bits
        )
        register_layer("linear")(CreatorLinearBuilder)
        ADAPTER_REGISTRY[(None, "linear")] = None

    def build_from_trial(self, trial, searchspace: SearchSpace) -> torch.nn.Module:
        return creator_nn.Sequential(*searchspace.create_model_layers(trial=trial))

    def get_supported_layers(self) -> tuple[type] | None:
        return (fixed_point.Linear,)

    def get_supported_activations(self) -> tuple[type] | None:
        return (fixed_point.ReLU,)

    def get_supported_quantization_schemes(
        self,
    ) -> tuple[type[QuantizationScheme]] | None:

        return (type(self.quantization_scheme),)
