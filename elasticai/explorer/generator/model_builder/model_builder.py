from abc import ABC, abstractmethod
import logging
import math
from typing import Any
from elasticai.creator import nn as creator_nn
from elasticai.creator.nn import fixed_point
import torch
from elasticai.explorer.generator.reflection import Reflective
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace
from elasticai.explorer.hw_nas.search_space.layer_builder import (
    LayerBuilder,
)
from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
    QuantizationScheme,
)
from elasticai.explorer.hw_nas.search_space.registry import (
    activation_mapping,
)


class ModelBuilder(ABC, Reflective):
    @abstractmethod
    def build_from_trial(self, trial, searchspace: SearchSpace) -> Any:
        pass


class TorchModelBuilder(ModelBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(
            "explorer.generator.model_builder.TorchModelBuilder"
        )

    def build_from_trial(self, trial, searchspace: SearchSpace) -> torch.nn.Module:
        return searchspace.create_native_torch_model_sample(trial)


class CreatorLinear(LayerBuilder):
    def build(self, num_layers: int, is_last_block: bool):
        for i in range(num_layers):
            if isinstance(self.input_shape, int):
                self.input_shape = self.input_shape
            else:
                self.input_shape = math.prod(self.input_shape)
            if isinstance(self.output_shape, int):
                self.output_shape = self.output_shape
            else:
                self.output_shape = math.prod(self.output_shape)

            self.layers.append(
                fixed_point.Linear(
                    in_features=self.input_shape,
                    out_features=self.output_shape,
                    total_bits=self.quantization_scheme.total_bits,
                    frac_bits=self.quantization_scheme.frac_bits,
                )
            )
            # self.add_activation(fixed_point.ReLU(total_bits=self.quantization_scheme.total_bits))

        return self.get_layers()


class CreatorModelBuilder(ModelBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.quantization_scheme = FixedPointInt8Scheme()
        self.logger = logging.getLogger(
            "explorer.generator.model_builder.CreatorModelBuilder"
        )
        activation_mapping["Relu"] = fixed_point.ReLU(
            total_bits=self.quantization_scheme.total_bits
        )

        # overwrite the Registry for the correct types

    def build_from_trial(self, trial, searchspace: SearchSpace) -> torch.nn.Module:

        try:
            if isinstance(searchspace.next_input_shape, int):
                flat_input = searchspace.next_input_shape
            else:
                flat_input = math.prod(searchspace.next_input_shape)  # type:ignore

        except Exception as e:
            self.logger.exception(
                f"The given searchspace.input_shape {searchspace.next_input_shape} is not formatted correctly!"
            )
            raise e
        # TODO Add these to the layer builder
        sequential = creator_nn.Sequential(
            fixed_point.Linear(
                in_features=flat_input,
                out_features=searchspace.output_shape,
                total_bits=self.quantization_scheme.total_bits,
                frac_bits=self.quantization_scheme.frac_bits,
            ),
            fixed_point.ReLU(total_bits=self.quantization_scheme.total_bits),
        )
        return sequential

    def get_supported_layers(self) -> tuple[type] | None:
        return (fixed_point.Linear,)

    def get_supported_activations(self) -> tuple[type] | None:
        return (fixed_point.ReLU,)

    def get_supported_quantization_schemes(
        self,
    ) -> tuple[type[QuantizationScheme]] | None:

        return (type(self.quantization_scheme),)
