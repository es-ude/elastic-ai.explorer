from abc import ABC, abstractmethod
import logging
import math
from typing import Any
from elasticai.creator import nn as creator_nn
from elasticai.creator.nn import fixed_point
import torch
from elasticai.explorer.generator.reflection import Reflective
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace
from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
    QuantizationScheme,
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


class CreatorModelBuilder(ModelBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.quantization_scheme = FixedPointInt8Scheme()
        self.logger = logging.getLogger(
            "explorer.generator.model_builder.CreatorModelBuilder"
        )

    def build_from_trial(self, trial, searchspace: SearchSpace) -> torch.nn.Module:

        try:
            if isinstance(searchspace.input_shape, int):
                flat_input = searchspace.input_shape
            else:
                flat_input = math.prod(searchspace.input_shape)  # type:ignore

        except Exception as e:
            self.logger.exception(
                f"The given searchspace.input_shape {searchspace.input_shape} is not formatted correctly!"
            )
            raise e

        searchspace.createLinear(
            trial, searchspace.blocks[0], 1, searchspace.blocks[0]["linear"]
        )
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

    def get_supported_quantization_schemes(
        self,
    ) -> tuple[type[QuantizationScheme]] | None:

        return (type(self.quantization_scheme),)
