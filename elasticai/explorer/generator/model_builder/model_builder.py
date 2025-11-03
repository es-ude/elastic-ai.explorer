from abc import ABC, abstractmethod
import logging
import math
from typing import Any
from elasticai.creator import nn as creator_nn
from elasticai.creator.nn import fixed_point
import torch
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace
from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
    QuantizationScheme,
)

logger = logging.getLogger("explorer.generator.model_builder")


class ModelBuilder(ABC):
    @abstractmethod
    def build_from_trial(self, trial, searchspace: SearchSpace) -> Any:
        pass

    def get_supported_layers(self) -> set[type] | None:
        """Override if necessary. "None" means no constraints."""
        return None

    def get_supported_quantization_schemes(self) -> set[QuantizationScheme] | None:
        """Override if necessary. "None" means no constraints."""
        return None

    def _validate_model(
        self, model: torch.nn.Module, quantization_scheme: QuantizationScheme
    ):
        """Override if necessary"""
        supported_layers = self.get_supported_layers()
        supported_quantization_schemes = self.get_supported_quantization_schemes()
        if supported_layers is not None:
            for layer in model.modules():
                if layer is model:
                    continue
                if type(layer) not in supported_layers:
                    raise NotImplementedError(
                        f"Layer {type(layer).__name__} wird von {self.__class__.__name__} nicht unterstützt"
                    )

        if supported_quantization_schemes is not None:
            if quantization_scheme not in supported_quantization_schemes:
                raise NotImplementedError(
                    f"Layer {quantization_scheme} wird von {self.__class__.__name__} nicht unterstützt"
                )


class TorchModelBuilder(ModelBuilder):
    def __init__(self) -> None:
        super().__init__()

    def build_from_trial(self, trial, searchspace: SearchSpace) -> torch.nn.Module:
        return searchspace.create_native_torch_model_sample(trial)


class CreatorModelBuilder(ModelBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.quantization_scheme = FixedPointInt8Scheme()

    def build_from_trial(self, trial, searchspace: SearchSpace) -> torch.nn.Module:

        try:
            if isinstance(searchspace.input_shape, int):
                flat_input = searchspace.input_shape
            else:
                flat_input = math.prod(searchspace.input_shape)  # type:ignore

        except Exception as e:
            logger.exception(
                f"The given searchspace.input_shape {searchspace.input_shape} is not formatted correctly!"
            )
            raise e

        layers = [
            fixed_point.Linear(
                in_features=flat_input,
                out_features=searchspace.output_shape,
                total_bits=self.quantization_scheme.total_bits,
                frac_bits=self.quantization_scheme.frac_bits,
            )
        ]
        return creator_nn.Sequential(*layers)

    def get_supported_layers(self) -> set[type] | None:
        return {
            fixed_point.Linear,
        }

    def get_supported_quantization_schemes(self) -> set[QuantizationScheme] | None:

        return {self.quantization_scheme}
