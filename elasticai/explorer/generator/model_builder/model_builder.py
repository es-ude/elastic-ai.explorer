from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
from elasticai.creator import nn as creator_nn
from elasticai.creator.nn import fixed_point
import torch
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace


class QuantizationSchemes(str, Enum):
    FULL_PRECISION_FLOAT32 = "full_precision_float32"
    INT8_UNIFORM = "int8_uniform"
    FIXED_POINT_INT8 = "fixed_point_int8"


class ModelBuilder(ABC):
    @abstractmethod
    def build_from_trial(self, trial, searchspace: SearchSpace) -> Any:
        pass

    def get_supported_layers(self) -> set[type] | None:
        """Override if necessary. "None" means no constraints."""
        return None

    def get_supported_quantization_schemes(self) -> set[QuantizationSchemes] | None:
        """Override if necessary. "None" means no constraints."""
        return None

    def _validate_model(
        self, model: torch.nn.Module, quantization_scheme: QuantizationSchemes
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

    def build_from_trial(self, trial, searchspace: SearchSpace) -> torch.nn.Module:

        layers = []

        # TODO create a fitting creator representation of the trials params

        return creator_nn.Sequential(*layers)

    def get_supported_layers(self) -> set[type] | None:
        return {
            fixed_point.Linear,
            fixed_point.Conv1d,
        }

    def get_supported_quantization_schemes(self) -> set[QuantizationSchemes] | None:

        return {QuantizationSchemes.FIXED_POINT_INT8}
