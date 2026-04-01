from abc import ABC, abstractmethod
from typing import Any
import torch
from torch import nn
from elasticai.explorer.hw_nas.search_space.layer_builder import (
    LayerBuilder,
)
from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme


class Reflective(ABC):
    @abstractmethod
    def get_activation_mappings(self) -> dict[str, nn.Module]:
        pass

    @abstractmethod
    def get_adapter_mappings(self) -> dict[tuple[str | None, str | None], None | type]:
        pass

    @abstractmethod
    def get_supported_quantization_schemes(
        self,
    ) -> dict[str, Any]:
        pass

    def get_layer_mappings(self) -> dict[str, type[LayerBuilder]]:
        """Override if necessary. Empty dict means all base layer builders are allowed."""
        return {}

    def get_supported_layers(self) -> list[type]:
        supported_layers = []
        for layer_name, layer_builder in self.get_layer_mappings().items():
            base_type = layer_builder.base_type
            supported_layers.append(base_type)
        return supported_layers

    def get_supported_activations(self) -> list[type]:
        supported_activations = []
        for name, activation in self.get_activation_mappings().items():
            supported_activations.append(type(activation))
        return supported_activations

    def validate_model(
        self, model: torch.nn.Module, quantization_scheme: QuantizationScheme
    ):
        """Override if necessary"""
        supported_layers = self.get_supported_layers()
        supported_activations = self.get_supported_activations()
        supported_quantization_schemes = self.get_supported_quantization_schemes()

        # modules gives back all modules recursively
        for module in model.modules():
            if module is model:
                continue

            # skip any container like Sequential
            if any(True for _ in module.children()):
                continue

            module_type = type(module)
            in_supported_layers = module_type in supported_layers
            in_supported_activations = module_type in supported_activations
            if not in_supported_layers and not in_supported_activations:
                raise NotImplementedError(
                    f"{type(module).__name__} is not supported by {self.__class__.__name__} "
                )

        if supported_quantization_schemes is not None:
            if type(quantization_scheme) not in supported_quantization_schemes:
                raise NotImplementedError(
                    f"{quantization_scheme.name()}  is not supported by {self.__class__.__name__}"
                )
