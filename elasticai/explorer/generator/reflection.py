from abc import ABC, abstractmethod
from typing import Any
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
    def get_supported_quantization(
        self,
    ) -> dict[str, Any]:
        """
        Return dictionary with the key of the quantization parameter (dtype, total_bits)
        and the corresponding allowed values as set or boolean lambda function (e.g. int8, float32). Unspecified parameters are ignored.
        """
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

    def _validate_model(self, model: nn.Module):
        """Override if necessary"""
        supported_layers = self.get_supported_layers()
        supported_activations = self.get_supported_activations()

        for module in model.modules():
            if module is model:
                continue

            # skip any non-leaf modules like Sequentials
            if any(True for _ in module.children()):
                continue

            module_type = type(module)
            in_supported_layers = module_type in supported_layers
            in_supported_activations = module_type in supported_activations
            if not in_supported_layers and not in_supported_activations:
                raise NotImplementedError(
                    f"{type(module).__name__} is not supported by {self.__class__.__name__} "
                )

    def _validate_quantization(self, quantization_scheme: QuantizationScheme):
        supported_quantization_parameters = self.get_supported_quantization()
        if supported_quantization_parameters is not None:
            for field, allowed_values in supported_quantization_parameters.items():
                value = getattr(quantization_scheme, field, None)
                if value is None or allowed_values is None:
                    continue

                if callable(allowed_values):
                    if not allowed_values(value):
                        raise NotImplementedError(
                            f"{field}={value} is not supported by {self.__class__.__name__}. "
                        )
                elif isinstance(allowed_values, set):
                    if value not in allowed_values:
                        raise NotImplementedError(
                            f"{field}={value} is not supported by {self.__class__.__name__}. "
                            f"Allowed: {allowed_values}"
                        )

    def validate_model(self, model: nn.Module, quantization_scheme: QuantizationScheme):
        """Override if necessary"""
        self._validate_model(model)
        self._validate_quantization(quantization_scheme)
