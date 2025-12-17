import torch
from torch import nn
from elasticai.explorer.hw_nas.search_space.layer_builder import (
    LayerBuilder,
)
from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme


class Reflective:
    def get_supported_layers(self) -> list[type]:
        """Override if necessary"""
        supported_layers = []
        for layer_name, layer_builder in self.get_layer_mappings().items():
            base_type = layer_builder.base_type
            supported_layers.append(base_type)
        return supported_layers

    def get_supported_activations(self) -> list[type]:
        """Override if necessary."""
        supported_activations = []
        for name, activation in self.get_activation_mappings().items():
            supported_activations.append(type(activation))
        return supported_activations

    def get_supported_quantization_schemes(
        self,
    ) -> list[type[QuantizationScheme]]:
        """Override if necessary."""
        return []

    def get_layer_mappings(self) -> dict[str, type[LayerBuilder]]:
        """Override to return layer name to builder mappings."""
        return {}

    def get_activation_mappings(self) -> dict[str, nn.Module]:
        """Override to return activation name to module mappings."""
        return {}

    def get_adapter_mappings(self) -> dict[tuple[str | None, str | None], None | type]:
        """Override to return adapter key to adapter class mappings."""
        return {}

    def _validate_model(
        self, model: torch.nn.Module, quantization_scheme: QuantizationScheme
    ):
        """Override if necessary"""
        sl = self.get_supported_layers()
        sa = self.get_supported_activations()
        sqs = self.get_supported_quantization_schemes()
        for module in model.modules():
            if module is model:
                continue
            module_type = type(module)
            in_supported_layers = module_type in sl
            in_supported_activations = module_type in sa
            if not in_supported_layers and not in_supported_activations:
                raise NotImplementedError(
                    f"{type(module).__name__} is not supported by {self.__class__.__name__} "
                )

        if sqs is not None:
            if type(quantization_scheme) not in sqs:
                raise NotImplementedError(
                    f"{quantization_scheme}  is not supported by {self.__class__.__name__}"
                )
