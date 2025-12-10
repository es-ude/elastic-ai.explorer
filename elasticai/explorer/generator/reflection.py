import torch

from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme


class Reflective:
    def get_supported_layers(self) -> tuple[type] | None:
        """Override if necessary. "None" means no constraints."""
        return None

    def get_supported_quantization_schemes(self) -> tuple[type[QuantizationScheme]] | None:
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
