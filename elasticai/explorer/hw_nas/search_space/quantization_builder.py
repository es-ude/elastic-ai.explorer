from abc import ABC, abstractmethod
from elasticai.explorer.hw_nas.search_space.quantization import (
    PTQFullyQuantizedInt8Scheme,
    FullPrecisionScheme,
    QuantizationScheme,
)

QUANT_REGISTRY: dict[str, type["QuantizationBuilder"]] = {}


def register_quantization_scheme(name: str):
    """Decorator to register new layer types."""

    def wrapper(cls):
        QUANT_REGISTRY[name] = cls
        return cls

    return wrapper


class QuantizationBuilder(ABC):
    base_type: type[QuantizationScheme]

    def __init__(self, trial, search_params: dict) -> None:
        self.trial = trial
        self.search_params = search_params

    def build(self) -> QuantizationScheme:
        return self.base_type()


@register_quantization_scheme("fully_quantized_int8")
class FullyQuantizedInt8Builder(QuantizationBuilder):
    base_type = PTQFullyQuantizedInt8Scheme


@register_quantization_scheme("full_precision")
class FullPrecisionBuilder(QuantizationBuilder):
    base_type = FullPrecisionScheme
