from abc import ABC, abstractmethod
from elasticai.explorer.hw_nas.search_space.layer_builder import parse_search_param
from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
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
    def __init__(self, trial, search_params: dict) -> None:
        self.trial = trial
        self.search_params = search_params

    @abstractmethod
    def build(self) -> QuantizationScheme:
        pass


@register_quantization_scheme("fixed_point_int8")
class FixedPointInt8Builder(QuantizationBuilder):
    base_type = FixedPointInt8Scheme

    def build(
        self,
    ) -> QuantizationScheme:
        total_bits = parse_search_param(
            self.trial,
            f"total_bits",
            self.search_params,
            "total_bits",
        )

        frac_bits = parse_search_param(
            self.trial,
            f"frac_bits",
            self.search_params,
            "frac_bits",
        )
        signed = bool(
            parse_search_param(
                self.trial,
                f"signed_quant",
                self.search_params,
                "signed",
            )
        )

        return FixedPointInt8Scheme(
            total_bits=total_bits, frac_bits=frac_bits, signed=signed
        )


@register_quantization_scheme("full_precision")
class FullPrecisionBuilder(QuantizationBuilder):
    base_type = FullPrecisionScheme

    def build(self) -> QuantizationScheme:
        return FullPrecisionScheme()
