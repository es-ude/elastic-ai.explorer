from abc import ABC
from elasticai.explorer.hw_nas.search_space.quantization import (
    CreatorFixedPointScheme,
    PTQFullyQuantizedInt8Scheme,
    FullPrecisionScheme,
    QuantizationScheme,
)
from elasticai.explorer.hw_nas.search_space.registry import quantization_registry
from elasticai.explorer.hw_nas.search_space.sample_blocks import parse_search_param


def register_quantization_scheme(name: str):
    """Decorator to register new layer types."""

    def wrapper(cls):
        quantization_registry[name] = cls
        return cls

    return wrapper


class QuantizationBuilder(ABC):
    base_type: type[QuantizationScheme]

    def __init__(self, trial, search_params: dict) -> None:
        self.trial = trial
        self.search_params = search_params

    def build(self) -> QuantizationScheme:
        return self.base_type()


@register_quantization_scheme("ptq_fully_quantized_int8")
class PTQFullyQuantizedInt8Builder(QuantizationBuilder):
    base_type = PTQFullyQuantizedInt8Scheme


@register_quantization_scheme("full_precision")
class FullPrecisionBuilder(QuantizationBuilder):
    base_type = FullPrecisionScheme


@register_quantization_scheme("creator_fixed_point")
class CreatorFixedPointBuilder(QuantizationBuilder):
    base_type = CreatorFixedPointScheme

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

        return CreatorFixedPointScheme(
            total_bits=total_bits, frac_bits=frac_bits, signed=signed
        )
