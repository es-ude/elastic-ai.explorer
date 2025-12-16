from elasticai.explorer.hw_nas.search_space.layer_builder import parse_search_param
from elasticai.explorer.hw_nas.search_space.quantization import FixedPointInt8Scheme

QUANT_REGISTRY = {}


def register_quantization_scheme(name: str):
    """Decorator to register new layer types."""

    def wrapper(cls):
        QUANT_REGISTRY[name] = cls
        return cls

    return wrapper


@register_quantization_scheme("fixed_point_int8")
class FixedPointInt8Builder:
    def __init__(self, trial, block: dict, search_params: dict, block_id) -> None:
        self.trial = trial
        self.block = block
        self.search_params = search_params
        self.block_id = block_id

    def build(
        self,
    ):
        total_bits = parse_search_param(
            self.trial,
            f"total_bits_b{self.block_id}",
            self.search_params,
            "total_bits",
        )

        frac_bits = parse_search_param(
            self.trial,
            f"frac_bits_b{self.block_id}",
            self.search_params,
            "frac_bits",
        )
        signed = bool(
            parse_search_param(
                self.trial,
                f"signed_quant_b{self.block_id}",
                self.search_params,
                "signed",
            )
        )

        return FixedPointInt8Scheme(
            total_bits=total_bits, frac_bits=frac_bits, signed=signed
        )
