from abc import ABC, abstractmethod
from dataclasses import dataclass

# TODO unify to only one quant scheme with all necessary parameters

class QuantizationScheme(ABC):
    dtype: str

    @staticmethod
    @abstractmethod
    def name() -> str: ...


@dataclass(frozen=True)
class PTQFullyQuantizedInt8Scheme(QuantizationScheme):
    dtype: str = "int8"

    @staticmethod
    def name() -> str:
        return "ptq_fully_quantized_int8"


@dataclass(frozen=True)
class FullPrecisionScheme(QuantizationScheme):
    dtype: str = "float32"

    @staticmethod
    def name() -> str:
        return f"full_precision"
