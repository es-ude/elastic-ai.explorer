from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy


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


@dataclass(frozen=True)
class CreatorFixedPointScheme(QuantizationScheme):
    dtype: str = "int8"
    total_bits: int = 8
    frac_bits: int = 2
    signed: bool = True

    @staticmethod
    def name() -> str:
        return f"creator_fixed_point"

