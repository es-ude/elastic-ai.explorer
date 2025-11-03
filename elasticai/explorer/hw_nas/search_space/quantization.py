from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

from numpy import dtype


class QuantizationScheme(ABC):

    @abstractmethod
    def name(self) -> str: ...

    def is_compatible_with_layer(self, layer_type: Type) -> bool:
        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self.name()}>"


@dataclass(frozen=True)
class FixedPointInt8Scheme(QuantizationScheme):
    total_bits: int = 8
    frac_bits: int = 2
    signed: bool = True

    def name(self) -> str:
        sign = "s" if self.signed else "u"
        return f"fixed_{sign}_total{self.total_bits}_frac{self.frac_bits}"


@dataclass(frozen=True)
class FullPrecisionScheme(QuantizationScheme):
    dtype: str = "float32"

    def name(self) -> str:
        return self.dtype

    def is_compatible_with_layer(self, layer_type: Type) -> bool:
        return True


@dataclass(frozen=True)
class Int8Uniform:
    dtype: str = "int8"

    def name(self) -> str:
        return self.dtype

    def is_compatible_with_layer(self, layer_type: Type) -> bool:
        return True
