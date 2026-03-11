from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type
import numpy
class QuantizationScheme(ABC):
    dtype: str
    total_bits: int
    frac_bits: int
    signed: bool

    @abstractmethod
    def name(self) -> str: ...


@dataclass(frozen=True)
class FixedPointInt8Scheme(QuantizationScheme):
    dtype: str = "int8"
    total_bits: int = 8
    frac_bits: int = 2
    signed: bool = True

    def name(self) -> str:
        sign = "s" if self.signed else "u"
        return f"fixed_{sign}_total{self.total_bits}_frac{self.frac_bits}"


@dataclass(frozen=True)
class FullPrecisionScheme(QuantizationScheme):
    dtype: str = "float32"
    total_bits: int = 32
    frac_bits: int = 23
    signed: bool = True

    def name(self) -> str:
        return self.dtype

    def is_compatible_with_layer(self, layer_type: Type) -> bool:
        return True


def tflite_quantize(array, scaling: float = 0.012728233821690083, zeropoint: float = 95): 
    return (array / scaling) + zeropoint   

def tflite_dequantize(array, scaling: float = 0.00390625, zeropoint: float = 128):
    return scaling * (array.astype(numpy.float32) - zeropoint)
    
