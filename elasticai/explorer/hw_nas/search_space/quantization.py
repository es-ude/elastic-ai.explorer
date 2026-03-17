from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type
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
        return "PTQFullyQuantizedInt8"


@dataclass(frozen=True)
class FullPrecisionScheme(QuantizationScheme):
    dtype: str = "float32"

    @staticmethod
    def name() -> str:
        return "FullPrecision"

    def is_compatible_with_layer(self, layer_type: Type) -> bool:
        return True


def tflite_quantize(
    array, scaling: float = 0.012728233821690083, zeropoint: float = 95
):
    return (array / scaling) + zeropoint


def tflite_dequantize(array, scaling: float = 0.00390625, zeropoint: float = 128):
    return scaling * (array.astype(numpy.float32) - zeropoint)
