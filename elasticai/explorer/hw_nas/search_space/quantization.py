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


def tflite_quantize(
    array, scaling: float = 0.012728233821690083, zeropoint: float = 95
):
    return (array / scaling) + zeropoint


def tflite_dequantize(array, scaling: float = 0.00390625, zeropoint: float = 128):
    return scaling * (array.astype(numpy.float32) - zeropoint)
