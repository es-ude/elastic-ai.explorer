from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QuantizationScheme:
    dtype: str = "float32"
    total_bits: int | None = None
    frac_bits: int | None = None
    signed: bool | None = None
    training_type: Literal["PTQ"] | Literal["QAT"] | None = None
