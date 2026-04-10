from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationScheme:
    dtype: str = "int8"
    total_bits: int | None = None
    frac_bits: int | None = None
    signed: bool | None = None
