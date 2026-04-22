from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationScheme:
    dtype: str = "int8"

    # Affine 
    scale: float | None = None
    zero_point: int | None = None

    # Fixed Point
    total_bits: int | None = None
    frac_bits: int | None = None
    signed: bool | None = None
