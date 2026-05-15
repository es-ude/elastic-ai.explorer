from dataclasses import dataclass

DEFAULT_PREPROCESSING_ORDER = (
    "filtering",
    "downsampling",
    "normalization",
    "windowing",
)


VALID_PREPROCESSING_STEPS = frozenset(DEFAULT_PREPROCESSING_ORDER)


FILTERBAND_CUTOFFS = {
    "lowpass": ("high_cut_hz",),
    "highpass": ("low_cut_hz",),
    "bandpass": ("low_cut_hz", "high_cut_hz"),
    "bandstop": ("low_cut_hz", "high_cut_hz"),
}


@dataclass(frozen=True)
class WindowingSample:
    window_ms: int


@dataclass(frozen=True)
class FilteringSample:
    low_cut_hz: int | float | None = None
    high_cut_hz: int | float | None = None
    gain: int = 1
    order: int = 2
    filter_type: str = "iir"
    filter_design: str = "butter"
    band_type: str = "bandpass"

    def __post_init__(self) -> None:
        if self.band_type not in FILTERBAND_CUTOFFS:
            raise ValueError(f"Unsupported filter candidate: {self.band_type}")

        required = FILTERBAND_CUTOFFS[self.band_type]
        missing = [key for key in required if getattr(self, key) is None]

        if missing:
            raise ValueError(
                f"{','.join(missing)} is required for {self.band_type} filtering"
            )

        if "low_cut_hz" in required and "high_cut_hz" in required:
            if self.low_cut_hz >= self.high_cut_hz:
                raise ValueError("low_cut_hz must be below high_cut_hz")


@dataclass(frozen=True)
class DownsamplingSample:
    factor: int
    drop_samples: bool = True


@dataclass(frozen=True)
class NormalizationSample:
    method: str


@dataclass(frozen=True)
class PreprocessingSample:
    order: tuple[str, ...]
    sample_rate_hz: float
    windowing: WindowingSample | None = None
    filtering: FilteringSample | None = None
    downsampling: DownsamplingSample | None = None
    normalization: NormalizationSample | None = None
