from dataclasses import dataclass

DEFAULT_PREPROCESSING_ORDER = (
    "filtering",
    "downsampling",
    "normalization",
    "windowing",
)


VALID_PREPROCESSING_STEPS = (
    "filtering",
    "downsampling",
    "normalization",
    "windowing",
)


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
