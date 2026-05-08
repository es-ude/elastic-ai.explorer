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
    low_cut_hz: int | float
    high_cut_hz: int | float


@dataclass(frozen=True)
class DownsamplingSample:
    factor: int


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
