import numpy as np
from elasticai.preprocessor.data_augmentation import augmentation_downsampling
from elasticai.preprocessor.filter import Filtering, SettingsFilter
from elasticai.preprocessor.normalization import DataNormalization

from elasticai.explorer.preprocessing.types import (
    FILTERBAND_CUTOFFS,
    DownsamplingSample,
    FilteringSample,
    NormalizationSample,
    PreprocessingSample,
    WindowingSample,
)
from elasticai.explorer.preprocessing.windowing import cut_windows_by_timestamp


def apply_preprocessing_step(
    step: str,
    preprocessing: PreprocessingSample,
    signal: np.ndarray,
    sample_rate_hz: float,
    timestamps_ms: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    match step:
        case "windowing":
            result = apply_windowing(
                signal=signal,
                timestamps_ms=timestamps_ms,
                windowing=preprocessing.windowing,
                sample_rate_hz=sample_rate_hz,
            )
        case "filtering":
            result = apply_filtering(
                signal=signal,
                filtering=preprocessing.filtering,
                sample_rate_hz=sample_rate_hz,
            )
        case "downsampling":
            result, sample_rate_hz = apply_downsampling(
                signal=signal,
                downsampling=preprocessing.downsampling,
                sample_rate_hz=sample_rate_hz,
            )
        case "normalization":
            result = apply_normalization(
                signal=signal,
                normalization=preprocessing.normalization,
            )
        case _:
            raise ValueError("Unknown preprocessing step.")

    return result, sample_rate_hz


def apply_windowing(
    signal: np.ndarray,
    timestamps_ms: np.ndarray | None,
    windowing: WindowingSample,
    sample_rate_hz: float,
) -> np.ndarray:
    if timestamps_ms is None:
        raise ValueError("timestamps_ms is required for windowing")

    return cut_windows_by_timestamp(
        signal=signal,
        timestamps_ms=timestamps_ms,
        sample_rate_hz=sample_rate_hz,
        window_ms=windowing.window_ms,
    )


def apply_downsampling(
    signal: np.ndarray,
    downsampling: DownsamplingSample,
    sample_rate_hz: float,
) -> tuple[np.ndarray, float]:
    result, _ = augmentation_downsampling(
        data=signal,
        label=np.zeros(shape=signal.shape[0]),
        n_downsampling=downsampling.factor,
        drop_samples=downsampling.drop_samples,
    )
    return result, sample_rate_hz / downsampling.factor


def apply_filtering(
    signal: np.ndarray,
    filtering: FilteringSample,
    sample_rate_hz: float,
) -> np.ndarray:
    dsp_filter = Filtering(
        setting=SettingsFilter(
            gain=filtering.gain,
            fs=sample_rate_hz,
            n_order=filtering.order,
            type=filtering.filter_type,
            f_type=filtering.filter_design,
            b_type=filtering.band_type,
            f_filt=filter_frequencies(filtering),
        )
    )
    return dsp_filter.filter(signal)


def filter_frequencies(filtering: FilteringSample) -> list[int | float]:
    return [getattr(filtering, key) for key in FILTERBAND_CUTOFFS[filtering.band_type]]


def apply_normalization(
    signal: np.ndarray,
    normalization: NormalizationSample,
) -> np.ndarray:
    normalizer = DataNormalization(method=normalization.method)
    return normalizer.normalize(signal)
