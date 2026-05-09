import numpy as np
import pytest

from elasticai.explorer.preprocessing.pipeline import apply_preprocessing_pipeline
from elasticai.explorer.preprocessing.types import (
    DownsamplingSample,
    FilteringSample,
    NormalizationSample,
    PreprocessingSample,
    WindowingSample,
)


def test_returns_signal_unchanged_when_pipeline_order_is_empty():
    signal = np.array([1, 2, 3])
    preprocessing = PreprocessingSample(
        sample_rate_hz=1000.0,
        order=(),
    )
    result = apply_preprocessing_pipeline(
        signal=signal,
        preprocessing=preprocessing,
    )

    np.testing.assert_array_equal(result, signal)


def test_rejects_unknown_preprocessing_steps():
    signal = np.array([1, 2, 3])
    preprocessing = PreprocessingSample(
        sample_rate_hz=1000.0,
        order=("yeet",),
    )

    with pytest.raises(ValueError, match="Unknown preprocessing step:"):
        _ = apply_preprocessing_pipeline(
            signal=signal,
            preprocessing=preprocessing,
        )


def test_applies_downsampling_factor_to_1d_signal():
    signal = np.array([0, 1, 2, 3, 4, 5])
    preprocessing = PreprocessingSample(
        sample_rate_hz=1000.0,
        order=("downsampling",),
        downsampling=DownsamplingSample(factor=2),
    )

    result = apply_preprocessing_pipeline(
        signal=signal,
        preprocessing=preprocessing,
    )

    np.testing.assert_array_equal(result, np.array([0, 2, 4]))


def test_applies_downsampling_factor_to_last_axis_of_2d_signal():
    signal = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [10, 11, 12, 13, 14, 15],
        ]
    )
    preprocessing = PreprocessingSample(
        sample_rate_hz=1000.0,
        order=("downsampling",),
        downsampling=DownsamplingSample(factor=2),
    )

    result = apply_preprocessing_pipeline(
        signal=signal,
        preprocessing=preprocessing,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [0, 2, 4],
                [10, 12, 14],
            ]
        ),
    )


def test_downsampling_before_windowing_updates_sample_rate_for_window_size():
    signal = np.arange(2000)
    preprocessing = PreprocessingSample(
        sample_rate_hz=1000.0,
        order=("downsampling", "windowing"),
        downsampling=DownsamplingSample(factor=2),
        windowing=WindowingSample(window_ms=1000),
    )

    result = apply_preprocessing_pipeline(
        signal=signal,
        preprocessing=preprocessing,
        timestamps_ms=np.array([0]),
    )

    assert result.shape == (1, 1, 500)
    assert result[0, 0, 0] == 0
    assert result[0, 0, -1] == 998


def test_rejects_windowing_without_timestamps():
    signal = np.arange(1000)
    preprocessing = PreprocessingSample(
        sample_rate_hz=1000.0,
        order=("windowing",),
        windowing=WindowingSample(window_ms=1000),
    )

    with pytest.raises(ValueError, match="timestamps_ms is required for windowing"):
        _ = apply_preprocessing_pipeline(
            signal=signal,
            preprocessing=preprocessing,
        )


def test_applies_zscore_normalization():
    signal = np.array([1.0, 2.0, 3.0])
    preprocessing = PreprocessingSample(
        sample_rate_hz=1000.0,
        order=("normalization",),
        normalization=NormalizationSample(method="zscore"),
    )

    result = apply_preprocessing_pipeline(signal, preprocessing)
    expected = (signal - np.mean(signal)) / np.std(signal)

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_applies_filtering_and_preserves_shape():
    signal = np.sin(np.linspace(0, 10, 1000))
    preprocessing = PreprocessingSample(
        sample_rate_hz=1000.0,
        order=("filtering",),
        filtering=FilteringSample(low_cut_hz=1.0, high_cut_hz=100.0),
    )

    result = apply_preprocessing_pipeline(signal, preprocessing)

    assert result.shape == signal.shape
    assert np.all(np.isfinite(result))
