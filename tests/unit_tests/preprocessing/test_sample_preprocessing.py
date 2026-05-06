import pytest
from optuna.trial import FixedTrial

from elasticai.explorer.preprocessing.sample import (
    PreprocessingSample,
    WindowingSample,
    sample_preprocessing,
)


def test_uses_constant_window_ms_from_config():
    trial = FixedTrial({})
    params = {
        "windowing": {
            "sample_rate_hz": 1000.0,
            "window_ms": 1000,
        }
    }

    sample = sample_preprocessing(
        trial=trial,
        params=params,
    )
    assert sample == PreprocessingSample(
        windowing=WindowingSample(
            sample_rate_hz=1000.0,
            window_ms=1000,
        )
    )


def test_samples_window_ms_from_categorical_config():
    trial = FixedTrial({"preprocessing/windowing/window_ms": 1000})
    params = {
        "windowing": {
            "sample_rate_hz": 1000.0,
            "window_ms": [1000, 2000],
        }
    }

    sample = sample_preprocessing(
        trial=trial,
        params=params,
    )
    assert sample == PreprocessingSample(
        windowing=WindowingSample(
            sample_rate_hz=1000.0,
            window_ms=1000,
        )
    )


def test_samples_window_ms_from_int_range_config():
    trial = FixedTrial({"preprocessing/windowing/window_ms": 1000})
    params = {
        "windowing": {
            "sample_rate_hz": 1000.0,
            "window_ms": {
                "start": 500,
                "end": 1500,
                "step": 250,
            },
        }
    }

    sample = sample_preprocessing(
        trial=trial,
        params=params,
    )
    assert sample == PreprocessingSample(
        windowing=WindowingSample(
            sample_rate_hz=1000.0,
            window_ms=1000,
        )
    )


def test_samples_filter_params_when_filter_config_exists():
    trial = FixedTrial(
        {
            "preprocessing/filtering/low_cut_hz": 1.0,
            "preprocessing/filtering/high_cut_hz": 100.0,
        }
    )
    params = {
        "windowing": {
            "sample_rate_hz": 1000.0,
            "window_ms": 1000,
        },
        "filtering": {
            "low_cut_hz": [0.5, 1.0],
            "high_cut_hz": [100.0, 200.0],
        },
    }

    sample = sample_preprocessing(
        trial=trial,
        params=params,
    )
    assert sample.filtering.low_cut_hz == 1.0
    assert sample.filtering.high_cut_hz == 100.0


def test_does_not_sample_filtering_when_filtering_config_is_missing():
    trial = FixedTrial({})
    params = {
        "windowing": {
            "sample_rate_hz": 1000.0,
            "window_ms": 1000,
        }
    }
    sample = sample_preprocessing(
        trial=trial,
        params=params,
    )
    assert sample.filtering is None


def test_rejects_filtering_when_low_cut_is_not_below_high_cut():
    trial = FixedTrial({})
    params = {
        "windowing": {
            "sample_rate_hz": 1000.0,
            "window_ms": 1000,
        },
        "filtering": {
            "low_cut_hz": 200.0,
            "high_cut_hz": 100.0,
        },
    }
    with pytest.raises(ValueError, match="low_cut_hz must be below high_cut_hz"):
        _ = sample_preprocessing(
            trial=trial,
            params=params,
        )


def test_does_not_sample_filter_values_when_filtering_is_disabled():
    trial = FixedTrial({"preprocessing/filtering/enabled": False})

    params = {
        "windowing": {
            "sample_rate_hz": 1000.0,
            "window_ms": 1000,
        },
        "filtering": {
            "enabled": [True, False],
            "low_cut_hz": [0.5, 1.0],
            "high_cut_hz": [100.0, 200.0],
        },
    }
    sample = sample_preprocessing(
        trial=trial,
        params=params,
    )
    assert sample.filtering is None


def test_rejects_downsampling_factor_below_one():
    trial = FixedTrial({})
    params = {
        "windowing": {
            "sample_rate_hz": 1000.0,
            "window_ms": 1000,
        },
        "downsampling": {
            "factor": 0,
        },
    }
    with pytest.raises(ValueError, match="downsampling factor must be at least 1"):
        _ = sample_preprocessing(
            trial=trial,
            params=params,
        )


def test_samples_downsampling_factor_when_config_exists():
    trial = FixedTrial(
        {
            "preprocessing/downsampling/factor": 5,
        }
    )
    params = {
        "windowing": {
            "sample_rate_hz": 1000.0,
            "window_ms": 1000,
        },
        "downsampling": {
            "factor": [1, 2, 3, 4, 5, 10],
        },
    }
    sample = sample_preprocessing(trial=trial, params=params)
    assert sample.downsampling.factor == 5


def test_does_not_sample_downsampling_when_downsampling_config_is_missing():
    trial = FixedTrial({})
    params = {
        "windowing": {
            "sample_rate_hz": 1000.0,
            "window_ms": 1000,
        }
    }

    sample = sample_preprocessing(
        trial=trial,
        params=params,
    )
    assert sample.downsampling is None


def test_does_not_sample_normalization_when_method_is_none():
    trial = FixedTrial({"preprocessing/normalization/method": "none"})
    params = {
        "windowing": {
            "sample_rate_hz": 1000.0,
            "window_ms": 1000,
        },
        "normalization": {
            "method": ["none", "zscore", "minmax"],
        },
    }
    sample = sample_preprocessing(
        trial=trial,
        params=params,
    )
    assert sample.normalization is None


def test_samples_normalization_method_when_normalization_config_exists():
    trial = FixedTrial({"preprocessing/normalization/method": "zscore"})
    params = {
        "windowing": {
            "sample_rate_hz": 1000.0,
            "window_ms": 1000,
        },
        "normalization": {
            "method": ["none", "zscore", "minmax"],
        },
    }
    sample = sample_preprocessing(
        trial=trial,
        params=params,
    )
    assert sample.normalization.method == "zscore"
