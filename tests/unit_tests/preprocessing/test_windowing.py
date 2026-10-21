import numpy as np
import pytest

from elasticai.explorer.preprocessing.windowing import (
    _ms_to_sample_idx,
    cut_windows_by_timestamp,
)


@pytest.mark.parametrize(
    ("ms", "fs_hz", "expected"),
    [
        (0, 1000, 0),
        (1, 1000, 1),
        (10, 1000, 10),
        (1000, 1000, 1000),
        (1500, 1000, 1500),
        (1000, 16000, 16000),
        (10, 48000, 480),
        (0.5, 1000, 0),
        (1.5, 1000, 2),
        (0.75, 1000, 1),
    ],
)
def test_ms_to_sample_idx(ms, fs_hz, expected):
    assert _ms_to_sample_idx(ms=ms, fs_hz=fs_hz) == expected


def test_cuts_window_from_timestamp_start_ms_single_channel():
    signal = np.arange(10000)

    windows = cut_windows_by_timestamp(
        signal=signal,
        timestamps_ms=np.array([5000]),
        sample_rate_hz=1000.0,
        window_ms=1000,
    )

    assert windows.shape == (1, 1, 1000)
    assert windows[0, 0, 0] == 5000
    assert windows[0, 0, -1] == 5999


def test_cuts_window_from_timestamp_start_ms_multi_channel():
    signal = np.stack(
        [
            np.arange(10000),
            np.arange(10000) + 10000,
        ],
        axis=0,
    )

    windows = cut_windows_by_timestamp(
        signal=signal,
        timestamps_ms=np.array([5000]),
        sample_rate_hz=1000.0,
        window_ms=1000,
    )

    assert windows.shape == (1, 2, 1000)
    assert windows[0, 0, 0] == 5000
    assert windows[0, 0, -1] == 5999
    assert windows[0, 1, 0] == 15000
    assert windows[0, 1, -1] == 15999


def test_pads_window_after_signal_end_with_zero():
    signal = np.arange(10000)

    windows = cut_windows_by_timestamp(
        signal=signal,
        timestamps_ms=np.array([9500]),
        sample_rate_hz=1000.0,
        window_ms=1000,
    )

    assert windows.shape == (1, 1, 1000)
    assert windows[0, 0, 0] == 9500
    assert windows[0, 0, -1] == 0
    assert windows[0, 0, 499] == 9999
    assert windows[0, 0, 500] == 0


def test_discards_negative_timestamps():
    signal = np.arange(10000)

    with pytest.raises(ValueError, match="timestamps_ms must be positive."):
        _ = cut_windows_by_timestamp(
            signal=signal,
            timestamps_ms=np.array([-500]),
            sample_rate_hz=1000.0,
            window_ms=1000,
        )
