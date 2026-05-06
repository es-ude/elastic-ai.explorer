import logging

import numpy as np

_logger = logging.getLogger("explorer.nas")


def _ms_to_sample_idx(
    ms: float | int,
    fs_hz: float | int,
) -> int:
    return int(round((ms / 1000.0) * fs_hz))


def _ensure_2d_channels_first(signal: np.ndarray) -> np.ndarray:
    if signal.ndim == 1:
        return signal[np.newaxis, :]
    elif signal.ndim == 2:
        return signal
    else:
        raise ValueError(
            "signal must have shape (n_samples, ) or (n_channels, n_samples)"
        )


def cut_windows_by_timestamp(
    signal: np.ndarray,
    timestamps_ms: np.ndarray,
    sample_rate_hz: float,
    window_ms: int,
) -> np.ndarray:
    signal = _ensure_2d_channels_first(signal)

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    if window_ms <= 0:
        raise ValueError("window_ms must be positive.")

    window_size = _ms_to_sample_idx(ms=window_ms, fs_hz=sample_rate_hz)

    if np.any(timestamps_ms < 0):
        raise ValueError("timestamps_ms must be positive.")

    event_idc = [_ms_to_sample_idx(ms=ms, fs_hz=sample_rate_hz) for ms in timestamps_ms]

    windows = []
    for event_idx in event_idc:
        window_start_idx = event_idx
        window_stop_idx = window_start_idx + window_size

        window = signal[:, window_start_idx:window_stop_idx]

        if window.shape[-1] < window_size:
            need = window_size - window.shape[-1]
            window = np.pad(window, ((0, 0), (0, need)), mode="constant")

        windows.append(window)

    return np.stack(windows, axis=0)
