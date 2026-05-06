from dataclasses import dataclass

import optuna

from elasticai.explorer.hw_nas.search_space.sample_blocks import parse_search_param


@dataclass(frozen=True)
class WindowingSample:
    window_ms: int
    sample_rate_hz: float


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
    windowing: WindowingSample
    filtering: FilteringSample | None = None
    downsampling: DownsamplingSample | None = None
    normalization: NormalizationSample | None = None


def parse_filtering_params(
    trial: optuna.Trial,
    filtering_params: dict | None,
) -> FilteringSample | None:
    if filtering_params is None:
        return None

    enabled = parse_search_param(
        trial=trial,
        name="preprocessing/filtering/enabled",
        params=filtering_params,
        key="enabled",
        default_value=True,
    )

    if not enabled:
        return None

    filtering_sample = FilteringSample(
        low_cut_hz=parse_search_param(
            trial=trial,
            name="preprocessing/filtering/low_cut_hz",
            params=filtering_params,
            key="low_cut_hz",
        ),
        high_cut_hz=parse_search_param(
            trial=trial,
            name="preprocessing/filtering/high_cut_hz",
            params=filtering_params,
            key="high_cut_hz",
        ),
    )

    if filtering_sample.low_cut_hz >= filtering_sample.high_cut_hz:
        raise ValueError("low_cut_hz must be below high_cut_hz")

    return filtering_sample


def parse_downsampling_params(
    trial: optuna.Trial,
    downsampling_params: dict | None,
) -> DownsamplingSample | None:
    downsampling_sample = None
    if downsampling_params is not None:
        downsampling_sample = DownsamplingSample(
            factor=parse_search_param(
                trial=trial,
                name="preprocessing/downsampling/factor",
                params=downsampling_params,
                key="factor",
            )
        )

        if downsampling_sample.factor < 1:
            raise ValueError("downsampling factor must be at least 1")

    return downsampling_sample


def parse_normalization_params(
    trial: optuna.Trial,
    normalization_params: dict | None,
) -> NormalizationSample | None:
    if normalization_params is None:
        return None

    method = parse_search_param(
        trial=trial,
        name="preprocessing/normalization/method",
        params=normalization_params,
        key="method",
        default_value="none",
    )

    if method == "none":
        return None

    return NormalizationSample(method=method)


def sample_preprocessing(
    trial: optuna.Trial,
    params: dict,
) -> PreprocessingSample:
    windowing_params = params["windowing"]
    filtering_params = params.get("filtering")
    downsampling_params = params.get("downsampling")
    normalization_params = params.get("normalization")

    windowing_sample = WindowingSample(
        window_ms=parse_search_param(
            trial=trial,
            name="preprocessing/windowing/window_ms",
            params=windowing_params,
            key="window_ms",
        ),
        sample_rate_hz=windowing_params["sample_rate_hz"],
    )

    filtering_sample = parse_filtering_params(
        trial=trial,
        filtering_params=filtering_params,
    )

    downsampling_sample = parse_downsampling_params(
        trial=trial,
        downsampling_params=downsampling_params,
    )

    normalization_sample = parse_normalization_params(
        trial=trial,
        normalization_params=normalization_params,
    )

    return PreprocessingSample(
        windowing=windowing_sample,
        filtering=filtering_sample,
        downsampling=downsampling_sample,
        normalization=normalization_sample,
    )
