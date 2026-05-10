import optuna

from elasticai.explorer.hw_nas.search_space.sample_blocks import parse_search_param
from elasticai.explorer.preprocessing.types import (
    DEFAULT_PREPROCESSING_ORDER,
    VALID_PREPROCESSING_STEPS,
    DownsamplingSample,
    FilteringSample,
    NormalizationSample,
    PreprocessingSample,
    WindowingSample,
)


def sample_preprocessing(
    trial: optuna.Trial,
    params: dict,
) -> PreprocessingSample:
    windowing_params = params.get("windowing")
    filtering_params = params.get("filtering")
    downsampling_params = params.get("downsampling")
    normalization_params = params.get("normalization")

    sample_rate_hz = parse_sample_rate_hz(params)

    pipeline_order = parse_preprocessing_order(
        trial=trial,
        params=params,
    )

    windowing_sample = (
        parse_windowing_params(
            trial=trial,
            windowing_params=windowing_params,
        )
        if "windowing" in pipeline_order
        else None
    )

    filtering_sample = (
        parse_filtering_params(
            trial=trial,
            filtering_params=filtering_params,
        )
        if "filtering" in pipeline_order
        else None
    )

    downsampling_sample = (
        parse_downsampling_params(
            trial=trial,
            downsampling_params=downsampling_params,
        )
        if "downsampling" in pipeline_order
        else None
    )

    normalization_sample = (
        parse_normalization_params(
            trial=trial,
            normalization_params=normalization_params,
        )
        if "normalization" in pipeline_order
        else None
    )

    return PreprocessingSample(
        sample_rate_hz=sample_rate_hz,
        windowing=windowing_sample,
        filtering=filtering_sample,
        downsampling=downsampling_sample,
        normalization=normalization_sample,
        order=pipeline_order,
    )


def parse_filtering_params(
    trial: optuna.Trial,
    filtering_params: dict | None,
) -> FilteringSample | None:
    if filtering_params is None:
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
        gain=parse_search_param(
            trial=trial,
            name="preprocessing/filtering/gain",
            params=filtering_params,
            key="gain",
            default_value=1,
        ),
        order=parse_search_param(
            trial=trial,
            name="preprocessing/filtering/order",
            params=filtering_params,
            key="order",
            default_value=2,
        ),
        filter_type=parse_search_param(
            trial=trial,
            name="preprocessing/filtering/filter_type",
            params=filtering_params,
            key="filter_type",
            default_value="iir",
        ),
        filter_design=parse_search_param(
            trial=trial,
            name="preprocessing/filtering/filter_design",
            params=filtering_params,
            key="filter_design",
            default_value="butter",
        ),
        band_type=parse_search_param(
            trial=trial,
            name="preprocessing/filtering/band_type",
            params=filtering_params,
            key="band_type",
            default_value="bandpass",
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
            ),
            drop_samples=parse_search_param(
                trial=trial,
                name="preprocessing/downsampling/drop_samples",
                params=downsampling_params,
                key="drop_samples",
                default_value=True,
            ),
        )

        if downsampling_sample.factor < 1:
            raise ValueError("downsampling factor must be at least 1")

    return downsampling_sample


def parse_windowing_params(
    trial: optuna.Trial,
    windowing_params: dict | None,
) -> WindowingSample | None:
    if windowing_params is None:
        return None

    windowing_sample = WindowingSample(
        window_ms=parse_search_param(
            trial=trial,
            name="preprocessing/windowing/window_ms",
            params=windowing_params,
            key="window_ms",
        ),
    )

    return windowing_sample


def parse_sample_rate_hz(params: dict) -> float:
    sample_rate_hz = params["sample_rate_hz"]

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    return sample_rate_hz


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
    )

    return NormalizationSample(method=method)


def _validate_preprocessing_order(
    order: tuple[str, ...],
    params: dict,
) -> tuple[str, ...]:
    unknown_steps = [step for step in order if step not in VALID_PREPROCESSING_STEPS]
    if unknown_steps:
        raise ValueError(f"Unknown preprocessing step(s): {unknown_steps}")

    duplicated_steps = [step for step in order if order.count(step) > 1]
    if duplicated_steps:
        raise ValueError(f"Duplicate preprocessing step(s): {duplicated_steps}")

    configured_steps = {
        key
        for key, value in params.items()
        if key in VALID_PREPROCESSING_STEPS and value is not None
    }

    extra_steps = set(order) - configured_steps
    if extra_steps:
        raise ValueError(
            f"Pipeline order contains unconfigured step(s): {sorted(extra_steps)}"
        )

    return order


def _default_processing_order(params: dict) -> tuple[str, ...]:
    return tuple(step for step in DEFAULT_PREPROCESSING_ORDER if step in params)


def parse_preprocessing_order(
    trial: optuna.Trial,
    params: dict,
) -> tuple[str, ...]:
    pipeline_params = params.get("pipeline")

    if pipeline_params is None:
        return _default_processing_order(params)

    order = parse_search_param(
        trial=trial,
        name="preprocessing/pipeline/order",
        params=pipeline_params,
        key="order",
        default_value=">".join(_default_processing_order(params)),
    )

    return _validate_preprocessing_order(
        order=tuple(order.split(">")),
        params=params,
    )
