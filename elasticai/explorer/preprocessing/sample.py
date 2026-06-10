from typing import Any

import optuna

from elasticai.explorer.hw_nas.search_space.sample_blocks import parse_search_param
from elasticai.explorer.preprocessing.types import (
    DEFAULT_PREPROCESSING_ORDER,
    FILTERBAND_CUTOFFS,
    VALID_PREPROCESSING_STEPS,
    DownsamplingSample,
    FilteringSample,
    NormalizationSample,
    PreprocessingSample,
    WindowingSample,
)

_PIPELINE_SEPARATOR = ">"
_PIPELINE_DISABLED = "none"


def _suggest(
    trial: optuna.Trial, step: str, params: dict, key: str, default_value: Any = None
) -> Any:
    return parse_search_param(
        trial=trial,
        name=f"preprocessing/{step}/{key}",
        params=params,
        key=key,
        default_value=default_value,
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

    # no further sampling needed for empty preprocessing pipeline
    if pipeline_order == ():
        return PreprocessingSample(
            order=pipeline_order,
            sample_rate_hz=sample_rate_hz,
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


def parse_preprocessing_order(
    trial: optuna.Trial,
    params: dict,
) -> tuple[str, ...]:
    pipeline_params = params.get("pipeline")

    # default case if no pipeline order config is found
    if pipeline_params is None:
        return _default_processing_order(params)

    order = _suggest(
        trial=trial,
        step="pipeline",
        params=pipeline_params,
        key="order",
        default_value=_PIPELINE_SEPARATOR.join(_default_processing_order(params)),
    )

    # skip preprocessing
    if order in (None, _PIPELINE_DISABLED):
        return ()

    return _validate_preprocessing_order(
        order=tuple(order.split(_PIPELINE_SEPARATOR)),
        params=params,
    )


def _default_processing_order(params: dict) -> tuple[str, ...]:
    return tuple(step for step in DEFAULT_PREPROCESSING_ORDER if step in params)


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


def parse_windowing_params(
    trial: optuna.Trial,
    windowing_params: dict | None,
) -> WindowingSample | None:
    if windowing_params is None:
        return None

    windowing_sample = WindowingSample(
        window_ms=_suggest(
            trial=trial,
            step="windowing",
            params=windowing_params,
            key="window_ms",
        ),
    )

    return windowing_sample


def parse_filtering_params(
    trial: optuna.Trial,
    filtering_params: dict | None,
) -> FilteringSample | None:
    if filtering_params is None:
        return None
    band_type = _suggest(
        trial=trial,
        step="filtering",
        params=filtering_params,
        key="filter_candidates",
        default_value="bandpass",
    )

    candidate_params = filtering_params.get(band_type)
    if candidate_params is None:
        raise ValueError(f"Missing config for filter candidate: {band_type}")

    cutoff_kwargs = _parse_filter_cutoffs(
        trial=trial,
        candidate_params=candidate_params,
        band_type=band_type,
    )

    filtering_sample = FilteringSample(
        band_type=band_type,
        **cutoff_kwargs,
        gain=_suggest(
            trial=trial,
            step="filtering",
            params=filtering_params,
            key="gain",
            default_value=1,
        ),
        order=_suggest(
            trial=trial,
            step="filtering",
            params=filtering_params,
            key="order",
            default_value=2,
        ),
        filter_type=_suggest(
            trial=trial,
            step="filtering",
            params=filtering_params,
            key="filter_type",
            default_value="iir",
        ),
        filter_design=_suggest(
            trial=trial,
            step="filtering",
            params=filtering_params,
            key="filter_design",
            default_value="butter",
        ),
    )

    return filtering_sample


def _parse_filter_cutoffs(
    trial: optuna.Trial,
    candidate_params: dict,
    band_type: str,
) -> dict[str, int | float]:
    if band_type not in FILTERBAND_CUTOFFS:
        raise ValueError(f"Unsupported filter candidate: {band_type}")

    parsed = {
        key: parse_search_param(
            trial=trial,
            name=f"preprocessing/filtering/{band_type}/{key}",
            params=candidate_params,
            key=key,
        )
        for key in FILTERBAND_CUTOFFS[band_type]
    }

    return parsed


def parse_downsampling_params(
    trial: optuna.Trial,
    downsampling_params: dict | None,
) -> DownsamplingSample | None:
    if downsampling_params is None:
        return None

    downsampling_sample = DownsamplingSample(
        factor=_suggest(
            trial=trial,
            step="downsampling",
            params=downsampling_params,
            key="factor",
        ),
        drop_samples=_suggest(
            trial=trial,
            step="downsampling",
            params=downsampling_params,
            key="drop_samples",
            default_value=True,
        ),
    )

    return downsampling_sample


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

    return NormalizationSample(
        method=_suggest(
            trial=trial,
            step="normalization",
            params=normalization_params,
            key="method",
        )
    )
