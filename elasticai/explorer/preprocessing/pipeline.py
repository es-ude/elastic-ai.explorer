import numpy as np

from elasticai.explorer.preprocessing.methods import apply_preprocessing_step
from elasticai.explorer.preprocessing.types import (
    VALID_PREPROCESSING_STEPS,
    PreprocessingSample,
)


def apply_preprocessing_pipeline(
    signal: np.ndarray,
    preprocessing: PreprocessingSample,
    timestamps_ms: np.ndarray | None = None,
) -> np.ndarray:
    result = signal
    current_sample_rate_hz = preprocessing.sample_rate_hz
    for step in preprocessing.order:
        if step not in VALID_PREPROCESSING_STEPS:
            raise ValueError(f"Unknown preprocessing step: {step}")

        result, current_sample_rate_hz = apply_preprocessing_step(
            step=step,
            preprocessing=preprocessing,
            signal=result,
            sample_rate_hz=current_sample_rate_hz,
            timestamps_ms=timestamps_ms,
        )

    return result
