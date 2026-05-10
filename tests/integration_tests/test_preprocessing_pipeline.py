import numpy as np
from optuna.trial import FixedTrial

from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.preprocessing.pipeline import apply_preprocessing_pipeline
from elasticai.explorer.preprocessing.sample import sample_preprocessing
from settings import ROOT_DIR

PREPROCESSING_SEARCH_SPACE_FILE = (
    ROOT_DIR / "tests/integration_tests/samples/preprocessing_search_space.yaml"
)


def test_samples_and_applies_preprocessing_pipeline_from_yaml_search_space():
    search_space = yaml_to_dict(PREPROCESSING_SEARCH_SPACE_FILE)

    trial = FixedTrial(
        {
            "preprocessing/windowing/window_ms": 1000,
            "preprocessing/downsampling/factor": 2,
            "preprocessing/downsampling/drop_samples": True,
            "preprocessing/normalization/method": "zscore",
            "preprocessing/pipeline/order": "downsampling>windowing>normalization",
        }
    )

    preprocessing = sample_preprocessing(
        trial=trial,
        params=search_space["preprocessing"],
    )

    signal = np.arange(2000, dtype=float)

    result = apply_preprocessing_pipeline(
        signal=signal,
        preprocessing=preprocessing,
        timestamps_ms=np.array([0]),
    )

    assert result.shape == (1, 1, 500)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(np.mean(result, axis=-1), 0.0, atol=1e-12)
