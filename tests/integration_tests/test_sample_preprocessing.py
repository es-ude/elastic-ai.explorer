from optuna.trial import FixedTrial

from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.preprocessing.sample import sample_preprocessing
from elasticai.explorer.preprocessing.types import (
    DownsamplingSample,
    FilteringSample,
    NormalizationSample,
    PreprocessingSample,
    WindowingSample,
)
from settings import ROOT_DIR

PREPROCESSING_SEARCH_SPACE_FILE = (
    ROOT_DIR / "tests/integration_tests/samples/preprocessing_search_space.yaml"
)


def test_samples_preprocessing_from_yaml_search_space():
    search_space = yaml_to_dict(PREPROCESSING_SEARCH_SPACE_FILE)

    trial = FixedTrial(
        {
            "preprocessing/windowing/window_ms": 1000,
            "preprocessing/filtering/low_cut_hz": 1.0,
            "preprocessing/filtering/high_cut_hz": 100.0,
            "preprocessing/downsampling/factor": 2,
            "preprocessing/normalization/method": "zscore",
            "preprocessing/pipeline/order": "normalization>windowing>filtering>downsampling",
        }
    )

    sample = sample_preprocessing(
        trial=trial,
        params=search_space["preprocessing"],
    )

    assert sample == PreprocessingSample(
        sample_rate_hz=1000.0,
        windowing=WindowingSample(window_ms=1000),
        filtering=FilteringSample(low_cut_hz=1.0, high_cut_hz=100.0),
        downsampling=DownsamplingSample(factor=2),
        normalization=NormalizationSample(method="zscore"),
        order=(
            "normalization",
            "windowing",
            "filtering",
            "downsampling",
        ),
    )
