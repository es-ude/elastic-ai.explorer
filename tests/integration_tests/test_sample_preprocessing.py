from optuna.trial import FixedTrial

from elasticai.explorer import get_path_to_project
from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.preprocessing.sample import sample_preprocessing
from elasticai.explorer.preprocessing.types import (
    DownsamplingSample,
    FilteringSample,
    NormalizationSample,
    PreprocessingSample,
    WindowingSample,
)

PREPROCESSING_SEARCH_SPACE_FILE = (
    get_path_to_project() / "tests/integration_tests/samples/preprocessing_search_space.yaml"
)


def test_samples_preprocessing_from_yaml_search_space():
    search_space = yaml_to_dict(PREPROCESSING_SEARCH_SPACE_FILE)

    trial = FixedTrial(
        {
            "preprocessing/windowing/window_ms": 1000,
            "preprocessing/filtering/filter_candidates": "bandpass",
            "preprocessing/filtering/bandpass/low_cut_hz": 1.0,
            "preprocessing/filtering/bandpass/high_cut_hz": 100.0,
            "preprocessing/filtering/gain": 1,
            "preprocessing/filtering/order": 4,
            "preprocessing/filtering/filter_type": "iir",
            "preprocessing/filtering/filter_design": "butter",
            "preprocessing/downsampling/factor": 2,
            "preprocessing/downsampling/drop_samples": True,
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
        filtering=FilteringSample(
            band_type="bandpass",
            low_cut_hz=1.0,
            high_cut_hz=100.0,
            gain=1,
            order=4,
            filter_type="iir",
            filter_design="butter",
        ),
        downsampling=DownsamplingSample(
            factor=2,
            drop_samples=True,
        ),
        normalization=NormalizationSample(method="zscore"),
        order=(
            "normalization",
            "windowing",
            "filtering",
            "downsampling",
        ),
    )
