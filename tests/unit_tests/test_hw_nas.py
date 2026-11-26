from optuna.trial import FixedTrial
from torch import nn

from elasticai.explorer import get_path_to_project
from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.hw_nas.hw_nas import sample_and_create_model, count_configurations


def test_sample_and_create_model_uses_given_input_shape():
    search_space = {
        "input": 20,
        "output": 2,
        "sequence": [
            {
                "block": "1",
                "op_candidates": "linear",
                "linear": {"width": 8},
            }
        ],
    }

    model = sample_and_create_model(
        trial=FixedTrial({}),
        search_space=search_space,
        input_shape=30,
    )

    linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    assert linear_layers[0].in_features == 30


def test_count_configurations_matches_actual_grid():
    path2yaml = get_path_to_project() / "tests" / "unit_tests" / "searchspace" / "files" / "mnist_search_space.yaml"
    search_space = yaml_to_dict(path2yaml.resolve())
    total, valid = count_configurations(search_space)
    assert total == 432
    assert valid == 360


def test_count_configurations_respects_limit():
    path2yaml = get_path_to_project() / "tests" / "unit_tests" / "searchspace" / "files" / "mnist_search_space.yaml"
    search_space = yaml_to_dict(path2yaml.resolve())
    total, valid = count_configurations(search_space, limit=10)
    assert total == 10
    assert valid == 10
