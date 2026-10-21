from optuna.trial import FixedTrial
from torch import nn

from elasticai.explorer.hw_nas.hw_nas import sample_and_create_model


def test_sample_and_create_model_uses_given_input_shape():
    search_space = {
        "input": 20,
        "output": 1,
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
