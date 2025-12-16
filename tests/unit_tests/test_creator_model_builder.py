import pytest
import yaml

from elasticai.explorer.generator import model_builder
from elasticai.explorer.generator.model_builder.model_builder import (
    CreatorModelBuilder,
    ModelBuilder,
)
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace
from optuna.trial import FixedTrial


class TestConstruct_SP:
    @pytest.fixture
    def search_space_dict(self):
        return yaml.safe_load(
            """input: [1, 6]
    output: 4
    blocks:
    - block:  "1" 
        op_candidates: ["linear"]
        activation: ["relu"]
        depth: [1, 2]
        linear:
            activation: [ "relu"]
            width: [4]
        """
        )

    def test_build_creator_model(self, search_space_dict):
        search_space = SearchSpace(search_space_dict)

        sample = {
            "num_layers_b1": 1,
            "operation_b1": "linear",
            "layer_width_b1_l0": 4,
            "activation_func_b1_l0": "relu",
        }

        model_builder = CreatorModelBuilder()
        result = model_builder.build_from_trial(FixedTrial(sample), search_space)
