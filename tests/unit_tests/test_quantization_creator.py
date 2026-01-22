import pytest
import yaml
from elasticai.explorer.generator.model_builder.model_builder import CreatorModelBuilder
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace
from optuna.trial import FixedTrial

from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
)


class TestConstruct_SP:
    @pytest.fixture
    def search_space_dict(self):
        return yaml.safe_load(
            """input: [1, 6]
output: 4
quantization:
  quant_candidates: ["fixed_point_int8"]
  fixed_point_int8:
        total_bits: [8]
        frac_bits: [2]
        signed: [1]

blocks:
  - block:  "1" 
    op_candidates: ["conv1d", "linear"]
    activation: ["relu"]
    depth: [1]
    conv1d:
      kernel_size: [1]
      signal_length: [1]
      out_channels: [4]
    linear:
      activation: [ "relu"]
      width: [4,6,8]
  - block:  "2" 
    op_candidates: ["linear", "linear"]
    activation: ["relu"]
    depth: [1]
    linear:
      activation: [ "relu"]
      width: [4,6,8]
  """
        )

    def test_build_creator_model(self, search_space_dict):
        search_space = SearchSpace(search_space_dict)
        sample = {
            "quant": "fixed_point_int8",
            "total_bits": 8,
            "frac_bits": 2,
            "signed_quant": 1,
            "num_layers_b1": 1,
            "operation_b1": "linear",
            "layer_width_b1_l0": 8,
            "activation_func_b1_l0": "relu",
            "num_layers_b2": 1,
            "operation_b2": "linear",
            "layer_width_b2_l0": 8,
            "activation_func_b2_l0": "relu",
        }

        model_builder = CreatorModelBuilder()
        model, _ = model_builder.build_from_trial(FixedTrial(sample), search_space)
        model_builder.validate_model(model, FixedPointInt8Scheme("int8", 8, 2, True))

        assert model != None
