import numpy as np
import pytest
import torch
import yaml

from elasticai.explorer.generator import model_builder
from elasticai.explorer.generator.model_builder.model_builder import (
    CreatorModelBuilder,
    ModelBuilder,
)
import elasticai.creator.nn.fixed_point as nn_creator
from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import MathOperations
from elasticai.explorer.generator.model_compiler.model_compiler import ModelCompiler
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace
from optuna.trial import FixedTrial

from elasticai.creator.nn import Sequential


class TestConstruct_SP:
    @pytest.fixture
    def search_space_dict(self):
        return yaml.safe_load(
            """input: [1, 2]
output: 2
quantization:
  quant_candidates: ["fixed_point_int8"]
  fixed_point_int8:
        total_bits: [8]
        frac_bits: [2]
        signed: [1]

blocks:
  - block:  "1" 
    op_candidates: ["linear"]
    activation: ["relu"]
    depth: [1]
    linear:
      activation: [ "relu"]
      width: [2]
    
        """
        )

    def test_build_creator_model(self, search_space_dict):
        search_space = SearchSpace(search_space_dict)

        sample = {
            "num_layers_b1": 1,
            "quant": "fixed_point_int8",
            "total_bits": 8,
            "frac_bits": 2,
            "signed_quant": 1,
            "operation_b1": "linear",
            "layer_width_b1_l0": 2,
            "activation_func_b1_l0": "relu",
        }

        model_builder = CreatorModelBuilder()
        result, _ = model_builder.build_from_trial(FixedTrial(sample), search_space)

    @pytest.mark.parametrize(
        "total_bits, frac_bits, features_in, features_out",
        [
            (8, 2, 2, 2),
        ],
    )
    def test_creator_model(
        self,
        total_bits: int,
        frac_bits: int,
        features_in: int,
        features_out: int,
    ):
        fxp = FxpArithmetic(FxpParams(total_bits=total_bits, frac_bits=frac_bits))
        math = MathOperations(fxp)
        model = Sequential(
            nn_creator.Linear(
                in_features=features_in,
                out_features=features_out,
                total_bits=total_bits,
                frac_bits=frac_bits,
            )
        )
        # --- Adapting values
        scale_amp = (fxp.maximum_as_rational - fxp.minimum_as_rational) / (
            2 * features_in
        )
        scale_min = -scale_amp / 2

        model[0].weight.data = torch.nn.Parameter(
            math.quantize(
                scale_amp
                * torch.rand(
                    size=(features_out, features_in),
                )
                + scale_min
                + torch.randint(low=-1, high=+1, size=(features_out, features_in))
                * fxp.config.minimum_step_as_rational
            )
        )
        model[0].bias.data = torch.nn.Parameter(
            math.quantize(
                scale_amp
                * torch.rand(
                    size=(features_out,),
                )
                + scale_min
                + torch.randint(low=-1, high=+1, size=(features_out,))
                * fxp.config.minimum_step_as_rational
            )
        )
        val_input = fxp.as_rational(
            torch.randint(
                low=-(2 ** (fxp.total_bits - 2)),
                high=2 ** (fxp.total_bits - 2),
                size=(20, features_in),
            )
        )

        weights_q, bias_q = model[0].get_params_quant()
       
        weights_f, bias_f = model[0].get_params()
        error_w = (
            np.array(weights_f)
            - np.array(weights_q) * fxp.config.minimum_step_as_rational
        )
        assert np.all(np.abs(error_w) < fxp.config.minimum_step_as_rational)
        error_b = (
            np.array(bias_f)
            - np.array(bias_q) * fxp.config.minimum_step_as_rational
        )
        assert np.all(np.abs(error_b) < fxp.config.minimum_step_as_rational)

        model.eval()
        with torch.no_grad():
            val_output = model(val_input)
        print(val_output)
        print(val_output)
        #ModelCompiler.