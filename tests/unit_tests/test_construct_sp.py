from pathlib import Path
import pytest
import torch
import yaml
from optuna.trial import FixedTrial
from torch import nn

from elasticai.explorer.hw_nas.search_space.construct_search_space import (
    calculate_conv_output_shape,
    SearchSpace,
    parse_search_param,
)
from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from settings import ROOT_DIR

yaml_mock = """input: 784
output: 10
blocks:
  - block:  "1"
    op_candidates: ["linear"]
    activation: [ "relu"]
    depth: [1,2, 4]
    linear:
        width: [16, 32, 5, 4]
    """


class TestConstruct_SP:

    @pytest.fixture
    def search_space_dict(self):
        return yaml.safe_load(
            """input: [1, 28, 28]
output: 10
blocks:
  - block:  "1" #namefield muss noch rein
    op_candidates: ["linear", "conv2d"]
    activation: ["relu", "sigmoid"]
    depth: [1, 2, 3]
    linear:
        activation: [ "relu", "sigmoid"]
        width: [16, 32, 5, 4]
    conv2D:
      kernel_size: [1, 2]
      stride: [1, 2]
      out_channels: [ 10, 4]"""
        )

    @pytest.fixture
    def search_space_dict_mult_blocks(self):
        return yaml.safe_load(
            """input: [1, 28, 28]
output: 10
blocks:
            - block:  "1" #namefield muss noch rein
              op_candidates: ["linear", "conv2d"]
              depth: [1, 2, 3]
              activation: [ "relu", "sigmoid"]
              linear:
                width: [16, 32, 5, 4]
              conv2D:
                kernel_size: [1, 2]
                stride: [1, 2]
                out_channels:
                    start: 4
                    end: 11
            - block:  "2"
              op_candidates: ["linear", "conv2d"] 
              depth: 1
              activation: [ "relu", "sigmoid"]
              linear:
                width: [16, 32]
              conv2D:
                kernel_size: [1, 2]
                stride: [1, 2]
                out_channels: [ 10, 4] """
        )

    @pytest.mark.parametrize(
        "shape, out_channels, kernel_size, stride, dilation",
        [
            ([4, 1, 28, 28], 16, 3, 1, 1),
            ([4, 1, 28, 28], 16, (3, 2), (1, 1), 2),
            ([5, 3, 4], 4, 2, 2, 1),
            ([28, 6, 6], 1, 6, 1, 1),
        ],
    )
    def test_calculate_conv_output_shape(
        self, shape, out_channels, kernel_size, stride, dilation
    ):
        actual = calculate_conv_output_shape(shape, out_channels, kernel_size, stride)
        layer = nn.Conv2d(
            in_channels=shape[-3],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        print(torch.ones(shape).shape)
        expected = list(layer(torch.ones(shape)).shape)
        print(expected)
        assert actual == expected

    def test_construct_one_block_linear_search_space_valid(self):
        search_space = SearchSpace(yaml.safe_load(yaml_mock))
        x = torch.randn(5, 784)
        sample = {
            "num_layers_b1": 2,
            "operation_b1": "linear",
            "layer_width_b1_l0": 16,
            "layer_width_b1_l1": 32,
            "activation_func_b1_l0": "relu",
            "activation_func_b1_l1": "relu",
        }
        result = search_space.create_native_torch_model_sample(FixedTrial(sample))
        assert result(x).shape == torch.Size([5, 10])

    @pytest.mark.parametrize(
        "b1, b2",
        [
            (
                {
                    "num_layers_b1": 2,
                    "operation_b1": "conv2d",
                    "activation_func_b1_l0": "relu",
                    "activation_func_b1_l1": "relu",
                    "out_channels_b1_l0": 6,
                    "out_channels_b1_l1": 10,
                    "stride_b1_l0": 1,
                    "stride_b1_l1": 1,
                    "kernel_size_b1_l0": 2,
                    "kernel_size_b1_l1": 1,
                },
                {
                    "operation_b2": "linear",
                    "activation_func_b2_l0": "relu",
                    "layer_width_b2_l0": 16,
                },
            ),
            (
                {
                    "num_layers_b1": 2,
                    "operation_b1": "linear",
                    "activation_func_b1_l0": "relu",
                    "activation_func_b1_l1": "sigmoid",
                    "layer_width_b1_l0": 16,
                    "layer_width_b1_l1": 32,
                },
                {
                    "operation_b2": "linear",
                    "activation_func_b2_l0": "relu",
                    "layer_width_b2_l0": 16,
                },
            ),
        ],
    )
    def test_construct_mixed_sp_multiple_blocks(
        self, search_space_dict_mult_blocks, b1, b2
    ):
        search_space = SearchSpace(search_space_dict_mult_blocks)
        x = torch.randn(5, 1, 28, 28)
        sample_model = search_space.create_native_torch_model_sample(
            FixedTrial({**b1, **b2})
        )
        assert sample_model(x).shape == torch.Size([5, 10])

    @pytest.mark.parametrize(
        "type, value, expected",
        [
            (list, [1, 2, 3], 3),
            (dict, {"start": 1, "end": 4}, 2),
            (int, 2, 2),
            (str, "Hello", "Hello"),
        ],
    )
    def test_parse_params(self, type, value, expected):
        trial = FixedTrial({f"type_{type}": expected})
        actual = parse_search_param(trial, "type_{}".format(type), value)
        assert actual == expected


def objective(trial):
    search_space = yaml_to_dict(
        ROOT_DIR / Path("elasticai/explorer/hw_nas/search_space/search_space.yaml")
    )
    search_space = SearchSpace(search_space)
    return search_space.create_native_torch_model_sample(trial)
