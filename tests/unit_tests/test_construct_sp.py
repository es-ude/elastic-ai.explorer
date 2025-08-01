import pytest
import torch
import yaml
from optuna.trial import FixedTrial
from torch import nn

from elasticai.explorer.hw_nas.search_space.construct_sp import (
    calculate_conv_output_shape,
    SearchSpace,
    yaml_to_dict,
)

yaml_mock = """input: 784
output: 10
blocks:
  - block:  "1" #namefield muss noch rein
    op_candidates: ["linear"]
    depth: [1, 4]
    linear:
      #überall range oder choices
      activation: [ "relu"]
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
              linear:
                #überall range oder choices
             #   activation: [ "relu", "sigmoid"]
                width: [16, 32, 5, 4]
              conv2D:
                kernel_size: [1, 2]
                stride: [1, 2]
                out_channels: [ 10, 4]
            - block:  "2"
              op_candidates: ["linear", "conv2d"] 
              depth: 1
              linear:
                width: [16, 32]
              conv2D:
                kernel_size: [1, 2]
                stride: [1, 2]
                out_channels: [ 10, 4] """
        )

    # @pytest.fixture
    # def search_space_dict__2_linear(self):
    #     return yaml.safe_load(
    #         """input: [1, 28, 28]
    # output: 10
    # blocks:
    #             - block:  "1"
    #               op_candidates: ["linear", "conv2d"]
    #               depth: [1, 2, 3]
    #               linear:
    #                 width: [16, 32, 5, 4]
    #               conv2D:
    #                 kernel_size: [1, 2]
    #                 stride: [1, 2]
    #                 out_channels: [ 10, 4]
    #             - block:  "2"
    #               op_candidates: ["linear"]
    #               depth: 1
    #               linear:
    #                 width: [16, 32]"""
    #     )

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

    def test_construct_linear_search_space_valid(self):

        search_space = SearchSpace(yaml.safe_load(yaml_mock))
        x = torch.randn(5, 784)

        result = search_space(x)

        assert result.shape == torch.Size([5, 10])

    @pytest.mark.parametrize(
        "op_1, op_2",
        [
            ("linear", "linear"),
            ("conv2d", "conv2d"),
            ("conv2d", "linear"),
        ],
    )
    def test_construct_mixed_sp_multiple_blocks(
        self, search_space_dict_mult_blocks, op_1, op_2
    ):
        search_space = CombinedSearchSpace(search_space_dict_mult_blocks)
        x = torch.randn(5, 1, 28, 28)

        sample_model = search_space.freeze(
            {
                "block_1/depth": 3,
                "block_1/layer_width_0": 16,
                "block_1/layer_width_1": 32,
                "block_1/layer_width_2": 5,
                "block_1/out_channels_0": 10,
                "block_1/out_channels_1": 4,
                "block_1/out_channels_2": 10,
                "block_1/kernel_size": 1,
                "block_1/stride": 1,
                "block_1/activation": 0,
                "block_1/candidate_op": op_1,
                "block_2/layer_width_0": 16,
                "block_2/layer_width_1": 32,
                "block_2/candidate_op": op_2,
                "block_2/out_channels_0": 10,
                "block_2/kernel_size": 1,
                "block_2/stride": 1,
            }
        )
        assert sample_model(x).shape == torch.Size([5, 10])

    @pytest.mark.parametrize(
        "depth, width_0, width_1, width_2, width_3",
        [(2, 16, 32, 5, 4), (3, 16, 32, 5, 4), (1, 16, 32, 5, 4)],
    )
    def test_construct_convolutional_and_linear_search_space_single_block(
        self, search_space_dict, depth, width_0, width_1, width_2, width_3
    ):
        sample = {
            "num_layers_b1": 2,
            "num_layers_b2": 1,
            "operation_b1": "conv2d",
            "operation_b2": "linear",
            "layer_width_b2_l0": 21,
            "out_channels_b1_l0": 4,
            "out_channels_b1_l1": 10,
            "stride_b1_l0": 1,
            "stride_b1_l1": 1,
            "kernel_size_b1_l0": 2,
            "kernel_size_b1_l1": 2,
            "activation_func_b1_l0": "relu",
            "activation_func_b1_l1": "relu",
            "activation_func_b2_l0": "sigmoid",
        }
        #  sample={"num_layers": 2,"layer_op_l1":"linear","layer_op_l0":"conv2d","layer_width_l1": 128,"out_channels_l0": 16,"stride_l0": 1,"kernel_size_l0": 2, "activation_func_l0": "relu", "activation_func_l1": "sigmoid" }
        model = objective(FixedTrial(sample))
        search_space = SearchSpace(search_space_dict)
        x = torch.randn(5, 1, 28, 28)
        sample_model = search_space.freeze(
            {
                "block_1/depth": depth,
                "block_1/layer_width_0": width_0,
                "block_1/layer_width_1": width_1,
                "block_1/layer_width_2": width_2,
                "block_1/out_channels_0": 10,
                "block_1/out_channels_1": 4,
                "block_1/out_channels_2": 10,
                "block_1/kernel_size": 1,
                "block_1/stride": 1,
                "block_1/activation": 0,
                "block_1/candidate_op": "conv2d",
            }
        )
        assert sample_model(x).shape == torch.Size([5, 10])

        sample_model2 = search_space.freeze(
            {
                "block_1/depth": depth,
                "block_1/layer_width_0": width_0,
                "block_1/layer_width_1": width_1,
                "block_1/layer_width_2": width_2,
                "block_1/out_channels_0": 10,
                "block_1/out_channels_1": 4,
                "block_1/out_channels_2": 10,
                "block_1/kernel_size": 1,
                "block_1/stride": 1,
                "block_1/activation": 0,
                "block_1/candidate_op": "linear",
            }
        )
        assert sample_model2(x).shape == torch.Size([5, 10])


def objective(trial):
    search_space = yaml_to_dict("search_space.yml")
    search_space = SearchSpace(search_space)
    return search_space.create_model_sample(trial)
