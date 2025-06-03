import pytest
import torch
import yaml
from torch import nn

from elasticai.explorer.hw_nas.search_space.construct_sp import (
    calculate_conv_output_shape,
    SearchSpace,
    compute_start_view,
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
        "depth, width_0, width_1, width_2, width_3",
        [(2, 16, 32, 5, 4), (3, 16, 32, 5, 4), (1, 16, 32, 5, 4)],
    )
    def test_construct_linear_search_space_mutable_depth(
        self, depth, width_0, width_1, width_2, width_3
    ):
        # [16, 32, 5]
        search_space = SearchSpace(yaml.safe_load(yaml_mock))
        print(search_space)
        # print(search_space.freeze({"depth_block_1":2}))
        # print(search_space)
        x = torch.randn(5, 784)
        result = search_space(x)
        # this is not as expected, but seems to work in practise(sadge)
        print(
            search_space.freeze(
                {
                    "depth_block_1": depth,
                    "layer_width_0_block_1": width_0,
                    "layer_width_1_block_1": width_1,
                    "layer_width_2_block_1": width_2,
                    "layer_width_3_block_1": width_3,
                    "activation_block_1": 0,
                }
            )
        )

        # print(net)
        assert result.shape == torch.Size([5, 10])

    # def test_construct_mixed_sp(self):
    #     yaml_mock = """input: (1, 28, 28)
    #     output: 10
    #     blocks:
    #       - block:  "1" #namefield muss noch rein
    #         op_candidates: ["conv2d"]
    #         depth: [1, 4]
    #             conv2D:
    #                 kernel_size: [5, 10]
    #                 stride: [1, 2]
    #                 out_channels: [ 10, 8]
    #       - block:  "2" #namefield muss noch rein
    #         op_candidates: ["linear"]
    #         depth: [1, 4]
    #         linear:
    #           #überall range oder choices
    #           activation: [ "relu", sigmoid"]
    #           width: [16, 32, 5, 4]
    #                 """
    #     search_space = CombinedSearchSpace(yaml.safe_load(yaml_mock))
    #     print(search_space)
    #     x = torch.randn(5, 1, 28, 28)
    #     result = search_space(x)

    def test_compute_start_view_for_linear(self):
        x = torch.randn(5, 1, 28, 28)
        self.blocks = nn.Sequential(
            nn.Sequential(nn.Linear(784, 20), nn.Linear(20, 10)), nn.Linear(10, 10)
        )
        x = compute_start_view(x, self.blocks)
        assert x.shape == torch.Size([5, 784])

    def test_compute_start_view_for_conv2d(self):
        x = torch.randn(5, 1, 28, 28)
        self.blocks = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 16, 4, 1), nn.Conv2d(16, 16, 4, 1)),
            nn.Linear(256, 10),
        )
        x = compute_start_view(x, self.blocks)
        assert x.shape == torch.Size([5, 1, 28, 28])
