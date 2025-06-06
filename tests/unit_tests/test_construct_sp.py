import pytest
import torch
import yaml
from nni.nas.execution import SequentialExecutionEngine
from torch import nn

from elasticai.explorer.hw_nas.search_space.construct_sp import (
    calculate_conv_output_shape,
    SearchSpace,
    CombinedSearchSpace,
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

    @pytest.fixture
    def search_space_dict(self):
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
              op_candidates: ["linear"] 
              depth: 1
              linear:
                width: [16, 32]"""
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

    def test_construct_linear_search_space_valid(self):

        search_space = SearchSpace(yaml.safe_load(yaml_mock))
        x = torch.randn(5, 784)

        result = search_space(x)

        assert result.shape == torch.Size([5, 10])

    def test_construct_mixed_sp_multiple_blocks(self, search_space_dict_mult_blocks):
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
                "block_1/candidate_op": "conv2d",
                "block_2/layer_width_0": 16,
                "block_2/layer_width_1": 32,
                "block_2/candidate_op": "linear",
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

        search_space = CombinedSearchSpace(search_space_dict)
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

        @pytest.fixture
        def engine(self):
            return SequentialExecutionEngine(max_model_count=30)

        # def test_random(self, engine):
        #
        #     model_space = CombinedSearchSpace(
        #         yml_to_dict(
        #             "/Users/mokou/Documents/transfair/toolbox/elastic-ai.explorer/elasticai/explorer/hw_nas/search_space/search_space.yml"
        #         )
        #     )
        #     evaluator = FunctionalEvaluator(evaluate_model, device="cpu")
        #
        #     model_space = SimplifiedModelSpace.from_model(model_space, evaluator)
        #     dedup = True
        #     # name, model_space = named_model_space
        #     strategy = Random(dedup=dedup)
        #     assert repr(strategy) == f"Random(dedup={dedup})"
        #     strategy(model_space, engine)
        #     assert len(list(engine.list_models())) == len(list(model_space.grid()))
        #     state_dict = strategy.state_dict()
        #     previous_submitted = len(list(engine.list_models()))
        #     strategy2 = Random(dedup=dedup)
        #     strategy2.load_state_dict(state_dict)
        #
        #     engine.max_model_count += 10
        #     strategy2(model_space, engine)
        #     if dedup:
        #         assert len(list(engine.list_models())) == previous_submitted
        #     else:
        #         assert len(list(engine.list_models())) == engine.max_model_count
