import math
from typing import OrderedDict
import torch
import pytest
from torch import nn

from elasticai.explorer.hw_nas.search_space.architecture_components import SimpleLSTM
from elasticai.explorer.hw_nas.search_space.build_model import construct_model
from elasticai.explorer.hw_nas.search_space.layer_adapter import (
    ToLinearAdapter,
    LSTMNoSequenceAdapter,
    Conv1dToLSTM,
)


# todo: fix toLinear schouldn't be inserted before each linear layer. But doesn't break anything
@pytest.mark.parametrize(
    "first_block, expected",
    [
        (
            OrderedDict(
                [
                    ("layer1", {"operation": "linear", "params": {"width": 64}}),
                    ("layer2", {"operation": "linear", "params": {}}),
                ]
            ),
            nn.Sequential(
                ToLinearAdapter(),
                nn.Linear(20, 64),
                ToLinearAdapter(),
                nn.Linear(64, 1),
            ),
        ),
        (
            OrderedDict(
                [("layer1", {"operation": "linear", "params": {"activation": "relu"}})]
            ),
            nn.Sequential(
                ToLinearAdapter(), nn.Sequential(nn.Linear(20, 1), nn.ReLU())
            ),
        ),
    ],
)
def test_build_fc_model(first_block, expected):

    sample = OrderedDict({"1": first_block})
    model = construct_model(sample, 20, 1)
    states = model.state_dict()
    expected.load_state_dict(states)
    input = torch.rand([16, 20])
    assert torch.equal(expected(input), model(input)) == True


def test_build_conv2d_model():
    block_1 = OrderedDict(
        [
            (
                "layer1",
                {
                    "operation": "conv2d",
                    "params": {
                        "kernel_size": 3,
                        "out_channels": 16,
                        "stride": 1,
                        "padding": 0,
                        "activation": "relu",
                    },
                },
            ),
            (
                "layer2",
                {
                    "operation": "conv2d",
                    "params": {
                        "kernel_size": 2,
                        "out_channels": 24,
                        "stride": 2,
                        "padding": 0,
                    },
                },
            ),
        ]
    )
    block_2 = OrderedDict(
        [
            (
                "layer1",
                {
                    "operation": "maxpool",
                    "params": {"kernel_size": 3, "stride": 1, "padding": 0},
                },
            ),
            ("layer2", {"operation": "activation", "params": {"op": "relu"}}),
            ("layer3", {"operation": "batch_norm", "params": {}}),
        ]
    )
    block_3 = OrderedDict(
        [("layer1", {"operation": "linear", "params": {"activation": "sigmoid"}})]
    )
    sample = OrderedDict({"1": block_1, "2": block_2, "3": block_3})

    model = construct_model(sample, [16, 30, 20], 1)
    print(model)
    first_part = [
        nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0), nn.ReLU()),
        nn.Conv2d(16, 24, kernel_size=2, stride=2, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(24),
    ]
    states = model.state_dict()
    input = torch.rand([16, 16, 30, 20])
    out = nn.Sequential(*first_part)(input)
    shape = math.prod(out.shape[-3:])
    expected = nn.Sequential(
        *first_part, ToLinearAdapter(), nn.Sequential(nn.Linear(shape, 1), nn.Sigmoid())
    )
    print(expected)
    expected.load_state_dict(states)
    assert torch.equal(expected(input), model(input)) == True


def test_build_conv1d_model():
    block_1 = OrderedDict(
        [
            (
                "layer1",
                {
                    "operation": "conv1d",
                    "params": {
                        "kernel_size": 3,
                        "out_channels": 16,
                        "stride": 1,
                        "padding": 0,
                        "activation": "relu",
                    },
                },
            ),
            (
                "layer2",
                {
                    "operation": "conv1d",
                    "params": {
                        "kernel_size": 2,
                        "out_channels": 24,
                        "stride": 2,
                        "padding": 0,
                    },
                },
            ),
        ]
    )
    block_2 = OrderedDict(
        [
            (
                "layer1",
                {
                    "operation": "maxpool",
                    "params": {"kernel_size": 3, "stride": 1, "padding": 0},
                },
            ),
            ("layer2", {"operation": "activation", "params": {"op": "relu"}}),
            ("layer3", {"operation": "batch_norm", "params": {}}),
        ]
    )
    block_3 = OrderedDict(
        [("layer1", {"operation": "linear", "params": {"activation": "sigmoid"}})]
    )
    sample = OrderedDict({"1": block_1, "2": block_2, "3": block_3})
    in_dim = [2, 20]
    model = construct_model(sample, in_dim, 1)
    print(model)
    first_part = [
        nn.Sequential(
            nn.Conv1d(in_dim[0], 16, kernel_size=3, stride=1, padding=0), nn.ReLU()
        ),
        nn.Conv1d(16, 24, kernel_size=2, stride=2, padding=0),
        nn.MaxPool1d(kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm1d(24),
    ]
    states = model.state_dict()
    input = torch.rand([16, 2, 20])
    out = nn.Sequential(*first_part)(input)
    shape = math.prod(out.shape[-2:])
    expected = nn.Sequential(
        *first_part, ToLinearAdapter(), nn.Sequential(nn.Linear(shape, 1), nn.Sigmoid())
    )
    print(expected)
    expected.load_state_dict(states)
    assert torch.equal(expected(input), model(input)) == True


def test_build_lstm_model():
    torch.manual_seed(40)
    block_1 = OrderedDict(
        [
            (
                "layer1",
                {
                    "operation": "lstm",
                    "params": {
                        "hidden_size": 64,
                        "num_layers": 2,
                        "bidirectional": False,
                    },
                },
            ),
            (
                "layer2",
                {
                    "operation": "lstm",
                    "params": {
                        "hidden_size": 32,
                        "num_layers": 1,
                        "bidirectional": True,
                    },
                },
            ),
        ]
    )
    block_2 = OrderedDict(
        [("layer1", {"operation": "linear", "params": {"activation": "sigmoid"}})]
    )
    sample = OrderedDict({"1": block_1, "2": block_2})
    # sequence_length, num_features
    in_dim = [2000, 2]
    model = construct_model(sample, in_dim, 1)
    print(model)
    states = model.state_dict()
    input = torch.rand([16, 2000, 2])
    print(input[0])
    expected = nn.Sequential(
        SimpleLSTM(
            in_dim[-1],
            hidden_size=64,
            num_layers=2,
            bidirectional=False,
            batch_first=True,
            bias=True,
            dropout=0,
        ),
        SimpleLSTM(
            64,
            hidden_size=32,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            bias=True,
            dropout=0,
        ),
        LSTMNoSequenceAdapter(),
        nn.Sequential(nn.Linear(64, 1), nn.Sigmoid()),
    )
    print(expected)
    expected.load_state_dict(states)

    assert torch.equal(model(input), expected(input)) == True


def test_lstm_to_conv1d():
    input = torch.rand([16, 6, 10])

    conv_layer = nn.Conv1d(6, 6, 2)
    shape = conv_layer(input).shape
    layer_adapter = Conv1dToLSTM()
    adapter_shape = layer_adapter.infer_output_shape(shape[1:])
    lstm_layer = SimpleLSTM(
        adapter_shape[1],
        6,
        1,
        bidirectional=False,
        batch_first=True,
        bias=True,
        dropout=0,
    )

    output = LSTMNoSequenceAdapter()((lstm_layer(layer_adapter(conv_layer(input)))))
    assert output.shape == torch.Size([16, 6])
