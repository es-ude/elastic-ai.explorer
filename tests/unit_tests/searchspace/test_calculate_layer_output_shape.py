import pytest
import torch
from torch import nn

from elasticai.explorer.hw_nas.search_space.utils import calculate_output_shape


@pytest.mark.parametrize(
    "shape, out_channels, kernel_size, stride, dilation, padding",
    [
        ([4, 1, 28, 28], 16, 3, 1, 2, 0),
        ([4, 1, 28, 28], 16, (3, 2), (1, 1), 1, 0),
        ([5, 3, 4], 4, 2, 2, 2, 0),
        ([5, 3, 4], 4, 2, 2, 1, 0),
        ([28, 6, 6], 1, 6, 1, 1, 0),
        ([28, 6, 6], 1, 6, 1, 1, "same"),
        ([28, 6, 6], 1, 6, 1, 1, 4),
        ([28, 6, 6], 1, 6, 1, 1, (4, 3)),
    ],
)
def test_calculate_conv2d_output_shape(
    shape,
    out_channels,
    kernel_size,
    stride,
    dilation,
    padding,
):
    actual = calculate_output_shape(
        shape,
        kernel_size=kernel_size,
        stride=stride,
        out_channels=out_channels,
        dilation=dilation,
        padding=padding,
        layer_type="conv2d",
    )
    layer = nn.Conv2d(
        in_channels=shape[-3],
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
    )
    expected = list(layer(torch.ones(shape)).shape)
    assert actual == expected


@pytest.mark.parametrize(
    "shape, out_channels, kernel_size, stride, dilation, padding",
    [
        ([4, 28, 28], 16, 3, 1, 2, 0),
        ([5, 4], 4, 2, 2, 2, 0),
        ([5, 4], 4, 2, 2, 1, 0),
        ([28, 6], 1, 6, 1, 1, 0),
        ([28, 6], 1, 6, 1, 1, "same"),
        ([28, 6], 1, 6, 1, 1, 4),
    ],
)
def test_calculate_conv1d_output_shape(
    shape, out_channels, kernel_size, stride, dilation, padding
):
    actual = calculate_output_shape(
        shape,
        kernel_size=kernel_size,
        stride=stride,
        out_channels=out_channels,
        dilation=dilation,
        padding=padding,
        layer_type="conv1d",
    )
    layer = nn.Conv1d(
        in_channels=shape[-2],
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
    )
    expected = list(layer(torch.ones(shape)).shape)
    assert actual == expected


@pytest.mark.parametrize(
    "shape, kernel_size, stride, dilation, padding, layer_type, layer",
    [
        ([4, 10, 28, 28], 3, 1, 2, 0, "pool2d", nn.MaxPool2d),
        ([5, 10, 4], 2, 2, 2, 0, "pool2d", nn.MaxPool2d),
        ([5, 10, 4], 2, 2, 1, 0, "pool2d", nn.MaxPool2d),
        ([28, 5, 60], 3, 1, 1, 0, "pool2d", nn.MaxPool2d),
        ([28, 5, 60], 8, 1, 1, 4, "pool2d", nn.MaxPool2d),
        ([4, 28, 28], 3, 1, 2, 0, "pool1d", nn.MaxPool1d),
        ([5, 4], 2, 2, 2, 0, "pool1d", nn.MaxPool1d),
        ([5, 4], 2, 2, 1, 0, "pool1d", nn.MaxPool1d),
        ([28, 6], 6, 1, 1, 0, "pool1d", nn.MaxPool1d),
        ([28, 6], 8, 1, 1, 4, "pool1d", nn.MaxPool1d),
    ],
)
def test_calculate_maxpool_output_shape(
    shape, kernel_size, stride, dilation, padding, layer_type, layer
):
    actual = calculate_output_shape(
        shape,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        layer_type=layer_type,
    )
    inst_layer = layer(
        kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding
    )
    expected = list(inst_layer(torch.ones(shape)).shape)
    assert actual == expected
