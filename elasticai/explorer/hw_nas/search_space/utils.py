import math
from pathlib import Path

import yaml

import copy
from typing import Sequence, Literal


def _to_tuple(value, dims: int) -> tuple[int, ...]:
    """Ensure a value is a tuple of length `dims`."""
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,) * dims



def _conv_output_dim(in_dim, kernel, stride, padding, dilation):
    """Compute the output dimension for one spatial axis."""
    return (in_dim + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def calculate_output_shape(
    shape: Sequence[int],
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int],
    padding: int | Sequence[int] | Literal["same"]= 0,
    dilation: int | Sequence[int] = 1,
    out_channels: int | None = None,
    layer_type: str = "conv2d",
):

    assert layer_type in {"conv1d", "conv2d", "pool1d", "pool2d"}, \
        f"Invalid layer type: {layer_type}"

    dims = 1 if "1d" in layer_type else 2


    kernel_size = _to_tuple(kernel_size, dims)
    stride = _to_tuple(stride, dims)

    dilation = _to_tuple(dilation, dims)
    new_shape = list(copy.deepcopy(shape))

    if "conv" in layer_type and out_channels is not None:
        new_shape[-(dims + 1)] = out_channels

    spatial_in = shape[-dims:]

    if padding == "same":
        new_spatial = [
            math.ceil(i / s)
            for i, s in zip(spatial_in, stride)
        ]

    else:
        padding = _to_tuple(padding, dims)
        new_spatial = [
            _conv_output_dim(i, k, s, p, d)
            for i, k, s, p, d in zip(spatial_in, kernel_size, stride, padding, dilation)
        ]

    new_shape[-dims:] = new_spatial
    return new_shape

def yaml_to_dict(file_path: Path) -> dict:

    with open(file_path) as stream:
        search_space = yaml.safe_load(stream)
        return search_space
