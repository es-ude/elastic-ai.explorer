import copy
from pathlib import Path

import yaml


def calculate_conv_output_shape(
    shape,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, int] = (0, 0),
):
    print(shape)
    kernel_size, stride, dilation, padding = _convert_to_tuples(
        [kernel_size, stride, dilation, padding]
    )
    new_shape = copy.deepcopy(shape)
    new_shape[-3] = out_channels
    new_shape[-2] = (
        shape[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    new_shape[-1] = (
        shape[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1
    return new_shape


def _convert_to_tuples(values) -> list[tuple[int, int]]:
    return [x if isinstance(x, tuple) else (x, x) for x in values]


def yaml_to_dict(file_path: Path) -> dict:

    with open(file_path) as stream:
        search_space = yaml.safe_load(stream)
        return search_space
