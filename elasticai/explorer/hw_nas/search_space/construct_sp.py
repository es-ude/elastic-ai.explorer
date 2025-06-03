import copy
from typing import Optional

import nni
import torch
import yaml
from nni.mutable import Mutable, MutableExpression, ensure_frozen
from nni.nas.nn.pytorch import (
    ModelSpace,
    LayerChoice,
    MutableLinear,
    Repeat,
    MutableConv2d,
)
from torch import nn
from torch.nn import Linear, ModuleList, Conv2d


class SearchSpace(ModelSpace):
    # def tuple_to_tensor(self):

    def __init__(self, parameters: dict):

        super().__init__()
        blocks: list[dict] = parameters["blocks"]
        block_sp = []

        last_out = None
        for block in blocks:
            input_width = parameters["input"] if last_out is None else last_out
            output_width = (
                parameters["output"] if self.is_last_block(block, blocks) else None
            )

            block, last_out = self.build_block(block, input_width, output_width)
            block_sp.append(block)

        self.block_sp = nn.Sequential(*block_sp)

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        x = self.block_sp(x)
        return x

    def get_preceeding_layer_width(self, layers):
        layer_new = []
        for layer in layers:
            layer_new = layer_new + [
                module
                for module in layer.modules()
                if not isinstance(module, nn.Sequential)
                and not isinstance(module, ModuleList)
                and not isinstance(module, Repeat)
            ]
        print(layer_new)
        index = -1
        while not (
            isinstance(layer_new[index], Linear)
            or isinstance(layer_new[index], Conv2d)
            or isinstance(layer_new[index], MutableLinear)
        ):
            index -= 1
        layer = layer_new[index]
        print(layer)
        last_layer = layer.out_features
        print(last_layer)
        return last_layer

    def build_block(
        self,
        block: dict,
        input_width: Optional[int | Mutable],
        output_width: Optional[int] = None,
    ):
        block_name = block["block"]
        # layer_op_mapping = [op_candidates[key] for key in block["op_candidates"]]
        activations: dict = block["linear"]["activation"]
        activation_mappings = [activation_candidates[key] for key in activations]

        depth = block["depth"]
        if isinstance(depth, int):
            max_depth = depth
        else:
            max_depth = depth[1]
            depth = tuple[int, int](depth)

        layers = []

        activation = LayerChoice(
            activation_mappings, label=f"activation_block_{block_name}"
        )

        self.h_l_widths = [input_width] + [
            nni.choice(f"layer_width_{i}_block_{block_name}", block["linear"]["width"])
            for i in range(max_depth)
        ]
        print(self.h_l_widths)
        repeat = Repeat(
            lambda index: nn.Sequential(
                MutableLinear(self.h_l_widths[index], self.h_l_widths[index + 1]),
                activation,
            ),
            depth,
            label=f"depth_block_{block_name}",
        )

        layers.append(repeat)

        #    last_layer = self.h_l_widths[ensure_frozen(repeat.depth_choice)]
        # layers.append(
        #     Repeat(
        #         lambda index: nn.Sequential(
        #             layer_choice(self.h_l_widths[index], self.h_l_widths[index + 1]),
        #             activation,
        #         ),
        #         depth,
        #         label=f"depth_block_{block_name}",
        #     )
        # )

        last_layer = self.get_preceeding_layer_width(layers)
        print(last_layer)
        if output_width is not None:

            layers.append(
                nn.Sequential(MutableLinear(last_layer, output_width), activation)
            )

        return nn.Sequential(*layers), last_layer

    def is_last_block(self, block, blocks):
        return blocks[-1]["block"] == block["block"]


class LinearFlattened(nn.Sequential):

    def __init__(
        self,
        in_feat,
        out_feat,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.lin = MutableLinear(in_feat, out_feat)

    def forward(self, x):
        new_shape = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        x = self.flatten(x)
        # x = x.view(-1, new_shape)

        return self.lin(x)


activation_candidates = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "identity": nn.Identity(),
}

supported_operations = {
    "linear": lambda input_width, output_width: MutableLinear(
        input_width, output_width
    ),
    "conv2d": lambda in_channels, out_channels: MutableConv2d(
        in_channels, out_channels, 3, 2
    ),
}


def build_choices(parameters: dict, layer, block):
    for key in parameters.keys():
        if isinstance(parameters[key], list):
            parameters[key] = nni.choice(
                f"{key}_layer_{layer}_block_{block}", parameters[key]
            )
    return parameters


def build_conv2d(block_name, in_channels, out_channels, kernel_size, stride):
    return MutableConv2d(in_channels, out_channels, kernel_size, stride)


def test_op_candidates():
    block = {
        "op_candidates": ["linear", "conv2d"],
        "linear": {"width": [16, 32]},
    }
    # überall range oder choices

    print(build_choices(block["linear"], 1, 3))


def yml_to_dict(file):

    with open(file) as stream:
        try:
            search_space = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return search_space


def simplify_net(blocks):
    layer_new = []
    for layer in blocks:
        layer_new = layer_new + [
            module
            for module in layer.modules()
            if not isinstance(module, nn.Sequential)
            and not isinstance(module, ModuleList)
            and not isinstance(module, Repeat)
        ]
    return layer_new


def compute_start_view(x: torch.Tensor, blocks):
    first_layer = simplify_net(blocks)[0]
    if isinstance(first_layer, Linear):
        x = x.view(-1, first_layer.in_features)
        print(first_layer.in_features)
    return x


def parse_depth(depth: int | str | list[int], block_id: str):
    if isinstance(depth, int):
        max_depth = depth
    elif isinstance(depth, str):
        max_depth = depth[1]
        depth = nni.choice(
            label=f"depth_block_{block_id}", choices=[d for d in range(*depth)]
        )
    elif isinstance(depth, list):
        max_depth = max(depth)
        depth = nni.choice(label=f"depth_block_{block_id}", choices=depth)
    else:
        raise ValueError("Depth must be int, tuple or list of ints")
    return depth, max_depth


def parse_op_candidates(op_candidates: list[str], block_id, block):
    layer_op_mapping = [supported_operations[key](*block[key]) for key in op_candidates]

    if len(layer_op_mapping) > 1:
        # LayerChoice(layer_op_mapping, label=label = f"Layer_choice_block_{block_id}")
        layer = LayerChoice(layer_op_mapping, label=f"activation_block_{block_id}")


class CombinedSearchSpace(ModelSpace):

    # same for both
    def __init__(self, parameters: dict):
        super().__init__()
        blocks: list[dict] = parameters["blocks"]
        block_sp = []

        last_out = None
        for block in blocks:
            input_width = parameters["input"] if last_out is None else last_out
            output_width = (
                parameters["output"] if self.is_last_block(block, blocks) else None
            )

            block, last_out = self.build_block(block, input_width, output_width)
            block_sp.append(block)

        self.block_sp = nn.Sequential(*block_sp)

    def build_block(
        self,
        block,
        input_width: Optional[int | Mutable],
        output_width: Optional[int] = None,
    ):
        depth, max_depth = parse_depth(block["depth"])
        layer_op_mapping = [supported_operations[key] for key in block["op_candidates"]]

    def build_block(
        self,
        block: dict,
        input_width: Optional[int | Mutable],
        output_width: Optional[int] = None,
    ):
        block_name = block["block"]
        # layer_op_mapping = [op_candidates[key] for key in block["op_candidates"]]
        activations: dict = block["linear"]["activation"]
        activation_mappings = [activation_candidates[key] for key in activations]

        depth = block["depth"]
        if isinstance(depth, int):
            max_depth = depth
        else:
            max_depth = depth[1]
            depth = tuple[int, int](depth)

        layers = []

        activation = LayerChoice(
            activation_mappings, label=f"activation_block_{block_name}"
        )

        self.h_l_widths = [input_width] + [
            nni.choice(f"layer_width_{i}_block_{block_name}", block["linear"]["width"])
            for i in range(max_depth)
        ]
        print(self.h_l_widths)
        repeat = Repeat(
            lambda index: nn.Sequential(
                MutableLinear(self.h_l_widths[index], self.h_l_widths[index + 1]),
                activation,
            ),
            depth,
            label=f"depth_block_{block_name}",
        )

        layers.append(repeat)

        #    last_layer = self.h_l_widths[ensure_frozen(repeat.depth_choice)]
        # layers.append(
        #     Repeat(
        #         lambda index: nn.Sequential(
        #             layer_choice(self.h_l_widths[index], self.h_l_widths[index + 1]),
        #             activation,
        #         ),
        #         depth,
        #         label=f"depth_block_{block_name}",
        #     )
        # )

        last_layer = self.get_preceeding_layer_width(layers)
        print(last_layer)
        if output_width is not None:

            layers.append(
                nn.Sequential(MutableLinear(last_layer, output_width), activation)
            )

        return nn.Sequential(*layers), last_layer

    def forward(self, x):
        x = compute_start_view(x, self.block_sp)
        x = self.block_sp(x)
        return x


class CNNSpace(ModelSpace):

    def __init__(self, parameters: dict):
        super().__init__()
        blocks: list[dict] = parameters["blocks"]
        block_sp = []

        last_out = None
        for block in blocks:
            input_width = parameters["input"] if last_out is None else last_out
            output_width = (
                parameters["output"] if self.is_last_block(block, blocks) else None
            )

            block, last_out = self.build_block(block, input_width, output_width)
            block_sp.append(block)
        self.block_sp = nn.Sequential(*block_sp)

    def forward(self, x):
        #  x = x.view(-1, 28 * 28)
        x = self.block_sp(x)
        return x

    def is_last_block(self, block, blocks):
        return blocks[-1]["block"] == block["block"]

    def build_block(
        self,
        block: dict,
        input_width: Optional[int | tuple | Mutable],
        output_width: Optional[int] = None,
    ):
        layers = []
        depth, max_depth = parse_depth(block["depth"], block["block"])

        block_name = "herro"
        self.out_channels = [1] + [
            nni.choice(
                label=f"out_channel_nr_{i}_block_{block_name}",
                choices=block["conv2D"]["out_channels"],
            )
            for i in range(max_depth)
        ]
        print(self.out_channels)

        kernel_size = block["conv2D"]["kernel_size"]
        if isinstance(kernel_size, list):
            self.kernel_size = nni.choice(
                label=f"kernel_size_block_{block_name}", choices=kernel_size
            )
        else:
            self.kernel_size = kernel_size
        stride = block["conv2D"]["stride"]
        if isinstance(stride, list):
            self.stride = nni.choice(label=f"stride_block_{block_name}", choices=stride)

        else:
            self.stride = stride

        layers.append(
            Repeat(
                lambda index: nn.Sequential(
                    MutableConv2d(
                        self.out_channels[index],
                        self.out_channels[index + 1],
                        self.kernel_size,
                        self.stride,
                    )
                ),
                depth,
            )
        )
        x_shape = [1, 28, 28]
        self.output_shapes = []
        for i in range(max_depth):
            x_shape = calculate_conv_output_shape(
                x_shape,
                self.out_channels[i + 1],
                kernel_size=self.kernel_size,
                stride=self.stride,
            )
            self.output_shapes.append(x_shape)
        print(len(self.output_shapes))
        #
        # print(x_shape)
        #
        # x_shape = calculate_conv_output_shape(
        #     x_shape,
        #     self.out_channels[2],
        #     kernel_size=self.kernel_size,
        #     stride=self.stride,
        # )
        # print(x_shape)
        #
        # print(x_shape)
        print(ensure_frozen(depth))

        x_shape = self.output_shapes[ensure_frozen(depth) - 1]
        input_width = x_shape[0] * x_shape[1] * x_shape[2]
        print(input_width)
        if output_width is not None:
            # layers.append(MutableLinear(self.out_channels[-1], 10))

            layers.append(LinearFlattened(input_width, 10))
        return nn.Sequential(*layers), None


def calculate_conv_output_shape(
    shape,
    out_channels: int | MutableExpression[int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = (1, 1),
    dilation: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, int] = (0, 0),
):
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


if __name__ == "__main__":
    x = torch.randn(5, 1, 28, 28)
    search_space = yml_to_dict("search_space.yml")

    search_space = CNNSpace(search_space)
    print(search_space(x).shape)
