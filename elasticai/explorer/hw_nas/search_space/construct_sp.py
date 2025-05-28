from math import floor
from typing import Optional

import nni
import yaml
from nni.mutable import Mutable
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
            isinstance(layer_new[index], Linear) or isinstance(layer_new[index], Conv2d)
        ):
            index -= 1
        layer = layer_new[index]
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
        layer_op_mapping = [op_candidates[key] for key in block["op_candidates"]]
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

        #   self.h_l_widths = [input_width] + [20 for i in range(max_depth)]
        layers.append(
            Repeat(
                lambda index: nn.Sequential(
                    MutableLinear(self.h_l_widths[index], self.h_l_widths[index + 1]),
                    activation,
                ),
                depth,
                label=f"depth_block_{block_name}",
            )
        )

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

op_candidates = {
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


def parse_block_op_candidates(block):
    ops = block["op_candidates"]
    mapping = [op_candidates[key] for key in ops]
    print(mapping)


def yml_to_dict(file):
    with open(file) as stream:
        try:
            search_space = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return search_space


def calc_shape(channel, prev_width, prev_height, kernel, stride):
    return (
        channel,
        calculate_2d_conv_pool_shape(prev_width, kernel, stride),
        calculate_2d_conv_pool_shape(prev_height, kernel, stride),
    )


def calculate_2d_conv_pool_shape(
    prev_shape, kernel_size, stride=1, dilation=1, padding=0
):
    return floor(
        ((prev_shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    )


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
        depth = 2
        max_depth = 2
        block_name = "herro"
        self.out_channels = [1] + [
            nni.choice(
                label=f"out_channel_nr_{i}_block_{block_name}",
                choices=block["conv2D"]["out_channels"],
            )
            for i in range(max_depth)
        ]
        print(self.out_channels)
        # layers.append(
        #     Repeat(
        #         lambda index: nn.Sequential(
        #             MutableLinear(
        #                 self.out_channels[index], self.out_channels[index + 1]
        #             )
        #         ),
        #         depth,
        #         label=f"depth_block_{block_name}",
        #     )
        # )
        layers.append(
            Repeat(
                lambda index: nn.Sequential(
                    MutableConv2d(
                        self.out_channels[index],
                        self.out_channels[index + 1],
                        block["conv2D"]["kernel_size"],
                        block["conv2D"]["stride"],
                    )
                ),
                depth,
                label=f"depth_block_{block_name}",
            )
        )
        kernel_size = [5, 5]
        x_shape = self.calc_shape_test(
            [1, 28, 28], self.out_channels[1], kernel_size=kernel_size, stride=[1, 1]
        )
        print(x_shape)

        x_shape = self.calc_shape_test(
            x_shape, self.out_channels[2], kernel_size=kernel_size, stride=[1, 1]
        )
        print(x_shape)

        print(x_shape)
        input_width = x_shape[0] * x_shape[1] * x_shape[2]
        print(input_width)
        if output_width is not None:
            # layers.append(MutableLinear(self.out_channels[-1], 10))

            layers.append(LinearFlattened(input_width, 10))
        return nn.Sequential(*layers), None

    def calc_shape_test(
        self,
        shape,
        out_channels,
        kernel_size=[5, 5],
        stride=[2, 2],
        dilation=[1, 1],
        padding=(0, 0),
    ):
        shape[-3] = out_channels
        # H_out and W_out
        shape[-2] = (
            shape[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
        ) // stride[0] + 1
        shape[-1] = (
            shape[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
        ) // stride[1] + 1
        return shape


if __name__ == "__main__":
    search_space = yml_to_dict("search_space.yml")
    search_space = SearchSpace(search_space)
    print(search_space)
