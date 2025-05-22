from typing import Optional, Callable

import nni
import yaml
from build.lib.nni.nas.nn.pytorch import MutableConv2d
from nni.mutable import Mutable
from nni.nas.nn.pytorch import ModelSpace, LayerChoice, MutableLinear, Repeat
from torch import nn
from torch.nn import Linear, ModuleList


class LinearActivation(nn.Sequential):

    def __init__(self, in_feat, out_feat, activation: Callable[..., nn.Module]):
        super().__init__(MutableLinear(in_feat, out_feat), activation())


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


class SearchSpace(ModelSpace):

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

        index = -1
        while not isinstance(layer_new[index], Linear):
            index -= 1
        layer = layer_new[index]
        last_layer = layer.out_features
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
            nni.choice(f"layer_width_{i}_block_{block_name}", block["width"])
            for i in range(max_depth)
        ]

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

        last_layer = self.get_preceeding_layer_width(layers)
        if output_width is not None:

            layers.append(
                nn.Sequential(MutableLinear(last_layer, output_width), activation)
            )

            return nn.Sequential(*layers), None

        return nn.Sequential(*layers), last_layer

    def is_last_block(self, block, blocks):
        return blocks[-1]["block"] == block["block"]

    def __init__(self, parameters: dict):
        in1 = nni.choice("i", [32, 16])
        out2 = nni.choice("o", [15, 11])
        layer_choice = LayerChoice(
            [MutableLinear(in1, out2), MutableConv2d(in1, out2, 3, 1)], label="l0"
        )
        print(layer_choice)
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


def yml_to_dict(file):
    with open(file) as stream:
        try:
            search_space = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return search_space


if __name__ == "__main__":
    search_space = yml_to_dict("search_space.yml")
    search_space = SearchSpace(search_space)
    # print(search_space)
# test_op_candidates()
