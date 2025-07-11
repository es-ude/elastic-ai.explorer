import copy
import math
from typing import Optional, Iterable

import optuna
import torch
import yaml
from optuna import trial, Trial
from optuna.trial import FixedTrial
from torch import nn


#
# import nni
# import torch
# import yaml
# from nni.mutable import (
#     Mutable,
#     MutableExpression,
#     ensure_frozen,
#     Sample,
#     label_scope,
# )
# from nni.nas.nn.pytorch import (
#     ModelSpace,
#     LayerChoice,
#     MutableLinear,
#     Repeat,
#     MutableConv2d,
# )
# from nni.nas.space import model_context
# from optuna.trial import FixedTrial
# from torch import nn
# from torch.nn import Linear, ModuleList, Conv2d
#
# from elasticai.explorer.hw_nas.search_space.operations import BlockFactory, LinearBlock
#
#
# class SearchSpace(ModelSpace):
#     # def tuple_to_tensor(self):
#
#     def __init__(self, parameters: dict):
#
#         super().__init__()
#         blocks: list[dict] = parameters["blocks"]
#         block_sp = []
#
#         last_out = None
#         for block in blocks:
#             input_width = parameters["input"] if last_out is None else last_out
#             output_width = (
#                 parameters["output"] if self.is_last_block(block, blocks) else None
#             )
#
#             block, last_out = self.build_block(block, input_width, output_width)
#             block_sp.append(block)
#
#         self.block_sp = nn.Sequential(*block_sp)
#
#     def forward(self, x):
#         x = x.view(-1, 28 * 28)
#
#         x = self.block_sp(x)
#         return x
#
#     def get_preceeding_layer_width(self, layers):
#         layer_new = []
#         for layer in layers:
#             layer_new = layer_new + [
#                 module
#                 for module in layer.modules()
#                 if not isinstance(module, nn.Sequential)
#                 and not isinstance(module, ModuleList)
#                 and not isinstance(module, Repeat)
#             ]
#
#         index = -1
#         while not (
#             isinstance(layer_new[index], Linear)
#             or isinstance(layer_new[index], Conv2d)
#             or isinstance(layer_new[index], MutableLinear)
#         ):
#             index -= 1
#         layer = layer_new[index]
#
#         last_layer = layer.out_features
#
#         return last_layer
#
#     def build_block(
#         self,
#         block: dict,
#         input_width: Optional[int | Mutable],
#         output_width: Optional[int] = None,
#     ):
#         block_name = block["block"]
#         # layer_op_mapping = [op_candidates[key] for key in block["op_candidates"]]
#         activations: dict = block["linear"]["activation"]
#         activation_mappings = [activation_candidates[key] for key in activations]
#
#         depth = block["depth"]
#         if isinstance(depth, int):
#             max_depth = depth
#         else:
#             max_depth = depth[1]
#             depth = tuple[int, int](depth)
#
#         layers = []
#
#         activation = LayerChoice(
#             activation_mappings, label=f"activation_block_{block_name}"
#         )
#
#         self.h_l_widths = [input_width] + [
#             nni.choice(f"layer_width_{i}_block_{block_name}", block["linear"]["width"])
#             for i in range(max_depth)
#         ]
#         print(self.h_l_widths)
#         repeat = Repeat(
#             lambda index: nn.Sequential(
#                 MutableLinear(self.h_l_widths[index], self.h_l_widths[index + 1]),
#                 activation,
#             ),
#             depth,
#             label=f"depth_block_{block_name}",
#         )
#
#         layers.append(repeat)
#
#         #    last_layer = self.h_l_widths[ensure_frozen(repeat.depth_choice)]
#         # layers.append(
#         #     Repeat(
#         #         lambda index: nn.Sequential(
#         #             layer_choice(self.h_l_widths[index], self.h_l_widths[index + 1]),
#         #             activation,
#         #         ),
#         #         depth,
#         #         label=f"depth_block_{block_name}",
#         #     )
#         # )
#
#         last_layer = self.get_preceeding_layer_width(layers)
#         if output_width is not None:
#
#             layers.append(
#                 nn.Sequential(MutableLinear(last_layer, output_width), activation)
#             )
#
#         return nn.Sequential(*layers), last_layer
#
#     def is_last_block(self, block, blocks):
#         return blocks[-1]["block"] == block["block"]
#
#
# class LinearFlattened(nn.Sequential):
#
#     def __init__(
#         self,
#         in_feat,
#         out_feat,
#     ):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.lin = MutableLinear(in_feat, out_feat)
#
#     def forward(self, x):
#         # new_shape = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
#         x = self.flatten(x)
#         # x = x.view(-1, new_shape)
#
#         return self.lin(x)
#
#
# activation_candidates = {
#     "relu": nn.ReLU(),
#     "sigmoid": nn.Sigmoid(),
#     "identity": nn.Identity(),
# }
#
# supported_operations = {
#     "linear": lambda input_width, output_width: MutableLinear(
#         input_width, output_width
#     ),
#     "conv2d": lambda in_channels, out_channels: MutableConv2d(
#         in_channels, out_channels, 3, 2
#     ),
# }
#
#
# def build_choices(parameters: dict, layer, block):
#     for key in parameters.keys():
#         if isinstance(parameters[key], list):
#             parameters[key] = nni.choice(
#                 f"{key}_layer_{layer}_block_{block}", parameters[key]
#             )
#     return parameters
#
#
# def build_conv2d(block_name, in_channels, out_channels, kernel_size, stride):
#     return MutableConv2d(in_channels, out_channels, kernel_size, stride)
#
#
# def test_op_candidates():
#     block = {
#         "op_candidates": ["linear", "conv2d"],
#         "linear": {"width": [16, 32]},
#     }
#     # überall range oder choices
#
#     print(build_choices(block["linear"], 1, 3))
#
#
def yml_to_dict(file):

    with open(file) as stream:
        try:
            search_space = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return search_space

#
# def parse_depth(depth: int | str | list[int]):
#     if isinstance(depth, int):
#         max_depth = depth
#     elif isinstance(depth, str):
#         max_depth = depth[1]
#         depth = nni.choice(label=f"depth", choices=[d for d in range(*depth)])
#     elif isinstance(depth, list):
#         max_depth = max(depth)
#         depth = nni.choice(label=f"depth", choices=depth)
#     else:
#         raise ValueError("Depth must be int, tuple or list of ints")
#     return depth, max_depth
#
#
# def parse_op_candidates(op_candidates: list[str], block_id, block):
#     layer_op_mapping = [supported_operations[key](*block[key]) for key in op_candidates]
#
#     if len(layer_op_mapping) > 1:
#         # LayerChoice(layer_op_mapping, label=label = f"Layer_choice_block_{block_id}")
#         layer = LayerChoice(layer_op_mapping, label=f"activation_block_{block_id}")
#
#
# def simplify_net(blocks):
#     layer_new = []
#     for layer in blocks:
#         layer_new = layer_new + [
#             module
#             for module in layer.modules()
#             if not isinstance(module, nn.Sequential)
#             and not isinstance(module, ModuleList)
#             and not isinstance(module, Repeat)
#         ]
#     return layer_new
#
#
# class CombinedSearchSpace(ModelSpace):
#
#     def is_last_block(self, block, blocks):
#         return blocks[-1]["block"] == block["block"]
#
#     # same for both
#     def __init__(self, parameters: dict) -> None:
#
#         super().__init__()
#         self.params = parameters
#         blocks: list[dict] = parameters["blocks"]
#         block_sp = []
#
#         last_out = None
#         for block in blocks:
#
#             input_width = parameters["input"] if last_out is None else last_out
#             print(input_width)
#             output_width = (
#                 parameters["output"] if self.is_last_block(block, blocks) else None
#             )
#             block_id = block["block"]
#             with label_scope(f"block_{block_id}"):
#                 block, last_out = self.build_block(block, input_width, output_width)
#                 block_sp.append(block)
#
#         self.block_sp = nn.Sequential(*block_sp)
#
#     def build_conv2d_block(
#         self, input_width, output_width, max_depth, block, activation
#     ):
#
#         layers = []
#
#         layers.append(
#             Repeat(
#                 lambda index: nn.Sequential(
#                     MutableConv2d(
#                         self.out_channels[index],
#                         self.out_channels[index + 1],
#                         self.kernel_size,
#                         self.stride,
#                     )
#                 ),
#                 self.depth,
#             )
#         )
#         print(f"input_width: {input_width}")
#         x_shape = input_width
#         self.output_shapes = []
#         for i in range(max_depth):
#             x_shape = calculate_conv_output_shape(
#                 x_shape,
#                 self.out_channels[i + 1],
#                 kernel_size=self.kernel_size,
#                 stride=self.stride,
#             )
#             self.output_shapes.append(x_shape)
#
#         x_shape = self.output_shapes[ensure_frozen(self.depth) - 1]
#         input_width = x_shape[0] * x_shape[1] * x_shape[2]
#         print(input_width)
#         if output_width is not None:
#             layers.append(LinearFlattened(input_width, 10))
#         return nn.Sequential(*layers), x_shape
#
#     def build_linear_block(
#         self,
#         input_width,
#         output_width,
#         max_depth,
#         block,
#         activation,
#     ):
#         layers = []
#         flattened_input = (
#             math.prod(input_width) if isinstance(input_width, Iterable) else input_width
#         )
#         print(flattened_input)
#         self.h_l_widths = [flattened_input] + [
#             nni.choice(f"layer_width_{i}", block["linear"]["width"])
#             for i in range(max_depth)
#         ]
#         for width in self.h_l_widths[1:]:
#             if isinstance(width, Mutable):
#                 self.add_mutable(width)
#
#         repeat = Repeat(
#             lambda index: (
#                 LinearFlattened(self.h_l_widths[index], self.h_l_widths[index + 1])
#                 if index == 0
#                 else nn.Sequential(
#                     MutableLinear(self.h_l_widths[index], self.h_l_widths[index + 1]),
#                     #    LayerChoice(activation, label=f"activation_block_{block_id}"),
#                 )
#             ),
#             self.depth,
#         )
#
#         layers.append(repeat)
#         last_layer_width = self.h_l_widths[ensure_frozen(self.depth)]
#         if output_width is not None:
#             layers.append(
#                 nn.Sequential(
#                     MutableLinear(last_layer_width, output_width),
#                     #     LayerChoice(activation, label=f"activation_block_{block_id}"),
#                 )
#             )
#
#         return nn.Sequential(*layers), last_layer_width
#
#     def build_block(self, block, input_width, output_width=None):
#
#         # input_width = [1, 28, 28]
#         # output_width = 10
#
#         self.depth, max_depth = parse_depth(block["depth"])
#         #    depth, max_depth = parse_depth(block["depth"], block_id)
#         if "conv2d" in block["op_candidates"]:
#             # print(input_width[0])
#             print(ensure_frozen(f"conv2d input width {input_width}"))
#             if isinstance(input_width, int):
#                 first_channel = input_width
#             else:
#                 first_channel = input_width[0]
#             self.out_channels = [first_channel] + [
#                 nni.choice(
#                     label=f"out_channels_{i}",
#                     choices=block["conv2D"]["out_channels"],
#                 )
#                 for i in range(max_depth)
#             ]
#             print("out_channels: ")
#             for width in self.out_channels[1:]:
#                 if isinstance(width, Mutable):
#                     print(width)
#                     self.add_mutable(width)
#             kernel_size = block["conv2D"]["kernel_size"]
#             if isinstance(kernel_size, list):
#                 self.kernel_size = nni.choice(label=f"kernel_size", choices=kernel_size)
#             else:
#                 self.kernel_size = kernel_size
#             stride = block["conv2D"]["stride"]
#             if isinstance(stride, list):
#                 self.stride = nni.choice(label=f"stride", choices=stride)
#
#             else:
#                 self.stride = stride
#             if isinstance(self.stride, Mutable):
#                 self.add_mutable(self.stride)
#                 if isinstance(self.kernel_size, Mutable):
#                     self.add_mutable(self.kernel_size)
#         # if "linear" in block["op_candidates"]:
#         #     flattened_input = (
#         #         math.prod(input_width)
#         #         if isinstance(input_width, Iterable)
#         #         else input_width
#         #     )
#         #     print(flattened_input)
#         #     self.h_l_widths = [flattened_input] + [
#         #         nni.choice(f"layer_width_{i}", block["linear"]["width"])
#         #         for i in range(max_depth)
#         #     ]
#         #     for width in self.h_l_widths[1:]:
#         #         if isinstance(width, Mutable):
#         #             self.add_mutable(width)
#         # activations: dict = block["linear"]["activation"]
#         # activation_mappings = [activation_candidates[key] for key in activations]
#
#         # if isinstance(activation, Mutable):
#         #     self.add_mutable(activation)
#         self.candidate_op = nni.choice(
#             label=f"candidate_op",
#             choices=block["op_candidates"],
#         )
#         if isinstance(self.candidate_op, Mutable):
#             self.add_mutable(self.candidate_op)
#
#         if ensure_frozen(self.candidate_op) == "conv2d":
#             return self.build_conv2d_block(
#                 input_width,
#                 output_width,
#                 max_depth,
#                 block,
#                 "activation_mappings",
#             )
#         elif ensure_frozen(self.candidate_op) == "linear":
#             linearb = LinearBlock(
#                 input_width, output_width, block, self.depth, max_depth
#             )
#             return linearb, linearb.output_shape
#             # return self.build_linear_block(
#             #     input_width,
#             #     output_width,
#             #     max_depth,
#             #     block,
#             #     "activation_mappings",
#             # )
#         return None
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.block_sp(x)
#         return x
#
#     def freeze(self, sample: Sample):
#         with model_context(sample):
#             comb_sp = CombinedSearchSpace(self.params)
#             print(comb_sp)
#             return comb_sp


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
    print(new_shape)
    return new_shape



activation_mapping={"relu": nn.ReLU(), "sigmoid": nn.Sigmoid()}


def map_layer_op(trial:Trial, layer_no, layer_op, in_size, op_parameter: dict):
    print(in_size)
    match layer_op:
        case "linear":
            layer=[]
            if isinstance(in_size, list):
                layer.append(nn.Flatten())
                in_size = math.prod(in_size)
                print(in_size)
            layer_width=trial.suggest_categorical("layer_width_l{}".format(layer_no), op_parameter["linear"]["width"])
            layer.append(nn.Linear(in_size,layer_width))
            return nn.Sequential(*layer), layer_width
        case "conv2d":
            out_channels=trial.suggest_categorical("out_channels_l{}".format(layer_no), op_parameter["conv2d"]["out_channels"])
            kernel_size = trial.suggest_categorical("kernel_size_l{}".format(layer_no),
                                                     op_parameter["conv2d"]["kernel_size"])
            stride= trial.suggest_categorical("stride_l{}".format(layer_no),
                                                     op_parameter["conv2d"]["stride"])
            return nn.Conv2d(in_size[0], out_channels, kernel_size, stride), calculate_conv_output_shape(in_size, out_channels, kernel_size, stride)

def _convert_to_tuples(values) -> list[tuple[int, int]]:
    return [x if isinstance(x, tuple) else (x, x) for x in values]


def create_model(trial):
    num_layers=trial.suggest_int("num_layers", 1, 3)
    in_size: int | list[int]=[1, 28, 28]
    layers=[]
    params= {"linear": {"width": [4, 16, 128]}, "conv2d":{"out_channels":[6, 10, 16], "kernel_size":[2, 3],  "stride":[1, 2]}}
    for i in range(num_layers):
      #  layer_width= trial.suggest_categorical("layer_width_l{}".format(i),[4, 16,128])
        layer_op= trial.suggest_categorical("layer_op_l{}".format(i), ["linear", "conv2d"])
        activation= trial.suggest_categorical("activation_func_l{}".format(i), ["relu","sigmoid"])
        layer_op, in_size=map_layer_op(trial, i, layer_op, in_size,params)
        layers.append(layer_op)
        #layers.append(nn.Linear(in_size, layer_width))
        layers.append(activation_mapping[activation])
        #in_size= layer_width
    if isinstance(in_size, list):
        layers.append(nn.Flatten())
        in_size=math.prod(in_size)
    layers.append(nn.Linear(in_size, 10))
    return nn.Sequential(*layers)


 # super().__init__()
#         self.params = parameters
#         blocks: list[dict] = parameters["blocks"]
#         block_sp = []
#
#         last_out = None
#         for block in blocks:
#
#             input_width = parameters["input"] if last_out is None else last_out
#             print(input_width)
#             output_width = (
#                 parameters["output"] if self.is_last_block(block, blocks) else None
#             )
#             block_id = block["block"]
#             with label_scope(f"block_{block_id}"):
#                 block, last_out = self.build_block(block, input_width, output_width)
#                 block_sp.append(block)
#
#         self.block_sp = nn.Sequential(*block_sp)


class OptunaSearchSpace:
    def __init__(self, search_space_cfg: dict):
        self.input_shape= search_space_cfg["input"]
        self.output_shape= search_space_cfg["output"]
        self.blocks: list[dict]= search_space_cfg["blocks"]
        self.layers=[]
    # def map_block_operation(self, operation:str):
    #     match operation:
    #         case "linear":
    #             layer = []
    #             if isinstance(in_size, list):
    #                 layer.append(nn.Flatten())
    #                 in_size = math.prod(in_size)
    #                 print(in_size)
    #             layer_width = trial.suggest_categorical("layer_width_l{}".format(layer_no),
    #                                                     op_parameter["linear"]["width"])
    #             layer.append(nn.Linear(in_size, layer_width))
    #             return nn.Sequential(*layer), layer_width
    #         case "conv2d":
    #             out_channels = trial.suggest_categorical("out_channels_l{}".format(layer_no),
    #                                                      op_parameter["conv2d"]["out_channels"])
    #             kernel_size = trial.suggest_categorical("kernel_size_l{}".format(layer_no),
    #                                                     op_parameter["conv2d"]["kernel_size"])
    #             stride = trial.suggest_categorical("stride_l{}".format(layer_no),
    #                                                op_parameter["conv2d"]["stride"])
    #             return nn.Conv2d(in_size[0], out_channels, kernel_size, stride), calculate_conv_output_shape(in_size,
    #                                                                                                          out_channels,
    #                                                                                                          kernel_size,
    #                                                                                                          stride)
    #


    def createLinear(self, trial, block ,num_layers, search_params):
        block_id=block["block"]
        if isinstance(self.input_shape, list):
            self.layers.append(nn.Flatten())
            self.input_shape = math.prod(self.input_shape)
        for i in range(num_layers):

            layer_width= trial.suggest_categorical("layer_width_b{}_l{}".format(block_id, i),search_params["width"])
            activation= trial.suggest_categorical("activation_func_b{}_l{}".format(block_id,i), block["activation"])
            self.layers.append(nn.Linear(self.input_shape, layer_width))
            self.layers.append(activation_mapping[activation])
            self.input_shape=layer_width
    def createConv2d(self, trial, block, num_layers, search_params):
        block_id = block["block"]
        for i in range(num_layers):
            out_channels = trial.suggest_categorical("out_channels_b{}_l{}".format(block_id, i),
                                                     search_params["out_channels"])
            kernel_size = trial.suggest_categorical("kernel_size_b{}_l{}".format(block_id, i),
                                                    search_params["kernel_size"])
            stride = trial.suggest_categorical("stride_l{}".format(block, i),
                                               search_params["stride"])
            activation = trial.suggest_categorical("activation_func_b{}_l{}".format(block_id, i), block["activation"])
            self.layers.append(nn.Conv2d(self.input_shape[0], out_channels, kernel_size, stride))
            self.layers.append(activation_mapping[activation])
            self.input_shape = calculate_conv_output_shape(self.input_shape,out_channels,kernel_size,stride)
    def create_block(self, trial, block: dict):
        operation_candidates=block["op_candidates"]

        num_layers = trial.suggest_categorical("depth_b{}".format(block["block"]), block["depth"])
        match operation_candidates:
            case "linear":
                self.createLinear(trial, block,num_layers,block["linear"])
            case "conv2d":
                self.createConv2d(trial, block, num_layers, block["linear"])



    def create_model_sample(self, trial):
        for block in self.blocks:
            self.create_block(trial, block)



search_space= yml_to_dict("search_space.yml")
search_space= OptunaSearchSpace(search_space)

def objective(trial):
    return search_space.create_model_sample(trial)

if __name__ == "__main__":
    search_space= yml_to_dict("search_space.yml")
    search_space=OptunaSearchSpace(search_space)
    sample={"num_layers": 2,"layer_op_l1":"linear","layer_op_l0":"conv2d","layer_width_l1": 128,"out_channels_l0": 16,"stride_l0": 1,"kernel_size_l0": 2, "activation_func_l0": "relu", "activation_func_l1": "sigmoid"  }
  #  sample={"num_layers": 2,"layer_op_l1":"linear","layer_op_l0":"conv2d","layer_width_l1": 128,"out_channels_l0": 16,"stride_l0": 1,"kernel_size_l0": 2, "activation_func_l0": "relu", "activation_func_l1": "sigmoid" }
    model=objective(FixedTrial(sample))
    print(model)
    test_sample= torch.ones(4, 1, 28, 28)
    print(model(test_sample))



    # x = torch.randn(5, 1, 28, 28)
    # search_space = yml_to_dict("search_space.yml")
    #
    # search_space = CombinedSearchSpace(search_space)

    # print(search_space)
    # search_space.freeze({"op_candidates": 0, depth})
    # print(search_space(x).shape)
