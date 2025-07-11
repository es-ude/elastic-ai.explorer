import math
import torch
from optuna import trial, Trial
from optuna.trial import FixedTrial
from torch import nn

from elasticai.explorer.hw_nas.search_space.utils import calculate_conv_output_shape, yml_to_dict

activation_mapping={"relu": nn.ReLU(),
                    "sigmoid": nn.Sigmoid()}

class SearchSpace:
    def __init__(self, search_space_cfg: dict):
        self.search_space_cfg = search_space_cfg
        self.input_shape = search_space_cfg["input"]
        self.output_shape = search_space_cfg["output"]
        self.blocks: list[dict] = search_space_cfg["blocks"]
        self.layers = []

    def createLinear(self, trial, block, num_layers, search_params):
        block_id = block["block"]
        if isinstance(self.input_shape, list):
            self.layers.append(nn.Flatten())
            self.input_shape = math.prod(self.input_shape)
        for i in range(num_layers):
            layer_width = trial.suggest_categorical("layer_width_b{}_l{}".format(block_id, i), search_params["width"])
            activation = trial.suggest_categorical("activation_func_b{}_l{}".format(block_id, i), block["activation"])
            self.layers.append(nn.Linear(self.input_shape, layer_width))
            self.layers.append(activation_mapping[activation])
            self.input_shape = layer_width

    def createConv2d(self, trial, block, num_layers, search_params):
        block_id = block["block"]
        for i in range(num_layers):
            out_channels = trial.suggest_categorical("out_channels_b{}_l{}".format(block_id, i),
                                                     search_params["out_channels"])
            kernel_size = trial.suggest_categorical("kernel_size_b{}_l{}".format(block_id, i),
                                                    search_params["kernel_size"])
            stride = trial.suggest_categorical("stride_b{}_l{}".format(block_id, i),
                                               search_params["stride"])
            activation = trial.suggest_categorical("activation_func_b{}_l{}".format(block_id, i), block["activation"])
            self.layers.append(nn.Conv2d(self.input_shape[0], out_channels, kernel_size, stride))
            self.layers.append(activation_mapping[activation])
            self.input_shape = calculate_conv_output_shape(self.input_shape, out_channels, kernel_size, stride)

    def create_block(self, trial, block: dict):
        operation_candidates = block["op_candidates"]
        num_layers = trial.suggest_categorical("num_layers_b{}".format(block["block"]), block["depth"])
        operation = trial.suggest_categorical("operation_b{}".format(block["block"]), operation_candidates)
        match operation:
            case "linear":
                self.createLinear(trial, block, num_layers, block["linear"])
            case "conv2d":
                self.createConv2d(trial, block, num_layers, block["conv2D"])

    def create_model_sample(self, trial):
        self.input_shape = self.search_space_cfg["input"]
        for block in self.blocks:
            self.create_block(trial, block)
        return nn.Sequential(*self.layers)





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








def objective(trial):
    search_space = yml_to_dict("search_space.yml")
    search_space = SearchSpace(search_space)
    return search_space.create_model_sample(trial)

if __name__ == "__main__":
    search_space= yml_to_dict("search_space.yml")
    search_space=SearchSpace(search_space)
    sample={"num_layers_b1": 2,"num_layers_b2": 1, "operation_b1":"conv2d","operation_b2":"linear","layer_width_b2_l0": 21,"out_channels_b1_l0": 4,"out_channels_b1_l1": 10,"stride_b1_l0": 1,"stride_b1_l1": 1,"kernel_size_b1_l0": 2, "kernel_size_b1_l1": 2,"activation_func_b1_l0": "relu","activation_func_b1_l1": "relu" ,"activation_func_b2_l0": "sigmoid"  }
  #  sample={"num_layers": 2,"layer_op_l1":"linear","layer_op_l0":"conv2d","layer_width_l1": 128,"out_channels_l0": 16,"stride_l0": 1,"kernel_size_l0": 2, "activation_func_l0": "relu", "activation_func_l1": "sigmoid" }
    model=objective(FixedTrial(sample))
    print(model)
    test_sample= torch.ones(4, 1, 28, 28)
    print(model(test_sample))



