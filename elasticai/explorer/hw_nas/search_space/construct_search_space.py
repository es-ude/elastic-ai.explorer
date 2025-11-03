import math
from torch import nn
from elasticai.explorer.hw_nas.search_space.utils import calculate_conv_output_shape

activation_mapping = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid()}


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
            layer_width = parse_search_param(
                trial, "layer_width_b{}_l{}".format(block_id, i), search_params["width"]
            )
            activation = parse_search_param(
                trial,
                "activation_func_b{}_l{}".format(block_id, i),
                block["activation"],
            )
            if self.is_last_block(block_id) and i == (num_layers - 1):
                self.layers.append(nn.Linear(self.input_shape, self.output_shape))
            else:
                self.layers.append(nn.Linear(self.input_shape, layer_width))
            self.input_shape = layer_width
            self.layers.append(activation_mapping[activation])

    def createConv2d(self, trial, block, num_layers, search_params):
        block_id = block["block"]
        for i in range(num_layers):
            out_channels = parse_search_param(
                trial,
                "out_channels_b{}_l{}".format(block_id, i),
                search_params["out_channels"],
            )
            kernel_size = parse_search_param(
                trial,
                "kernel_size_b{}_l{}".format(block_id, i),
                search_params["kernel_size"],
            )
            stride = parse_search_param(
                trial, "stride_b{}_l{}".format(block_id, i), search_params["stride"]
            )
            activation = parse_search_param(
                trial,
                "activation_func_b{}_l{}".format(block_id, i),
                block["activation"],
            )
            self.layers.append(
                nn.Conv2d(self.input_shape[0], out_channels, kernel_size, stride)
            )
            self.layers.append(activation_mapping[activation])
            self.input_shape = calculate_conv_output_shape(
                self.input_shape, out_channels, kernel_size, stride
            )

    def is_last_block(self, block_id):
        return self.blocks[-1]["block"] == block_id

    def create_block(self, trial, block: dict):
        num_layers = parse_search_param(
            trial, "num_layers_b{}".format(block["block"]), block["depth"]
        )
        operation = parse_search_param(
            trial, "operation_b{}".format(block["block"]), block["op_candidates"]
        )

        match operation:
            case "linear":
                self.createLinear(trial, block, num_layers, block["linear"])
            case "conv2d":
                self.createConv2d(trial, block, num_layers, block["conv2D"])

    def create_native_torch_model_sample(self, trial) -> nn.Module:
        self.input_shape = self.search_space_cfg["input"]
        self.output_shape = self.search_space_cfg["output"]
        self.layers = []
        for block in self.blocks:
            self.create_block(trial, block)

        return nn.Sequential(*self.layers)


def parse_search_param(trial, name, param):
    if isinstance(param, list):
        return trial.suggest_categorical(name, param)
    elif isinstance(param, dict) and "start" in param and "end" in param:
        print(param)
        if isinstance(param["start"], int):
            return trial.suggest_int(name, param["start"], param["end"])
    else:
        return param
    return ValueError("Search space parameter '{}' is invalid".format(name))
