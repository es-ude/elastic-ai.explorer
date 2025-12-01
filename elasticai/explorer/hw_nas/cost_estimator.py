from numbers import Number
from typing import List, Any, Optional

import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from fvcore.nn.jit_handles import get_shape


def get_values(vals: List[Any]) -> Optional[List[Any]]:
    return [v.toIValue() for v in vals]


def lstm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    num_timesteps, batch_size, feature_width = get_shape(inputs[0])
    *_, proj_size = get_shape(outputs[1])
    *_, hidden_size = get_shape(outputs[2])

    *_, _, num_layers, _, _, bidirectional, batch_first = get_values(inputs)
    num_directions = 2 if bidirectional else 1
    sigmoid_flops = 1
    tanh_flops = 1
    gate_flops = 2 * (feature_width * hidden_size + hidden_size * hidden_size)
    all_gate_flops = 4 * gate_flops
    hadamard = 3 * hidden_size
    activations = 3 * sigmoid_flops * hidden_size + 2 * tanh_flops * hidden_size
    flops = (
        num_timesteps
        * num_directions
        * num_layers
        * batch_size
        * (all_gate_flops + hadamard + activations)
    )
    return flops


class CostEstimator:
    def estimate_flops(self, model: torch.nn.Module, data_sample) -> int:
        handlers = {"aten::sigmoid": None, "aten::lstm": lstm_flop_jit}
        flops = FlopCountAnalysis(model, data_sample).set_op_handle(**handlers)

        return flops.total()

    def compute_num_params(self, model_sample: torch.nn.Module) -> float:
        return parameter_count(model_sample)[""]
