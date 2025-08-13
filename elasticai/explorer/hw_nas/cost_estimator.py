import torch
from fvcore.nn import FlopCountAnalysis, parameter_count


class CostEstimator:
    def estimate_flops(self, model_sample: torch.nn.Module, data_sample) -> int:
        handlers = {"aten::sigmoid": None}
        flops = FlopCountAnalysis(model_sample, data_sample).set_op_handle(**handlers)
        return flops.total()

    def compute_num_params(self, model_sample: torch.nn.Module) -> float:
        return parameter_count(model_sample)[""]
