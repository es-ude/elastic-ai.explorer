import torch
from fvcore.nn import FlopCountAnalysis

class FlopsEstimator:
    def estimate_flops(self, model: torch.nn.Module, data_sample) -> int:
        handlers = {"aten::sigmoid": None}
        flops = FlopCountAnalysis(model, data_sample).set_op_handle(**handlers)
        return flops.total()
