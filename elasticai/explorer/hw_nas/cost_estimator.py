import torch
from fvcore.nn import FlopCountAnalysis
import multiprocessing


class FlopsEstimator:
    def estimate_flops(self, model: torch.nn.Module, data_sample) -> int:
        handlers = {"aten::sigmoid": None}
        flops = FlopCountAnalysis(model, data_sample).set_op_handle(**handlers)
        return flops.total()

    @staticmethod
    def _safe_estimate(model, data_sample):
        handlers = {"aten::sigmoid": None}
        flops = FlopCountAnalysis(model, data_sample).set_op_handle(**handlers)
        return flops.total()
