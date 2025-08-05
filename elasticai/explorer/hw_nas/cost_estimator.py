import torch
from fvcore.nn import FlopCountAnalysis
import multiprocessing


class FlopsEstimator:
    def estimate_flops(self, model: torch.nn.Module, data_sample) -> int:
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            return pool.apply(self._safe_estimate, args=(model, data_sample))

    @staticmethod
    def _safe_estimate(model, data_sample):
        handlers = {"aten::sigmoid": None}
        flops = FlopCountAnalysis(model, data_sample).set_op_handle(**handlers)
        return flops.total()
