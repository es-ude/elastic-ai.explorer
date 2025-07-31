
import torch

import torch
from fvcore.nn import FlopCountAnalysis
class FlopsEstimator:
    """Wrapper for FlopsProfiler could extend in future"""


    def estimate_flops(self, model: torch.nn.Module, data_sample) -> int:
        """Computes FLOPS for a single module.

        Args:
            model (torch.nn.Module): A frozen sample from ModelSpace
            data_sample: a sample of the dataset
        Returns:
            int: The FLOPS-estimate
        """
        


        # Compute FLOPs
        flops = FlopCountAnalysis(model, data_sample)
        print("Total FLOPs:", flops.total())
        print("FLOPs by operation:", flops.by_operator())

        # register_flops_formula(nn.Linear, linear_flops)
        # profiler = FlopsProfiler(model, data_sample)

        # return float(profiler.expression)
        return flops.total()  