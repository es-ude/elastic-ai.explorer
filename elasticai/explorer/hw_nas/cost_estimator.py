import torch
from fvcore.nn import FlopCountAnalysis


class FlopsEstimator:
    """Wrapper for FlopsProfiler"""

    def estimate_flops(self, model: torch.nn.Module, data_sample) -> int:
        """Computes FLOPS for a single module.

        Args:
            model (torch.nn.Module): A frozen sample from ModelSpace
            data_sample: a sample of the dataset
        Returns:
            int: The FLOPS-estimate
        """

        flops = FlopCountAnalysis(model, data_sample)

        return flops.total()
