import torch
from nni.nas.profiler.pytorch.flops import FlopsProfiler
from nni.nas.nn.pytorch import ModelSpace

class FlopsEstimator:
    """Wrapper for FlopsProfiler could extend in future"""

    def estimate_flops(self, model_sample: ModelSpace) -> int:
        """Computes FLOPS for a single module.

        Args:
            model_sample (torch.nn.Module): A frozen sample from ModelSpace

        Returns:
            int: The FLOPS-estimate
        """

        first_parameter = next(model_sample.parameters())
        input_shape = first_parameter.size()
        data_sample = torch.full(input_shape, 1.0)
        profiler = FlopsProfiler(model_sample, data_sample)

        return int(profiler.expression)
