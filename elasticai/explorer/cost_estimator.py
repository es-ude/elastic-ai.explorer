from typing import Any
import torch
from nni.nas.profiler.pytorch.flops import FlopsProfiler
from nni.nas.nn.pytorch import ModelSpace

class FlopsEstimator:
    """Wrapper for FlopsProfiler could extend in future"""

    def estimate_flops(self, model_sample: ModelSpace) -> float:
        """Computes FLOPS for a single module.

        Returns:
            The FLOPS-estimate
        """

        first_parameter = next(model_sample.parameters())
        input_shape = first_parameter.size()
        data_sample = torch.full(input_shape, 1.0)
        profiler = FlopsProfiler(model_sample, data_sample)

        return float(profiler.expression)
