from typing import Any
import torch
from nni.nas.profiler.pytorch.flops import FlopsProfiler, NumParamsProfiler
from nni.nas.nn.pytorch import ModelSpace

class CostEstimator:
    """Wrapper for FlopsParamsProfiler, we could extend in the future"""

    def _data_sample(self, model_sample: ModelSpace) -> torch.Tensor:
        """Generates a data sample for the model sample.

        Args:
            model_sample: The model sample to generate a data sample for.

        Returns:
            A tensor filled with ones, matching the size of the first parameter.
        """
        first_parameter = next(model_sample.parameters())
        input_shape = first_parameter.size()
        return torch.full(input_shape, 1.0)

    def estimate_flops(self, model_sample: ModelSpace) -> float:
        """Computes FLOPS for a single module.

        Returns:
            The FLOPS-estimate
        """

        data_sample = self._data_sample(model_sample)
        profiler = FlopsProfiler(model_sample, data_sample)
        return float(profiler.expression)

    def compute_num_params(self, model_sample: ModelSpace) -> float:
        """Computes number of parameters for a single module.

        Returns:
            The number of parameters.
        """

        data_sample = self._data_sample(model_sample)
        profiler = NumParamsProfiler(model_sample, data_sample)
        return float(profiler.expression)