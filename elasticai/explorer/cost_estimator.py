import torch
from nni.nas.profiler.pytorch.flops import FlopsProfiler
from nni.nas.nn.pytorch import ModelSpace
from nni.mutable.symbol import SymbolicExpression


class FlopsEstimator:
    """Wrapper for FlopsProfiler"""

    def estimate_flops(self, model_space: ModelSpace) -> SymbolicExpression | float:
        """Computes FLOPS for a frozen model space."""

        first_parameter = next(model_space.parameters())
        input_shape = first_parameter.size()
        data_sample = torch.full(input_shape, 1.0)
        profiler = FlopsProfiler(model_space, data_sample)

        return profiler.expression
