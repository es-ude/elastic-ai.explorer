
import torch
#from nni.nas.profiler.pytorch.flops import FlopsProfiler


class FlopsEstimator:
    """Wrapper for FlopsProfiler could extend in future"""


    def estimate_flops(self, model: torch.nn.Module, data_sample) -> float:
        """Computes FLOPS for a single module.

        Args:
            model (torch.nn.Module): A frozen sample from ModelSpace
            data_sample: a sample of the dataset
        Returns:
            int: The FLOPS-estimate
        """

        # profiler = FlopsProfiler(model, data_sample)

        # return float(profiler.expression)
        return 1.0  # FIXME: Dummy just to make it work for now, later should be implemented without NNI