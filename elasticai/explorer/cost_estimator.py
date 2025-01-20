import math
from typing import Any
from nni.nas.nn.pytorch import ModelSpace
from nni.nas.profiler.pytorch.flops import FlopsProfiler
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch


"""
Wrapper for FlopsProfiler could extend in future
"""


class FlopsEstimator:

    def estimate_flops_grid(self, model_space: ModelSpace):
        """Print FLOPS-Information over the whole searchspace.

        Args:
            model_space (ModelSpace): A mutable ModelSpace
        """

        data_sample = torch.full((1, 1, 28, 28), 1.0)

        profiler = FlopsProfiler(model_space, data_sample)

        print("General Flops, Formular: ", profiler.expression)
        for model_sample in model_space.grid():

            profiler = FlopsProfiler(model_sample, data_sample)
            print("Flops for", model_sample, ": ", profiler.expression)

    def estimate_flops_single_module(self, model_sample: torch.nn.Module) -> int:
        """Computes FLOPS for a single module.

        Args:
            model_sample (torch.nn.Module): A frozen sample from ModelSpace

        Returns:
            int: The FLOPS-estimate
        """

        data_sample = torch.full((1, 1, 28, 28), 1.0)
        profiler = FlopsProfiler(model_sample, data_sample)

        return profiler.expression
