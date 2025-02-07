import logging

from nni.nas.nn.pytorch import ModelSpace
from nni.nas.profiler.pytorch.flops import FlopsProfiler
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch


"""
Wrapper for FlopsProfiler could extend in future
"""


class FlopsEstimator:

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
