import math
from typing import Any
from nni.nas.nn.pytorch import ModelSpace
from nni.nas.profiler.pytorch.flops import FlopsProfiler
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader



'''
Wrapper for FlopsProfiler could extend in future
'''
class FlopsEstimator():
    def __init__(self, model_space: ModelSpace):
        self.model_space = model_space
        self.data_loader = self.prepare_dataloader()

    def prepare_dataloader(self):
        transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data_loader = DataLoader(MNIST("data/mnist", download=True, transform=transf), batch_size=1, shuffle=True)
        return data_loader
    
    def estimate_flops_grid(self):
        """
        Print FLOPS-Information over the whole searchspace.
        """
        
        data_sample, _target= next(iter(self.data_loader))

        profiler = FlopsProfiler(self.model_space, data_sample)

        print("General Flops, Formular: ", profiler.expression)
        for model_sample in self.model_space.grid():

            profiler = FlopsProfiler(model_sample, data_sample)
            print("Flops for",model_sample ,": ", profiler.expression)
        
        
    
    def estimate_flops_single_module(self) -> int:
        """Computes FLOPS for a single module. 

        Returns:
            int: The FLOPS-estimate
        """
        
        data_sample, _target= next(iter(self.data_loader))
        model_sample = next(iter(self.model_space.grid()))
        profiler = FlopsProfiler(model_sample, data_sample)

        return profiler.expression



        
