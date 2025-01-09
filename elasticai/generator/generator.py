from abc import ABC, abstractmethod

import torch
from torch import nn
import os

class Generator(ABC):
    @abstractmethod
    def generate(self, model : nn.Module, path: str)-> any:
        pass


class PIGenerator(Generator):
    def generate(self, model: nn.Module, path: str):
        model.eval()
        
        if not os.path.exists(path):
            os.makedirs(path)

        ts_model= torch.jit.script(model)
        ts_model.save(path+".pt")
        return ts_model




