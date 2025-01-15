from abc import ABC, abstractmethod

import torch
from torch import nn
import os
from pathlib import Path


class Generator(ABC):
    @abstractmethod
    def generate(self, model: nn.Module, path: str) -> any:
        pass


class PIGenerator(Generator):
    def generate(self, model: nn.Module, path: str):
        model.eval()

        dir_path = os.path.dirname(os.path.realpath(path))
        
        if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        ts_model= torch.jit.script(model)
        path = Path(os.path.realpath(path)).with_suffix(".pt")
        ts_model.save(path)
        
        return ts_model
