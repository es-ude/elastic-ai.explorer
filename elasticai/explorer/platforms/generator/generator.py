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

        try:
            dir_path = os.path.dirname(os.path.realpath(path))
            if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
            ts_model= torch.jit.script(model)
            path = Path(path).with_suffix(".pt")
            ts_model.save(path)
        except:
            print("Could not create or find path to model.")
            exit(-1)

        
        
        return ts_model
