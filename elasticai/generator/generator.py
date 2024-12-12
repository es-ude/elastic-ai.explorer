from abc import ABC, abstractmethod

import torch
from torch import nn


class Generator(ABC):
    @abstractmethod
    def generate(self, model : nn.Module, path: str)-> any:
        pass


class PIGenerator(Generator):
    def generate(self, model: nn.Module, path: str):
        model.eval()
        ts_model= torch.jit.script(model)
        ts_model.save(path+".pt")
        return ts_model




