from abc import ABC, abstractmethod

import torch
from torch import nn
import os
import datetime


class Generator(ABC):
    @abstractmethod
    def generate(self, model: nn.Module, path: str) -> any:
        pass


class PIGenerator(Generator):
    def generate(self, model: nn.Module, path: str):
        model.eval()
        dir_path = os.path.dirname(os.path.realpath(path))
        file_name = os.path.basename(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        now = datetime.datetime.now()
        current_timestamp =now.strftime('%Y-%m-%dT%H:%M:%S')
        ts_model= torch.jit.script(model)
        ts_model.save(dir_path + file_name + "_" + str(current_timestamp) + ".pt")
        
        return ts_model
