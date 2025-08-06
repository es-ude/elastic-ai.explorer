import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import torch
from torch import nn


class Generator(ABC):
    @abstractmethod
    def generate(self, model: nn.Module, path: Path) -> Any:
        pass


class PIGenerator(Generator):
    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.platforms.generator.generator.PIGenerator"
        )

    def generate(self, model: nn.Module, path: Path):
        self.logger.info("Generate torchscript model from %s", model)
        model.eval()

        dir_path = os.path.dirname(os.path.realpath(path))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model.to("cpu")
        ts_model = torch.jit.script(model)
        path = Path(os.path.realpath(path)).with_suffix(".pt")
        self.logger.info("Save model to %s", path)
        ts_model.save(path)  # type: ignore

        return ts_model
