from elasticai.explorer.generator.model_translator.model_translator import (
    ModelTranslator,
)
from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme


import torch
from torch import nn


import logging
import os
from pathlib import Path


class TorchscriptModelTranslator(ModelTranslator):

    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.generator.model_translator.model_translator.TorchscriptModelTranslator"
        )

    def translate(
        self,
        model: nn.Module,
        output_path: Path,
        sample: torch.Tensor,
        quantization_scheme: QuantizationScheme | None = None,
    ):

        self.logger.info("Generate torchscript model from %s", model)
        model.eval()

        dir_path = os.path.dirname(os.path.realpath(output_path))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model.to("cpu")
        ts_model = torch.jit.script(model)
        output_path = Path(os.path.realpath(output_path)).with_suffix(".pt")
        self.logger.info("Save model to %s", output_path)
        ts_model.save(output_path)  # type: ignore

        return ts_model
