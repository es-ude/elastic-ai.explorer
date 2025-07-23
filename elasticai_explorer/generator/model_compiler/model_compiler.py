from enum import Enum
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import torch
from torch import nn

from elasticai.creator.torch2ir.torch2ir import get_default_converter


class QuantizationSchemes(str, Enum):
    FULL_PRECISION_FLOAT32 = "full_precision_float32"
    INT8_UNIFORM = "int8_uniform"


class ModelCompiler(ABC):
    @abstractmethod
    def generate(
        self,
        model: nn.Module,
        path: Path,
        quantization_scheme: QuantizationSchemes = QuantizationSchemes.FULL_PRECISION_FLOAT32,
    ) -> Any:
        pass

    def get_supported_layers(self) -> Optional[set[type]]:
        """Override if necessary. None means no constraints."""
        return None

    def get_supported_quantization_schemes(self) -> Optional[set[QuantizationSchemes]]:
        """Override if necessary. None means no constraints."""
        return None

    def _validate_model(
        self, model: nn.Module, quantization_scheme: QuantizationSchemes
    ):
        """Override if necessary"""
        supported_layers = self.get_supported_layers()
        supported_quantization_schemes = self.get_supported_quantization_schemes()
        if supported_layers is not None:
            for layer in model.modules():
                if layer is model:
                    continue
                if type(layer) not in supported_layers:
                    raise NotImplementedError(
                        f"Layer {type(layer).__name__} wird von {self.__class__.__name__} nicht unterstützt"
                    )

        if supported_quantization_schemes is not None:
            if quantization_scheme not in supported_quantization_schemes:
                raise NotImplementedError(
                    f"Layer {quantization_scheme} wird von {self.__class__.__name__} nicht unterstützt"
                )


class TorchscriptModelCompiler(ModelCompiler):
    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.generator.model_compiler.model_compiler.TorchscriptModelCompiler"
        )

    def get_supported_quantization_schemes(self) -> Optional[set[QuantizationSchemes]]:
        return {QuantizationSchemes.FULL_PRECISION_FLOAT32}

    def generate(
        self,
        model: nn.Module,
        path: Path,
        quantization_scheme: QuantizationSchemes = QuantizationSchemes.FULL_PRECISION_FLOAT32,
    ):

        self._validate_model(model, quantization_scheme)
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


class CreatorModelCompiler(ModelCompiler):
    def __init__(self) -> None:
        self.logger = logging.getLogger(
            "explorer.generator.model_compiler.model_compiler.CreatorModelCompiler"
        )

    def generate(
        self,
        model: nn.Module,
        path: Path,
        quantization_scheme: QuantizationSchemes = QuantizationSchemes.FULL_PRECISION_FLOAT32,
    ):

        self._validate_model(model, quantization_scheme)
        default_converter = get_default_converter()
        ir = default_converter.convert(model)

    def get_supported_layers(self) -> Optional[set[type]]:
        return {
            nn.Linear,
            nn.Conv1d,
            nn.ReLU,
            nn.MaxPool1d,
            nn.BatchNorm1d,
            nn.Flatten,
            nn.Sigmoid,
        }
