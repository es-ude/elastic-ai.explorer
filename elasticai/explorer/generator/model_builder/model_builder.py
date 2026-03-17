from abc import ABC, abstractmethod
import logging

from torch import nn
from typing import Any
import torch
from elasticai.explorer.generator.reflection import Reflective
from elasticai.explorer.hw_nas.search_space.build_model import construct_model

from elasticai.explorer.hw_nas.search_space.quantization import (
    FullPrecisionScheme,
    QuantizationScheme,
)

from elasticai.explorer.hw_nas.search_space.registry import (
    ACTIVATION_REGISTRY,
    ADAPTER_REGISTRY,
    DEFAULT_ACTIVATION,
    DEFAULT_ADAPTER,
    LAYER_REGISTRY,
)
from elasticai.explorer.hw_nas.search_space.sample_blocks import Sampler


class ModelBuilder(Reflective, ABC):
    @abstractmethod
    def build_from_trial(self, trial, search_space: dict) -> Any:
        pass

    def setup_registries(self, replace=False):
        if replace:
            ACTIVATION_REGISTRY.clear()
            ADAPTER_REGISTRY.clear()
            LAYER_REGISTRY.clear()

        ACTIVATION_REGISTRY.update(self.get_activation_mappings())
        ADAPTER_REGISTRY.update(self.get_adapter_mappings())
        LAYER_REGISTRY.update(self.get_layer_mappings())


class DefaultModelBuilder(ModelBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(
            "explorer.generator.model_builder.TorchModelBuilder"
        )
        self.setup_registries()

    def get_activation_mappings(self) -> dict[str, Any]:
        return DEFAULT_ACTIVATION

    def get_adapter_mappings(self) -> dict[tuple[str | None, str | None], type | None]:
        return DEFAULT_ADAPTER

    def get_supported_quantization_schemes(self) -> list[type[QuantizationScheme]]:
        return [FullPrecisionScheme]

    def build_from_trial(
        self, trial, search_space: dict
    ) -> tuple[torch.nn.Module, QuantizationScheme]:
        sampler = Sampler(trial)
        sample, quant_scheme = sampler.construct_sample(search_space)
        return (
            nn.Sequential(
                *construct_model(sample, search_space["input"], search_space["output"])
            ),
            quant_scheme,
        )
