from abc import ABC, abstractmethod
import logging
import math

from torch import nn
from typing import Any
from elasticai.creator import nn as creator_nn
from elasticai.creator.nn import fixed_point
import torch
from elasticai.explorer.generator.reflection import Reflective
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace
from elasticai.explorer.hw_nas.search_space.layer_builder import (
    LayerBuilder,
    parse_search_param,
)
from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
    FullPrecisionScheme,
    QuantizationScheme,
)
from elasticai.creator.nn.fixed_point.lstm.layer import (
    FixedPointLSTMWithHardActivations,
)
from elasticai.explorer.hw_nas.search_space.registry import (
    ACTIVATION_REGISTRY,
    ADAPTER_REGISTRY,
    DEFAULT_ACTIVATION,
    DEFAULT_ADAPTER,
    LAYER_REGISTRY,
)


class ModelBuilder(Reflective, ABC):
    @abstractmethod
    def build_from_trial(self, trial, searchspace: SearchSpace) -> Any:
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

    def get_activation_mappings(self) -> dict[str, nn.Module]:
        return DEFAULT_ACTIVATION

    def get_adapter_mappings(self) -> dict[tuple[str | None, str | None], type | None]:
        return DEFAULT_ADAPTER

    def get_supported_quantization_schemes(self) -> list[type[QuantizationScheme]]:
        return [FullPrecisionScheme]

    def build_from_trial(self, trial, searchspace: SearchSpace) -> torch.nn.Module:
        return nn.Sequential(*searchspace.create_model_layers(trial))


class CreatorLinearBuilder(LayerBuilder):
    base_type = fixed_point.Linear

    def build(self, num_layers: int, is_last_block: bool) -> Any:
        for i in range(num_layers):
            if isinstance(self.input_shape, int):
                self.input_shape = self.input_shape
            else:
                self.input_shape = math.prod(self.input_shape)
            if isinstance(self.output_shape, int):
                self.output_shape = self.output_shape
            else:
                self.output_shape = math.prod(self.output_shape)
            width = parse_search_param(
                self.trial,
                f"layer_width_b{self.block_id}_l{i}",
                self.search_params,
                "width",
            )
            activation = parse_search_param(
                self.trial,
                f"activation_func_b{self.block_id}_l{i}",
                self.block,
                "activation",
                "identity",
            )

            if is_last_block and i == num_layers - 1:
                self.layers.append(
                    fixed_point.Linear(
                        self.input_shape,
                        self.output_shape,
                        total_bits=self.quantization_scheme.total_bits,
                        frac_bits=self.quantization_scheme.frac_bits,
                    )
                )
            else:
                self.layers.append(
                    fixed_point.Linear(
                        self.input_shape,
                        width,
                        total_bits=self.quantization_scheme.total_bits,
                        frac_bits=self.quantization_scheme.frac_bits,
                    )
                )
                self.input_shape = width
            self.add_activation(activation)

        return self.get_layers()


class CreatorLSTMBuilder(LayerBuilder):
    base_type = FixedPointLSTMWithHardActivations

    def build(self, num_layers: int, is_last_block: bool):
        for i in range(num_layers):
            hidden_size = parse_search_param(
                self.trial,
                f"hidden_size_b{self.block_id}_l{i}",
                self.search_params,
                "hidden_size",
                default_value=None,
            )
            num_lstm_layers = parse_search_param(
                self.trial,
                f"num_lstm_layers_b{self.block_id}_l{i}",
                self.search_params,
                "num_lstm_layers",
                default_value=None,
            )
            bidirectional = parse_search_param(
                self.trial,
                f"bidirectional_b{self.block_id}_l{i}",
                self.search_params,
                "bidirectional",
                default_value=False,
            )
            dropout = parse_search_param(
                self.trial,
                f"dropout_b{self.block_id}_l{i}",
                self.search_params,
                key="dropout",
                default_value=0.0,
            )
            if is_last_block and i == num_layers - 1:
                hidden_size = self.output_shape
                if bidirectional & ((self.output_shape % 2) != 0):
                    raise NotImplementedError
                elif bidirectional:
                    hidden_size = self.output_shape / 2

            self.layers.append(
                FixedPointLSTMWithHardActivations(
                    total_bits=self.quantization_scheme.total_bits,
                    frac_bits=self.quantization_scheme.frac_bits,
                    input_size=self.input_shape[-1],
                    hidden_size=hidden_size,
                    bias=True,
                )
            )
            self.input_shape = [
                self.input_shape[0],
                hidden_size * 2 if bidirectional else hidden_size,
            ]
        return self.get_layers()


class CreatorModelBuilder(ModelBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.quantization_scheme = FixedPointInt8Scheme()
        self.logger = logging.getLogger(
            "explorer.generator.model_builder.CreatorModelBuilder"
        )
        self.setup_registries(True)

    def get_layer_mappings(self) -> dict[str, type[LayerBuilder]]:
        return {"linear": CreatorLinearBuilder}

    def get_activation_mappings(self) -> dict[str, nn.Module]:
        return {
            "relu": fixed_point.ReLU(total_bits=self.quantization_scheme.total_bits)
        }

    def get_adapter_mappings(self) -> dict[tuple[str | None, str | None], None | type]:
        return {(None, "linear"): None}

    def build_from_trial(self, trial, searchspace: SearchSpace) -> torch.nn.Module:
        model = creator_nn.Sequential(*searchspace.create_model_layers(trial=trial))
        self._validate_model(model, self.quantization_scheme)
        return model

    def get_supported_quantization_schemes(
        self,
    ) -> list[type[QuantizationScheme]]:
        return [type(self.quantization_scheme)]
