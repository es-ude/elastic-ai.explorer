import math

from elasticai.explorer.generator.model_builder.model_builder import (
    DefaultModelBuilder,
)
from elasticai.explorer.hw_nas.search_space.layer_builder import LayerBuilder

from abc import ABC, abstractmethod
import logging
from typing import Any, Sequence
from elasticai.creator import nn as creator_nn
from elasticai.creator.nn import fixed_point
import torch


from elasticai.explorer.hw_nas.search_space.layer_adapter import ToLinearAdapter
from elasticai.explorer.hw_nas.search_space.layer_builder import (
    LayerBuilder,
)
from elasticai.explorer.hw_nas.search_space.quantization import (
    CreatorFixedPointScheme,
    QuantizationScheme,
)
from elasticai.explorer.hw_nas.search_space.quantization_builder import (
    CreatorFixedPointBuilder,
)
from elasticai.explorer.hw_nas.search_space.registry import (
    activation_registry,
)
from elasticai.explorer.hw_nas.search_space.sample_blocks import (
    Sampler,
)


def shape_to_prod(shape: int | Sequence | None):
    if isinstance(shape, int):
        return shape
    elif shape:
        return math.prod(shape)
    return None


class CreatorLinearBuilder(LayerBuilder):
    base_type = fixed_point.Linear

    def build(
        self,
        input_shape,
        search_parameters: dict,
        output_shape=None,
        quantization_scheme=CreatorFixedPointScheme,
    ) -> Any:
        self.quantization_scheme = quantization_scheme
        activation_builder = search_parameters.get("activation", None)

        input_shape = shape_to_prod(input_shape)
        output_shape = shape_to_prod(output_shape)

        if output_shape is None:
            layer, shape = self.build_layer(input_shape, search_parameters)
        else:
            layer, shape = self.get_last_layer(
                input_shape, search_parameters, output_shape
            )

        if activation_builder is not None:
            return (
                [
                    layer,
                    activation_registry[activation_builder].build(
                        self.quantization_scheme
                    ),
                ],
                shape,
            )
        return layer, shape

    def build_layer(
        self,
        input_shape,
        search_parameters: dict,
    ):

        linear = fixed_point.Linear(
            input_shape,
            search_parameters["width"],
            self.quantization_scheme.total_bits,
            self.quantization_scheme.frac_bits,
        )
        return linear, search_parameters["width"]

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        linear = fixed_point.Linear(
            input_shape,
            output_shape,
            self.quantization_scheme.total_bits,
            self.quantization_scheme.frac_bits,
        )

        return linear, output_shape


class ActivationBuilder(ABC):
    @abstractmethod
    def build(self, quantization_scheme: CreatorFixedPointScheme) -> Any:
        pass


class CreatorReluBuilder(ActivationBuilder):
    def build(self, quantization_scheme: CreatorFixedPointScheme) -> Any:
        return fixed_point.ReLU(
            total_bits=quantization_scheme.total_bits, use_clock=False
        )


class CreatorSigmoidBuilder(ActivationBuilder):
    def build(self, quantization_scheme: CreatorFixedPointScheme) -> Any:
        return fixed_point.HardSigmoid(
            total_bits=quantization_scheme.total_bits,
            frac_bits=quantization_scheme.frac_bits,
        )


class CreatorTanhBuilder(ActivationBuilder):
    def build(self, quantization_scheme: CreatorFixedPointScheme) -> Any:
        return fixed_point.HardTanh(
            total_bits=quantization_scheme.total_bits,
            frac_bits=quantization_scheme.frac_bits,
        )


class CreatorModelBuilder(DefaultModelBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(
            "explorer.generator.model_builder.CreatorModelBuilder"
        )
        self.setup_registries(True)

    def get_layer_mappings(self) -> dict[str, type[LayerBuilder]]:
        return {
            "linear": CreatorLinearBuilder,
        }

    def get_activation_mappings(self) -> dict[str, Any]:
        return {
            "relu": CreatorReluBuilder(),
            "sigmoid": CreatorSigmoidBuilder(),
            "tanh": CreatorTanhBuilder(),
        }

    def get_adapter_mappings(self) -> dict[tuple[str | None, str | None], None | type]:
        return {
            (None, "linear"): None,
            ("conv1d", "linear"): ToLinearAdapter,
            ("linear", "conv1d"): None,
        }

    def build_from_trial(
        self, trial, search_space: dict
    ) -> tuple[torch.nn.Module, QuantizationScheme]:

        sampler = Sampler(trial)
        sample = sampler.construct_sample(search_space)
        quant_scheme = sampler.get_quantization_scheme(search_space)
        layers = self.construct_model(
            sample, search_space["input"], search_space["output"], quant_scheme
        )
        model = creator_nn.Sequential(*layers)

        self.validate_model(model, quant_scheme)
        return model, quant_scheme

    def get_supported_quantization_schemes(
        self,
    ) -> dict[str, Any]:
        return {CreatorFixedPointScheme.name(): CreatorFixedPointBuilder}

    def validate_model(
        self, model: torch.nn.Module, quantization_scheme: QuantizationScheme
    ):
        """Override if necessary"""
        sl = self.get_supported_layers()
        sa = [fixed_point.ReLU, fixed_point.Tanh, fixed_point.Sigmoid]
        sqs = [CreatorFixedPointScheme]
        for module in model.modules():
            if module is model:
                continue
            module_type = type(module)
            in_supported_layers = module_type in sl
            in_supported_activations = module_type in sa
            if not in_supported_layers and not in_supported_activations:
                raise NotImplementedError(
                    f"{type(module).__name__} is not supported by {self.__class__.__name__} "
                )

        if sqs is not None:
            if type(quantization_scheme) not in sqs:
                raise NotImplementedError(
                    f"{quantization_scheme}  is not supported by {self.__class__.__name__}"
                )
