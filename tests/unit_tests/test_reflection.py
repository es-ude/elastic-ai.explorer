import pytest
import torch.nn as nn

from elasticai.explorer.generator.reflection import Reflective
from elasticai.explorer.hw_nas.search_space.build_model import DefaultModelBuilder
from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme


class DummyLayerBuilder:
    build_return_types = [nn.Linear]


class ReflectiveExample(Reflective):

    def get_activation_mappings(self):  # type: ignore
        return {
            "relu": nn.ReLU(),
        }

    def get_layer_mappings(self):  # type:ignore
        return {
            "linear_test": DummyLayerBuilder,
        }

    def get_adapter_mappings(self):
        return {}

    def get_supported_quantization(self):
        return {"dtype": {"int8"}, "total_bits": lambda x: x <= 32}


def test_simple_supported_model():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
    )

    reflective = ReflectiveExample()
    assert reflective.validate_model(model, QuantizationScheme(total_bits=32))


def test_deeply_nested_sequential_supported():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 10),
                nn.ReLU(),
            )
        ),
    )

    reflective = ReflectiveExample()
    assert reflective.validate_model(model, QuantizationScheme())


def test_no_quant_scheme():
    model = nn.Sequential(
        nn.Linear(10, 10),
    )

    reflective = ReflectiveExample()

    assert reflective.validate_model(model, None)


def test_unsupported_leaf_layer():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.L1Loss(),
    )

    reflective = ReflectiveExample()

    with pytest.raises(NotImplementedError, match="L1Loss"):
        reflective.validate_model(model, QuantizationScheme())


def test_unsupported_layer_inside_nested_sequential():
    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(10, 10),
            nn.L1Loss(),
        )
    )

    reflective = ReflectiveExample()

    with pytest.raises(NotImplementedError, match="L1Loss"):
        reflective.validate_model(model, QuantizationScheme())


def test_unsupported_quantization_scheme():
    model = nn.Sequential(
        nn.Linear(10, 10),
    )

    reflective = ReflectiveExample()

    with pytest.raises(NotImplementedError, match="total_bits=80"):
        reflective.validate_model(model, QuantizationScheme(total_bits=80))


def test_non_overwrite():
    class L1Builder:
        build_return_types = [nn.L1Loss]

    class DummyBuilder(DefaultModelBuilder):
        def get_layer_mappings(self):  # type:ignore
            return {"L1Loss": L1Builder}

    model_builder = DummyBuilder()
    model = nn.Sequential(
        nn.Linear(10, 10),
    )
    assert model_builder.validate_model(model, None)
    assert nn.L1Loss in model_builder.get_supported_layers()
    assert (
        nn.Linear in model_builder.get_supported_layers()
    )  # nn.Linear is from the default layers that should not be overwritten
