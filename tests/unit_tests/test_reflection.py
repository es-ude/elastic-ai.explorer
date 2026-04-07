import pytest
import torch.nn as nn

from elasticai.explorer.generator.reflection import Reflective
from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme


class DummyLayerBuilder:
    base_type = nn.Linear


class ReflectiveExample(Reflective):
    def get_activation_mappings(self):  # type: ignore
        return {
            "relu": nn.ReLU(),
        }

    def get_adapter_mappings(self):
        return {}

    def get_supported_quantization(self):
        return {"dtype": {"float32"}, "total_bits": lambda x: x <= 32}

    def get_layer_mappings(self):  # type:ignore
        return {
            "linear": DummyLayerBuilder,
        }


def test_simple_supported_model():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
    )

    reflective = ReflectiveExample()
    reflective.validate_model(model, QuantizationScheme(total_bits=32))


def test_deeply_nested_sequential_supported():
    model = nn.Sequential(
        nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 10),
                nn.ReLU(),
            )
        )
    )

    reflective = ReflectiveExample()
    reflective.validate_model(model, QuantizationScheme())


def test_unsupported_leaf_layer():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Sigmoid(),
    )

    reflective = ReflectiveExample()

    with pytest.raises(NotImplementedError, match="Sigmoid"):
        reflective.validate_model(model, QuantizationScheme())


def test_unsupported_layer_inside_nested_sequential():
    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(10, 10),
            nn.Sigmoid(),
        )
    )

    reflective = ReflectiveExample()

    with pytest.raises(NotImplementedError, match="Sigmoid"):
        reflective.validate_model(model, QuantizationScheme())


def test_unsupported_quantization_scheme():
    model = nn.Sequential(
        nn.Linear(10, 10),
    )

    reflective = ReflectiveExample()

    with pytest.raises(NotImplementedError, match="total_bits=80"):
        reflective.validate_model(model, QuantizationScheme(total_bits=80))
