from numpy import dtype
import pytest
import torch
import torch.nn as nn

from elasticai.explorer.generator.reflection import Reflective
from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme

# --- Minimal stubs ---


class DummyQuantScheme(QuantizationScheme):
    dtype: str = ""

    @staticmethod
    def name() -> str:
        return ""


class DummyLayerBuilder:
    base_type = nn.Linear


class TestReflective(Reflective):
    def get_activation_mappings(self):
        return {
            "relu": nn.ReLU(),
        }

    def get_adapter_mappings(self):
        return {}

    def get_supported_quantization_schemes(self):
        return {DummyQuantScheme: {}}

    def get_layer_mappings(self):
        return {
            "linear": DummyLayerBuilder,
        }


def test_simple_supported_model():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
    )

    reflective = TestReflective()
    reflective.validate_model(model, DummyQuantScheme())


def test_deeply_nested_sequential_supported():
    model = nn.Sequential(
        nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 10),
                nn.ReLU(),
            )
        )
    )

    reflective = TestReflective()
    reflective.validate_model(model, DummyQuantScheme())


def test_unsupported_leaf_layer():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Sigmoid(),
    )

    reflective = TestReflective()

    with pytest.raises(NotImplementedError, match="Sigmoid"):
        reflective.validate_model(model, DummyQuantScheme())


def test_unsupported_layer_inside_nested_sequential():
    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(10, 10),
            nn.Sigmoid(),
        )
    )

    reflective = TestReflective()

    with pytest.raises(NotImplementedError, match="Sigmoid"):
        reflective.validate_model(model, DummyQuantScheme())


def test_unsupported_quantization_scheme():
    class OtherQuantScheme(QuantizationScheme):
        dtype: str = ""

        @staticmethod
        def name() -> str:
            return "other"

    model = nn.Sequential(
        nn.Linear(10, 10),
    )

    reflective = TestReflective()

    with pytest.raises(NotImplementedError, match="other"):
        reflective.validate_model(model, OtherQuantScheme())
