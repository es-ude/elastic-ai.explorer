from unittest.mock import MagicMock
from optuna.trial import FrozenTrial
import pytest

from elasticai.explorer.hw_nas.search_space.sample_blocks import QuantizationBuilder


@pytest.mark.parametrize(
    "params",
    [
        {
            "dtype": "float32",
            "total_bits": 32,
            "frac_bits": 8,
            "signed": True,
        },
        {
            "dtype": "int8",
            "total_bits": 8,
            "frac_bits": 2,
            "signed": True,
        },
        {
            "dtype": "float32",
            "total_bits": 16,
            "frac_bits": 32,
            "signed": True,
        },
    ],
)
def test_quantization_scheme_builder(params):
    trial = MagicMock(FrozenTrial)
    builder = QuantizationBuilder(trial, params)

    if params["total_bits"] == 16:
        with pytest.raises(ValueError, match="(16)"):
            quant_scheme = builder.build()
    else:
        quant_scheme = builder.build()
        assert quant_scheme.dtype == params["dtype"]
        assert quant_scheme.total_bits == params["total_bits"]
        assert quant_scheme.frac_bits == params["frac_bits"]
        assert quant_scheme.signed == params["signed"]
