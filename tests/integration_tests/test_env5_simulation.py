from os import environ
import pytest
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import Linear, ReLU

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.nn import Sequential

from elasticai.explorer_plugins.creator_generator.simulation.simulation import (
    simulate_sequential_module,
)


@pytest.mark.simulation
@pytest.mark.parametrize(
    "total_bits, frac_bits, features_in, features_out",
    [
        (4, 2, 12, 6),
        (10, 8, 24, 20),
    ],
)
def test_build_test_linear_relu(
    total_bits: int,
    frac_bits: int,
    features_in: int,
    features_out: int,
) -> None:
    file_name = f"TestLinearReLU_{total_bits}_{frac_bits}_{features_in}x{features_out}"
    fxp = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )

    dut = Sequential(
        Linear(
            in_features=features_in,
            out_features=features_out,
            total_bits=total_bits,
            frac_bits=frac_bits,
        ),
        ReLU(total_bits=total_bits),
    )
    environ["SIM_RESULT_FILE"] = (
        "/home/robin/code/elastic-ai.explorer/tests/integration_tests/test_experiment/simulation/sim_results.json"
    )
    simulate_sequential_module(
        dut=dut,
        file_name=file_name,
        fxp=fxp,
        feat_in=features_in,
    )
