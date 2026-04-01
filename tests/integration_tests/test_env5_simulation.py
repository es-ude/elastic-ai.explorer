from glob import glob
from os import environ
from os.path import exists
from pathlib import Path
from elasticai.creator.file_generation import find_project_root
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.testing import build_report_folder_and_testdata, run_cocotb_sim
import pytest
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import Linear, ReLU

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.nn import Sequential
import torch
from torch.utils.data import DataLoader


def simulate_sequential_module(
    dut: Sequential,
    feat_in: int,
    fxp: FxpArithmetic,
    file_name: str,
    check_quant: bool = True,
    data_loader: DataLoader | None = None,
) -> None:

    if not data_loader:
        val_input = fxp.as_rational(
            torch.randint(
                low=-(2 ** (fxp.total_bits - 2)),
                high=2 ** (fxp.total_bits - 2),
                size=(20, feat_in),
            )
        )

    else:
        val_input = next(iter(data_loader))

    dut.eval()
    val_output = dut(val_input)

    output_dir = build_report_folder_and_testdata(
        dut_name=file_name,
        testdata={
            "in": fxp.cut_as_integer(val_input).int().tolist(),
            "target": fxp.round_to_integer(val_output).int().tolist(),
        },
    )

    output_dir_sub = Path(*output_dir.parts[len(find_project_root().parts) :])
    destination = OnDiskPath(str(output_dir_sub), parent=find_project_root())
    dut.create_design(file_name).save_to(destination)
    assert exists(output_dir / f"{file_name}.vhd")
    assert exists(output_dir / "testdata.json")

    # --- Prepare and start cocotb runner
    src_folders = [folder for folder in output_dir.iterdir() if folder.is_dir()]
    src_files = [output_dir / f"{file_name}.vhd"]
    for folder in src_folders:
        src_files.extend(glob(str(output_dir / folder / "*.vhd")))
    run_cocotb_sim(
        src_files=src_files,
        top_module_name=file_name,
        cocotb_test_module="elasticai.explorer_plugins.creator_generator.simulation.simulation",
        waveform_save_dst=str(output_dir),
    )


@pytest.mark.simulation
@pytest.mark.parametrize(
    "total_bits, frac_bits, features_in, features_out",
    [
        (4, 2, 12, 6),
        (10, 8, 24, 20),
    ],
)
def test_simulate_linear_relu(
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
        "/home/robin/code/elastic-ai.explorer/tests/integration_tests/test_experiment/simulation/sim_resultsy.json"
    )
    simulate_sequential_module(
        dut=dut,
        file_name=file_name,
        fxp=fxp,
        feat_in=features_in,
    )
