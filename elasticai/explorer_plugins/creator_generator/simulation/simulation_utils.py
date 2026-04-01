from glob import glob
import json
import os
from pathlib import Path

import torch

from elasticai.explorer.generator.deployment.device_communication import (
    Host,
    SerialHost,
)
from elasticai.explorer.generator.deployment.hw_manager import HWManager
from elasticai.explorer.hw_nas.search_space.quantization import CreatorFixedPointScheme

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from torch.utils.data import DataLoader


def build_simulation_folder_and_test_data(build_dir: Path, testdata: dict) -> Path:
    """Building the test/simulation folder which contains the test data and hardware design for testing in cocotb
    :param dut_name:        The name of the Top Module
    :param testdata:        Dictionary with test data/params data
    :return:                Path to the report folder containing hardware design and testpattern data
    """
    build_dir = build_dir
    build_dir.mkdir(exist_ok=True, parents=True)

    file_name = "testdata.json"
    with open(build_dir / file_name, "w") as f0:
        json.dump(testdata, f0, indent=1)

    return build_dir


def read_testdata(build_dir: Path) -> dict:
    """Reading the data as testpattern in the cocotb testbench
    :param dut_name:        The name of the Top Module DUT in the cocotb testbench (using dut._name)
    :return:                Dictionary with testpattern for testing the DUT
    """
    path_to_file = build_dir
    file_name = "testdata.json".lower()
    with open(
        path_to_file / file_name,
        "r",
    ) as f:
        data = json.load(f)
    return data


def _prep_simulation(
    host: Host,
    hw_manager: HWManager,
    path_to_model: Path,
    input_dim: int,
    output_dim: int,
):
    assert type(hw_manager.quantization_scheme) is CreatorFixedPointScheme

    if not hw_manager.test_loader:
        raise TypeError("Testloader not defined.")
    if (
        not hw_manager.quantization_scheme
        or type(hw_manager.quantization_scheme) is not CreatorFixedPointScheme
    ):
        raise TypeError("Quantization Scheme is not defined correctly.")
    if not isinstance(host, SerialHost):
        raise TypeError("Need Serialhost for this test.")

    fxp = FxpArithmetic(
        FxpParams(
            total_bits=hw_manager.quantization_scheme.total_bits,
            frac_bits=hw_manager.quantization_scheme.frac_bits,
            signed=True,
        )
    )

    if not hw_manager.test_loader:
        val_input = fxp.as_rational(
            torch.randint(
                low=-(2 ** (fxp.total_bits - 2)),
                high=2 ** (fxp.total_bits - 2),
                size=(20, input_dim),
            )
        )

    else:
        test_loader = DataLoader(
            hw_manager.test_loader.dataset, batch_size=256, shuffle=False
        )
        val_input, target = next(iter(test_loader))

    output_dir = build_simulation_folder_and_test_data(
        build_dir=path_to_model / "srcs",
        testdata={
            "in": fxp.cut_as_integer(val_input).int().tolist(),
            "target": target.int().tolist(),
            "out_dim": output_dim,
        },
    )

    file_name = f"sequential"
    src_folders = [folder for folder in output_dir.iterdir() if folder.is_dir()]
    src_files = [output_dir / f"{file_name}.vhd"]
    for folder in src_folders:
        src_files.extend(glob(str(output_dir / folder / "*.vhd")))

    result_file = path_to_model / "sim_results.json"
    os.environ["SIM_RESULT_FILE"] = str(result_file)
    os.environ["TEST_DIR"] = str(output_dir)
    return src_files, file_name, output_dir, result_file
