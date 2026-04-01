import json
from pathlib import Path


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
