import json
import os
import torch
from elasticai.creator.arithmetic import FxpArithmetic
from elasticai.creator.file_generation import find_project_root
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn import Sequential
from elasticai.creator.testing import build_report_folder_and_testdata, run_cocotb_sim


from glob import glob
from os.path import exists
from pathlib import Path

from elasticai.creator.testing.cocotb_prepare import read_testdata
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer


from torch.utils.data import DataLoader


@cocotb.test()
async def accuracy_test(dut):
    data = read_testdata(dut._name)
    num_feat_output = len(data["out"][-1])
    clock_period_ns = 10

    dut.enable.value = 0  # Has no impact
    dut.clock.value = 0  # has no impact
    dut.x_address.value = 0
    dut.y_address.value = 0
    dut.x.value = 0

    cocotb.start_soon(Clock(dut.clock, period=clock_period_ns, unit="ns").start())
    await Timer(4 * clock_period_ns, unit="ns")
    await RisingEdge(dut.clock)
    chck_test = list()
    for ite, (sig_in, ref_out) in enumerate(zip(data["in"], data["out"])):
        result = list()
        chck_ite = list()
        dut.enable.value = 1

        # --- Apply data for inference
        while dut.done.value == 0:
            dut.x.value = sig_in[dut.x_address.value]
            await FallingEdge(dut.clock)

        # --- Getting data
        await RisingEdge(dut.clock)
        for idx in range(num_feat_output):
            dut.y_address.value = idx
            for _ in range(2):
                await RisingEdge(dut.clock)
            result.append(dut.y.value.signed_integer)
            chck_ite.append(
                dut.y.value.signed_integer
                in [ref_out[idx] - 1, ref_out[idx], ref_out[idx] + 1]
            )
            for _ in range(2):
                await RisingEdge(dut.clock)
        dut.y_address.value = 0
        chck_test.extend(chck_ite)

        if not all(chck_ite):
            print(f"\n--- Run {ite} ---")
            print(f"Chck: {chck_ite}")
            print(f"Pred: {result}")
            print(f"True: {ref_out}")

        # --- Do reset
        for _ in range(2):
            await RisingEdge(dut.clock)

        dut.enable.value = 0
        for _ in range(2):
            await RisingEdge(dut.clock)

    accuracy = sum(chck_test) / len(chck_test)

    result_path = Path(
        os.getenv(
            "SIM_RESULT_FILE",
            "tests/integration_tests/test_experiment/simulation/sim_results.json",
        )
    ).resolve()
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps({"accuracy": accuracy, "accuracy_percent": accuracy * 100})
    )
    limit = 0.9
    assert accuracy >= limit, f"Accuracy of {accuracy * 100:.2f}%"


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
            "out": fxp.round_to_integer(val_output).int().tolist(),
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
