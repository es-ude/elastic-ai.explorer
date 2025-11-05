import os
from pathlib import Path
from random import randint
import time
from typing import Any
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import Linear
from elasticai.creator.vhdl.system_integrations.firmware_env5 import FirmwareENv5
from elasticai.runtime.env5.usb import UserRemoteControl, get_env5_port

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
import serial
import torch
import tarfile

from elasticai.explorer.utils import synthesis_utils
from elasticai.explorer.hw_nas.search_space.quantization import (
    parse_fxp_tensor_to_bytearray,
)
from elasticai.explorer.hw_nas.search_space.quantization import (
    parse_bytearray_to_fxp_tensor,
)


def build_vhdl_files(
    inputs: int,
    outputs: int,
    total_bits: int,
    frac_bits: int,
    skeleton_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
) -> Sequential:

    output_dir = "examples/designs"
    destination = OnDiskPath(output_dir)

    nn = Sequential(
        Linear(
            in_features=inputs,
            out_features=outputs,
            total_bits=total_bits,
            frac_bits=frac_bits,
        ),
        # ReLU(total_bits=total_bits),
    )
    my_design = nn.create_design("myNetwork")
    my_design.save_to(destination.create_subpath("srcs"))

    firmware = FirmwareENv5(
        network=my_design,
        x_num_values=inputs,
        y_num_values=outputs,
        id=skeleton_id,
        skeleton_version="v2",
    )
    firmware.save_to(destination)
    nn[0].weight.data = torch.ones_like(torch.Tensor(nn[0].weight)) * 2
    nn[0].bias.data = torch.ones_like(torch.Tensor(nn[0].bias)) * 0.5
    return nn


def simulate_result(
    total_bits: int,
    frac_bits: int,
    nn: Sequential,
    input_tensors: torch.Tensor,
):
    fxp_params = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    fxp_conf = FxpArithmetic(fxp_params)
    inputs_rational = fxp_conf.as_rational(fxp_conf.cut_as_integer(input_tensors))
    expected_outputs = nn(inputs_rational)
    print("Expected Outputs", expected_outputs)
    return inputs_rational, expected_outputs


def synthesis(
    output_dir: Path = Path("examples"),
    design_dir: Path = Path("examples/designs"),
):
    try:
        os.remove(str(output_dir) + "/vivado_run_results.tar.gz")
    except:
        pass
    time.sleep(1)

    synthesis_utils.run_vhdl_synthesis(
        src_dir=design_dir,
        remote_working_dir="/home/vivado/robin-build/",
        host="65.108.38.237",
        ssh_user="vivado",
    )

    tar = tarfile.open(str(output_dir) + "/vivado_run_results.tar.gz")
    tar.extractall(output_dir)
    tar.close()


def test_on_device(
    outputs: int,
    total_bits: int,
    frac_bits: int,
    skeleton_id_as_bytearray: bytearray,
    inputs_rational: Any,
    expected_outputs: torch.Tensor,
    binfile_path: Path = Path("examples/results/impl/env5_top_reconfig.bin"),
):

    dev_address = get_env5_port()

    # --- Doing the test
    with serial.Serial(dev_address) as serial_con:
        flash_start_address = 0
        urc = UserRemoteControl(device=serial_con)
        urc.send_and_deploy_model(
            binfile_path, flash_start_address, skeleton_id_as_bytearray
        )

        skeleton_id_on_device = bytearray(urc._enV5RCP.read_skeleton_id())
        print(
            "Skeleton-ID on device",
            skeleton_id_on_device,
            "\nExpected Skeleton-ID",
            skeleton_id_as_bytearray,
        )
        print(
            "Correct design on FPGA: ",
            skeleton_id_on_device == skeleton_id_as_bytearray,
        )

        # --- Doing the test
        batch_data = parse_fxp_tensor_to_bytearray(
            inputs_rational, total_bits, frac_bits
        )
        inference_result = list()
        state = False

        print("LEDs: now 1,0,0,0")
        urc.fpga_leds(True, False, False, False)
        for i, sample in enumerate(batch_data):
            urc.fpga_leds(True, False, False, state)
            state = False if state else True

            raw_result = urc.inference_with_data(sample, outputs)
            print("Raw Result: ", raw_result)
            fxp_tensor_result = parse_bytearray_to_fxp_tensor(
                [raw_result], total_bits, frac_bits, (1, 1, outputs)
            )

            dev_inp = fxp_tensor_result
            dev_out = expected_outputs.data[i].view((1, 1, outputs))
            if not torch.equal(dev_inp, dev_out):
                print(
                    f"Batch #{i:02d}: \t{dev_inp} == {dev_out}, (Delta ="
                    f" {dev_inp - dev_out}) \t\t\t\tfor input {inputs_rational[i]}"
                )

                print("\n")

            inference_result.append(raw_result)

        urc.fpga_leds(False, False, False, False)
        actual_result = parse_bytearray_to_fxp_tensor(
            inference_result, total_bits, frac_bits, expected_outputs.shape
        )

        assert torch.equal(actual_result, expected_outputs)


def main():
    # input 2 output 2 does not work
    inputs = 6
    outputs = 4
    total_bits = 8
    frac_bits = 2

    batchsize = 10

    skeleton_id = [randint(0, 16) for i in range(16)]
    skeleton_id_as_bytearray = bytearray()
    for x in skeleton_id:
        skeleton_id_as_bytearray.extend(
            x.to_bytes(length=1, byteorder="little", signed=False)
        )

    nn = build_vhdl_files(
        inputs=inputs,
        outputs=outputs,
        total_bits=total_bits,
        frac_bits=frac_bits,
        skeleton_id=skeleton_id,
    )

    input_tensors = torch.rand([batchsize, 1, inputs])
    inputs_rational, expected_outputs = simulate_result(
        total_bits=total_bits,
        frac_bits=frac_bits,
        nn=nn,
        input_tensors=input_tensors,
    )

    output_dir = "examples"
    synthesis(output_dir)

    test_on_device(
        outputs=outputs,
        total_bits=total_bits,
        frac_bits=frac_bits,
        inputs_rational=inputs_rational,
        expected_outputs=expected_outputs,
        skeleton_id_as_bytearray=skeleton_id_as_bytearray,
        binfile_path=Path(output_dir + "/results/impl/env5_top_reconfig.bin"),
    )


if __name__ == "__main__":
    main()
