from pathlib import Path

import numpy as np

import serial  # type: ignore
import torch

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)

from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import Linear, ReLU
from elasticai.runtime.env5.usb import UserRemoteControl, get_env5_port

from tests.system_tests.helper.parse_tensors_to_bytearray import (
    parse_bytearray_to_fxp_tensor,
    parse_fxp_tensor_to_bytearray,
)


# --- Open Serial Communication to Device
def main():

    num_inputs = 2
    num_outputs = 2
    total_bits = 8
    frac_bits = 4

    nn = Sequential(
        Linear(
            in_features=num_inputs,
            out_features=num_outputs,
            total_bits=total_bits,
            frac_bits=frac_bits,
        ),
        ReLU(total_bits=total_bits),
    )
    skelecton_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    skeleton_id_as_bytearray = bytearray()
    for x in skelecton_id:
        skeleton_id_as_bytearray.extend(
            x.to_bytes(length=1, byteorder="little", signed=False)
        )
    binfile_dir = "./tests/system_tests/samples/results"
    binfile_path = Path(binfile_dir + "/impl/env5_top_reconfig.bin")
    dev_address = get_env5_port()

    # --- Processing
    # --- Creating the dummy
    input_tensor = torch.Tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]])

    for idx_array in range(0, num_inputs):
        for value in np.arange(-2, 2, 0.5):
            list_zeros = [0.0 for idx in range(0, num_inputs)]
            list_zeros[idx_array] = value
            input_tensor = torch.cat(
                (input_tensor, torch.Tensor([[list_zeros]])), dim=0
            )
        # urc.deploy_model(flash_start_address, skeleton_id_as_bytearray)

    fxp_params = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    fxp_conf = FxpArithmetic(fxp_params)
    inputs = fxp_conf.as_rational(fxp_conf.cut_as_integer(input_tensor))
    #expected_outputs = nn(inputs)

    # --- Doing the test
    with serial.Serial(dev_address) as serial_con:
        flash_start_address = 0
        urc = UserRemoteControl(device=serial_con)
        urc.send_and_deploy_model(
            binfile_path, flash_start_address, skeleton_id_as_bytearray
        )
        # urc.deploy_model(flash_start_address, skeleton_id_as_bytearray)

        # --- Doing the test
        batch_data = parse_fxp_tensor_to_bytearray(inputs, total_bits, frac_bits)
        inference_result = list()
        state = False
        urc.fpga_leds(True, False, False, False)
        for i, sample in enumerate(batch_data):
            urc.fpga_leds(True, False, False, state)
            state = False if state else True

            batch_result = urc.inference_with_data(sample, num_outputs)
            my_result = parse_bytearray_to_fxp_tensor(
                [batch_result], total_bits, frac_bits, (1, 1, 3)
            )

            dev_inp = my_result
            dev_out = expected_outputs.data[i].view((1, 1, 3))
            if not torch.equal(dev_inp, dev_out):
                print(
                    f"Batch #{i:02d}: \t{dev_inp} == {dev_out}, (Delta ="
                    f" {dev_inp - dev_out}) \t\t\t\tfor input {inputs[i]}"
                )
                if i % 4 == 3:
                    print("\n")

            inference_result.append(batch_result)

        urc.fpga_leds(False, False, False, False)
        actual_result = parse_bytearray_to_fxp_tensor(
            inference_result, total_bits, frac_bits, expected_outputs.shape
        )

        assert torch.equal(actual_result, expected_outputs)


if __name__ == "__main__":
    main()
