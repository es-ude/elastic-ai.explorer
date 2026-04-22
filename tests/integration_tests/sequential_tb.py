import fnmatch

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from elasticai.creator.testing.cocotb_prepare import read_testdata


@cocotb.test()
async def layer_computation_test(dut):
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
    limit = 0.9
    assert accuracy >= limit, f"Accuracy of {accuracy * 100:.2f}%"

