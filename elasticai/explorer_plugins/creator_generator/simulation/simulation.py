import json
import os


from pathlib import Path


import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from numpy import argmax

from elasticai.explorer_plugins.creator_generator.simulation.simulation_utils import (
    read_testdata,
)
from cocotb.utils import get_sim_time


@cocotb.test()
async def accuracy_latency_test(dut):

    test_dir = Path(
        os.getenv(
            "TEST_DIR",
            "tests/integration_tests/test_experiment/simulation/sim_results.json",
        )
    )
    data = read_testdata(test_dir)

    num_feat_output = data["out_dim"]
    clock_period_ns = 10

    dut.enable.value = 0  # Has no impact
    dut.clock.value = 0  # has no impact
    dut.x_address.value = 0
    dut.y_address.value = 0
    dut.x.value = 0

    cocotb.start_soon(Clock(dut.clock, period=clock_period_ns, unit="ns").start())
    await Timer(4 * clock_period_ns, unit="ns")
    await RisingEdge(dut.clock)
    iterations = 0
    correct = 0

    start_time = get_sim_time(unit="ns")
    for ite, (sig_in, target) in enumerate(zip(data["in"], data["target"])):
        result = list()

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
            result.append(dut.y.value.to_signed())
            for _ in range(2):
                await RisingEdge(dut.clock)

        dut.y_address.value = 0

        # --- Do reset
        for _ in range(2):
            await RisingEdge(dut.clock)

        dut.enable.value = 0
        for _ in range(2):
            await RisingEdge(dut.clock)
        if argmax(result) == target:
            correct += 1
        iterations += 1
    end_time = get_sim_time(unit="ns")

    total_latency = end_time - start_time
    latency_per_sample = total_latency / iterations
    accuracy = correct / iterations

    result_path = Path(
        os.getenv(
            "SIM_RESULT_FILE",
            "tests/integration_tests/test_experiment/simulation/sim_results.json",
        )
    ).resolve()
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps({"accuracy_percent": accuracy * 100, "latency_ns": latency_per_sample})
    )
