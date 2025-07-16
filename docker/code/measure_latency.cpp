#include <ATen/ATen.h>
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <cinttypes>
#include <chrono>

#define NUM_WARMUP_RUNS 5
#define NUM_MEASURE_RUNS 10

int main(int argc, const char *argv[])
{
    at::globalContext().setQEngine(at::QEngine::QNNPACK);

    if (argc != 2)
    {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error &e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 1, 28, 28}));

    c10::IValue output;

    for (uint32_t runIdx = 0; runIdx < NUM_WARMUP_RUNS; runIdx++)
    {
        output = module.forward(inputs);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t runIdx = 0; runIdx < NUM_MEASURE_RUNS; runIdx++)
    {
        output = module.forward(inputs);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed = t1 - t0;
    uint64_t microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    microseconds = microseconds / NUM_MEASURE_RUNS;

    std::printf("{ \"Latency\": { \"value\":  %u, \"unit\": \"microseconds\"}}", microseconds);
}