#include <cstdio>
#include <cstdint>
#include <cmath>
#include <memory>

#include "pico/stdio.h"
#include "pico/stdlib.h"
#include "pico/time.h"
#include "pico/bootrom.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "mnist_test_images.h"
#include "mnist_labels.h"

#include "model.h"
#include "tflite_interpreter.h"
#include "signal_queue.h"
#include "processing_functions.h"
// #include "led.h"
#include "hardware_setup.h"
#include "adxl345.h"

const uint32_t TENSOR_ARENA_SIZE = (50 * 1024);
const uint32_t CHANNEL_COUNT = 1;
const uint32_t INPUT_FEATURE_COUNT = CHANNEL_COUNT * 784;
const uint32_t OUTPUT_FEATURE_COUNT = 10;
const uint32_t INFERENCE_EVERY_NTH_POINTS = 10;

std::unique_ptr<TfLiteInterpreter> interpreter = nullptr;

std::unique_ptr<TfLiteInterpreter> getInterpreter()
{
    std::unique_ptr<tflite::MicroMutableOpResolver<11>> resolver(new tflite::MicroMutableOpResolver<11>());

    resolver->AddAdd();
    resolver->AddRelu();
    resolver->AddFullyConnected();

    // printf("Added layers\n");
    std::unique_ptr<TfLiteInterpreter> interpreter(new TfLiteInterpreter(model_tflite, *resolver, TENSOR_ARENA_SIZE));

    // printf("Created Interpreter pointer.\n");
    interpreter->initialize();

    // printf("Initialized Interpreter.\n");
    return interpreter;
}

void doFirmwareUpgradeReset()
{
    reset_usb_boot(1, 0);
}
int runInference(int dataset_size)
{
    static float inputBuffer[INPUT_FEATURE_COUNT];
    float outputBuffer[OUTPUT_FEATURE_COUNT];

    int correct = 0;
    for (uint32_t sample_index = 0; sample_index < dataset_size; sample_index++)
    {
        // printf("Counter: %d\n", sample_index);
        memcpy(inputBuffer, mnist_test_images[sample_index], sizeof(float) * INPUT_FEATURE_COUNT);
        centerChannels(inputBuffer, INPUT_FEATURE_COUNT, CHANNEL_COUNT);
        int result = interpreter->runInference(inputBuffer, outputBuffer);
        sleep_ms(15);
        if (mnist_labels[sample_index] == result)
        {
            correct++;
        }
    }

    return correct;
}

int main()
{
    stdio_init_all();
    initializePeripherals();
    setup_adxl345();
    sleep_ms(2000);

    int dataset_size = 128;

    interpreter = getInterpreter();

    uint64_t current_time, previous_time;
    previous_time = to_us_since_boot(get_absolute_time());
    int correct = runInference(dataset_size);
    current_time = to_us_since_boot(get_absolute_time());

    printf("{ \"Latency\": { \"value\": %llu, \"unit\": \"microseconds\"}}", current_time - previous_time);
    printf("|");
    printf("{\"Accuracy\": { \"value\":  %.3f, \"unit\": \"percent\"}}", static_cast<double>(correct) / dataset_size);

    sleep_ms(2000);
    doFirmwareUpgradeReset();
    return 0;
}
