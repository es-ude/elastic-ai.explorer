#include <cstdio>
#include <cstdint>
#include <cmath>
#include <memory>

#include "pico/stdio.h"
#include "pico/stdlib.h"
#include "pico/time.h"
#include "pico/bootrom.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "mnist_features.h"
#include "mnist_labels.h"

#include "model.h"
#include "tflite_interpreter.h"
#include "signal_queue.h"
#include "output_functions.h"
#include "hardware_setup.h"
#include "adxl345.h"

const uint32_t TENSOR_ARENA_SIZE = (100 * 1024);
const uint32_t CHANNEL_COUNT = 1;
const uint32_t INPUT_FEATURE_COUNT = CHANNEL_COUNT * 784;
const uint32_t OUTPUT_FEATURE_COUNT = 10;
const uint32_t INFERENCE_EVERY_NTH_POINTS = 10;

std::unique_ptr<TfLiteInterpreter> interpreter = nullptr;

std::unique_ptr<TfLiteInterpreter> getInterpreter()
{
    std::unique_ptr<tflite::MicroMutableOpResolver<11>> resolver(new tflite::MicroMutableOpResolver<11>());
#if __has_include("resolver_ops.h")
#include "resolver_ops.h"
#endif
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

        memcpy(inputBuffer, mnist_images[sample_index], sizeof(float) * INPUT_FEATURE_COUNT);
        int result = interpreter->runInference(inputBuffer, outputBuffer);
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
    sleep_ms(2000);
    int dataset_size = 256;

    interpreter = getInterpreter();
    int correct = runInference(dataset_size);
    printf("{\"Accuracy\": { \"value\":  %.3f, \"unit\": \"percent\"}}", (static_cast<double>(correct) / dataset_size) * 100);

    sleep_ms(2000);
    doFirmwareUpgradeReset();

    return 0;
}
