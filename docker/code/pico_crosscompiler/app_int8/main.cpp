#include <cstdio>
#include <cstdint>
#include <cmath>
#include <memory>

#include "pico/stdio.h"
#include "pico/stdlib.h"
#include "pico/time.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "model.h"
#include "tflite_interpreter.h"
#include "signal_queue.h"
#include "processing_functions.h"
//#include "led.h"
#include "hardware_setup.h"
#include "adxl345.h"

#define DEBUG_PRINT_FPS false
#define DEBUG_PRINT_CLASS_PROBS true

const uint32_t TENSOR_ARENA_SIZE = (140 * 1024);
const uint32_t CHANNEL_COUNT = 1;
const uint32_t INPUT_FEATURE_COUNT = CHANNEL_COUNT * 784;
const uint32_t OUTPUT_FEATURE_COUNT = 10;
const uint32_t INFERENCE_EVERY_NTH_POINTS = 10;
enum TargetClasses
{
    clsIdle,
    clsSnake,
    clsUpDown,
    clsWave,
    clsUndefined
};
std::unique_ptr<TfLiteInterpreter> interpreter = nullptr;

std::unique_ptr<TfLiteInterpreter> getInterpreter()
{
    std::unique_ptr<tflite::MicroMutableOpResolver<11>> resolver(new tflite::MicroMutableOpResolver<11>());

    resolver->AddAdd();
    resolver->AddRelu();
    resolver->AddFullyConnected();

    printf("Added layers\n");
    std::unique_ptr<TfLiteInterpreter> interpreter(new TfLiteInterpreter(model_tflite, *resolver, TENSOR_ARENA_SIZE));

    printf("Created Interpreter pointer.\n");
    interpreter->initialize();

    printf("Initialized Interpreter.\n");
    return interpreter;
}

void displayPredictedClass(float *predictions)
{
}

void runInference(SignalQueue &queue)
{
    static float inputBuffer[INPUT_FEATURE_COUNT];
    float outputBuffer[OUTPUT_FEATURE_COUNT];

    queue.copyToBuffer(inputBuffer);
    centerChannels(inputBuffer, INPUT_FEATURE_COUNT, CHANNEL_COUNT);
    interpreter->runInference(inputBuffer, outputBuffer);

#if DEBUG_PRINT_CLASS_PROBS
    printf(
        "Idle: %.04f ; Snake: %.04f ; UpDown: %.04f ; Wave: %.04f\n",
        outputBuffer[0], outputBuffer[1], outputBuffer[2], outputBuffer[3]);
#endif

}

int main()
{
    stdio_init_all();
    initializePeripherals();
    setup_adxl345();
    sleep_ms(5000);

    interpreter = getInterpreter();

    // SignalQueue queue(INPUT_FEATURE_COUNT, CHANNEL_COUNT);
    // queue.notifyOnOverflowingElement(INFERENCE_EVERY_NTH_POINTS, runInference);

    // int16_t accel[CHANNEL_COUNT];

#if DEBUG_PRINT_FPS
    uint64_t current_time, previous_time;
    previous_time = 0;
#endif

    while (true)
    {
#if DEBUG_PRINT_FPS
        current_time = to_us_since_boot(get_absolute_time());
        printf("FPS: %f\n", 1.0f / (current_time - previous_time) / 1e-6);
        previous_time = current_time;
#endif

        //     adxl345_readData(&accel[0], &accel[1], &accel[2]);
        //     queue.add(accel);

        sleep_ms(150);
    }

    return 0;
}