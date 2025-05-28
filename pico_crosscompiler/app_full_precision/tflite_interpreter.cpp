#include <cstdio>
#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "tflite_interpreter.h"

TfLiteInterpreter::TfLiteInterpreter(
    const uint8_t *const modelBuffer,
    tflite::MicroOpResolver &resolver,
    const uint32_t tensorArenaSize) : modelBuffer(modelBuffer),
                                      resolver(&resolver),
                                      tensorArenaSize(tensorArenaSize),
                                      tensorArena(new uint8_t[tensorArenaSize]),
                                      initialized(false) {}

int TfLiteInterpreter::initialize()
{
    tflite::InitializeTarget();

    this->model = tflite::GetModel(this->modelBuffer);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        printf(
            "Model provided is schema version %d not equal "
            "to supported version %d.\n",
            model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    interpreter = new tflite::MicroInterpreter(
        this->model, *this->resolver, this->tensorArena, this->tensorArenaSize);

    TfLiteStatus allocateStatus = interpreter->AllocateTensors();
    if (allocateStatus != kTfLiteOk)
    {
        printf("AllocateTensors() failed\n");
        return -2;
    }

    this->input = this->interpreter->input(0);
    this->output = this->interpreter->output(0);

    if ((this->input->type != kTfLiteFloat32) ||
        (this->output->type != kTfLiteFloat32))
    {
        printf("Expect model with Float32 input/output tensor\n");
    }

    this->inputFeatureCount = this->input->bytes;
    this->outputFeatureCount = this->output->bytes;

    this->initialized = true;

    return 0;
}

int TfLiteInterpreter::runInference(float *const inputBuffer, float *const outputBuffer)
{
    printf("Try inference. \n");

    if (!initialized)
    {
        printf("Interpreter must be initialized\n");
        return -1;
    }

    for (uint32_t inputIdx = 0; inputIdx < 784; inputIdx++)
    {
        // const float x = inputBuffer[inputIdx];
        this->input->data.f[inputIdx] = 1.0f;
    }

    TfLiteStatus invokeStatus = this->interpreter->Invoke();
    if (invokeStatus != kTfLiteOk)
    {
        printf("Invoke failed\n");
        return -2;
    }

    for (uint32_t outputIdx = 0; outputIdx < 10; outputIdx++)
    {
        float output_y = this->output->data.f[outputIdx];
        outputBuffer[outputIdx] = output_y;
        printf("Output is  %.04f \n", output_y);
    }

    printf("Got out \n");
    return 0;
}
