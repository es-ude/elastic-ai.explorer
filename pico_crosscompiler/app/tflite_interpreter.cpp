#include <cstdio>
#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "tflite_interpreter.h"

#define DEBUG_SIZES false
#define DEBUG_QUANT true

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

    if ((this->input->type != kTfLiteInt8) ||
        (this->output->type != kTfLiteInt8))
    {
        printf("Expect model with int8 input/output tensor\n");
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

    for (uint32_t inputIdx = 0; inputIdx < this->inputFeatureCount; inputIdx++)
    {
#if DEBUG_SIZES
        printf("inputFeatureCount %u\n", this->inputFeatureCount);
#endif
        // const float x = inputBuffer[inputIdx];
        this->input->data.int8[inputIdx] = quantize(1.0f);

    }

    TfLiteStatus invokeStatus = this->interpreter->Invoke();
    if (invokeStatus != kTfLiteOk)
    {
        printf("Invoke failed\n");
        return -2;
    }

    for (uint32_t outputIdx = 0; outputIdx < this->outputFeatureCount; outputIdx++)
    {
#if DEBUG_SIZES
        printf("outputFeatureCount %u\n", this->outputFeatureCount);
#endif

        const int8_t quant_y = this->output->data.int8[outputIdx];
        printf("Quantized Output is  %d \n", quant_y);

        outputBuffer[outputIdx] = dequantize(quant_y);
        int8_t output_y = dequantize(quant_y);
        printf("Dequantized Output is  %d \n", output_y);
    }

#if DEBUG_QUANT
    printf("output param.scale: %.04f\n", this->output->params.scale);
    printf("output param.zeropoint: %d\n", this->output->params.zero_point);
    printf("input param.scale: %.04f \n", this->input->params.scale);
    printf("input param.zeropoint: %d\n", this->input->params.zero_point);
#endif
    printf("Got out \n");
    return 0;
}

int8_t TfLiteInterpreter::quantize(float x)
{

    return x / this->input->params.scale + this->input->params.zero_point;
}

float TfLiteInterpreter::dequantize(int8_t x)
{

    return (x - this->output->params.zero_point) * this->output->params.scale;
}
