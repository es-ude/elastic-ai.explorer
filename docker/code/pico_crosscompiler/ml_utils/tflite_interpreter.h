#ifndef TFLITE_INTERPRETER_H
#define TFLITE_INTERPRETER_H

#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

class TfLiteInterpreter
{
public:
    TfLiteInterpreter(
        const uint8_t *const modelBuffer,
        tflite::MicroOpResolver &resolver,
        const uint32_t tensorArenaSize,
        bool is_quant = false);
    int initialize();
    int8_t quantize(float x);
    float dequantize(int8_t x);
    int runInference(float *const inputBuffer, float *const outputBuffer);
    TfLiteTensor *input;
    TfLiteTensor *output;
private:
    const uint32_t tensorArenaSize;
    uint8_t *const tensorArena;
    uint32_t inputFeatureCount, outputFeatureCount;
    const uint8_t *const modelBuffer;
    bool is_quant;
    const tflite::Model *model;
    tflite::MicroOpResolver *resolver;
    tflite::MicroInterpreter *interpreter;
    bool initialized;
};

#endif