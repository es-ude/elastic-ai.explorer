#ifndef TFLITE_INTERPRETER_H
#define TFLITE_INTERPRETER_H

#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"


class TfLiteInterpreter {
    public:
        TfLiteInterpreter(
            const uint8_t* const modelBuffer,
            tflite::MicroOpResolver& resolver,
            const uint32_t tensorArenaSize);
        int initialize();
        int runInference(float* const inputBuffer, float* const outputBuffer);
    private:
        const uint32_t tensorArenaSize;
        uint8_t* const tensorArena;
        uint32_t inputFeatureCount, outputFeatureCount;
        const uint8_t* const modelBuffer;
        const tflite::Model* model;
        tflite::MicroOpResolver* resolver;
        tflite::MicroInterpreter* interpreter;
        TfLiteTensor* input;
        TfLiteTensor* output;
        bool initialized;

        uint8_t quantize(float x);
        float dequantize(uint8_t x);
};

#endif