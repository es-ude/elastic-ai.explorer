#include <cstdint>
#include "processing_functions.h"


void calculateChannelwiseMean(
        const float inValues[],
        const uint32_t length,
        const uint32_t channels,
        float outMeans[]) {

    for (uint32_t chIdx = 0; chIdx < channels; chIdx++) {
        outMeans[chIdx] = 0;

        for (uint32_t valIdx = chIdx; valIdx < length; valIdx += channels) {
            outMeans[chIdx] += inValues[valIdx];
        }

        outMeans[chIdx] /= length / channels;
    }
}


void centerChannels(
        float values[],
        const uint32_t length,
        const uint32_t channels) {

    float means[channels];
    calculateChannelwiseMean(values, length, channels, means);
    
    for (uint32_t chIdx = 0; chIdx < channels; chIdx++) {
        for (uint32_t valIdx = chIdx; valIdx < length; valIdx += channels) {
            values[valIdx] = values[valIdx] - means[chIdx];
        }
    }
}


uint32_t argmax(const float values[], const uint32_t length) {
    uint32_t maxIdx = 0;
    for (uint32_t idx = 1; idx < length; idx++) {
        if (values[idx] > values[maxIdx]) {
            maxIdx = idx;
        }
    }
    return maxIdx;
}
