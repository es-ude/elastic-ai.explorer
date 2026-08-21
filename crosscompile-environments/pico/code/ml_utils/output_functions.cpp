#include <cstdint>
#include "output_functions.h"


uint32_t argmax(const float values[], const uint32_t length) {
    uint32_t maxIdx = 0;
    for (uint32_t idx = 1; idx < length; idx++) {
        if (values[idx] > values[maxIdx]) {
            maxIdx = idx;
        }
    }
    return maxIdx;
}
