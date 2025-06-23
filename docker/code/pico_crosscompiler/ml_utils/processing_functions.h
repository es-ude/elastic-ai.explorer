#ifndef PROCESSING_FUNCTIONS_H
#define PROCESSING_FUNCTIONS_H

#include <cstdint>


void centerChannels(float values[], const uint32_t length, const uint32_t channels);
uint32_t argmax(const float values[], const uint32_t length);

#endif