cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

set(PICO_SDK_FETCH_FROM_GIT Off)
set(PICO_SDK_PATH ${CMAKE_SOURCE_DIR}/third_party/pico-sdk)

set(PICO_PLATFORM rp2040)
# set(PICO_PLATFORM rp2350-arm-s)

include(pico_sdk_import.cmake)

project(gesture_recognition CXX C ASM)

pico_sdk_init()

add_subdirectory(lstm_on_pico)
#add_subdirectory(measure_accuracy)
#add_subdirectory(measure_latency)
add_subdirectory(data)
add_subdirectory(hw_lib)
add_subdirectory(ml_utils)
add_subdirectory(third_party)