add_executable(measure_latency)

file(GLOB_RECURSE SRC 
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)
target_sources(measure_latency PRIVATE ${SRC})
target_include_directories(measure_latency PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(measure_latency
    hardware_setup
    sensor_lib
    led_lib
    data
    ml_utils
    pico_stdlib
    pico-tflmicro
)

pico_enable_stdio_usb(measure_latency 1)
pico_enable_stdio_uart(measure_latency 0)
pico_add_extra_outputs(measure_latency)