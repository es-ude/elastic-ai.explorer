add_executable(measure_accuracy)

file(GLOB_RECURSE SRC 
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)
target_sources(measure_accuracy PRIVATE ${SRC})
target_include_directories(measure_accuracy PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(measure_accuracy
    hardware_setup
    sensor_lib
    led_lib
    ml_utils
    data
    pico_stdlib
    pico-tflmicro
)

pico_enable_stdio_usb(measure_accuracy 1)
pico_enable_stdio_uart(measure_accuracy 0)
pico_add_extra_outputs(measure_accuracy)