add_executable(lstm_on_pico)

file(GLOB_RECURSE SRC 
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)
target_sources(lstm_on_pico PRIVATE ${SRC})
target_include_directories(lstm_on_pico PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(lstm_on_pico
    hardware_setup
    sensor_lib
    led_lib
    ml_utils
    data
    pico_stdlib
    pico-tflmicro
)

pico_enable_stdio_usb(lstm_on_pico 1)
pico_enable_stdio_uart(lstm_on_pico 0)
pico_add_extra_outputs(lstm_on_pico)