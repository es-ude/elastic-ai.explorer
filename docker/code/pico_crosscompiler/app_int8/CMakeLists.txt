add_executable(app_int8)

file(GLOB_RECURSE SRC 
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)
target_sources(app_int8 PRIVATE ${SRC})
target_include_directories(app_int8 PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(app_int8
    hardware_setup
    sensor_lib
    led_lib
    ml_utils
    pico_stdlib
    pico-tflmicro
)

pico_enable_stdio_usb(app_int8 1)
pico_enable_stdio_uart(app_int8 0)
pico_add_extra_outputs(app_int8)