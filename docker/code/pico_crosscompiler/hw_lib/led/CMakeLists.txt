add_library(led_lib INTERFACE)
target_sources(led_lib INTERFACE ${CMAKE_CURRENT_LIST_DIR}/led.cpp)
target_include_directories(led_lib INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(led_lib INTERFACE
    pico_stdlib
    hardware_pwm
)
