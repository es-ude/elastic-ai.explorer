add_library(sensor_lib INTERFACE)
target_sources(sensor_lib INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/adxl345.cpp
)
target_include_directories(sensor_lib INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(sensor_lib INTERFACE pico_stdlib)
