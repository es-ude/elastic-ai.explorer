add_library(hardware_setup INTERFACE)

target_sources(hardware_setup INTERFACE
	${CMAKE_CURRENT_LIST_DIR}/hardware_setup.cpp
)

target_include_directories(hardware_setup INTERFACE
	${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(hardware_setup INTERFACE
	hardware_adc 
	hardware_dma 
	pico_stdlib 
	hardware_i2c 
	sensor_lib
)

