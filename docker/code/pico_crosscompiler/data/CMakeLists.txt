add_library(data INTERFACE)
file(GLOB_RECURSE SRC 
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)
target_sources(data INTERFACE ${SRC})
target_include_directories(data INTERFACE ${CMAKE_CURRENT_LIST_DIR})