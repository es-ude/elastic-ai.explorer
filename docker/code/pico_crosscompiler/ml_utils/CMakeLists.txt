add_library(ml_utils INTERFACE)
file(GLOB_RECURSE SRC 
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)
target_sources(ml_utils INTERFACE ${SRC})
target_include_directories(ml_utils INTERFACE ${CMAKE_CURRENT_LIST_DIR})