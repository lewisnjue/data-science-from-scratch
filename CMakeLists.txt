cmake_minimum_required(VERSION 3.16)

# Project name and C++ standard
project(ds_from_scratch VERSION 1.0 LANGUAGES CXX)

# Use modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------
# Library Target
# -----------------------------------------

# Automatically collect all .cpp files in src/
file(GLOB_RECURSE DS_SOURCES CONFIGURE_DEPENDS
     src/*.cpp)

add_library(ds ${DS_SOURCES})

# Tell the compiler where headers are
target_include_directories(ds
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# -----------------------------------------
# Example Executable
# -----------------------------------------

add_executable(example_linear_algebra
    examples/example_linear_algebra.cpp
)

target_link_libraries(example_linear_algebra PRIVATE ds)
target_include_directories(example_linear_algebra PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(example_statistics
    examples/statistics.cpp)

target_link_libraries(example_statistics PRIVATE ds)
target_include_directories(example_statistics PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(test_statistics 
tests/test_statistics.cpp
)
target_link_libraries(test_statistics PRIVATE ds)
target_include_directories(test_statistics PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_executable(
    test_linear_algebra
    tests/test_linear_algebra.cpp
)
target_link_libraries(test_linear_algebra PRIVATE ds)
target_include_directories(test_linear_algebra PRIVATE ${PROJECT_SOURCE_DIR}/include)
