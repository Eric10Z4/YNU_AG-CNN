cmake_minimum_required(VERSION 3.18)
project(gat_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Single-config generators (e.g., Ninja) should default to Release for runtime speed.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Try to discover pybind11 from the active Python environment.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(mcts_cpp
    src/bindings.cpp
    src/board.cpp
    src/mcts_stub.cpp
)

target_include_directories(mcts_cpp PRIVATE src)

if(MSVC)
    target_compile_options(mcts_cpp PRIVATE /W4 /EHsc /O2 /GL)
    target_link_options(mcts_cpp PRIVATE /LTCG)
else()
    target_compile_options(
        mcts_cpp PRIVATE
        -Wall -Wextra -Wpedantic
        -O3 -march=native -mtune=native
    )
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported)
    if(ipo_supported)
        set_property(TARGET mcts_cpp PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()
