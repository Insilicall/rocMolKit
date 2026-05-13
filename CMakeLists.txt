cmake_minimum_required(VERSION 3.21)

project(rocMolKit
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated RDKit operations on AMD via HIP/ROCm"
    LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ROCMOLKIT_BUILD_TESTS "Build unit tests" OFF)
option(ROCMOLKIT_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ROCMOLKIT_BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(NOT DEFINED GPU_TARGETS)
    set(GPU_TARGETS "gfx1100;gfx90a;gfx942" CACHE STRING "AMD GPU targets")
endif()

include(ROCmTargets)

if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${ROCM_PATH}/lib/cmake")

find_package(hip QUIET)
find_package(hipblas QUIET)
find_package(hiprand QUIET)
find_package(rocsolver QUIET)

if(NOT hip_FOUND)
    message(WARNING "rocMolKit: HIP runtime not found — only header-only / CPU targets will build. "
                    "Install ROCm SDK to build GPU targets.")
endif()

message(STATUS "rocMolKit: ROCm at ${ROCM_PATH}")
message(STATUS "rocMolKit: GPU_TARGETS = ${GPU_TARGETS}")

add_subdirectory(rocmolkit)

if(ROCMOLKIT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ROCMOLKIT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
