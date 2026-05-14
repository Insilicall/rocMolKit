cmake_minimum_required(VERSION 3.21)

project(rocMolKit
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated RDKit operations on AMD via HIP/ROCm"
    LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# HIP language inherits CXX standard, but be explicit:
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ROCMOLKIT_BUILD_TESTS "Build unit tests" OFF)
option(ROCMOLKIT_BUILD_BENCHMARKS "Build benchmarks" OFF)
# Phase 2 - default ON since _embedMolecules.so links cleanly. Other bindings
# (fingerprints, clustering, substructure, TFD) stay off in nvmolkit/CMakeLists.txt
# until their backing kernels port over (Phases 4-7).
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

include(CheckLanguage)
check_language(HIP)
if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    find_package(hip QUIET)
    find_package(hipblas QUIET)
    find_package(hiprand QUIET)
    find_package(rocsolver QUIET)
else()
    message(WARNING "rocMolKit: HIP compiler not found — only CPU/header targets will build. "
                    "Install ROCm SDK (hipcc) to build GPU code.")
endif()

# RDKit: install via conda-forge (micromamba) so we always get a recent
# release (>=2024.09). Pass -DRDKit_ROOT=/opt/rdkit (or set CMAKE_PREFIX_PATH).
# Avoid Ubuntu's librdkit-dev — it ships ancient 2021.09 and lacks DistGeom
# helpers used by ETKDG.
if(DEFINED RDKit_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${RDKit_ROOT}")
endif()
find_package(RDKit QUIET CONFIG)
if(NOT RDKit_FOUND)
    set(_RDKIT_HINTS
        ${RDKit_ROOT}
        $ENV{RDKIT_ROOT}
        $ENV{CONDA_PREFIX}
        /opt/rdkit
        /opt/conda
        /usr/local
    )
    find_path(RDKIT_INCLUDE_DIR
        NAMES rdkit/GraphMol/RDKitBase.h GraphMol/RDKitBase.h
        HINTS ${_RDKIT_HINTS}
        PATH_SUFFIXES include include/rdkit
    )
    file(GLOB _RDKIT_LIBS
        "${RDKit_ROOT}/lib/libRDKit*.so"
        "$ENV{CONDA_PREFIX}/lib/libRDKit*.so"
        "/opt/rdkit/lib/libRDKit*.so"
    )
    if(RDKIT_INCLUDE_DIR AND _RDKIT_LIBS)
        message(STATUS "rocMolKit: RDKit at ${RDKIT_INCLUDE_DIR} (manual probe)")
        set(RDKit_FOUND TRUE)
        # Export both `<root>/include` and `<root>/include/rdkit` so that
        # both `#include <GraphMol/...>` and `#include <rdkit/GraphMol/...>` work.
        set(RDKit_INCLUDE_DIRS ${RDKIT_INCLUDE_DIR} ${RDKIT_INCLUDE_DIR}/rdkit)
        set(RDKit_LIBRARIES ${_RDKIT_LIBS})
    else()
        message(WARNING "rocMolKit: RDKit not found. Install via "
                        "'micromamba install -c conda-forge rdkit librdkit-dev' "
                        "and pass -DRDKit_ROOT=<prefix>.")
    endif()
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
