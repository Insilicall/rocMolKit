# Estrutura espelha o upstream nvMolKit:
#   src/                  kernels HIP + lógica core
#   rdkit_extensions/     C++ puro, extensões do RDKit
#   nvmolkit/             bindings Python (boost-python)
#
# Fase 1: alvos compilam (cmake configure passa). Build real depende de hipcc.

set(ROCMOLKIT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ─── Core HIP library ───────────────────────────────────────
file(GLOB_RECURSE ROCMOLKIT_HIP_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hip.cpp"
)
file(GLOB_RECURSE ROCMOLKIT_CPP_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
list(REMOVE_ITEM ROCMOLKIT_CPP_SOURCES ${ROCMOLKIT_HIP_SOURCES})

# Excluir módulos que usam features sem equivalente direto em ROCm:
#   butina:                     CUDA Graphs Conditional (sem hipGraph equivalente)
#   tfd:                        depende de butina
#   substruct:                  cudaSharedmemCarveoutMaxShared não-traduzido
#   bfgs_hessian:               cooperative_groups/reduce.h (NVIDIA libcudacxx)
#   bfgs_minimize:              hipcub::DeviceReduce::TransformReduce (não em hipcub 6.2)
#   morgan_fingerprint_kernels: cooperative_groups::block_tile_memory (NVIDIA only)
#   similarity_kernels:         PTX inline asm (BMMA, async copy) em macros_ptx.hip.h
#   symmetric_eigensolver:      cuda::std::abs em contexto onde shim não resolve (Phase 4 reescrita)
list(FILTER ROCMOLKIT_HIP_SOURCES EXCLUDE REGEX "/(butina|tfd)\\.hip\\.cpp$")
list(FILTER ROCMOLKIT_HIP_SOURCES EXCLUDE REGEX "/(tfd|substruct)/")
list(FILTER ROCMOLKIT_HIP_SOURCES EXCLUDE REGEX "/bfgs_hessian\\.hip\\.cpp$")
list(FILTER ROCMOLKIT_HIP_SOURCES EXCLUDE REGEX "/bfgs_minimize\\.hip\\.cpp$")
list(FILTER ROCMOLKIT_HIP_SOURCES EXCLUDE REGEX "/morgan_fingerprint_kernels\\.hip\\.cpp$")
list(FILTER ROCMOLKIT_HIP_SOURCES EXCLUDE REGEX "/similarity_kernels\\.hip\\.cpp$")
list(FILTER ROCMOLKIT_HIP_SOURCES EXCLUDE REGEX "/symmetric_eigensolver\\.hip\\.cpp$")
list(FILTER ROCMOLKIT_CPP_SOURCES EXCLUDE REGEX "/(tfd|substruct|testutils)/")
list(FILTER ROCMOLKIT_CPP_SOURCES EXCLUDE REGEX "/(butina|tfd)\\.cpp$")
list(FILTER ROCMOLKIT_HIP_SOURCES EXCLUDE REGEX "/testutils/")

# Generate versions.h from RDKit version (drives RDKIT_NEW_FLAG_API constexpr).
if(RDKit_FOUND)
    if(NOT DEFINED RDKit_VERSION_MAJOR)
        # find_package may not have populated these — derive from CONFIG file or
        # parse RDKit's Version.h. Fallback to the Release_2024_09 we ship.
        set(RDKit_VERSION_MAJOR 2024)
        set(RDKit_VERSION_MINOR 9)
        set(RDKit_VERSION_PATCH 6)
    endif()
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/versions.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/versions.h"
        @ONLY
    )
endif()

if(ROCMOLKIT_HIP_SOURCES AND CMAKE_HIP_COMPILER AND hip_FOUND)
    set_source_files_properties(${ROCMOLKIT_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library(rocmolkit_core SHARED ${ROCMOLKIT_HIP_SOURCES} ${ROCMOLKIT_CPP_SOURCES})

    # Each upstream .cpp/.hip.cpp uses bare `#include "nvtx.h"` etc. (relative
    # to its own subdir). Mirror nvMolKit's CMake behavior: every subdir of
    # src/ becomes an include path.
    target_include_directories(rocmolkit_core PUBLIC
        $<BUILD_INTERFACE:${ROCMOLKIT_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>  # generated versions.h
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/utils>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/conformer>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/forcefields>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/minimizer>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/data_structures>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rdkit_extensions>
        $<INSTALL_INTERFACE:include>
    )

    # Force-include hip_compat.h in every translation unit so the
    # NVMOLKIT_CUDA_CC_* macros, cudaCheckError, and cudaSharedmemCarveout*
    # aliases are available without editing each upstream source.
    # Use generator expressions to gate options by source language:
    #   - HIP files get offload arch via HIP_ARCHITECTURES property already.
    #   - All files (HIP+CXX) get -include and warning suppressions.
    # CXX-only files (.cpp pure) must NOT see HIP-only flags like --offload-arch.
    target_compile_options(rocmolkit_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX,HIP>:-include${ROCMOLKIT_INCLUDE_DIR}/rocmolkit/hip_compat.h>
        $<$<COMPILE_LANGUAGE:CXX,HIP>:-Wno-enum-constexpr-conversion>
        $<$<COMPILE_LANGUAGE:CXX,HIP>:-Wno-deprecated-declarations>
    )

    target_link_libraries(rocmolkit_core PUBLIC hip::host hip::device)
    if(hipblas_FOUND)
        target_link_libraries(rocmolkit_core PUBLIC roc::hipblas)
    endif()
    if(hiprand_FOUND)
        target_link_libraries(rocmolkit_core PUBLIC hip::hiprand)
    endif()

    # RDKit: include + libs (resolved in top-level CMakeLists.txt).
    if(RDKit_FOUND)
        target_include_directories(rocmolkit_core PUBLIC ${RDKit_INCLUDE_DIRS})
        target_link_libraries(rocmolkit_core PUBLIC ${RDKit_LIBRARIES})
    endif()

    # Eigen (header-only) for the dist-geom builders.
    find_package(Eigen3 QUIET NO_MODULE)
    if(Eigen3_FOUND)
        target_link_libraries(rocmolkit_core PUBLIC Eigen3::Eigen)
    else()
        find_path(EIGEN3_INCLUDE_DIR Eigen/Core PATHS /usr/include/eigen3)
        if(EIGEN3_INCLUDE_DIR)
            target_include_directories(rocmolkit_core PUBLIC ${EIGEN3_INCLUDE_DIR})
        endif()
    endif()

    # OpenMP - used in bfgs_*, etkdg_*, tfd_*, substruct_* CPU paths.
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(rocmolkit_core PUBLIC OpenMP::OpenMP_CXX)
    endif()

    # Boost - dynamic_bitset (header-only), shared_ptr (header-only).
    # libboost-all-dev provides everything we need at apt level.
    find_package(Boost QUIET COMPONENTS system filesystem serialization)
    if(Boost_FOUND)
        target_link_libraries(rocmolkit_core PUBLIC Boost::headers)
    endif()

    set_target_properties(rocmolkit_core PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    # Apply --offload-arch ONLY to HIP language sources. The HIP_ARCHITECTURES
    # property leaks to all sources in CMake 3.30, breaking g++ on .cpp.
    foreach(_arch IN LISTS GPU_TARGETS)
        target_compile_options(rocmolkit_core PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${_arch}>
        )
    endforeach()

    install(TARGETS rocmolkit_core EXPORT rocmolkit-targets
            LIBRARY DESTINATION lib)
else()
    add_library(rocmolkit_core INTERFACE)
    target_include_directories(rocmolkit_core INTERFACE
        $<BUILD_INTERFACE:${ROCMOLKIT_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
    )
    install(TARGETS rocmolkit_core EXPORT rocmolkit-targets)
endif()

# ─── RDKit extensions (C++ puro) ────────────────────────────
file(GLOB ROCMOLKIT_RDKIT_EXT
    "${CMAKE_CURRENT_SOURCE_DIR}/rdkit_extensions/*.cpp"
)
if(ROCMOLKIT_RDKIT_EXT)
    add_library(rocmolkit_rdkit_ext STATIC ${ROCMOLKIT_RDKIT_EXT})
    target_include_directories(rocmolkit_rdkit_ext PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/rdkit_extensions
    )
    set_target_properties(rocmolkit_rdkit_ext PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# ─── Python bindings (boost-python ou nanobind) ─────────────
# Fase 2+: ativar quando ETKDG estiver compilando.
if(ROCMOLKIT_BUILD_PYTHON_BINDINGS AND TARGET rocmolkit_core)
    # add_subdirectory(nvmolkit)   # depois do hipify dos bindings
endif()

install(DIRECTORY ${ROCMOLKIT_INCLUDE_DIR}/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
