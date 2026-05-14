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
#   butina:    CUDA Graphs Conditional (sem hipGraph equivalente)
#   tfd:       depende de butina
#   substruct: cudaSharedmemCarveoutMaxShared não-traduzido (Fase 7)
list(FILTER ROCMOLKIT_HIP_SOURCES EXCLUDE REGEX "/(butina|tfd)\\.hip\\.cpp$")
list(FILTER ROCMOLKIT_HIP_SOURCES EXCLUDE REGEX "/(tfd|substruct)/")
list(FILTER ROCMOLKIT_CPP_SOURCES EXCLUDE REGEX "/(tfd|substruct)/")
list(FILTER ROCMOLKIT_CPP_SOURCES EXCLUDE REGEX "/(butina|tfd)\\.cpp$")

if(ROCMOLKIT_HIP_SOURCES AND CMAKE_HIP_COMPILER AND hip_FOUND)
    set_source_files_properties(${ROCMOLKIT_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library(rocmolkit_core SHARED ${ROCMOLKIT_HIP_SOURCES} ${ROCMOLKIT_CPP_SOURCES})

    # Each upstream .cpp/.hip.cpp uses bare `#include "nvtx.h"` etc. (relative
    # to its own subdir). Mirror nvMolKit's CMake behavior: every subdir of
    # src/ becomes an include path.
    target_include_directories(rocmolkit_core PUBLIC
        $<BUILD_INTERFACE:${ROCMOLKIT_INCLUDE_DIR}>
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
    target_compile_options(rocmolkit_core PRIVATE
        -include "${ROCMOLKIT_INCLUDE_DIR}/rocmolkit/hip_compat.h"
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
        HIP_ARCHITECTURES "${GPU_TARGETS}"
    )

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
