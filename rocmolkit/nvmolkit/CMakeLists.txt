# Python bindings via boost-python.
# Each MODULE library becomes a `_<Name>.so` extension importable from Python.
#
# Phase 2 strategy: enable only bindings whose backing C++ targets exist in
# rocmolkit_core (i.e. modules NOT excluded for Phase 4-7). Fingerprints,
# clustering, substructure, TFD bindings stay disabled until their kernels
# port over.

if(NOT ROCMOLKIT_BUILD_PYTHON_BINDINGS OR NOT TARGET rocmolkit_core)
    return()
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)

set(CMAKE_SHARED_MODULE_PREFIX "")

# Bindings active in Phase 2:
set(ROCMOLKIT_PHASE2_BINDINGS
    embedMolecules
    mmffOptimization
    uffOptimization
    batchedForcefield
    conformerRmsd
    array_helpers
)

# Each name corresponds to a `_<NameCamelCase>.so` plus `<source>.cpp`.
foreach(_binding IN LISTS ROCMOLKIT_PHASE2_BINDINGS)
    set(_module "_${_binding}")
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${_binding}.cpp")

    if(NOT EXISTS "${_src}")
        message(WARNING "rocMolKit: binding source missing: ${_src}")
        continue()
    endif()

    add_library(${_module} MODULE ${_src})
    target_include_directories(${_module}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${Python_INCLUDE_DIRS}
        SYSTEM PRIVATE
            ${Boost_INCLUDE_DIRS}
    )
    target_link_libraries(${_module}
        PRIVATE
            rocmolkit_core
            Boost::python
            Python::Module
    )
    set_target_properties(${_module} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        PREFIX ""
        SUFFIX ".so"
    )
    install(TARGETS ${_module}
        LIBRARY DESTINATION ${Python_SITELIB}/rocmolkit
    )
endforeach()
