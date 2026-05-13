# Mapping de gfx targets e flags HIP comuns.
# Use -DGPU_TARGETS="gfx1100;gfx90a" para customizar.

if(NOT DEFINED GPU_TARGETS)
    set(GPU_TARGETS "gfx1100;gfx90a;gfx942")
endif()

# AMDGPU_TARGETS é o nome legado; HIP CMake aceita ambos
set(AMDGPU_TARGETS "${GPU_TARGETS}" CACHE STRING "AMD GPU targets" FORCE)
set(GPU_TARGETS "${GPU_TARGETS}" CACHE STRING "AMD GPU targets" FORCE)

# Wavefront 64 é o default em CDNA (MI*); RDNA usa 32 mas HIP normaliza via warpSize.
# Não fixar -mwavefrontsize; deixar o compilador decidir por target.

function(rocmolkit_add_hip_library name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LINK_LIBRARIES" ${ARGN})

    set_source_files_properties(${ARG_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library(${name} ${ARG_SOURCES})

    target_link_libraries(${name} PUBLIC
        hip::host
        hip::device
        ${ARG_LINK_LIBRARIES}
    )

    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-result>
        $<$<COMPILE_LANGUAGE:HIP>:-fno-gpu-rdc>
    )

    set_target_properties(${name} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        HIP_ARCHITECTURES "${GPU_TARGETS}"
    )
endfunction()
