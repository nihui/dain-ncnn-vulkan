cmake_policy(SET CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

project(dain-ncnn-vulkan)

cmake_minimum_required(VERSION 3.9)

set(CMAKE_BUILD_TYPE Release)

find_package(Threads)
find_package(OpenMP)
find_package(Vulkan REQUIRED)

macro(ncnn_add_shader SHADER_SRC)
    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    set(SHADER_COMP_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.comp.hex.h)

    add_custom_command(
        OUTPUT ${SHADER_COMP_HEADER}
        COMMAND ${CMAKE_COMMAND} -DSHADER_SRC=${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC} -DSHADER_COMP_HEADER=${SHADER_COMP_HEADER} -P "${CMAKE_CURRENT_SOURCE_DIR}/generate_shader_comp_header.cmake"
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC}
        COMMENT "Preprocessing shader source ${SHADER_SRC_NAME_WE}.comp"
        VERBATIM
    )
    set_source_files_properties(${SHADER_COMP_HEADER} PROPERTIES GENERATED TRUE)

    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_COMP_HEADER})
endmacro()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(GLSLANG_TARGET_DIR "GLSLANG-NOTFOUND" CACHE PATH "Absolute path to glslangTargets.cmake directory")
if(NOT GLSLANG_TARGET_DIR AND NOT DEFINED ENV{GLSLANG_TARGET_DIR})
    message(FATAL_ERROR "GLSLANG_TARGET_DIR must be defined!")
else()
    message(STATUS "Using glslang install located at ${GLSLANG_TARGET_DIR}")

    find_package(Threads)

    include("${GLSLANG_TARGET_DIR}/OSDependentTargets.cmake")
    include("${GLSLANG_TARGET_DIR}/OGLCompilerTargets.cmake")
    if(EXISTS "${GLSLANG_TARGET_DIR}/HLSLTargets.cmake")
        # hlsl support can be optional
        include("${GLSLANG_TARGET_DIR}/HLSLTargets.cmake")
    endif()
    include("${GLSLANG_TARGET_DIR}/glslangTargets.cmake")
    include("${GLSLANG_TARGET_DIR}/SPIRVTargets.cmake")

    if (NOT TARGET glslang OR NOT TARGET SPIRV)
        message(FATAL_ERROR "glslang or SPIRV target not found!")
    endif()
endif()

find_package(ncnn)

ncnn_add_shader(dain_preproc.comp)
ncnn_add_shader(dain_postproc.comp)
ncnn_add_shader(correlation.comp)
ncnn_add_shader(correlation_pack4to1.comp)
ncnn_add_shader(depthflowprojection.comp)
ncnn_add_shader(depthflowprojection_fillhole.comp)
ncnn_add_shader(filterinterpolation.comp)
ncnn_add_shader(filterinterpolation_pack4.comp)
ncnn_add_shader(opticalflowwarp.comp)
ncnn_add_shader(opticalflowwarp_pack4.comp)

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

add_executable(dain-ncnn-vulkan
    correlation.cpp
    dain.cpp
    depthflowprojection.cpp
    filterinterpolation.cpp
    main.cpp
    opticalflowwarp.cpp
)

add_dependencies(dain-ncnn-vulkan generate-spirv)

target_link_libraries(dain-ncnn-vulkan ncnn ${Vulkan_LIBRARY})
