cmake_minimum_required(VERSION 3.20)
project(rfdetr VERSION 2.0.0 LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(RFDETR_BUILD_DEMO     "Build the demo CLI"                 ON)
option(RFDETR_BUILD_SHARED   "Build rfdetr as a shared library"   OFF)
option(RFDETR_INSTALL        "Generate install/export rules"      ON)
option(RFDETR_ENABLE_CUDA    "Enable CUDA toolkit integration"    ON)
option(RFDETR_ENABLE_OPENCV_CUDA "Use OpenCV CUDA modules for preprocessing if available" ON)

set(RFDETR_SIMD_LEVEL "AVX2" CACHE STRING "CPU SIMD level (NONE|AVX2|AVX512)")
set_property(CACHE RFDETR_SIMD_LEVEL PROPERTY STRINGS NONE AVX2 AVX512)

set(RFDETR_ONNXRUNTIME_ROOT "" CACHE PATH "Root of ONNX Runtime install (include/ + lib/)")
set(RFDETR_OPENCV_ROOT       "" CACHE PATH "OpenCV build or install root (optional)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# -----------------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------------
if(MSVC)
    add_compile_options(/W3 /permissive- /bigobj)
    add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/Oi> $<$<CONFIG:Release>:/Ot>)
    add_link_options   ($<$<CONFIG:Release>:/OPT:REF> $<$<CONFIG:Release>:/OPT:ICF>)
    if(RFDETR_SIMD_LEVEL STREQUAL "AVX512")
        add_compile_options(/arch:AVX512)
        add_compile_definitions(__AVX512F__=1 __AVX2__=1 __FMA__=1)
    elseif(RFDETR_SIMD_LEVEL STREQUAL "AVX2")
        add_compile_options(/arch:AVX2)
        add_compile_definitions(__AVX2__=1 __FMA__=1)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
    add_compile_options($<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-ffast-math>)
    if(RFDETR_SIMD_LEVEL STREQUAL "AVX512")
        add_compile_options(-mavx512f -mavx512dq -mfma -mavx2)
    elseif(RFDETR_SIMD_LEVEL STREQUAL "AVX2")
        add_compile_options(-mavx2 -mfma)
    endif()
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_out)
    if(_ipo_ok)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    endif()
endif()

find_package(OpenMP)

# -----------------------------------------------------------------------------
# OpenCV
# -----------------------------------------------------------------------------
if(RFDETR_OPENCV_ROOT)
    set(OpenCV_DIR "${RFDETR_OPENCV_ROOT}")
elseif(NOT OpenCV_DIR AND EXISTS "C:/Users/brone/Documents/Programming/opencv/build")
    set(OpenCV_DIR "C:/Users/brone/Documents/Programming/opencv/build")
endif()
find_package(OpenCV REQUIRED)

set(_rfdetr_have_opencv_cuda OFF)
if(RFDETR_ENABLE_OPENCV_CUDA)
    foreach(m cudaarithm cudaimgproc cudawarping)
        if(NOT TARGET opencv_${m} AND NOT ";${OpenCV_LIBS};" MATCHES ";opencv_${m};")
            set(_has_${m} FALSE)
        else()
            set(_has_${m} TRUE)
        endif()
    endforeach()
    if(_has_cudaarithm AND _has_cudaimgproc AND _has_cudawarping)
        set(_rfdetr_have_opencv_cuda ON)
    endif()
endif()

# -----------------------------------------------------------------------------
# CUDA Toolkit (optional but recommended for GPU preprocessing / device IO)
# -----------------------------------------------------------------------------
set(_rfdetr_have_cuda OFF)
if(RFDETR_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(_rfdetr_have_cuda ON)
        message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION} at ${CUDAToolkit_LIBRARY_ROOT}")
    else()
        message(STATUS "CUDA Toolkit not found; GPU I/O binding disabled.")
    endif()
endif()

# -----------------------------------------------------------------------------
# ONNX Runtime (find headers + lib, bundled redistributable)
# -----------------------------------------------------------------------------
if(NOT RFDETR_ONNXRUNTIME_ROOT)
    if(EXISTS "C:/Users/brone/Documents/Programming/onnxruntime-win-x64-gpu-1.24.4")
        set(RFDETR_ONNXRUNTIME_ROOT "C:/Users/brone/Documents/Programming/onnxruntime-win-x64-gpu-1.24.4")
    endif()
endif()
if(NOT RFDETR_ONNXRUNTIME_ROOT OR NOT EXISTS "${RFDETR_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime not found. Set -DRFDETR_ONNXRUNTIME_ROOT=<path-to-onnxruntime>")
endif()

set(ORT_INC "${RFDETR_ONNXRUNTIME_ROOT}/include")
set(ORT_LIB "${RFDETR_ONNXRUNTIME_ROOT}/lib")
find_library(_ORT_LIB NAMES onnxruntime PATHS "${ORT_LIB}" NO_DEFAULT_PATH REQUIRED)

# -----------------------------------------------------------------------------
# Library target: rfdetr
# -----------------------------------------------------------------------------
set(_rfdetr_sources
    src/engine.cpp
    src/preprocess.cpp
)

if(RFDETR_BUILD_SHARED)
    add_library(rfdetr SHARED ${_rfdetr_sources})
    target_compile_definitions(rfdetr
        PUBLIC  RFDETR_SHARED
        PRIVATE RFDETR_BUILDING)
else()
    add_library(rfdetr STATIC ${_rfdetr_sources})
endif()
add_library(rfdetr::rfdetr ALIAS rfdetr)

target_include_directories(rfdetr
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(rfdetr PRIVATE "${ORT_INC}")
target_link_libraries(rfdetr
    PUBLIC  ${OpenCV_LIBS}
    PRIVATE "${_ORT_LIB}")

if(OpenMP_CXX_FOUND)
    target_link_libraries(rfdetr PRIVATE OpenMP::OpenMP_CXX)
endif()

if(_rfdetr_have_cuda)
    target_compile_definitions(rfdetr PRIVATE RFDETR_HAVE_CUDA=1)
    target_link_libraries(rfdetr PRIVATE CUDA::cudart)
endif()
if(_rfdetr_have_opencv_cuda)
    target_compile_definitions(rfdetr PRIVATE RFDETR_HAVE_OPENCV_CUDA=1)
endif()

set_target_properties(rfdetr PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME rfdetr)

# -----------------------------------------------------------------------------
# Demo executable
# -----------------------------------------------------------------------------
if(RFDETR_BUILD_DEMO)
    add_executable(rfdetr_demo apps/demo.cpp)
    target_link_libraries(rfdetr_demo PRIVATE rfdetr::rfdetr)
    set_target_properties(rfdetr_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/bin")

    # Copy runtime DLLs next to the demo on Windows
    if(WIN32)
        set(_rt_dlls
            "${ORT_LIB}/onnxruntime.dll"
            "${ORT_LIB}/onnxruntime_providers_shared.dll"
            "${ORT_LIB}/onnxruntime_providers_cuda.dll"
            "${ORT_LIB}/onnxruntime_providers_tensorrt.dll")
        foreach(_dll IN LISTS _rt_dlls)
            if(EXISTS "${_dll}")
                add_custom_command(TARGET rfdetr_demo POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_dll}" "$<TARGET_FILE_DIR:rfdetr_demo>")
            endif()
        endforeach()
        foreach(_t IN LISTS OpenCV_LIBS)
            if(TARGET ${_t})
                add_custom_command(TARGET rfdetr_demo POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "$<TARGET_FILE:${_t}>" "$<TARGET_FILE_DIR:rfdetr_demo>")
            endif()
        endforeach()
    endif()
endif()

# -----------------------------------------------------------------------------
# Install / export
# -----------------------------------------------------------------------------
if(RFDETR_INSTALL)
    install(TARGETS rfdetr
        EXPORT  rfdetrTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/rfdetr DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    if(RFDETR_BUILD_DEMO)
        install(TARGETS rfdetr_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    install(EXPORT rfdetrTargets
        FILE rfdetrTargets.cmake
        NAMESPACE rfdetr::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rfdetr)

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rfdetrConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/rfdetrConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rfdetr)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/rfdetrConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/rfdetrConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/rfdetrConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rfdetr)
endif()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "================ rfdetr ${PROJECT_VERSION} ================")
message(STATUS "  C++ standard         : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD level           : ${RFDETR_SIMD_LEVEL}")
message(STATUS "  Shared library       : ${RFDETR_BUILD_SHARED}")
message(STATUS "  Build demo           : ${RFDETR_BUILD_DEMO}")
message(STATUS "  OpenCV version       : ${OpenCV_VERSION} (${OpenCV_DIR})")
message(STATUS "  OpenCV CUDA preproc  : ${_rfdetr_have_opencv_cuda}")
message(STATUS "  CUDA toolkit         : ${_rfdetr_have_cuda}")
message(STATUS "  ONNX Runtime         : ${RFDETR_ONNXRUNTIME_ROOT}")
message(STATUS "======================================================")
message(STATUS "")

