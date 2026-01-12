cmake_minimum_required(VERSION 3.15)
project(RF-DETR-ONNXRuntime-CPP LANGUAGES CXX)

# C++20 Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# SIMD Configuration - Choose your CPU's instruction set
# Options: NONE, AVX2, AVX512
# Usage: cmake -DSIMD_LEVEL=AVX512 ..
set(SIMD_LEVEL "AVX2" CACHE STRING "SIMD instruction set level (NONE, AVX2, AVX512)")
set_property(CACHE SIMD_LEVEL PROPERTY STRINGS NONE AVX2 AVX512)

# Aggressive optimization flags for maximum performance
if(MSVC)
    # Base MSVC optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /Ot")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF")

    # Add SIMD flags based on selection
    if(SIMD_LEVEL STREQUAL "AVX512")
        message(STATUS "Building with AVX-512 support")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX512")
    elseif(SIMD_LEVEL STREQUAL "AVX2")
        message(STATUS "Building with AVX2 support")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
    else()
        message(STATUS "Building without SIMD (scalar only)")
    endif()

    # OpenMP support
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
else()
    # GCC/Clang flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native")

    # Add SIMD flags based on selection
    if(SIMD_LEVEL STREQUAL "AVX512")
        message(STATUS "Building with AVX-512 support")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx512f -mavx512dq")
    elseif(SIMD_LEVEL STREQUAL "AVX2")
        message(STATUS "Building with AVX2 support")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2")
    else()
        message(STATUS "Building without SIMD (scalar only)")
    endif()

    # Link-time optimization
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

    # OpenMP support
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Find OpenCV
set(OpenCV_DIR "C:/Users/brone/Documents/Programming/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime
set(ONNXRUNTIME_ROOT "C:/Users/brone/Documents/Programming/onnxruntime-win-x64-gpu-1.23.2")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

# Check if ONNX Runtime exists
if(NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR})
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}")
endif()

include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

# CUDA Support
set(CUDA_TOOLKIT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
file(GLOB CUDA_VERSION_DIRS "${CUDA_TOOLKIT_ROOT}/v*")
if(CUDA_VERSION_DIRS)
    list(SORT CUDA_VERSION_DIRS)
    list(REVERSE CUDA_VERSION_DIRS)
    list(GET CUDA_VERSION_DIRS 0 CUDA_TOOLKIT_DIR)
    set(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_DIR}/include")
    set(CUDA_LIB_DIR "${CUDA_TOOLKIT_DIR}/lib/x64")

    if(EXISTS ${CUDA_INCLUDE_DIR})
        message(STATUS "Found CUDA at: ${CUDA_TOOLKIT_DIR}")
        include_directories(${CUDA_INCLUDE_DIR})
        link_directories(${CUDA_LIB_DIR})
        set(CUDA_FOUND TRUE)
    else()
        message(WARNING "CUDA headers not found, GPU support may be limited")
        set(CUDA_FOUND FALSE)
    endif()
else()
    message(WARNING "CUDA Toolkit not found at ${CUDA_TOOLKIT_ROOT}")
    set(CUDA_FOUND FALSE)
endif()

# Source files
set(SOURCES
    engine.cpp
    main.cpp
)

# Create main executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    onnxruntime
)

# Link CUDA runtime if available
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} cudart)
endif()

# Copy ONNX Runtime DLL to output directory (Windows)
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )

    # Also copy ONNX Runtime providers DLL if it exists
    if(EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        )
    endif()
endif()

# Print configuration
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Configuration Summary")
message(STATUS "========================================")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD Level: ${SIMD_LEVEL}")
message(STATUS "  OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "========================================")
message(STATUS "")
