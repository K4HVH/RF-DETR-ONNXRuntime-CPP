// ------------------------------------------------------------------------
// RF-DETR ONNX Runtime C++ Implementation
// Copyright (c) 2025. All Rights Reserved.
// Licensed under the Apache License, Version 2.0
// ------------------------------------------------------------------------

#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <array>

/**
 * @brief Detection result structure
 *
 * Represents a single object detection with bounding box, class, and confidence
 */
struct Detection {
    cv::Rect box;        ///< Bounding box in pixel coordinates (x, y, width, height)
    int class_id;        ///< Class ID (0-90 for RF-DETR with 91 classes)
    float confidence;    ///< Detection confidence score [0.0-1.0]
};

/**
 * @brief Execution provider type enumeration
 *
 * Defines available execution providers for ONNX Runtime inference
 */
enum class ExecutionProvider {
    CPU,            ///< CPU execution with SIMD optimizations
    CUDA,           ///< NVIDIA CUDA execution with CUDA graphs
    TensorRT,       ///< NVIDIA TensorRT with engine caching
    TensorRT_RTX    ///< NVIDIA TensorRT-RTX optimized for RTX GPUs
};

/**
 * @brief RF-DETR Inference Engine
 *
 * High-performance C++20 implementation of RF-DETR object detection using ONNX Runtime.
 * Features:
 * - SIMD-optimized preprocessing (AVX-512/AVX2)
 * - No NMS required (DETR architecture)
 * - Fixed 300 query detections
 * - Sigmoid activation for class probabilities
 * - ImageNet normalization
 */
class RFDETREngine {
public:
    /**
     * @brief Construct RF-DETR inference engine
     *
     * @param model_path Path to ONNX model file
     * @param log_id Logging identifier for ONNX Runtime
     * @param provider Execution provider ("CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider")
     * @param opt_level Graph optimization level ("disable", "basic", "extended", "all")
     * @param cpu_mode CPU threading mode ("auto" or "high-thread-count")
     * @param use_fp16 Enable FP16 mode for TensorRT (GPU providers only)
     */
    RFDETREngine(const std::wstring& model_path,
                 const char* log_id = "RF-DETR",
                 const char* provider = "CPUExecutionProvider",
                 const char* opt_level = "extended",
                 const char* cpu_mode = "auto",
                 bool use_fp16 = false);

    /**
     * @brief Run inference on an image
     *
     * @param image Input image (any size, BGR format)
     * @param conf_threshold Confidence threshold for filtering detections [0.0-1.0]
     * @return Vector of Detection objects
     */
    std::vector<Detection> infer(cv::Mat& image, float conf_threshold = 0.5f);

    /**
     * @brief Destructor - RAII cleanup of ONNX Runtime and CUDA resources
     */
    ~RFDETREngine();

private:
    // Model configuration - dynamically inferred from ONNX model (const after init)
    int INPUT_WIDTH;
    int INPUT_HEIGHT;
    int INPUT_CHANNELS;
    int NUM_QUERIES;
    int NUM_CLASSES;
    int BOX_DIM;

    // ImageNet normalization constants
    static constexpr std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
    static constexpr std::array<float, 3> STD = {0.229f, 0.224f, 0.225f};

    // ONNX Runtime objects
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;  // CPU or CUDA memory
    Ort::AllocatorWithDefaultOptions allocator_;

    // Provider tracking
    ExecutionProvider provider_type_;

    // CUDA-specific memory info (only initialized if using CUDA)
    std::unique_ptr<Ort::MemoryInfo> cuda_memory_info_;

    // Input/Output tensor names (stored as strings to ensure lifetime)
    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // Pre-allocated buffers for inference (aligned for SIMD)
    alignas(64) std::vector<float> input_tensor_values_;
    std::vector<int64_t> input_tensor_shape_;

    // Pre-allocated output buffers for ZERO-COPY
    alignas(64) std::vector<float> output_boxes_buffer_;
    alignas(64) std::vector<float> output_logits_buffer_;
    std::vector<int64_t> output_boxes_shape_;
    std::vector<int64_t> output_logits_shape_;

    // CUDA device buffers (only allocated if using GPU providers)
    void* cuda_input_buffer_;
    void* cuda_output_boxes_buffer_;
    void* cuda_output_logits_buffer_;
    size_t cuda_input_size_;
    size_t cuda_output_boxes_size_;
    size_t cuda_output_logits_size_;

    // CUDA stream for async operations
    void* cuda_stream_;  // cudaStream_t, stored as void* to avoid including cuda headers in .hpp

    // Original image dimensions (for denormalization)
    int original_width_;
    int original_height_;

    /**
     * @brief Preprocess image with SIMD optimization
     *
     * Steps:
     * 1. Resize to 384x384 (SquareResize - stretch)
     * 2. Convert BGR → RGB
     * 3. Normalize with ImageNet statistics
     * 4. Convert HWC → CHW layout
     *
     * @param image Input image
     */
    void preprocess(const cv::Mat& image);

    /**
     * @brief Run ONNX Runtime inference
     *
     * @return Vector of output tensors [pred_boxes, pred_logits]
     */
    std::vector<Ort::Value> forward();

    /**
     * @brief Postprocess model outputs
     *
     * Steps:
     * 1. Apply sigmoid to logits
     * 2. Find max class per query
     * 3. Filter by confidence threshold
     * 4. Convert cxcywh → xyxy
     * 5. Denormalize to pixel coordinates
     * 6. NO NMS (DETR architecture)
     *
     * @param output_tensors Model outputs [pred_boxes, pred_logits]
     * @param conf_threshold Confidence threshold
     * @return Vector of detections
     */
    std::vector<Detection> postprocess(std::vector<Ort::Value>& output_tensors,
                                       float conf_threshold);

    /**
     * @brief Detect CPU SIMD capabilities
     *
     * @return 2 for AVX-512, 1 for AVX2, 0 for scalar
     */
    int detect_simd_support();

    /**
     * @brief Fast sigmoid using intrinsics
     *
     * @param x Input value
     * @return Sigmoid(x)
     */
    inline float fast_sigmoid(float x) const {
        // Clamp to avoid overflow in exp
        x = std::max(-88.0f, std::min(88.0f, x));
        return 1.0f / (1.0f + std::exp(-x));
    }

    /**
     * @brief SIMD sigmoid for 8 floats (AVX2)
     */
    #if defined(__AVX2__)
    inline void sigmoid_avx2(__m256& values) const;
    #endif

    /**
     * @brief Clamp value to range [min, max]
     */
    template<typename T>
    inline T clamp(T value, T min_val, T max_val) const {
        return std::min(std::max(value, min_val), max_val);
    }
};
