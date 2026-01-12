// ------------------------------------------------------------------------
// RF-DETR ONNX Runtime C++ Implementation
// Copyright (c) 2025. All Rights Reserved.
// Licensed under the Apache License, Version 2.0
// ------------------------------------------------------------------------

#include "engine.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <chrono>

// SIMD intrinsics
#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(__GNUC__)
    #include <cpuid.h>
    #include <x86intrin.h>
#endif

// =============================================================================
// Constructor
// =============================================================================

RFDETREngine::RFDETREngine(const std::wstring& model_path,
                           const char* log_id,
                           const char* provider,
                           const char* opt_level,
                           const char* cpu_mode)
    : env_(ORT_LOGGING_LEVEL_WARNING, log_id),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

    std::cout << "Initializing RF-DETR Engine..." << std::endl;

    // Set graph optimization level based on parameter
    GraphOptimizationLevel opt_level_enum;
    std::string opt_level_str(opt_level);

    if (opt_level_str == "disable") {
        opt_level_enum = GraphOptimizationLevel::ORT_DISABLE_ALL;
        std::cout << "Graph optimization: DISABLED" << std::endl;
    } else if (opt_level_str == "basic") {
        opt_level_enum = GraphOptimizationLevel::ORT_ENABLE_BASIC;
        std::cout << "Graph optimization: BASIC" << std::endl;
    } else if (opt_level_str == "extended") {
        opt_level_enum = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
        std::cout << "Graph optimization: EXTENDED" << std::endl;
    } else if (opt_level_str == "all") {
        opt_level_enum = GraphOptimizationLevel::ORT_ENABLE_ALL;
        std::cout << "Graph optimization: ALL (hardware-specific)" << std::endl;
    } else {
        std::cerr << "Warning: Unknown optimization level '" << opt_level_str
                  << "', defaulting to EXTENDED" << std::endl;
        opt_level_enum = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
    }

    session_options_.SetGraphOptimizationLevel(opt_level_enum);

    // Configure execution provider
    if (std::string(provider) == "CUDAExecutionProvider") {
        OrtCUDAProviderOptions cuda_options{};
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "Using CUDA execution provider" << std::endl;
    } else {
        std::cout << "Using CPU execution provider" << std::endl;

        // Configure CPU threading mode
        std::string cpu_mode_str(cpu_mode);
        if (cpu_mode_str == "high-thread-count") {
            // Optimal for high core count systems (>16 cores)
            session_options_.SetIntraOpNumThreads(8);
            session_options_.SetInterOpNumThreads(1);
            std::cout << "  CPU Mode: HIGH-THREAD-COUNT" << std::endl;
        } else {
            // Let ONNX Runtime auto-detect optimal settings
            std::cout << "  CPU Mode: AUTO" << std::endl;
        }
    }

    // Create session
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

    // Get input names and infer input shape
    size_t num_input_nodes = session_->GetInputCount();
    std::cout << "Model has " << num_input_nodes << " input(s)" << std::endl;

    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session_->GetInputNameAllocated(i, allocator_);
        input_names_storage_.push_back(std::string(input_name.get()));

        // Infer input shape from model
        auto type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout << "  Input " << i << ": " << input_name.get() << " - Shape: [";
        for (size_t j = 0; j < shape.size(); j++) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Assume NCHW format: [batch, channels, height, width]
        if (i == 0 && shape.size() == 4) {
            INPUT_CHANNELS = static_cast<int>(shape[1] > 0 ? shape[1] : 3);
            INPUT_HEIGHT = static_cast<int>(shape[2] > 0 ? shape[2] : 384);
            INPUT_WIDTH = static_cast<int>(shape[3] > 0 ? shape[3] : 384);
        }
    }

    // Update char* pointers from storage
    for (const auto& name : input_names_storage_) {
        input_names_.push_back(name.c_str());
    }

    // Get output names and infer output shapes
    size_t num_output_nodes = session_->GetOutputCount();
    std::cout << "Model has " << num_output_nodes << " output(s)" << std::endl;

    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_storage_.push_back(std::string(output_name.get()));

        // Infer output shape from model
        auto type_info = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout << "  Output " << i << ": " << output_name.get() << " - Shape: [";
        for (size_t j = 0; j < shape.size(); j++) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Infer dimensions from output shapes
        // pred_boxes: [batch, num_queries, box_dim]
        // pred_logits: [batch, num_queries, num_classes]
        if (shape.size() == 3) {
            if (i == 0) {  // pred_boxes
                NUM_QUERIES = static_cast<int>(shape[1] > 0 ? shape[1] : 300);
                BOX_DIM = static_cast<int>(shape[2] > 0 ? shape[2] : 4);
            } else if (i == 1) {  // pred_logits
                NUM_CLASSES = static_cast<int>(shape[2] > 0 ? shape[2] : 91);
            }
        }
    }

    // Update char* pointers from storage
    for (const auto& name : output_names_storage_) {
        output_names_.push_back(name.c_str());
    }

    // Pre-allocate input tensor buffer
    const size_t input_tensor_size = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
    input_tensor_values_.resize(input_tensor_size);
    input_tensor_shape_ = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};

    // Pre-allocate output tensor buffers for ZERO-COPY
    const size_t output_boxes_size = 1 * NUM_QUERIES * BOX_DIM;
    const size_t output_logits_size = 1 * NUM_QUERIES * NUM_CLASSES;
    output_boxes_buffer_.resize(output_boxes_size);
    output_logits_buffer_.resize(output_logits_size);
    output_boxes_shape_ = {1, NUM_QUERIES, BOX_DIM};
    output_logits_shape_ = {1, NUM_QUERIES, NUM_CLASSES};

    std::cout << "\nInferred model configuration:" << std::endl;
    std::cout << "  Input: [1, " << INPUT_CHANNELS << ", " << INPUT_HEIGHT << ", " << INPUT_WIDTH << "]" << std::endl;
    std::cout << "  Output boxes: [1, " << NUM_QUERIES << ", " << BOX_DIM << "]" << std::endl;
    std::cout << "  Output logits: [1, " << NUM_QUERIES << ", " << NUM_CLASSES << "]" << std::endl;
    std::cout << "RF-DETR Engine initialized successfully" << std::endl;
    std::cout << std::endl;
}

// =============================================================================
// Main Inference Entry Point
// =============================================================================

std::vector<Detection> RFDETREngine::infer(cv::Mat& image, float conf_threshold) {
    // Save original dimensions for denormalization
    original_width_ = image.cols;
    original_height_ = image.rows;

    // Preprocess with timing
    auto t1 = std::chrono::high_resolution_clock::now();
    preprocess(image);
    auto t2 = std::chrono::high_resolution_clock::now();

    // Forward pass with timing
    auto output_tensors = forward();
    auto t3 = std::chrono::high_resolution_clock::now();

    // Postprocess with timing
    auto detections = postprocess(output_tensors, conf_threshold);
    auto t4 = std::chrono::high_resolution_clock::now();

    // Print timing breakdown (only in debug)
    static int call_count = 0;
    if (++call_count <= 3) {  // Print first 3 calls
        std::cout << "\nTiming breakdown:" << std::endl;
        std::cout << "  Preprocessing: "
                  << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms" << std::endl;
        std::cout << "  Inference: "
                  << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms" << std::endl;
        std::cout << "  Postprocessing: "
                  << std::chrono::duration<double, std::milli>(t4 - t3).count() << " ms" << std::endl;
        std::cout << "  Total: "
                  << std::chrono::duration<double, std::milli>(t4 - t1).count() << " ms" << std::endl;
    }

    return detections;
}

// =============================================================================
// Preprocessing with SIMD Optimization
// =============================================================================

int RFDETREngine::detect_simd_support() {
    // Detect AVX-512 and AVX2 support from CPU
    int cpu_info[4] = {0};

#if defined(_MSC_VER)
    __cpuidex(cpu_info, 7, 0);
    bool has_avx512 = (cpu_info[1] & (1 << 16)) != 0;  // AVX-512F

    __cpuid(cpu_info, 1);
    bool has_avx2 = (cpu_info[2] & (1 << 5)) != 0;  // AVX2 via ECX bit 5
#elif defined(__GNUC__)
    __cpuid_count(7, 0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
    bool has_avx512 = (cpu_info[1] & (1 << 16)) != 0;

    __cpuid(1, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
    bool has_avx2 = (cpu_info[2] & (1 << 5)) != 0;
#else
    bool has_avx512 = false;
    bool has_avx2 = false;
#endif

    // Determine SIMD level based on both CPU support AND compile-time flags
    int level = 0;

    if (has_avx512) level = 2;
    else if (has_avx2) level = 1;

    // Clamp to what was actually compiled - can't use SIMD code that doesn't exist!
#if !defined(__AVX512F__)
    if (level > 1) level = 1;  // AVX-512 not compiled, fall back to AVX2
#endif
#if !defined(__AVX2__)
    if (level > 0) level = 0;  // No SIMD compiled, use scalar
#endif

    return level;
}

void RFDETREngine::preprocess(const cv::Mat& image) {
    // Step 1: Resize to model input size (SquareResize - direct stretch, no letterbox)
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), 0, 0, cv::INTER_LINEAR);

    // Step 2: Convert BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Step 3 & 4: Normalize and convert HWC → CHW with SIMD
    const int image_area = INPUT_WIDTH * INPUT_HEIGHT;

    // Detect SIMD support
    static int simd_level = detect_simd_support();

    // ImageNet normalization: (pixel/255.0 - mean) / std
    const float inv_std_r = 1.0f / (255.0f * STD[0]);
    const float inv_std_g = 1.0f / (255.0f * STD[1]);
    const float inv_std_b = 1.0f / (255.0f * STD[2]);
    const float mean_r = MEAN[0];
    const float mean_g = MEAN[1];
    const float mean_b = MEAN[2];

#if defined(__AVX512F__)
    if (simd_level == 2) {
        // AVX-512 path: Process 16 pixels at a time
        const uint8_t* data = rgb.data;

        // Hoist constant vectors outside loop to avoid recreation
        const __m512 inv_std_r_vec = _mm512_set1_ps(inv_std_r);
        const __m512 inv_std_g_vec = _mm512_set1_ps(inv_std_g);
        const __m512 inv_std_b_vec = _mm512_set1_ps(inv_std_b);
        const __m512 mean_r_vec = _mm512_set1_ps(mean_r);
        const __m512 mean_g_vec = _mm512_set1_ps(mean_g);
        const __m512 mean_b_vec = _mm512_set1_ps(mean_b);

        for (int i = 0; i < image_area; i += 16) {
            const int remaining = std::min(16, image_area - i);

            // Gather RGB values for 16 pixels
            alignas(64) uint8_t r_vals[16] = {0};
            alignas(64) uint8_t g_vals[16] = {0};
            alignas(64) uint8_t b_vals[16] = {0};

            for (int j = 0; j < remaining; ++j) {
                const int pixel_idx = (i + j) * 3;
                r_vals[j] = data[pixel_idx + 0];
                g_vals[j] = data[pixel_idx + 1];
                b_vals[j] = data[pixel_idx + 2];
            }

            // Load uint8 → int32 → float
            __m512i r_u8 = _mm512_loadu_si512((__m512i*)r_vals);
            __m512i g_u8 = _mm512_loadu_si512((__m512i*)g_vals);
            __m512i b_u8 = _mm512_loadu_si512((__m512i*)b_vals);

            // Convert to float (first 16 bytes)
            __m512 r_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(r_u8)));
            __m512 g_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(g_u8)));
            __m512 b_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(b_u8)));

            // Normalize: (pixel / 255.0 - mean) / std
            r_f = _mm512_sub_ps(_mm512_mul_ps(r_f, inv_std_r_vec), mean_r_vec);
            g_f = _mm512_sub_ps(_mm512_mul_ps(g_f, inv_std_g_vec), mean_g_vec);
            b_f = _mm512_sub_ps(_mm512_mul_ps(b_f, inv_std_b_vec), mean_b_vec);

            // Store in CHW layout
            _mm512_storeu_ps(&input_tensor_values_[i], r_f);
            _mm512_storeu_ps(&input_tensor_values_[image_area + i], g_f);
            _mm512_storeu_ps(&input_tensor_values_[2 * image_area + i], b_f);
        }
    } else
#endif
#if defined(__AVX2__)
    if (simd_level == 1) {
        // AVX2 path: Process 8 pixels at a time
        const uint8_t* data = rgb.data;

        // Hoist constant vectors outside loop to avoid recreation
        const __m256 inv_std_r_vec = _mm256_set1_ps(inv_std_r);
        const __m256 inv_std_g_vec = _mm256_set1_ps(inv_std_g);
        const __m256 inv_std_b_vec = _mm256_set1_ps(inv_std_b);
        const __m256 mean_r_vec = _mm256_set1_ps(mean_r);
        const __m256 mean_g_vec = _mm256_set1_ps(mean_g);
        const __m256 mean_b_vec = _mm256_set1_ps(mean_b);

        for (int i = 0; i < image_area; i += 8) {
            const int remaining = std::min(8, image_area - i);

            alignas(32) uint8_t r_vals[8] = {0};
            alignas(32) uint8_t g_vals[8] = {0};
            alignas(32) uint8_t b_vals[8] = {0};

            for (int j = 0; j < remaining; ++j) {
                const int pixel_idx = (i + j) * 3;
                r_vals[j] = data[pixel_idx + 0];
                g_vals[j] = data[pixel_idx + 1];
                b_vals[j] = data[pixel_idx + 2];
            }

            // Load and convert to float
            __m128i r_u8 = _mm_loadl_epi64((__m128i*)r_vals);
            __m128i g_u8 = _mm_loadl_epi64((__m128i*)g_vals);
            __m128i b_u8 = _mm_loadl_epi64((__m128i*)b_vals);

            __m256 r_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r_u8));
            __m256 g_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g_u8));
            __m256 b_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b_u8));

            // Normalize
            r_f = _mm256_sub_ps(_mm256_mul_ps(r_f, inv_std_r_vec), mean_r_vec);
            g_f = _mm256_sub_ps(_mm256_mul_ps(g_f, inv_std_g_vec), mean_g_vec);
            b_f = _mm256_sub_ps(_mm256_mul_ps(b_f, inv_std_b_vec), mean_b_vec);

            // Store
            _mm256_storeu_ps(&input_tensor_values_[i], r_f);
            _mm256_storeu_ps(&input_tensor_values_[image_area + i], g_f);
            _mm256_storeu_ps(&input_tensor_values_[2 * image_area + i], b_f);
        }
    } else
#endif
    {
        // Fallback scalar path
        const uint8_t* data = rgb.data;

        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < image_area; ++i) {
                const uint8_t pixel_val = data[i * 3 + c];
                float normalized;

                if (c == 0) {  // R
                    normalized = (pixel_val * inv_std_r) - mean_r;
                } else if (c == 1) {  // G
                    normalized = (pixel_val * inv_std_g) - mean_g;
                } else {  // B
                    normalized = (pixel_val * inv_std_b) - mean_b;
                }

                input_tensor_values_[c * image_area + i] = normalized;
            }
        }
    }
}

// =============================================================================
// Forward Pass (ONNX Runtime Inference) - ZERO-COPY with IOBinding
// =============================================================================

std::vector<Ort::Value> RFDETREngine::forward() {
    // Create IOBinding for zero-copy
    Ort::IoBinding io_binding(*session_);

    // Bind input tensor (no copy)
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_tensor_values_.data(),
        input_tensor_values_.size(),
        input_tensor_shape_.data(),
        input_tensor_shape_.size()
    );
    io_binding.BindInput(input_names_[0], input_tensor);

    // Bind pre-allocated output tensors (zero-copy!)
    auto output_boxes_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        output_boxes_buffer_.data(),
        output_boxes_buffer_.size(),
        output_boxes_shape_.data(),
        output_boxes_shape_.size()
    );

    auto output_logits_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        output_logits_buffer_.data(),
        output_logits_buffer_.size(),
        output_logits_shape_.data(),
        output_logits_shape_.size()
    );

    io_binding.BindOutput(output_names_[0], output_boxes_tensor);
    io_binding.BindOutput(output_names_[1], output_logits_tensor);

    // Run inference with IOBinding (zero-copy)
    session_->Run(Ort::RunOptions{nullptr}, io_binding);

    // Return outputs (these point to our pre-allocated buffers)
    std::vector<Ort::Value> outputs;
    outputs.push_back(std::move(output_boxes_tensor));
    outputs.push_back(std::move(output_logits_tensor));

    return outputs;
}

// =============================================================================
// Postprocessing (Sigmoid + Box Conversion + No NMS)
// =============================================================================

std::vector<Detection> RFDETREngine::postprocess(std::vector<Ort::Value>& output_tensors,
                                                  float conf_threshold) {
    // Work directly on pre-allocated buffers (zero-copy!)
    float* __restrict pred_boxes = output_boxes_buffer_.data();
    float* __restrict pred_logits = output_logits_buffer_.data();

    // Pre-allocate detection vector (avoid reallocation)
    std::vector<Detection> detection_buffer;
    detection_buffer.reserve(static_cast<size_t>(NUM_QUERIES));

    const float width_f = static_cast<float>(original_width_);
    const float height_f = static_cast<float>(original_height_);

    // Process all queries
    for (int i = 0; i < NUM_QUERIES; ++i) {
        // Prefetch next query's data for cache efficiency
        if (i + 1 < NUM_QUERIES) {
#ifdef _MSC_VER
            _mm_prefetch((const char*)&pred_boxes[(i + 1) * BOX_DIM], _MM_HINT_T0);
            _mm_prefetch((const char*)&pred_logits[(i + 1) * NUM_CLASSES], _MM_HINT_T0);
#else
            __builtin_prefetch(&pred_boxes[(i + 1) * BOX_DIM], 0, 1);
            __builtin_prefetch(&pred_logits[(i + 1) * NUM_CLASSES], 0, 1);
#endif
        }

        // Get box coordinates (normalized cxcywh)
        const float cx_norm = pred_boxes[i * BOX_DIM + 0];
        const float cy_norm = pred_boxes[i * BOX_DIM + 1];
        const float w_norm = pred_boxes[i * BOX_DIM + 2];
        const float h_norm = pred_boxes[i * BOX_DIM + 3];

        // Find max class probability using SIGMOID with SIMD
        int max_class = 0;
        float max_prob = 0.0f;

        const float* logits_ptr = &pred_logits[i * NUM_CLASSES];

#if defined(__AVX2__)
        // AVX2 SIMD path: Process 8 classes at a time
        // Use fixed-size array to avoid heap allocation in hot loop
        alignas(32) float probs[256];  // Max classes we'll support

        for (int c = 0; c < NUM_CLASSES; c += 8) {
            const int remaining = std::min(8, NUM_CLASSES - c);

            // Load 8 logits
            __m256 logits = _mm256_loadu_ps(&logits_ptr[c]);

            // Fast sigmoid approximation for AVX2
            // sigmoid(x) ≈ 0.5 + 0.5 * tanh(0.5 * x)
            // Or use lookup table for better accuracy

            // Clamp to avoid overflow: max(-88, min(88, x))
            __m256 min_val = _mm256_set1_ps(-88.0f);
            __m256 max_val = _mm256_set1_ps(88.0f);
            logits = _mm256_max_ps(min_val, _mm256_min_ps(max_val, logits));

            // Store for now (will optimize exp later if needed)
            _mm256_storeu_ps(&probs[c], logits);

            // Scalar sigmoid for accuracy (can be optimized with fast exp)
            for (int j = 0; j < remaining; ++j) {
                const float prob = 1.0f / (1.0f + std::exp(-probs[c + j]));
                probs[c + j] = prob;

                if (prob > max_prob) {
                    max_prob = prob;
                    max_class = c + j;
                }
            }
        }
#else
        // Scalar fallback
        for (int c = 0; c < NUM_CLASSES; ++c) {
            const float logit = logits_ptr[c];
            const float prob = fast_sigmoid(logit);

            if (prob > max_prob) {
                max_prob = prob;
                max_class = c;
            }
        }
#endif

        // Early exit if below threshold
        if (max_prob < conf_threshold) [[unlikely]] {
            continue;
        }

        // Convert cxcywh → xyxy (normalized [0,1])
        const float half_w = 0.5f * w_norm;
        const float half_h = 0.5f * h_norm;

        float x1_norm = cx_norm - half_w;
        float y1_norm = cy_norm - half_h;
        float x2_norm = cx_norm + half_w;
        float y2_norm = cy_norm + half_h;

        // Clamp to [0, 1] - using fminf/fmaxf for potential SIMD
        x1_norm = std::fmax(0.0f, std::fmin(1.0f, x1_norm));
        y1_norm = std::fmax(0.0f, std::fmin(1.0f, y1_norm));
        x2_norm = std::fmax(0.0f, std::fmin(1.0f, x2_norm));
        y2_norm = std::fmax(0.0f, std::fmin(1.0f, y2_norm));

        // Denormalize to pixel coordinates
        const int x1 = static_cast<int>(x1_norm * width_f);
        const int y1 = static_cast<int>(y1_norm * height_f);
        const int x2 = static_cast<int>(x2_norm * width_f);
        const int y2 = static_cast<int>(y2_norm * height_f);

        // Add detection to buffer
        Detection det;
        det.box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        det.class_id = max_class;
        det.confidence = max_prob;
        detection_buffer.push_back(det);
    }

    // NO NMS - DETR architecture guarantees uniqueness!
    return detection_buffer;
}
