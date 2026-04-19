// ------------------------------------------------------------------------
// RF-DETR high-performance inference library - Public API
// Copyright (c) 2025. Licensed under the Apache License, Version 2.0
// ------------------------------------------------------------------------
#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#if defined(_WIN32) && defined(RFDETR_SHARED)
    #if defined(RFDETR_BUILDING)
        #define RFDETR_API __declspec(dllexport)
    #else
        #define RFDETR_API __declspec(dllimport)
    #endif
#else
    #define RFDETR_API
#endif

namespace rfdetr {

// ---------- Version ----------
struct RFDETR_API Version {
    static constexpr int major = 2;
    static constexpr int minor = 0;
    static constexpr int patch = 0;
    static const char* string() noexcept;
};

// ---------- Enums ----------
enum class Device {
    Auto,          // Pick best available: TensorRT > CUDA > CPU
    CPU,
    CUDA,
    TensorRT,
    TensorRTRTX    // TensorRT with RTX-specific optimisations (falls back to TensorRT if unsupported)
};

enum class Precision {
    FP32,
    FP16,  // Recommended default on GPU
    INT8
};

enum class Int8Mode {
    None,            // No INT8 (ignored unless Precision::INT8)
    Calibrated,      // Use pre-generated calibration table
    EmbeddedQDQ,     // ONNX model already has QDQ nodes (explicit precision)
    NoCalibration    // INT8 without calibration table (default dynamic ranges). May degrade accuracy.
};

enum class OptLevel { Disable, Basic, Extended, All };

enum class LogLevel { Verbose, Info, Warning, Error, Fatal };

// ---------- Detection result ----------
struct RFDETR_API Detection {
    cv::Rect box;         // pixel coords
    int      class_id;
    float    confidence;
};

// ---------- Engine configuration ----------
struct RFDETR_API EngineConfig {
    // Model
    std::filesystem::path model_path;

    // Provider
    Device     device     = Device::Auto;
    Precision  precision  = Precision::FP16;
    Int8Mode   int8_mode  = Int8Mode::NoCalibration;
    std::filesystem::path int8_calibration_table;  // used when int8_mode == Calibrated

    // GPU options
    int  device_id           = 0;
    bool enable_cuda_graph   = true;   // Capture & replay (CUDA EP) — huge latency win on steady input
    bool enable_gpu_preprocess = true; // Keep preprocessing on GPU when possible
    size_t trt_workspace_bytes = 4ull * 1024 * 1024 * 1024;  // 4 GiB
    std::filesystem::path trt_cache_dir = "trt_cache";       // TRT engine + timing cache dir
    bool trt_builder_optimization_level_max = true;          // set optimization_level=5

    // Batch
    int max_batch_size = 1;

    // CPU options
    int  intra_op_threads = 0;   // 0 = ORT auto
    int  inter_op_threads = 0;
    OptLevel opt_level    = OptLevel::All;

    // Logging / diagnostics
    std::string log_id     = "RF-DETR";
    LogLevel    log_level  = LogLevel::Warning;
    bool        verbose    = false;

    // Normalisation constants (ImageNet defaults)
    std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> std  = {0.229f, 0.224f, 0.225f};

    // Fallback policy: if the requested provider fails, automatically try weaker ones (TRT->CUDA->CPU).
    bool auto_fallback = true;
};

// ---------- Timings (last call) ----------
struct RFDETR_API Timings {
    double preprocess_ms  = 0.0;
    double inference_ms   = 0.0;
    double postprocess_ms = 0.0;
    double total_ms() const noexcept { return preprocess_ms + inference_ms + postprocess_ms; }
};

// ---------- Model metadata ----------
struct RFDETR_API ModelInfo {
    int input_width   = 0;
    int input_height  = 0;
    int input_channels= 3;
    int num_queries   = 0;
    int num_classes   = 0;
    int box_dim       = 4;
    int max_batch     = 1;
};

// ---------- Main engine ----------
class RFDETR_API Engine {
public:
    explicit Engine(EngineConfig config);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    // Single-image inference (BGR cv::Mat, any size).
    std::vector<Detection> infer(const cv::Mat& image, float conf_threshold = 0.5f);

    // Batched inference. Returns per-image detections. Throws if batch > max_batch_size.
    std::vector<std::vector<Detection>> infer_batch(const std::vector<cv::Mat>& images,
                                                    float conf_threshold = 0.5f);

    // Accessors
    Device     active_device() const noexcept;   // device after fallback resolution
    Precision  active_precision() const noexcept;
    const ModelInfo& model_info() const noexcept;
    Timings    last_timings() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------- Async pipelined engine (throughput-oriented) ----------
//
// Feeds frames through a pipeline of preprocess -> infer -> postprocess stages that overlap across
// frames. Optimal for video / camera streams.
class RFDETR_API PipelinedEngine {
public:
    struct FrameResult {
        uint64_t                frame_id;
        std::vector<Detection>  detections;
        Timings                 timings;
    };

    explicit PipelinedEngine(EngineConfig config, size_t queue_depth = 4);
    ~PipelinedEngine();

    PipelinedEngine(PipelinedEngine&&) noexcept;
    PipelinedEngine& operator=(PipelinedEngine&&) noexcept;
    PipelinedEngine(const PipelinedEngine&)            = delete;
    PipelinedEngine& operator=(const PipelinedEngine&) = delete;

    // Returns false if pipeline is shutting down. Blocks if queue is full.
    bool submit(uint64_t frame_id, cv::Mat frame, float conf_threshold = 0.5f);

    // Blocks until next result is ready. Returns std::nullopt if pipeline is drained & stopped.
    std::optional<FrameResult> next_result();

    // Signal no more frames & wait for pending work to flush.
    void stop();

    const ModelInfo& model_info() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------- Drawing helpers ----------
RFDETR_API void draw_detections(cv::Mat& image,
                                const std::vector<Detection>& detections,
                                const std::vector<std::string>& class_names = {},
                                float label_scale = 0.5f,
                                int   box_thickness = 2);

} // namespace rfdetr
