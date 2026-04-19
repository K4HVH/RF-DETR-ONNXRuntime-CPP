// ------------------------------------------------------------------------
// RF-DETR Engine implementation (TensorRT / CUDA / CPU)
// ------------------------------------------------------------------------
#include "rfdetr/rfdetr.hpp"
#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef RFDETR_HAVE_CUDA
    #include <cuda_runtime_api.h>
    #define RFDETR_CUDA_CHECK(x)                                             \
        do {                                                                 \
            cudaError_t _e = (x);                                            \
            if (_e != cudaSuccess) {                                         \
                throw std::runtime_error(std::string("CUDA error: ") +       \
                                         cudaGetErrorString(_e));            \
            }                                                                \
        } while (0)
#endif

#if defined(_MSC_VER)
    #include <intrin.h>
    #include <immintrin.h>
#elif defined(__GNUC__)
    #include <x86intrin.h>
#endif

namespace rfdetr {

// ---------- Version ----------
const char* Version::string() noexcept { return "2.0.0"; }

// =============================================================================
// Utilities
// =============================================================================
namespace {

using Clock = std::chrono::high_resolution_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::wstring to_wstring(const std::filesystem::path& p) { return p.wstring(); }

std::string dev_to_str(Device d) {
    switch (d) {
        case Device::CPU: return "CPU";
        case Device::CUDA: return "CUDA";
        case Device::TensorRT: return "TensorRT";
        case Device::TensorRTRTX: return "TensorRT-RTX";
        case Device::Auto: return "Auto";
    }
    return "?";
}
std::string prec_to_str(Precision p) {
    switch (p) {
        case Precision::FP32: return "FP32";
        case Precision::FP16: return "FP16";
        case Precision::INT8: return "INT8";
    }
    return "?";
}

GraphOptimizationLevel to_ort_opt(OptLevel o) {
    switch (o) {
        case OptLevel::Disable:  return GraphOptimizationLevel::ORT_DISABLE_ALL;
        case OptLevel::Basic:    return GraphOptimizationLevel::ORT_ENABLE_BASIC;
        case OptLevel::Extended: return GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
        case OptLevel::All:      return GraphOptimizationLevel::ORT_ENABLE_ALL;
    }
    return GraphOptimizationLevel::ORT_ENABLE_ALL;
}

OrtLoggingLevel to_ort_log(LogLevel l) {
    switch (l) {
        case LogLevel::Verbose: return ORT_LOGGING_LEVEL_VERBOSE;
        case LogLevel::Info:    return ORT_LOGGING_LEVEL_INFO;
        case LogLevel::Warning: return ORT_LOGGING_LEVEL_WARNING;
        case LogLevel::Error:   return ORT_LOGGING_LEVEL_ERROR;
        case LogLevel::Fatal:   return ORT_LOGGING_LEVEL_FATAL;
    }
    return ORT_LOGGING_LEVEL_WARNING;
}

// Sorted, ordered list of fallback providers starting from `requested`.
std::vector<Device> build_fallback_chain(Device requested, bool auto_fallback) {
    std::vector<Device> chain;
    if (requested == Device::Auto) {
        chain = {Device::TensorRT, Device::CUDA, Device::CPU};
    } else {
        chain.push_back(requested);
        if (auto_fallback) {
            if (requested == Device::TensorRTRTX || requested == Device::TensorRT) {
                if (requested == Device::TensorRTRTX) chain.push_back(Device::TensorRT);
                chain.push_back(Device::CUDA);
                chain.push_back(Device::CPU);
            } else if (requested == Device::CUDA) {
                chain.push_back(Device::CPU);
            }
        }
    }
    // de-dup while preserving order
    std::vector<Device> out;
    for (Device d : chain) {
        if (std::find(out.begin(), out.end(), d) == out.end()) out.push_back(d);
    }
    return out;
}

// Check which providers ORT has compiled in.
bool provider_available(Device d) {
    auto providers = Ort::GetAvailableProviders();
    auto has = [&](const char* name) {
        return std::find(providers.begin(), providers.end(), name) != providers.end();
    };
    switch (d) {
        case Device::CPU:         return true;
        case Device::CUDA:        return has("CUDAExecutionProvider");
        case Device::TensorRT:    return has("TensorrtExecutionProvider");
        case Device::TensorRTRTX: return has("NvTensorRTRTXExecutionProvider");
        case Device::Auto:        return true;
    }
    return false;
}

void apply_cpu_options(Ort::SessionOptions& so, const EngineConfig& cfg) {
    if (cfg.intra_op_threads > 0) so.SetIntraOpNumThreads(cfg.intra_op_threads);
    if (cfg.inter_op_threads > 0) so.SetInterOpNumThreads(cfg.inter_op_threads);
}

// Apply TensorRT EP options on a SessionOptions.
// `rtx_tuned` enables RTX-oriented latency tweaks.
void apply_tensorrt(Ort::SessionOptions& so, const EngineConfig& cfg, bool rtx_tuned) {
    const auto& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* trt_opts = nullptr;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt_opts));

    std::filesystem::create_directories(cfg.trt_cache_dir);
    const std::string cache_path = cfg.trt_cache_dir.string();
    const std::string device_id_s = std::to_string(cfg.device_id);
    const std::string workspace_s = std::to_string(cfg.trt_workspace_bytes);
    const std::string opt_level_s = cfg.trt_builder_optimization_level_max ? "5" : "3";

    const bool fp16 = (cfg.precision == Precision::FP16) || (cfg.precision == Precision::INT8);
    const bool int8 =  cfg.precision == Precision::INT8;

    std::vector<std::pair<std::string, std::string>> kv;
    auto push = [&](std::string k, std::string v) { kv.emplace_back(std::move(k), std::move(v)); };

    push("device_id", device_id_s);
    push("trt_max_workspace_size", workspace_s);
    push("trt_fp16_enable", fp16 ? "1" : "0");
    push("trt_int8_enable", int8 ? "1" : "0");
    push("trt_engine_cache_enable", "1");
    push("trt_engine_cache_path", cache_path);
    push("trt_timing_cache_enable", "1");
    push("trt_timing_cache_path",  cache_path);
    push("trt_builder_optimization_level", opt_level_s);
    push("trt_dla_enable", "0");
    push("trt_force_sequential_engine_build", "0");


    if (int8) {
        if (cfg.int8_mode == Int8Mode::Calibrated && !cfg.int8_calibration_table.empty()) {
            push("trt_int8_calibration_table_name", cfg.int8_calibration_table.string());
            push("trt_int8_use_native_calibration_table", "0");
        } else if (cfg.int8_mode == Int8Mode::EmbeddedQDQ) {
            // ONNX model has QDQ ops; TRT will use explicit precision. No calibration table needed.
        } else {
            // NoCalibration: TRT will fall back to per-layer default ranges and may auto-select
            // FP16/FP32 for unquantisable layers. Noisy but runs without calibration data.
        }
    }

    if (rtx_tuned) {
        // Latency-oriented tuning ("RTX mode"): maximum builder level + reduce aux streams.
        push("trt_builder_optimization_level", "5");
        push("trt_auxiliary_streams", "0");
    }

    std::vector<const char*> keys, vals;
    keys.reserve(kv.size()); vals.reserve(kv.size());
    for (auto& p : kv) { keys.push_back(p.first.c_str()); vals.push_back(p.second.c_str()); }

    Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt_opts, keys.data(), vals.data(), keys.size()));

    // V2 append. Uses the raw C API because the C++ wrapper does not expose this overload consistently.
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
        static_cast<OrtSessionOptions*>(so), trt_opts));
    api.ReleaseTensorRTProviderOptions(trt_opts);
}

void apply_cuda(Ort::SessionOptions& so, const EngineConfig& cfg, bool allow_graph = true) {
    const auto& api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_opts));

    const std::string device_id_s = std::to_string(cfg.device_id);
    std::vector<std::pair<std::string, std::string>> kv;
    auto push = [&](std::string k, std::string v) { kv.emplace_back(std::move(k), std::move(v)); };

    push("device_id", device_id_s);
    push("arena_extend_strategy", "kSameAsRequested");
    push("cudnn_conv_algo_search", "EXHAUSTIVE");
    push("cudnn_conv_use_max_workspace", "1");
    push("do_copy_in_default_stream", "1");
    if (cfg.enable_cuda_graph && allow_graph) push("enable_cuda_graph", "1");

    std::vector<const char*> keys, vals;
    keys.reserve(kv.size()); vals.reserve(kv.size());
    for (auto& p : kv) { keys.push_back(p.first.c_str()); vals.push_back(p.second.c_str()); }

    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda_opts, keys.data(), vals.data(), keys.size()));
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
        static_cast<OrtSessionOptions*>(so), cuda_opts));
    api.ReleaseCUDAProviderOptions(cuda_opts);
}

// TensorRT-RTX EP (NvTensorRTRTXExecutionProvider) — registered by provider-name
// via the generic string-based API. Only usable when the ORT build bundles
// onnxruntime_providers_tensorrt_rtx; otherwise provider_available() returns false.
void apply_tensorrt_rtx(Ort::SessionOptions& so, const EngineConfig& cfg) {
    std::filesystem::create_directories(cfg.trt_cache_dir);
    const std::string cache_path = cfg.trt_cache_dir.string();
    const std::string device_id_s = std::to_string(cfg.device_id);
    const std::string workspace_s = std::to_string(cfg.trt_workspace_bytes);

    std::vector<std::pair<std::string, std::string>> kv;
    auto push = [&](std::string k, std::string v) { kv.emplace_back(std::move(k), std::move(v)); };

    push("device_id", device_id_s);
    push("nv_max_workspace_size", workspace_s);
    push("nv_runtime_cache_path", cache_path);
    if (cfg.enable_cuda_graph) push("enable_cuda_graph", "1");

    std::vector<const char*> keys, vals;
    keys.reserve(kv.size()); vals.reserve(kv.size());
    for (auto& p : kv) { keys.push_back(p.first.c_str()); vals.push_back(p.second.c_str()); }

    so.AppendExecutionProvider("NvTensorRTRTX",
        std::unordered_map<std::string, std::string>(kv.begin(), kv.end()));
    (void)keys; (void)vals;
}

} // anonymous namespace

// =============================================================================
// Engine::Impl
// =============================================================================
struct Engine::Impl {
    EngineConfig              cfg;
    Ort::Env                  env;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo           cpu_mem_info{ Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault) };

    Device    active_dev  = Device::CPU;
    Precision active_prec = Precision::FP32;
    ModelInfo mi;
    mutable Timings last_timings{};

    // IO names (owned strings)
    std::vector<std::string>  in_name_store, out_name_store;
    std::vector<const char*>  in_names,      out_names;

    // Host-side preprocess buffer (NCHW float). When GPU I/O is active these are
    // pinned (cudaHostAlloc) so H2D/D2H copies saturate PCIe; otherwise plain heap.
    float*  host_input  = nullptr;
    float*  host_boxes  = nullptr;
    float*  host_logits = nullptr;
    size_t  host_input_count = 0, host_boxes_count = 0, host_logits_count = 0;
    bool    host_pinned = false;

    // Persistent IoBinding (re-used across forward() calls).
    std::unique_ptr<Ort::IoBinding> binding;
    int     bound_batch = -1;   // batch size currently bound; -1 ⇒ no binding

    // Device buffers (when GPU EP active)
#ifdef RFDETR_HAVE_CUDA
    void*   d_input  = nullptr;
    void*   d_boxes  = nullptr;
    void*   d_logits = nullptr;
    cudaStream_t stream = nullptr;
    Ort::MemoryInfo cuda_mem_info{ "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault };
#endif
    bool gpu_io = false;

    Impl(EngineConfig c)
        : cfg(std::move(c)),
          env(to_ort_log(cfg.log_level), cfg.log_id.c_str())
    {
        if (!std::filesystem::exists(cfg.model_path)) {
            throw std::runtime_error("Model file not found: " + cfg.model_path.string());
        }

        const auto chain = build_fallback_chain(cfg.device, cfg.auto_fallback);
        std::vector<std::string> errors;

        for (Device dev : chain) {
            if (!provider_available(dev)) {
                const std::string msg = dev_to_str(dev) + ": not available in this ORT build";
                errors.push_back(msg);
                std::cerr << "[rfdetr] skip " << msg << std::endl;
                continue;
            }
            try {
                if (cfg.verbose) {
                    std::cerr << "[rfdetr] trying provider " << dev_to_str(dev) << std::endl;
                }
                init_session(dev);
                active_dev = dev;
                break;
            } catch (const std::exception& e) {
                errors.push_back(dev_to_str(dev) + ": " + e.what());
                session.reset();
                std::cerr << "[rfdetr] " << dev_to_str(dev)
                          << " init failed: " << e.what() << std::endl;
                continue;
            }
        }
        if (!session) {
            std::ostringstream oss;
            oss << "Failed to initialise any execution provider:";
            for (const auto& e : errors) oss << "\n  - " << e;
            throw std::runtime_error(oss.str());
        }

        resolve_model_shape();
        allocate_buffers();

        if (cfg.verbose) {
            std::cout << "[rfdetr] Engine ready: " << dev_to_str(active_dev)
                      << " / " << prec_to_str(active_prec)
                      << " / input " << mi.input_width << "x" << mi.input_height
                      << " / queries " << mi.num_queries << " / classes " << mi.num_classes
                      << std::endl;
        }
    }

    ~Impl() {
        binding.reset();
#ifdef RFDETR_HAVE_CUDA
        if (host_pinned) {
            if (host_input)  cudaFreeHost(host_input);
            if (host_boxes)  cudaFreeHost(host_boxes);
            if (host_logits) cudaFreeHost(host_logits);
        } else
#endif
        {
            delete[] host_input;
            delete[] host_boxes;
            delete[] host_logits;
        }
#ifdef RFDETR_HAVE_CUDA
        if (d_input)  cudaFree(d_input);
        if (d_boxes)  cudaFree(d_boxes);
        if (d_logits) cudaFree(d_logits);
        if (stream)   cudaStreamDestroy(stream);
#endif
    }

    void alloc_host(float*& p, size_t n, bool pinned) {
#ifdef RFDETR_HAVE_CUDA
        if (pinned) {
            void* ptr = nullptr;
            RFDETR_CUDA_CHECK(cudaHostAlloc(&ptr, n * sizeof(float), cudaHostAllocDefault));
            std::memset(ptr, 0, n * sizeof(float));
            p = static_cast<float*>(ptr);
            return;
        }
#endif
        p = new float[n]();
    }

    void init_session(Device dev) {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(to_ort_opt(cfg.opt_level));
        so.EnableMemPattern();
        so.EnableCpuMemArena();
        apply_cpu_options(so, cfg);

        switch (dev) {
            case Device::TensorRT:
                apply_tensorrt(so, cfg, /*rtx_tuned=*/false);
                apply_cuda   (so, cfg, /*allow_graph=*/false);  // CUDA EP as fallback for ops TRT can't compile
                active_prec = cfg.precision;
                break;
            case Device::TensorRTRTX:
                apply_tensorrt_rtx(so, cfg);
                apply_cuda(so, cfg, /*allow_graph=*/false);
                active_prec = cfg.precision;
                break;
            case Device::CUDA:
                apply_cuda(so, cfg);
                active_prec = (cfg.precision == Precision::FP16) ? Precision::FP16 : Precision::FP32;
                break;
            case Device::CPU:
            case Device::Auto:
                active_prec = Precision::FP32;
                break;
        }

        session = std::make_unique<Ort::Session>(env, to_wstring(cfg.model_path).c_str(), so);

        // IO names
        in_name_store.clear();  in_names.clear();
        out_name_store.clear(); out_names.clear();
        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            auto n = session->GetInputNameAllocated(i, allocator);
            in_name_store.emplace_back(n.get());
        }
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            auto n = session->GetOutputNameAllocated(i, allocator);
            out_name_store.emplace_back(n.get());
        }
        for (auto& s : in_name_store)  in_names.push_back(s.c_str());
        for (auto& s : out_name_store) out_names.push_back(s.c_str());
    }

    void resolve_model_shape() {
        auto in_info  = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto in_shape = in_info.GetShape();
        if (in_shape.size() != 4) {
            throw std::runtime_error("Expected 4-D input tensor (NCHW)");
        }
        mi.input_channels = static_cast<int>(in_shape[1] > 0 ? in_shape[1] : 3);
        mi.input_height   = static_cast<int>(in_shape[2] > 0 ? in_shape[2] : 384);
        mi.input_width    = static_cast<int>(in_shape[3] > 0 ? in_shape[3] : 384);
        mi.max_batch      = cfg.max_batch_size;

        if (session->GetOutputCount() < 2) {
            throw std::runtime_error("Expected at least 2 outputs (pred_boxes, pred_logits)");
        }
        auto box_info  = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        auto log_info  = session->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
        mi.num_queries = static_cast<int>(box_info.size() >= 2 && box_info[1] > 0 ? box_info[1] : 300);
        mi.box_dim     = static_cast<int>(box_info.size() >= 3 && box_info[2] > 0 ? box_info[2] : 4);
        mi.num_classes = static_cast<int>(log_info.size() >= 3 && log_info[2] > 0 ? log_info[2] : 91);
    }

    void allocate_buffers() {
        const size_t per_in  = size_t(mi.input_channels) * mi.input_height * mi.input_width;
        const size_t per_box = size_t(mi.num_queries) * mi.box_dim;
        const size_t per_log = size_t(mi.num_queries) * mi.num_classes;

        gpu_io = (active_dev == Device::CUDA ||
                  active_dev == Device::TensorRT ||
                  active_dev == Device::TensorRTRTX);

        host_pinned = false;
#ifdef RFDETR_HAVE_CUDA
        host_pinned = gpu_io;  // pinned pages let H2D/D2H hit PCIe peak bandwidth
#endif
        host_input_count  = per_in  * cfg.max_batch_size;
        host_boxes_count  = per_box * cfg.max_batch_size;
        host_logits_count = per_log * cfg.max_batch_size;
        alloc_host(host_input,  host_input_count,  host_pinned);
        alloc_host(host_boxes,  host_boxes_count,  host_pinned);
        alloc_host(host_logits, host_logits_count, host_pinned);

#ifdef RFDETR_HAVE_CUDA
        if (gpu_io) {
            RFDETR_CUDA_CHECK(cudaSetDevice(cfg.device_id));
            RFDETR_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            RFDETR_CUDA_CHECK(cudaMalloc(&d_input,  per_in  * cfg.max_batch_size * sizeof(float)));
            RFDETR_CUDA_CHECK(cudaMalloc(&d_boxes,  per_box * cfg.max_batch_size * sizeof(float)));
            RFDETR_CUDA_CHECK(cudaMalloc(&d_logits, per_log * cfg.max_batch_size * sizeof(float)));
            cuda_mem_info = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, cfg.device_id, OrtMemTypeDefault);
        }
#else
        gpu_io = false;  // no CUDA toolkit at build: always bind host memory
#endif
    }

    // --- sigmoid helpers ---
    static inline float sigmoid(float x) {
        x = std::max(-88.f, std::min(88.f, x));
        return 1.f / (1.f + std::exp(-x));
    }

    // Preprocess `batch` images into host_input (and upload if GPU).
    void preprocess_batch(const std::vector<cv::Mat>& images,
                          std::vector<std::pair<int,int>>& orig_sizes) {
        orig_sizes.clear();
        orig_sizes.reserve(images.size());

        const int C = mi.input_channels, H = mi.input_height, W = mi.input_width;
        const size_t per = size_t(C) * H * W;

        for (size_t b = 0; b < images.size(); ++b) {
            const auto& img = images[b];
            if (img.empty()) throw std::runtime_error("Empty input image");
            orig_sizes.emplace_back(img.cols, img.rows);

#ifdef RFDETR_HAVE_OPENCV_CUDA
            if (gpu_io && cfg.enable_gpu_preprocess) {
                float* d_slot = reinterpret_cast<float*>(d_input) + b * per;
                detail::preprocess_gpu_opencv(img, W, H, cfg.mean, cfg.std, d_slot, stream);
                continue;
            }
#endif
            detail::preprocess_cpu(img, W, H, cfg.mean, cfg.std, host_input + b * per);
        }

#ifdef RFDETR_HAVE_CUDA
        // If we used CPU preproc but GPU EP is active, upload host_input to device.
        const bool used_gpu_preproc =
#ifdef RFDETR_HAVE_OPENCV_CUDA
            (gpu_io && cfg.enable_gpu_preprocess);
#else
            false;
#endif
        if (gpu_io && !used_gpu_preproc) {
            const size_t bytes = per * images.size() * sizeof(float);
            RFDETR_CUDA_CHECK(cudaMemcpyAsync(d_input, host_input, bytes,
                                              cudaMemcpyHostToDevice, stream));
        }
        if (gpu_io) RFDETR_CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
    }

    // Run inference for `batch` items. Fills host_boxes/host_logits.
    void forward(int batch) {
        const int C = mi.input_channels, H = mi.input_height, W = mi.input_width;
        std::array<int64_t, 4> in_shape = {batch, C, H, W};
        std::array<int64_t, 3> box_shape = {batch, mi.num_queries, mi.box_dim};
        std::array<int64_t, 3> log_shape = {batch, mi.num_queries, mi.num_classes};
        const size_t in_count  = size_t(batch) * C * H * W;
        const size_t box_count = size_t(batch) * mi.num_queries * mi.box_dim;
        const size_t log_count = size_t(batch) * mi.num_queries * mi.num_classes;

        if (!binding || bound_batch != batch) {
            binding = std::make_unique<Ort::IoBinding>(*session);
#ifdef RFDETR_HAVE_CUDA
            if (gpu_io) {
                auto in_t  = Ort::Value::CreateTensor<float>(cuda_mem_info,
                                   reinterpret_cast<float*>(d_input),  in_count,
                                   in_shape.data(), in_shape.size());
                auto box_t = Ort::Value::CreateTensor<float>(cuda_mem_info,
                                   reinterpret_cast<float*>(d_boxes),  box_count,
                                   box_shape.data(), box_shape.size());
                auto log_t = Ort::Value::CreateTensor<float>(cuda_mem_info,
                                   reinterpret_cast<float*>(d_logits), log_count,
                                   log_shape.data(), log_shape.size());
                binding->BindInput (in_names[0],  in_t);
                binding->BindOutput(out_names[0], box_t);
                binding->BindOutput(out_names[1], log_t);
            } else
#endif
            {
                auto in_t  = Ort::Value::CreateTensor<float>(cpu_mem_info,
                                   host_input,  in_count,
                                   in_shape.data(),  in_shape.size());
                auto box_t = Ort::Value::CreateTensor<float>(cpu_mem_info,
                                   host_boxes,  box_count,
                                   box_shape.data(), box_shape.size());
                auto log_t = Ort::Value::CreateTensor<float>(cpu_mem_info,
                                   host_logits, log_count,
                                   log_shape.data(), log_shape.size());
                binding->BindInput (in_names[0],  in_t);
                binding->BindOutput(out_names[0], box_t);
                binding->BindOutput(out_names[1], log_t);
            }
            bound_batch = batch;
        }

        session->Run(Ort::RunOptions{nullptr}, *binding);

#ifdef RFDETR_HAVE_CUDA
        if (gpu_io) {
            RFDETR_CUDA_CHECK(cudaMemcpyAsync(host_boxes,  d_boxes,
                                              box_count * sizeof(float),
                                              cudaMemcpyDeviceToHost, stream));
            RFDETR_CUDA_CHECK(cudaMemcpyAsync(host_logits, d_logits,
                                              log_count * sizeof(float),
                                              cudaMemcpyDeviceToHost, stream));
            RFDETR_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
#endif
    }

    // Postprocess one image's outputs.
    void postprocess_one(int batch_idx, int orig_w, int orig_h, float conf_th,
                         std::vector<Detection>& out) {
        const int Q = mi.num_queries, K = mi.num_classes, B = mi.box_dim;
        const float* boxes  = host_boxes  + size_t(batch_idx) * Q * B;
        const float* logits = host_logits + size_t(batch_idx) * Q * K;
        const float Wf = static_cast<float>(orig_w), Hf = static_cast<float>(orig_h);

        out.clear();
        out.reserve(Q / 4);

        for (int i = 0; i < Q; ++i) {
            int   best_c = 0;
            float best_l = -1e30f;
            const float* lp = logits + i * K;

#if defined(__AVX2__)
            // Scan the max logit with AVX2 (no sigmoid yet — sigmoid is monotonic, so argmax of
            // logits == argmax of sigmoid). We only compute sigmoid for the winner.
            __m256 vmax = _mm256_set1_ps(-1e30f);
            __m256i vidx = _mm256_set1_epi32(-1);
            __m256i viota = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
            int k = 0;
            const int Ke = (K / 8) * 8;
            for (; k < Ke; k += 8) {
                __m256 v = _mm256_loadu_ps(lp + k);
                __m256i kbase = _mm256_add_epi32(_mm256_set1_epi32(k), viota);
                __m256 mask = _mm256_cmp_ps(v, vmax, _CMP_GT_OQ);
                vmax = _mm256_blendv_ps(vmax, v, mask);
                vidx = _mm256_castps_si256(
                    _mm256_blendv_ps(_mm256_castsi256_ps(vidx),
                                     _mm256_castsi256_ps(kbase), mask));
            }
            alignas(32) float  tmp_vals[8];
            alignas(32) int32_t tmp_idx [8];
            _mm256_store_ps(tmp_vals, vmax);
            _mm256_store_si256((__m256i*)tmp_idx, vidx);
            for (int j = 0; j < 8; ++j) {
                if (tmp_vals[j] > best_l) { best_l = tmp_vals[j]; best_c = tmp_idx[j]; }
            }
            for (; k < K; ++k) {
                if (lp[k] > best_l) { best_l = lp[k]; best_c = k; }
            }
#else
            for (int k = 0; k < K; ++k) {
                if (lp[k] > best_l) { best_l = lp[k]; best_c = k; }
            }
#endif
            const float best_p = sigmoid(best_l);
            if (best_p < conf_th) continue;

            const float cx = boxes[i*B + 0];
            const float cy = boxes[i*B + 1];
            const float w  = boxes[i*B + 2];
            const float h  = boxes[i*B + 3];
            float x1 = std::clamp(cx - 0.5f*w, 0.f, 1.f) * Wf;
            float y1 = std::clamp(cy - 0.5f*h, 0.f, 1.f) * Hf;
            float x2 = std::clamp(cx + 0.5f*w, 0.f, 1.f) * Wf;
            float y2 = std::clamp(cy + 0.5f*h, 0.f, 1.f) * Hf;
            Detection d;
            d.box = cv::Rect(cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)));
            d.class_id = best_c;
            d.confidence = best_p;
            out.push_back(d);
        }
    }

    // High-level
    std::vector<std::vector<Detection>> infer_batch_impl(const std::vector<cv::Mat>& imgs,
                                                          float conf_th) {
        if (imgs.empty()) return {};
        if ((int)imgs.size() > cfg.max_batch_size) {
            throw std::runtime_error("Batch size " + std::to_string(imgs.size()) +
                                     " exceeds max_batch_size " + std::to_string(cfg.max_batch_size));
        }

        std::vector<std::pair<int,int>> orig;
        auto t0 = Clock::now();
        preprocess_batch(imgs, orig);
        auto t1 = Clock::now();
        forward(static_cast<int>(imgs.size()));
        auto t2 = Clock::now();

        std::vector<std::vector<Detection>> out(imgs.size());
        for (size_t i = 0; i < imgs.size(); ++i) {
            postprocess_one(int(i), orig[i].first, orig[i].second, conf_th, out[i]);
        }
        auto t3 = Clock::now();

        last_timings.preprocess_ms  = ms_since(t0) - ms_since(t1);
        last_timings.inference_ms   = ms_since(t1) - ms_since(t2);
        last_timings.postprocess_ms = ms_since(t2) - ms_since(t3);
        // ms_since(tN) subtraction is awkward; recompute cleanly:
        last_timings.preprocess_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        last_timings.inference_ms   = std::chrono::duration<double, std::milli>(t2 - t1).count();
        last_timings.postprocess_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        return out;
    }
};

// =============================================================================
// Engine public methods
// =============================================================================
Engine::Engine(EngineConfig cfg) : impl_(std::make_unique<Impl>(std::move(cfg))) {}
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

std::vector<Detection> Engine::infer(const cv::Mat& image, float conf_threshold) {
    auto out = impl_->infer_batch_impl({image}, conf_threshold);
    return out.empty() ? std::vector<Detection>{} : std::move(out[0]);
}

std::vector<std::vector<Detection>>
Engine::infer_batch(const std::vector<cv::Mat>& images, float conf_threshold) {
    return impl_->infer_batch_impl(images, conf_threshold);
}

Device    Engine::active_device()    const noexcept { return impl_->active_dev; }
Precision Engine::active_precision() const noexcept { return impl_->active_prec; }
const ModelInfo& Engine::model_info() const noexcept { return impl_->mi; }
Timings   Engine::last_timings()     const noexcept { return impl_->last_timings; }

// =============================================================================
// PipelinedEngine — producer/consumer queues with worker threads
// =============================================================================
struct PipelinedEngine::Impl {
    Engine                     engine;
    size_t                     queue_depth;

    struct InJob  { uint64_t id; cv::Mat frame; float conf; };
    struct OutJob { uint64_t id; std::vector<Detection> dets; Timings t; };

    std::mutex                 in_mu, out_mu;
    std::condition_variable    in_cv, out_cv;
    std::queue<InJob>          in_q;
    std::queue<OutJob>         out_q;
    std::atomic<bool>          stopping{false};
    std::atomic<size_t>        in_flight{0};  // queued + currently being processed
    std::thread                worker;

    Impl(EngineConfig c, size_t qd)
        : engine(std::move(c)), queue_depth(std::max<size_t>(qd, 1))
    {
        worker = std::thread([this]{ this->run(); });
    }

    ~Impl() {
        stop();
        if (worker.joinable()) worker.join();
    }

    void run() {
        while (true) {
            InJob job;
            {
                std::unique_lock<std::mutex> lk(in_mu);
                in_cv.wait(lk, [&]{ return stopping || !in_q.empty(); });
                if (stopping && in_q.empty()) return;
                job = std::move(in_q.front()); in_q.pop();
            }
            in_cv.notify_all();

            OutJob oj{job.id, {}, {}};
            try {
                oj.dets = engine.infer(job.frame, job.conf);
                oj.t    = engine.last_timings();
            } catch (...) {
                // Swallow — push an empty result so downstream consumers don't deadlock.
            }
            {
                std::lock_guard<std::mutex> lk(out_mu);
                out_q.push(std::move(oj));
            }
            in_flight.fetch_sub(1, std::memory_order_acq_rel);
            out_cv.notify_all();
        }
    }

    bool submit(uint64_t id, cv::Mat f, float conf) {
        std::unique_lock<std::mutex> lk(in_mu);
        in_cv.wait(lk, [&]{ return stopping || in_q.size() < queue_depth; });
        if (stopping) return false;
        in_q.push({id, std::move(f), conf});
        in_flight.fetch_add(1, std::memory_order_acq_rel);
        in_cv.notify_all();
        return true;
    }

    std::optional<FrameResult> next_result() {
        std::unique_lock<std::mutex> lk(out_mu);
        out_cv.wait(lk, [&]{
            return !out_q.empty() ||
                   (stopping && in_flight.load(std::memory_order_acquire) == 0);
        });
        if (out_q.empty()) return std::nullopt;
        auto j = std::move(out_q.front()); out_q.pop();
        return FrameResult{j.id, std::move(j.dets), j.t};
    }

    void stop() {
        stopping.store(true, std::memory_order_release);
        in_cv.notify_all();
        out_cv.notify_all();
    }
};

PipelinedEngine::PipelinedEngine(EngineConfig cfg, size_t queue_depth)
    : impl_(std::make_unique<Impl>(std::move(cfg), queue_depth)) {}
PipelinedEngine::~PipelinedEngine() = default;
PipelinedEngine::PipelinedEngine(PipelinedEngine&&) noexcept = default;
PipelinedEngine& PipelinedEngine::operator=(PipelinedEngine&&) noexcept = default;

bool PipelinedEngine::submit(uint64_t id, cv::Mat frame, float conf_threshold) {
    return impl_->submit(id, std::move(frame), conf_threshold);
}
std::optional<PipelinedEngine::FrameResult> PipelinedEngine::next_result() {
    return impl_->next_result();
}
void PipelinedEngine::stop() { impl_->stop(); }
const ModelInfo& PipelinedEngine::model_info() const noexcept {
    return impl_->engine.model_info();
}

// =============================================================================
// Drawing helpers
// =============================================================================
void draw_detections(cv::Mat& image,
                     const std::vector<Detection>& detections,
                     const std::vector<std::string>& class_names,
                     float label_scale, int box_thickness) {
    static const cv::Scalar kPalette[] = {
        {56,56,255},{151,157,255},{31,112,255},{29,178,255},
        {49,210,207},{10,249,72},{23,204,146},{134,219,61},
        {52,147,26},{187,212,0},{168,153,44},{255,194,0},
        {147,69,52},{255,115,100},{236,24,0},{255,56,132},
        {133,0,82},{255,56,203},{200,149,255},{199,55,255},
    };
    const size_t kColors = sizeof(kPalette) / sizeof(kPalette[0]);

    for (const auto& d : detections) {
        const cv::Scalar color = kPalette[d.class_id % kColors];
        cv::rectangle(image, d.box, color, box_thickness);

        std::string label = (d.class_id < (int)class_names.size())
                          ? class_names[d.class_id]
                          : ("cls_" + std::to_string(d.class_id));
        char buf[32];
        std::snprintf(buf, sizeof(buf), " %.1f%%", d.confidence * 100.f);
        label += buf;

        int base = 0;
        cv::Size ls = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, label_scale, 1, &base);
        int top = std::max(d.box.y, ls.height + 3);
        cv::rectangle(image,
                      cv::Point(d.box.x, top - ls.height - 3),
                      cv::Point(d.box.x + ls.width + 2, top + base - 2),
                      color, cv::FILLED);
        cv::putText(image, label, cv::Point(d.box.x + 1, top - 2),
                    cv::FONT_HERSHEY_SIMPLEX, label_scale, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    }
}

} // namespace rfdetr
