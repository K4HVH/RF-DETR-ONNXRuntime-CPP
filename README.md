# rfdetr — high-performance RF-DETR inference for C++

A reusable C++20 library (+ demo CLI) for running [RF-DETR](https://github.com/roboflow/rf-detr)
object detection via ONNX Runtime with first-class support for **TensorRT** (FP16 / INT8),
**CUDA** (with CUDA graph capture), and a **CPU** fallback with AVX2/AVX-512 SIMD preprocessing.

## Highlights

- **Four execution providers** with automatic fallback: `TensorRT-RTX → TensorRT → CUDA → CPU`
  (TensorRT-RTX (`NvTensorRTRTXExecutionProvider`) requires an ORT build that bundles
  `onnxruntime_providers_tensorrt_rtx`. The standard ORT 1.24.x GPU redistributable does
  **not** ship this provider — build ORT from source with `--use_nv_tensorrt_rtx` to
  enable it. When absent, the chain falls through to TensorRT seamlessly.)
- **TensorRT**: FP16 default, FP32, INT8 (calibrated, QDQ, or no-calibration), on-disk engine + timing cache, builder optimisation level 5
- **CUDA**: cuDNN `EXHAUSTIVE` algo search, optional CUDA-graph capture, IOBinding with device memory
- **GPU preprocessing**: upload + resize + normalize + HWC→CHW entirely on the GPU when OpenCV CUDA is available
- **Batched inference** and **async pipelined inference** (`PipelinedEngine`) for video / camera streams
- **AVX-512 / AVX2 SIMD** preprocess + argmax fallback on CPU
- **Zero-copy IO**: `IoBinding` with pre-allocated host or device buffers
- Clean pimpl-based public API, shipped as a static or shared library with full CMake install/export
- Drop-in demo CLI: `image`, `video`, `benchmark` modes

## Performance (RTX 5080, 384×384 input, 2-class RF-DETR, single image)

| Provider | Precision | Total ms | FPS |
|----------|-----------|----------|-----|
| CPU (AVX2) | FP32 | ~56 | ~18 |
| CUDA | FP16 | ~6.0 | ~165 |
| TensorRT | FP16 | ~2.1 | **~480** |

Numbers from `rfdetr_demo --mode benchmark`, 50 iterations after warmup.

## Repository layout

```
include/rfdetr/rfdetr.hpp   # public API
src/                        # implementation
apps/demo.cpp               # CLI demo / benchmark
cmake/                      # package config for find_package(rfdetr)
scripts/
  optimize_onnx.py          # ONNX graph optimiser
  quantize_int8.py          # INT8 static quantisation (ORT QDQ)
  benchmark.py              # multi-provider benchmark harness
models/inference_model.onnx # (user-supplied) RF-DETR model
```

## Build

Requirements:

- CMake ≥ 3.20, a C++20 compiler (MSVC 19.3+ / GCC 11+ / Clang 14+)
- OpenCV 4.x (CUDA-enabled build recommended for GPU preprocessing)
- ONNX Runtime GPU ≥ 1.22 (1.24 tested; must ship the TensorRT / CUDA provider DLLs)
- (optional) CUDA Toolkit for GPU IO + OpenCV CUDA preprocessing
- (optional) TensorRT — shipped inside ORT's `onnxruntime_providers_tensorrt.dll`; use a matching CUDA/cuDNN

### Windows (MSVC)

```powershell
cmake -S . -B build -A x64 ^
      -DRFDETR_ONNXRUNTIME_ROOT="C:/path/onnxruntime-win-x64-gpu-1.24.4" ^
      -DRFDETR_SIMD_LEVEL=AVX2
cmake --build build --config Release -j
```

### Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DRFDETR_ONNXRUNTIME_ROOT=/opt/onnxruntime \
      -DRFDETR_SIMD_LEVEL=AVX2
cmake --build build -j
```

### CMake options

| Option                      | Default | Notes                                      |
|-----------------------------|---------|--------------------------------------------|
| `RFDETR_BUILD_DEMO`         | `ON`    | Build the `rfdetr_demo` CLI                |
| `RFDETR_BUILD_SHARED`       | `OFF`   | Build as shared library                    |
| `RFDETR_INSTALL`            | `ON`    | Emit install rules + CMake package config  |
| `RFDETR_ENABLE_CUDA`        | `ON`    | Enable CUDA Toolkit integration            |
| `RFDETR_ENABLE_OPENCV_CUDA` | `ON`    | Use OpenCV CUDA modules for preproc        |
| `RFDETR_SIMD_LEVEL`         | `AVX2`  | `NONE`, `AVX2`, `AVX512`                   |
| `RFDETR_ONNXRUNTIME_ROOT`   | auto    | Path to an ORT install                     |

## Consuming the library

After `cmake --install`, downstream projects can simply:

```cmake
find_package(rfdetr CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE rfdetr::rfdetr)
```

## C++ API

```cpp
#include <rfdetr/rfdetr.hpp>

rfdetr::EngineConfig cfg;
cfg.model_path      = "models/inference_model.onnx";
cfg.device          = rfdetr::Device::Auto;      // TRT > CUDA > CPU
cfg.precision       = rfdetr::Precision::FP16;
cfg.max_batch_size  = 1;

rfdetr::Engine engine(cfg);

cv::Mat img = cv::imread("test.png");
auto dets   = engine.infer(img, /*conf=*/0.5f);

for (const auto& d : dets)
    std::cout << d.class_id << " " << d.confidence << " " << d.box << "\n";
```

### Async pipeline for video

```cpp
rfdetr::PipelinedEngine pipe(cfg, /*queue_depth=*/4);

std::thread feeder([&] {
    uint64_t id = 0; cv::Mat f;
    while (cap.read(f)) pipe.submit(id++, f.clone());
    pipe.stop();
});

while (auto r = pipe.next_result()) {
    std::cout << "frame " << r->frame_id << " -> " << r->detections.size() << " dets\n";
}
feeder.join();
```

## Demo CLI

```
rfdetr_demo --mode image     --device auto     --precision fp16  --input test2.png
rfdetr_demo --mode benchmark --device tensorrt --precision fp16  --iters 200
rfdetr_demo --mode video     --device tensorrt --input video.mp4 --output out.mp4
```

See `rfdetr_demo --help` for the full flag set.

## INT8 quantisation

Two supported paths:

**1. TensorRT-internal INT8 (no calibration)** — fastest to try, may lose accuracy:

```bash
rfdetr_demo --device tensorrt --precision int8 --int8-mode nocal --input test2.png
```

**2. Calibrated INT8** — generate a calibration table with the script, then:

```bash
python scripts/quantize_int8.py --model models/inference_model.onnx \
    --calib-dir path/to/calibration_images \
    --output   models/inference_model_int8.onnx
rfdetr_demo --model models/inference_model_int8.onnx \
    --device tensorrt --precision int8 --int8-mode qdq
```

The script writes an ONNX model with QDQ nodes (TensorRT explicit-precision mode), so no separate
calibration table file is required at inference time.

## Troubleshooting

- **"This session cannot use the graph capture feature"** — disable CUDA graph when the TensorRT
  provider is active (`cfg.enable_cuda_graph = false;`, or `--no-graph` on the CLI). The library
  already does this automatically when TRT is selected.
- **TensorRT takes 30+ seconds to start on first run** — expected; it builds an engine and writes
  it to `trt_cache_dir`. Subsequent runs are instant.
- **Model not found / ORT DLLs missing** — ensure the ORT redistributable DLLs are copied next to
  the executable. CMake does this automatically on Windows for the demo.

## License

Apache 2.0.  See `LICENSE`.
