// ------------------------------------------------------------------------
// Preprocess (CPU SIMD + optional GPU) -> planar NCHW float buffer
// ------------------------------------------------------------------------
#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <cstdint>

namespace rfdetr::detail {

// Detect SIMD support at runtime, clamped by compile-time availability.
// Returns: 2=AVX512, 1=AVX2, 0=scalar
int detect_simd_level();

// CPU preprocess: resize image to (dst_w, dst_h), BGR->RGB, normalise with mean/std,
// write as NCHW float32 into `dst` (size = 3 * dst_w * dst_h).
void preprocess_cpu(const cv::Mat& image,
                    int dst_w, int dst_h,
                    const std::array<float, 3>& mean,
                    const std::array<float, 3>& std,
                    float* dst);

#ifdef RFDETR_HAVE_OPENCV_CUDA
// GPU preprocess with OpenCV CUDA. Writes NCHW float into device pointer `d_dst`.
// Caller owns d_dst (must be at least 3*dst_w*dst_h floats, on device 0 or passed stream's device).
void preprocess_gpu_opencv(const cv::Mat& image,
                           int dst_w, int dst_h,
                           const std::array<float, 3>& mean,
                           const std::array<float, 3>& std,
                           void* d_dst,
                           void* cuda_stream_ptr); // cudaStream_t as void*
#endif

} // namespace rfdetr::detail
