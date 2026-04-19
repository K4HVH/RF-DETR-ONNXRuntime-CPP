// ------------------------------------------------------------------------
// Preprocess implementation (CPU SIMD + optional GPU)
// ------------------------------------------------------------------------
#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>

#if defined(_MSC_VER)
    #include <intrin.h>
    #include <immintrin.h>
#elif defined(__GNUC__)
    #include <cpuid.h>
    #include <x86intrin.h>
#endif

#ifdef RFDETR_HAVE_OPENCV_CUDA
    #include <opencv2/core/cuda.hpp>
    #include <opencv2/core/cuda_stream_accessor.hpp>
    #include <opencv2/cudaarithm.hpp>
    #include <opencv2/cudaimgproc.hpp>
    #include <opencv2/cudawarping.hpp>
    #include <cuda_runtime_api.h>
#endif

namespace rfdetr::detail {

int detect_simd_level() {
    int cpu_info[4] = {0};

#if defined(_MSC_VER)
    __cpuidex(cpu_info, 7, 0);
    const bool has_avx512 = (cpu_info[1] & (1 << 16)) != 0;
    __cpuid(cpu_info, 1);
    const bool has_avx2   = (cpu_info[2] & (1 << 5)) != 0;
#elif defined(__GNUC__)
    __cpuid_count(7, 0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
    const bool has_avx512 = (cpu_info[1] & (1 << 16)) != 0;
    __cpuid(1, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
    const bool has_avx2   = (cpu_info[2] & (1 << 5)) != 0;
#else
    const bool has_avx512 = false;
    const bool has_avx2   = false;
#endif

    int level = has_avx512 ? 2 : (has_avx2 ? 1 : 0);
#if !defined(__AVX512F__)
    if (level > 1) level = 1;
#endif
#if !defined(__AVX2__)
    if (level > 0) level = 0;
#endif
    return level;
}

void preprocess_cpu(const cv::Mat& image,
                    int dst_w, int dst_h,
                    const std::array<float, 3>& mean,
                    const std::array<float, 3>& std,
                    float* dst) {
    cv::Mat resized;
    if (image.cols == dst_w && image.rows == dst_h) {
        resized = image;
    } else {
        cv::resize(image, resized, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);
    }

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    const int area = dst_w * dst_h;
    const float inv_std_r = 1.0f / (255.0f * std[0]);
    const float inv_std_g = 1.0f / (255.0f * std[1]);
    const float inv_std_b = 1.0f / (255.0f * std[2]);
    const float mean_r = mean[0];
    const float mean_g = mean[1];
    const float mean_b = mean[2];

    static const int simd = detect_simd_level();
    const uint8_t* data = rgb.data;

#if defined(__AVX512F__)
    if (simd == 2) {
        const __m512 v_invr = _mm512_set1_ps(inv_std_r);
        const __m512 v_invg = _mm512_set1_ps(inv_std_g);
        const __m512 v_invb = _mm512_set1_ps(inv_std_b);
        const __m512 v_mr   = _mm512_set1_ps(mean_r);
        const __m512 v_mg   = _mm512_set1_ps(mean_g);
        const __m512 v_mb   = _mm512_set1_ps(mean_b);

        const int simd_end = (area / 16) * 16;
        for (int i = 0; i < simd_end; i += 16) {
            alignas(64) uint8_t r[16], g[16], b[16];
            for (int j = 0; j < 16; ++j) {
                const int p = (i + j) * 3;
                r[j] = data[p + 0];
                g[j] = data[p + 1];
                b[j] = data[p + 2];
            }
            __m128i r8 = _mm_load_si128((const __m128i*)r);
            __m128i g8 = _mm_load_si128((const __m128i*)g);
            __m128i b8 = _mm_load_si128((const __m128i*)b);
            __m512 rf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(r8));
            __m512 gf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(g8));
            __m512 bf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b8));
            rf = _mm512_fmsub_ps(rf, v_invr, v_mr);
            gf = _mm512_fmsub_ps(gf, v_invg, v_mg);
            bf = _mm512_fmsub_ps(bf, v_invb, v_mb);
            _mm512_storeu_ps(dst + 0 * area + i, rf);
            _mm512_storeu_ps(dst + 1 * area + i, gf);
            _mm512_storeu_ps(dst + 2 * area + i, bf);
        }
        for (int i = simd_end; i < area; ++i) {
            const int p = i * 3;
            dst[0 * area + i] = data[p + 0] * inv_std_r - mean_r;
            dst[1 * area + i] = data[p + 1] * inv_std_g - mean_g;
            dst[2 * area + i] = data[p + 2] * inv_std_b - mean_b;
        }
        return;
    }
#endif

#if defined(__AVX2__)
    if (simd >= 1) {
        const __m256 v_invr = _mm256_set1_ps(inv_std_r);
        const __m256 v_invg = _mm256_set1_ps(inv_std_g);
        const __m256 v_invb = _mm256_set1_ps(inv_std_b);
        const __m256 v_mr   = _mm256_set1_ps(mean_r);
        const __m256 v_mg   = _mm256_set1_ps(mean_g);
        const __m256 v_mb   = _mm256_set1_ps(mean_b);

        const int simd_end = (area / 8) * 8;
        for (int i = 0; i < simd_end; i += 8) {
            alignas(32) uint8_t r[8], g[8], b[8];
            for (int j = 0; j < 8; ++j) {
                const int p = (i + j) * 3;
                r[j] = data[p + 0];
                g[j] = data[p + 1];
                b[j] = data[p + 2];
            }
            __m128i r8 = _mm_loadl_epi64((const __m128i*)r);
            __m128i g8 = _mm_loadl_epi64((const __m128i*)g);
            __m128i b8 = _mm_loadl_epi64((const __m128i*)b);
            __m256 rf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r8));
            __m256 gf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8));
            __m256 bf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8));
            rf = _mm256_fmsub_ps(rf, v_invr, v_mr);
            gf = _mm256_fmsub_ps(gf, v_invg, v_mg);
            bf = _mm256_fmsub_ps(bf, v_invb, v_mb);
            _mm256_storeu_ps(dst + 0 * area + i, rf);
            _mm256_storeu_ps(dst + 1 * area + i, gf);
            _mm256_storeu_ps(dst + 2 * area + i, bf);
        }
        for (int i = simd_end; i < area; ++i) {
            const int p = i * 3;
            dst[0 * area + i] = data[p + 0] * inv_std_r - mean_r;
            dst[1 * area + i] = data[p + 1] * inv_std_g - mean_g;
            dst[2 * area + i] = data[p + 2] * inv_std_b - mean_b;
        }
        return;
    }
#endif

    // Scalar fallback
    for (int i = 0; i < area; ++i) {
        const int p = i * 3;
        dst[0 * area + i] = data[p + 0] * inv_std_r - mean_r;
        dst[1 * area + i] = data[p + 1] * inv_std_g - mean_g;
        dst[2 * area + i] = data[p + 2] * inv_std_b - mean_b;
    }
}

#ifdef RFDETR_HAVE_OPENCV_CUDA
void preprocess_gpu_opencv(const cv::Mat& image,
                           int dst_w, int dst_h,
                           const std::array<float, 3>& mean,
                           const std::array<float, 3>& std,
                           void* d_dst,
                           void* cuda_stream_ptr) {
    cv::cuda::Stream stream =
        cuda_stream_ptr
            ? cv::cuda::StreamAccessor::wrapStream(static_cast<cudaStream_t>(cuda_stream_ptr))
            : cv::cuda::Stream::Null();

    cv::cuda::GpuMat g_src;
    g_src.upload(image, stream);

    cv::cuda::GpuMat g_resized;
    if (image.cols == dst_w && image.rows == dst_h) {
        g_resized = g_src;
    } else {
        cv::cuda::resize(g_src, g_resized, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR, stream);
    }

    cv::cuda::GpuMat g_rgb;
    cv::cuda::cvtColor(g_resized, g_rgb, cv::COLOR_BGR2RGB, 0, stream);

    cv::cuda::GpuMat g_float;
    g_rgb.convertTo(g_float, CV_32FC3, 1.0 / 255.0, 0.0, stream);

    // Subtract mean, divide by std (per-channel) in-place.
    cv::cuda::subtract(g_float,
                       cv::Scalar(mean[0], mean[1], mean[2]),
                       g_float, cv::noArray(), -1, stream);
    cv::cuda::divide  (g_float,
                       cv::Scalar(std[0], std[1], std[2]),
                       g_float, 1.0, -1, stream);

    // HWC -> CHW via split into 3 planes aimed at d_dst.
    const size_t plane_bytes = static_cast<size_t>(dst_w) * dst_h * sizeof(float);
    cv::cuda::GpuMat planes[3] = {
        cv::cuda::GpuMat(dst_h, dst_w, CV_32FC1, (void*)((char*)d_dst + 0 * plane_bytes)),
        cv::cuda::GpuMat(dst_h, dst_w, CV_32FC1, (void*)((char*)d_dst + 1 * plane_bytes)),
        cv::cuda::GpuMat(dst_h, dst_w, CV_32FC1, (void*)((char*)d_dst + 2 * plane_bytes)),
    };
    cv::cuda::split(g_float, planes, stream);
}
#endif

} // namespace rfdetr::detail
