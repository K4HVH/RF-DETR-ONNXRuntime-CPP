#!/usr/bin/env python3
"""
Provider comparison benchmark (CUDA vs CPU) with multiple runs
Compares different execution providers on AVX-512
"""
import subprocess
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
BENCHMARK_RUNS = 3  # Number of complete benchmark cycles
SIMD_LEVEL = "AVX512"  # Fixed to AVX-512 for this comparison

# Model configurations for each provider
PROVIDER_CONFIGS = {
    "CPU": [
        {
            "model": "models/rf-detr-nano-cpu-fp32.onnx",
            "name": "CPU-FP32",
            "provider": "CPUExecutionProvider",
            "extra_args": "extended high-thread-count"
        },
        {
            "model": "models/rf-detr-nano-cpu-int8.onnx",
            "name": "CPU-INT8",
            "provider": "CPUExecutionProvider",
            "extra_args": "extended high-thread-count"
        }
    ],
    "CUDA": [
        {
            "model": "models/rf-detr-nano.onnx",
            "name": "CUDA-FP32",
            "provider": "CUDAExecutionProvider",
            "extra_args": "extended"
        }
    ],
    "TensorRT": [
        {
            "model": "models/rf-detr-nano.onnx",
            "name": "TensorRT-FP32",
            "provider": "TensorrtExecutionProvider",
            "extra_args": "extended auto 0"
        },
        {
            "model": "models/rf-detr-nano.onnx",
            "name": "TensorRT-FP16",
            "provider": "TensorrtExecutionProvider",
            "extra_args": "extended auto 1"
        }
    ]
    # Note: Both FP32 and FP16 use the same FP32 model (rf-detr-nano.onnx)
    # TensorRT converts FP32->FP16 internally when fp16 flag is enabled
    # First run will be slow (engine building), subsequent runs fast (cache loaded)
}

def run_command(cmd, cwd=None):
    """Run command and return output"""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    return result.stdout + result.stderr

def configure_and_build(simd_level):
    """Configure and build with specific SIMD level"""
    print(f"  Configuring with SIMD_LEVEL={simd_level}...", flush=True)

    # Configure
    output = run_command(f"cmake -DSIMD_LEVEL={simd_level} ..", cwd=BUILD_DIR)
    if "error" in output.lower():
        print(f"  Configuration failed!")
        return False

    # Build
    print(f"  Building...", flush=True)
    output = run_command("cmake --build . --config Release --clean-first", cwd=BUILD_DIR)
    if "error" in output.lower() and "0 error" not in output.lower():
        print(f"  Build failed!")
        return False

    return True

def run_single_benchmark(config):
    """Run single benchmark and extract time"""
    exe_path = BUILD_DIR / "Release" / "RF-DETR-ONNXRuntime-CPP.exe"
    cmd = f'"{exe_path}" {config["model"]} test.jpg 0.5 {config["provider"]} {config["extra_args"]}'

    output = run_command(cmd)

    match = re.search(r'Average inference time: ([\d.]+) ms', output)
    if match:
        return float(match.group(1))

    # Return output for debugging if failed
    return {"error": output}

def benchmark_config(config, runs=BENCHMARK_RUNS):
    """Benchmark a configuration multiple times"""
    print(f"\nBenchmarking {config['name']}...", flush=True)
    print(f"  Model: {config['model']}")
    print(f"  Provider: {config['provider']}")

    times = []
    error_output = None
    for i in range(runs):
        result = run_single_benchmark(config)
        if isinstance(result, dict) and "error" in result:
            print(f"  Run {i+1}/{runs}: FAILED", flush=True)
            if not error_output:
                error_output = result["error"]
        elif result:
            times.append(result)
            print(f"  Run {i+1}/{runs}: {result:.2f}ms", flush=True)
        else:
            print(f"  Run {i+1}/{runs}: FAILED", flush=True)

    if not times:
        if error_output:
            print(f"  Error output:")
            # Print first 500 chars of error
            print(f"  {error_output[:500]}")
        return None

    avg = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0

    print(f"  Average: {avg:.2f}ms ± {stdev:.2f}ms")

    return {
        'times': times,
        'avg': avg,
        'stdev': stdev,
        'min': min(times),
        'max': max(times)
    }

def main():
    print("="*80)
    print("Provider Comparison Benchmark (AVX-512)")
    print("="*80)
    print(f"SIMD Level: {SIMD_LEVEL}")
    print(f"Benchmark runs per config: {BENCHMARK_RUNS}")
    print(f"Inference runs per benchmark: 100")
    print("="*80)

    # Build once with AVX-512
    print("\n" + "="*80)
    print("BUILDING")
    print("="*80)
    if not configure_and_build(SIMD_LEVEL):
        print("Build failed! Exiting.")
        return

    # Run benchmarks
    print("\n" + "="*80)
    print("RUNNING BENCHMARKS")
    print("="*80)

    results = {}

    for provider_name, configs in PROVIDER_CONFIGS.items():
        print(f"\n--- {provider_name} Provider ---")
        for config in configs:
            result = benchmark_config(config)
            if result:
                results[config['name']] = result

    # Print comprehensive results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    print(f"\n{'Config':<15} {'Mean':>10} {'StdDev':>10} {'Min':>10} {'Max':>10}")
    print("-"*80)

    # Group results by provider
    for provider_name in ["CPU", "CUDA", "TensorRT"]:
        if provider_name not in PROVIDER_CONFIGS:
            continue
        print(f"\n{provider_name}:")
        for config in PROVIDER_CONFIGS[provider_name]:
            name = config['name']
            if name in results:
                r = results[name]
                print(f"  {name:<20} {r['avg']:>10.2f} {r['stdev']:>10.2f} "
                      f"{r['min']:>10.2f} {r['max']:>10.2f}")

    # Comparison
    if len(results) >= 2:
        print("\n" + "="*80)
        print("PROVIDER COMPARISON")
        print("="*80)

        # Compare CPU-FP32 vs CUDA-FP32
        if "CPU-FP32" in results and "CUDA-FP32" in results:
            cpu_fp32 = results["CPU-FP32"]['avg']
            cuda_fp32 = results["CUDA-FP32"]['avg']
            speedup = ((cpu_fp32 - cuda_fp32) / cpu_fp32) * 100
            print(f"CPU-FP32 vs CUDA-FP32: {speedup:+.1f}% "
                  f"({'CUDA faster' if speedup > 0 else 'CPU faster'})")

        # Compare CPU-INT8 vs CUDA-FP32
        if "CPU-INT8" in results and "CUDA-FP32" in results:
            cpu_int8 = results["CPU-INT8"]['avg']
            cuda_fp32 = results["CUDA-FP32"]['avg']
            speedup = ((cpu_int8 - cuda_fp32) / cpu_int8) * 100
            print(f"CPU-INT8 vs CUDA-FP32: {speedup:+.1f}% "
                  f"({'CUDA faster' if speedup > 0 else 'CPU faster'})")

        # CUDA vs TensorRT FP32
        if "CUDA-FP32" in results and "TensorRT-FP32" in results:
            cuda_fp32 = results["CUDA-FP32"]['avg']
            trt_fp32 = results["TensorRT-FP32"]['avg']
            speedup = ((cuda_fp32 - trt_fp32) / cuda_fp32) * 100
            print(f"CUDA-FP32 vs TensorRT-FP32: {speedup:+.1f}% "
                  f"({'TensorRT faster' if speedup > 0 else 'CUDA faster'})")

        # TensorRT FP32 vs FP16
        if "TensorRT-FP32" in results and "TensorRT-FP16" in results:
            trt_fp32 = results["TensorRT-FP32"]['avg']
            trt_fp16 = results["TensorRT-FP16"]['avg']
            speedup = ((trt_fp32 - trt_fp16) / trt_fp32) * 100
            print(f"TensorRT-FP32 vs TensorRT-FP16: {speedup:+.1f}% "
                  f"({'FP16 faster' if speedup > 0 else 'FP32 faster'})")

        # Best overall
        best = min(results.items(), key=lambda x: x[1]['avg'])
        print(f"\nBest overall: {best[0]} (Average: {best[1]['avg']:.2f}ms)")

        print("="*80)

if __name__ == "__main__":
    main()
