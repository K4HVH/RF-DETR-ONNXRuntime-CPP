#!/usr/bin/env python3
"""
Comprehensive SIMD benchmark with multiple runs
"""
import subprocess
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
MODEL_PATH = "models/rf-detr-nano-cpu-int8.onnx"
BENCHMARK_RUNS = 3  # Number of complete benchmark cycles

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
    # Configure
    output = run_command(f"cmake -DSIMD_LEVEL={simd_level} ..", cwd=BUILD_DIR)
    if "error" in output.lower():
        return False

    # Build
    output = run_command("cmake --build . --config Release --clean-first", cwd=BUILD_DIR)
    if "error" in output.lower() and "0 error" not in output.lower():
        return False

    return True

def run_single_benchmark():
    """Run single benchmark and extract time"""
    exe_path = BUILD_DIR / "Release" / "RF-DETR-ONNXRuntime-CPP.exe"
    cmd = f'"{exe_path}" {MODEL_PATH} test.jpg 0.5 CPUExecutionProvider extended high-thread-count'

    output = run_command(cmd)

    match = re.search(r'Average inference time: ([\d.]+) ms', output)
    if match:
        return float(match.group(1))
    return None

def benchmark_config(simd_level, runs=BENCHMARK_RUNS):
    """Benchmark a configuration multiple times"""
    print(f"\nBenchmarking {simd_level}...", flush=True)

    if not configure_and_build(simd_level):
        print(f"  Build failed!")
        return None

    times = []
    for i in range(runs):
        time = run_single_benchmark()
        if time:
            times.append(time)
            print(f"  Run {i+1}/{runs}: {time:.2f}ms", flush=True)
        else:
            print(f"  Run {i+1}/{runs}: FAILED", flush=True)

    if not times:
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
    print("Comprehensive SIMD Benchmark")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Benchmark runs per config: {BENCHMARK_RUNS}")
    print(f"Inference runs per benchmark: 100")
    print("="*80)

    results = {}

    for simd_level in ["NONE", "AVX2", "AVX512"]:
        result = benchmark_config(simd_level)
        if result:
            results[simd_level] = result

    # Print comprehensive results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    print(f"\n{'Config':<10} {'Mean':>10} {'StdDev':>10} {'Min':>10} {'Max':>10} {'Speedup':>10}")
    print("-"*80)

    baseline = results.get("NONE", {}).get('avg')

    for simd_level in ["NONE", "AVX2", "AVX512"]:
        if simd_level in results:
            r = results[simd_level]

            if baseline and baseline > 0 and simd_level != "NONE":
                speedup = ((baseline - r['avg']) / baseline) * 100
                speedup_str = f"{speedup:+.1f}%"
            else:
                speedup_str = "-"

            print(f"{simd_level:<10} {r['avg']:>10.2f} {r['stdev']:>10.2f} "
                  f"{r['min']:>10.2f} {r['max']:>10.2f} {speedup_str:>10}")

    # Recommendation
    if results:
        best = min(results.items(), key=lambda x: x[1]['avg'])
        print(f"\n{'='*80}")
        print(f"RECOMMENDATION: {best[0]} (Average: {best[1]['avg']:.2f}ms)")
        print(f"{'='*80}")

        # Set to recommended
        print(f"\nConfiguring to {best[0]}...")
        configure_and_build(best[0])
        print("Done!")

if __name__ == "__main__":
    main()
