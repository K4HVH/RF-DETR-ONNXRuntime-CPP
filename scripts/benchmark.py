#!/usr/bin/env python3
"""
Multi-provider / multi-precision benchmark harness for rfdetr_demo.

Runs the demo binary across every (device, precision) combination available on the
host, collects per-iteration stats, and prints a comparison table.

Usage:
    python scripts/benchmark.py [--iters 100] [--warmup 10] [--input test2.png]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def locate_demo() -> Path:
    candidates = [
        ROOT / "build" / "bin" / "Release" / "rfdetr_demo.exe",
        ROOT / "build" / "bin" / "rfdetr_demo",
        ROOT / "build" / "Release" / "rfdetr_demo.exe",
        ROOT / "build" / "rfdetr_demo",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find rfdetr_demo. Build first: cmake --build build --config Release"
    )


def run_one(demo: Path, device: str, precision: str, img: str, iters: int, warmup: int,
            model: str, extra: list[str]) -> dict | None:
    cmd = [
        str(demo),
        "--mode", "benchmark",
        "--model", model,
        "--input", img,
        "--device", device,
        "--precision", precision,
        "--iters", str(iters),
        "--warmup", str(warmup),
    ] + extra
    print(f"\n$ {' '.join(cmd)}", flush=True)
    try:
        out = subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.STDOUT, text=True,
                                      timeout=1800)
    except subprocess.CalledProcessError as e:
        print(e.output)
        return None
    print(out)

    def pick(label: str) -> dict | None:
        m = re.search(rf"^{label}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
                      out, flags=re.MULTILINE)
        if not m: return None
        cols = ["mean", "stdev", "p50", "p90", "p99", "min", "max"]
        return dict(zip(cols, map(float, m.groups())))

    dev_m = re.search(r"^Device\s*:\s*(\S+)", out, flags=re.MULTILINE)
    fps_m = re.search(r"FPS \(mean\)\s*:\s*([\d.]+)", out)
    return {
        "requested": f"{device}/{precision}",
        "active":    dev_m.group(1) if dev_m else "?",
        "fps":       float(fps_m.group(1)) if fps_m else 0.0,
        "preprocess": pick("preprocess"),
        "inference":  pick("inference"),
        "postprocess":pick("postprocess"),
        "total":      pick("total"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters",  type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--input",  default="test2.png")
    ap.add_argument("--model",  default="models/inference_model.onnx")
    ap.add_argument("--cache-dir", default="trt_cache")
    ap.add_argument("--skip",  nargs="*", default=[], help="Configs to skip, e.g. cpu/fp32")
    args = ap.parse_args()

    demo = locate_demo()
    print(f"demo: {demo}")

    matrix = [
        ("cpu",          "fp32"),
        ("cuda",         "fp16"),
        ("cuda",         "fp32"),
        ("tensorrt",     "fp16"),
        ("tensorrt",     "fp32"),
        ("tensorrt",     "int8"),   # no-calibration mode
        ("tensorrt-rtx", "fp16"),
    ]

    results = []
    for dev, prec in matrix:
        tag = f"{dev}/{prec}"
        if tag in args.skip: continue
        extra = ["--cache-dir", args.cache_dir]
        if prec == "int8":
            extra += ["--int8-mode", "nocal"]
        r = run_one(demo, dev, prec, args.input, args.iters, args.warmup, args.model, extra)
        if r: results.append(r)

    # Table
    print("\n" + "=" * 88)
    print(f"{'Requested':<18} {'Active':<14} {'Pre(ms)':>9} {'Inf(ms)':>9} {'Post(ms)':>9} {'Total(ms)':>10} {'FPS':>8}")
    print("-" * 88)
    for r in results:
        def g(k, sub="mean"):
            v = r.get(k) or {}
            return f"{v.get(sub, 0):>9.2f}" if v else "         -"
        print(f"{r['requested']:<18} {r['active']:<14} "
              f"{g('preprocess')} {g('inference')} {g('postprocess')} "
              f"{(r['total'] or {}).get('mean', 0):>10.2f} {r['fps']:>8.1f}")

    out = ROOT / "benchmark_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nJSON results -> {out}")


if __name__ == "__main__":
    sys.exit(main())
