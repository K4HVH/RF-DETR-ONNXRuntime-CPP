#!/usr/bin/env python3
"""
Static INT8 quantisation for RF-DETR ONNX models via onnxruntime's
QDQ quantiser + entropy calibration.

Output model has Q/DQ nodes inlined so TensorRT can run it in explicit-precision
INT8 mode without a separate calibration table.

Usage:
    python quantize_int8.py \
        --model models/inference_model.onnx \
        --calib-dir path/to/calibration_images \
        --output models/inference_model_int8.onnx
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantType,
    quantize_static,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img: np.ndarray, w: int, h: int) -> np.ndarray:
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(img, (2, 0, 1))[None, :, :, :].astype(np.float32)


class ImageFolderReader(CalibrationDataReader):
    def __init__(self, folder: str, input_name: str, w: int, h: int, limit: int):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder, "**", e), recursive=True))
        files.sort()
        if limit > 0:
            files = files[:limit]
        if not files:
            raise RuntimeError(f"No calibration images found under {folder}")
        print(f"[calib] {len(files)} images from {folder}")
        self.files = files
        self.i = 0
        self.input_name = input_name
        self.w, self.h = w, h

    def get_next(self):
        if self.i >= len(self.files):
            return None
        p = self.files[self.i]; self.i += 1
        img = cv2.imread(p)
        if img is None:
            return self.get_next()
        return {self.input_name: preprocess(img, self.w, self.h)}

    def rewind(self):
        self.i = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     required=True, type=Path)
    ap.add_argument("--output",    required=True, type=Path)
    ap.add_argument("--calib-dir", required=True, type=Path)
    ap.add_argument("--limit",     type=int, default=200, help="Max calibration images")
    ap.add_argument("--method",    choices=["entropy", "minmax", "percentile"], default="entropy")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    # Infer input shape
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0]
    _, _, h, w = (d if isinstance(d, int) and d > 0 else 384 for d in inp.shape)
    print(f"[calib] model input: {inp.name} [{inp.shape}]  -> using {h}x{w}")

    reader = ImageFolderReader(str(args.calib_dir), inp.name, w, h, args.limit)

    method = {
        "entropy":    CalibrationMethod.Entropy,
        "minmax":     CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
    }[args.method]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(args.model),
        model_output=str(args.output),
        calibration_data_reader=reader,
        calibrate_method=method,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    print(f"[calib] INT8 model written: {args.output}")

    # Sanity-check
    m = onnx.load(str(args.output))
    qdq = sum(1 for n in m.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear"))
    print(f"[calib] QDQ node count: {qdq}")


if __name__ == "__main__":
    main()
