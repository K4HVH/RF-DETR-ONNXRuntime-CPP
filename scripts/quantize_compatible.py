#!/usr/bin/env python3
"""
Quantize with settings optimized for CPU compatibility
Uses per-tensor quantization with simpler zero points
"""
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import sys

def quantize_cpu_compatible(input_model, output_model):
    """Quantize model with CPU-compatible settings"""
    print(f"\n{'='*80}")
    print(f"Quantizing model for CPU compatibility: {input_model}")
    print(f"{'='*80}\n")

    print("Using per-tensor quantization (not per-channel)...")
    print("This should produce simpler zero point values...")

    try:
        quantize_dynamic(
            model_input=input_model,
            model_output=output_model,
            weight_type=QuantType.QUInt8,
            per_channel=False,  # Per-tensor for compatibility
            reduce_range=True,   # Reduce range for CPU compatibility
        )

        print(f"[OK] Quantized model saved to: {output_model}")

        # Get file sizes
        original_size = os.path.getsize(input_model) / (1024 * 1024)
        quantized_size = os.path.getsize(output_model) / (1024 * 1024)

        print(f"\nModel sizes:")
        print(f"  Original:  {original_size:.2f} MB")
        print(f"  Quantized: {quantized_size:.2f} MB")
        print(f"  Reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")

        return True

    except Exception as e:
        print(f"[ERROR] Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    original_model = "rf-detr-nano.onnx"
    quantized_model = "rf-detr-nano-cpu-compat.onnx"

    if not os.path.exists(original_model):
        print(f"Error: {original_model} not found!")
        sys.exit(1)

    success = quantize_cpu_compatible(original_model, quantized_model)

    if success:
        print("\n" + "="*80)
        print("Quantization complete! Test with:")
        print(f"  .\\build\\Release\\RF-DETR-ONNXRuntime-CPP.exe {quantized_model} test.jpg")
        print("="*80)
        sys.exit(0)
    else:
        sys.exit(1)
