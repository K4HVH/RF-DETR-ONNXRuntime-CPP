#!/usr/bin/env python3
"""
Optimize and quantize RF-DETR ONNX model for maximum CPU performance
"""
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import sys

def optimize_model(input_model, output_model):
    """Apply ONNX Runtime graph optimizations"""
    print(f"\n{'='*80}")
    print(f"Optimizing model: {input_model}")
    print(f"{'='*80}\n")

    # Create session options with maximum graph optimization
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = output_model

    # Load and optimize
    print("Loading model and applying graph optimizations...")
    session = ort.InferenceSession(input_model, sess_options, providers=['CPUExecutionProvider'])

    print(f"[OK] Optimized model saved to: {output_model}")

    # Get file sizes
    original_size = os.path.getsize(input_model) / (1024 * 1024)
    optimized_size = os.path.getsize(output_model) / (1024 * 1024)

    print(f"\nModel sizes:")
    print(f"  Original:  {original_size:.2f} MB")
    print(f"  Optimized: {optimized_size:.2f} MB")
    print(f"  Reduction: {((original_size - optimized_size) / original_size * 100):.1f}%")

def quantize_model(input_model, output_model):
    """Quantize model to INT8 for faster inference"""
    print(f"\n{'='*80}")
    print(f"Quantizing model to INT8: {input_model}")
    print(f"{'='*80}\n")

    print("Applying dynamic INT8 quantization...")
    print("This will convert FP32 weights to INT8...")

    try:
        quantize_dynamic(
            model_input=input_model,
            model_output=output_model,
            weight_type=QuantType.QUInt8,  # Quantize to 8-bit unsigned integers
            per_channel=True,  # Per-channel quantization for better accuracy
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
        return False

def verify_model(model_path):
    """Verify model can be loaded and check its properties"""
    print(f"\nVerifying model: {model_path}")

    try:
        # Load with ONNX
        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        # Load with ONNX Runtime
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        print(f"[OK] Model is valid")
        print(f"  Inputs: {[(i.name, i.shape, i.type) for i in inputs]}")
        print(f"  Outputs: {[(o.name, o.shape, o.type) for o in outputs]}")

        return True

    except Exception as e:
        print(f"[ERROR] Model verification failed: {e}")
        return False

if __name__ == "__main__":
    original_model = "rf-detr-nano.onnx"
    optimized_model = "rf-detr-nano-optimized.onnx"
    quantized_model = "rf-detr-nano-quantized.onnx"

    if not os.path.exists(original_model):
        print(f"Error: {original_model} not found!")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("RF-DETR Model Optimization Pipeline")
    print(f"{'='*80}")

    # Step 1: Optimize graph
    optimize_model(original_model, optimized_model)
    verify_model(optimized_model)

    # Step 2: Quantize to INT8
    success = quantize_model(original_model, quantized_model)
    if success:
        verify_model(quantized_model)

    # Step 3: Quantize the optimized model
    quantized_optimized_model = "rf-detr-nano-optimized-quantized.onnx"
    print(f"\n{'='*80}")
    print("Bonus: Quantizing the optimized model")
    print(f"{'='*80}")
    success = quantize_model(optimized_model, quantized_optimized_model)
    if success:
        verify_model(quantized_optimized_model)

    print(f"\n{'='*80}")
    print("Summary of generated models:")
    print(f"{'='*80}")

    models = [
        (original_model, "Original FP32"),
        (optimized_model, "Graph-optimized FP32"),
        (quantized_model, "INT8 quantized"),
        (quantized_optimized_model, "Optimized + INT8 quantized (BEST)")
    ]

    print(f"\n{'Model':<45} {'Size':<15} {'Description'}")
    print("-" * 80)

    for model_path, description in models:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"{os.path.basename(model_path):<45} {size_mb:>8.2f} MB    {description}")

    print(f"\n{'='*80}")
    print("Done! Test each model to find the best speed/accuracy trade-off")
    print(f"{'='*80}\n")
