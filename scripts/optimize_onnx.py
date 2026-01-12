#!/usr/bin/env python3
"""
Comprehensive ONNX Model Optimization Script
Supports multiple precisions (FP32, FP16, INT8), targets (CPU, GPU), and optimization levels
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Install with: pip install onnx onnxruntime onnxruntime-gpu")
    sys.exit(1)


class ONNXOptimizer:
    """Comprehensive ONNX model optimizer"""

    # Optimization level mapping
    OPT_LEVELS = {
        'disable': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        'extended': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        'all': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    # Quantization types
    QUANT_TYPES = {
        'int8': QuantType.QInt8,
        'uint8': QuantType.QUInt8,
    }

    def __init__(self, input_model: str, output_model: str, precision: str,
                 target: str, opt_level: str, tensorrt_path: Optional[str] = None,
                 per_channel: bool = True, reduce_range: bool = False):
        """
        Initialize optimizer

        Args:
            input_model: Path to input ONNX model
            output_model: Path for output optimized model
            precision: One of 'fp32', 'fp16', 'int8'
            target: One of 'cpu', 'gpu'
            opt_level: One of 'disable', 'basic', 'extended', 'all'
            tensorrt_path: Optional path to TensorRT installation
            per_channel: Use per-channel quantization (better accuracy)
            reduce_range: Reduce quantization range for CPU compatibility
        """
        self.input_model = input_model
        self.output_model = output_model
        self.precision = precision.lower()
        self.target = target.lower()
        self.opt_level = opt_level.lower()
        self.tensorrt_path = tensorrt_path
        self.per_channel = per_channel
        self.reduce_range = reduce_range

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters"""
        if not os.path.exists(self.input_model):
            raise FileNotFoundError(f"Input model not found: {self.input_model}")

        if self.precision not in ['fp32', 'fp16', 'int8']:
            raise ValueError(f"Invalid precision: {self.precision}. Must be one of: fp32, fp16, int8")

        if self.target not in ['cpu', 'gpu']:
            raise ValueError(f"Invalid target: {self.target}. Must be one of: cpu, gpu")

        if self.opt_level not in self.OPT_LEVELS:
            raise ValueError(f"Invalid optimization level: {self.opt_level}. Must be one of: {', '.join(self.OPT_LEVELS.keys())}")

        # INT8 quantization is NOT supported on GPU
        if self.precision == 'int8' and self.target == 'gpu':
            raise ValueError(
                "INT8 quantization is not supported on GPU.\n"
                "INT8 creates CPU-only operators (DynamicQuantizeLinear, ConvInteger) that have no GPU implementation.\n"
                "For GPU acceleration, use --precision fp16 or --precision fp32 instead."
            )

        # Check for GPU support if target is GPU
        if self.target == 'gpu':
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in available_providers and 'TensorrtExecutionProvider' not in available_providers:
                print(f"Warning: GPU requested but no GPU providers available. Available: {available_providers}")

    def _get_execution_providers(self):
        """Get appropriate execution providers for target"""
        if self.target == 'cpu':
            return ['CPUExecutionProvider']
        else:  # gpu
            providers = []
            available = ort.get_available_providers()

            # Prefer TensorRT, then CUDA, fallback to CPU
            if 'TensorrtExecutionProvider' in available:
                trt_options = {}
                if self.tensorrt_path:
                    trt_options['trt_engine_cache_path'] = str(Path(self.output_model).parent)
                providers.append(('TensorrtExecutionProvider', trt_options))

            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')

            # Always add CPU as fallback
            providers.append('CPUExecutionProvider')

            return providers

    def _print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{'='*80}")
        print(f"{title}")
        print(f"{'='*80}\n")

    def _get_model_size_mb(self, model_path: str) -> float:
        """Get model file size in MB"""
        return os.path.getsize(model_path) / (1024 * 1024)

    def _verify_model(self, model_path: str) -> bool:
        """Verify model is valid and can be loaded"""
        print(f"Verifying model: {model_path}")

        try:
            # Load with ONNX
            model = onnx.load(model_path)
            onnx.checker.check_model(model)

            # Load with ONNX Runtime
            providers = self._get_execution_providers()
            session = ort.InferenceSession(model_path, providers=providers)

            inputs = session.get_inputs()
            outputs = session.get_outputs()

            print(f"[OK] Model is valid")
            print(f"  Inputs:  {[(i.name, i.shape, i.type) for i in inputs]}")
            print(f"  Outputs: {[(o.name, o.shape, o.type) for o in outputs]}")
            print(f"  Providers: {session.get_providers()}")

            return True

        except Exception as e:
            print(f"[ERROR] Model verification failed: {e}")
            return False

    def _optimize_graph(self, input_path: str, output_path: str):
        """Apply graph optimization"""
        print(f"Applying graph optimization level: {self.opt_level.upper()}")

        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = self.OPT_LEVELS[self.opt_level]
        sess_options.optimized_model_filepath = output_path

        # For GPU, enable more optimizations
        if self.target == 'gpu':
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1

        # Get providers
        # Note: Exclude TensorRT for graph optimization (it creates compiled nodes that can't be serialized)
        if self.target == 'gpu':
            available = ort.get_available_providers()
            providers = []
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
        else:
            providers = ['CPUExecutionProvider']

        print(f"Using providers: {providers}")

        session = ort.InferenceSession(input_path, sess_options, providers=providers)

        print(f"[OK] Graph optimization complete")

    def _convert_to_fp16(self, input_path: str, output_path: str):
        """Convert model to FP16"""
        print("Converting to FP16 precision...")

        try:
            from onnxconverter_common import float16

            model = onnx.load(input_path)
            model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
            onnx.save(model_fp16, output_path)

            print("[OK] FP16 conversion complete")

        except ImportError:
            print("[ERROR] onnxconverter-common not installed. Install with: pip install onnxconverter-common")
            raise
        except Exception as e:
            print(f"[ERROR] FP16 conversion failed: {e}")
            raise

    def _quantize_to_int8(self, input_path: str, output_path: str):
        """Quantize model to INT8"""
        print("Quantizing to INT8...")
        print(f"  Per-channel: {self.per_channel}")
        print(f"  Reduce range: {self.reduce_range}")

        # Choose quantization type based on target
        # CPU typically prefers QUInt8, GPU can use QInt8
        if self.target == 'cpu':
            quant_type = QuantType.QUInt8
        else:
            quant_type = QuantType.QInt8

        print(f"  Quantization type: {quant_type}")

        try:
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=quant_type,
                per_channel=self.per_channel,
                reduce_range=self.reduce_range,
            )

            print("[OK] INT8 quantization complete")

        except Exception as e:
            print(f"[ERROR] Quantization failed: {e}")
            raise

    def optimize(self) -> bool:
        """
        Run the optimization pipeline

        Returns:
            True if successful, False otherwise
        """
        try:
            self._print_header(f"ONNX Model Optimization Pipeline")
            print(f"Input model:  {self.input_model}")
            print(f"Output model: {self.output_model}")
            print(f"Precision:    {self.precision.upper()}")
            print(f"Target:       {self.target.upper()}")
            print(f"Opt level:    {self.opt_level.upper()}")

            original_size = self._get_model_size_mb(self.input_model)
            print(f"Original size: {original_size:.2f} MB")

            # Create temporary file for intermediate steps
            temp_model = self.output_model + ".temp.onnx"
            temp_model2 = self.output_model + ".temp2.onnx"
            current_model = self.input_model

            # For INT8: quantize first, then optimize (quantizer can't handle optimized models with custom ops)
            # For FP16/FP32: optimize first, then convert

            if self.precision == 'int8':
                # Step 1: INT8 Quantization (on original model)
                self._print_header("Step 1: INT8 Quantization")
                self._quantize_to_int8(self.input_model, temp_model)
                current_model = temp_model

                # Step 2: Graph Optimization (on quantized model)
                if self.opt_level != 'disable':
                    self._print_header("Step 2: Graph Optimization")
                    self._optimize_graph(current_model, self.output_model)
                else:
                    print("\n[SKIP] Graph optimization disabled")
                    import shutil
                    shutil.move(current_model, self.output_model)

            else:
                # Step 1: Graph Optimization
                if self.opt_level != 'disable':
                    self._print_header("Step 1: Graph Optimization")
                    self._optimize_graph(current_model, temp_model)
                    current_model = temp_model
                else:
                    print("\n[SKIP] Graph optimization disabled")

                # Step 2: Precision Conversion
                if self.precision == 'fp16':
                    self._print_header("Step 2: FP16 Conversion")
                    self._convert_to_fp16(current_model, self.output_model)

                else:  # fp32
                    # Just copy/rename if we did graph optimization
                    if current_model != self.input_model:
                        import shutil
                        shutil.move(current_model, self.output_model)
                    else:
                        import shutil
                        shutil.copy2(current_model, self.output_model)

            # Clean up temp files if they exist
            if os.path.exists(temp_model):
                os.remove(temp_model)
            if os.path.exists(temp_model2):
                os.remove(temp_model2)

            # Step 3: Verification
            self._print_header("Step 3: Model Verification")
            if not self._verify_model(self.output_model):
                return False

            # Print summary
            self._print_header("Optimization Summary")
            optimized_size = self._get_model_size_mb(self.output_model)
            size_reduction = ((original_size - optimized_size) / original_size * 100)

            print(f"Original model:  {original_size:>10.2f} MB")
            print(f"Optimized model: {optimized_size:>10.2f} MB")
            print(f"Size change:     {size_reduction:>10.1f}%")
            print(f"\n[SUCCESS] Optimized model saved to: {self.output_model}")

            return True

        except Exception as e:
            print(f"\n[ERROR] Optimization failed: {e}")
            import traceback
            traceback.print_exc()

            # Clean up temp files
            temp_model = self.output_model + ".temp.onnx"
            temp_model2 = self.output_model + ".temp2.onnx"
            if os.path.exists(temp_model):
                os.remove(temp_model)
            if os.path.exists(temp_model2):
                os.remove(temp_model2)

            return False


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive ONNX Model Optimization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize for CPU with FP32 (graph optimization only)
  python optimize_onnx.py model.onnx model_opt.onnx --precision fp32 --target cpu --opt-level extended

  # Quantize to INT8 for CPU
  python optimize_onnx.py model.onnx model_int8.onnx --precision int8 --target cpu --opt-level all

  # Convert to FP16 for GPU with TensorRT
  python optimize_onnx.py model.onnx model_fp16.onnx --precision fp16 --target gpu --opt-level all

  # CPU-compatible INT8 with per-tensor quantization
  python optimize_onnx.py model.onnx model_int8.onnx --precision int8 --target cpu --no-per-channel --reduce-range
        """
    )

    # Required arguments
    parser.add_argument('input_model', type=str, help='Path to input ONNX model')
    parser.add_argument('output_model', type=str, help='Path for output optimized model')

    # Optimization settings
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'int8'],
                        help='Target precision (default: fp32)')

    parser.add_argument('--target', type=str, default='cpu',
                        choices=['cpu', 'gpu'],
                        help='Target device (default: cpu)')

    parser.add_argument('--opt-level', type=str, default='extended',
                        choices=['disable', 'basic', 'extended', 'all'],
                        help='Graph optimization level (default: extended)')

    # Quantization settings
    parser.add_argument('--no-per-channel', dest='per_channel', action='store_false',
                        help='Disable per-channel quantization (use per-tensor instead)')

    parser.add_argument('--reduce-range', action='store_true',
                        help='Reduce quantization range for better CPU compatibility')

    # Advanced settings
    parser.add_argument('--tensorrt-path', type=str, default=None,
                        help='Path to TensorRT installation (default: auto-detect)')

    parser.set_defaults(per_channel=True)

    args = parser.parse_args()

    # Set TensorRT path if not provided and target is GPU
    if args.target == 'gpu' and not args.tensorrt_path:
        default_trt_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit"
        if os.path.exists(default_trt_path):
            args.tensorrt_path = default_trt_path

    # Validate precision/target combinations
    if args.precision == 'fp16' and args.target == 'cpu':
        print("Warning: FP16 on CPU may not provide performance benefits")
        print("Consider using FP32 or INT8 for CPU targets")

    # Create optimizer and run
    try:
        optimizer = ONNXOptimizer(
            input_model=args.input_model,
            output_model=args.output_model,
            precision=args.precision,
            target=args.target,
            opt_level=args.opt_level,
            tensorrt_path=args.tensorrt_path,
            per_channel=args.per_channel,
            reduce_range=args.reduce_range
        )

        success = optimizer.optimize()
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
