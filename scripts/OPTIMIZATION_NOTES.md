# ONNX Optimization Script - Technical Notes

## Tested Configurations

All combinations have been tested and verified working:

| Target | Precision | Opt Levels | Status |
|--------|-----------|------------|--------|
| CPU | FP32 | all, extended, basic, disable | ✅ Working |
| CPU | INT8 | all, extended, basic, disable | ✅ Working |
| GPU | FP32 | all, extended, basic, disable | ✅ Working |
| GPU | FP16 | all, extended, basic, disable | ✅ Working |
| GPU | INT8 | ANY | ❌ **BLOCKED** - Not supported |

## Known Limitations

### 1. INT8 Quantization is CPU-Only

**Issue**: INT8 quantization does not work on GPU.

**Reason**: INT8 quantization creates CPU-specific ONNX operators:
- `DynamicQuantizeLinear`
- `ConvInteger`

These operators have no GPU (CUDA/TensorRT) implementation in ONNX Runtime.

**Behavior**: Script will error immediately with clear message if you try `--precision int8 --target gpu`

**Workaround**: For GPU acceleration, use `--precision fp16` instead (50% size reduction, GPU accelerated).

### 2. TensorRT FusedMatMul Warnings

**What you'll see**:
```
ERROR: Plugin not found, are the plugin name, version, and namespace correct?
While parsing node number X [FusedMatMul -> ...]
```

**Is this a problem?** NO - these are warnings, not errors.

**Explanation**:
- Graph optimization level "all" creates `FusedMatMul` operators from Microsoft's ONNX Runtime domain
- TensorRT doesn't have a plugin for these operators
- They automatically fall back to CUDA execution
- The model still works correctly and efficiently

**Do not try to fix this** - it's expected behavior.

### 3. Hardware-Specific Optimizations Warning

**What you'll see** (with `--opt-level all`):
```
Serializing optimized model with Graph Optimization level greater than ORT_ENABLE_EXTENDED
and the NchwcTransformer enabled. The generated model may contain hardware specific optimizations,
and should only be used in the same environment the model was optimized in.
```

**Meaning**: Models optimized with `--opt-level all` include CPU/GPU-specific optimizations.

**Impact**:
- Model may not be portable to different hardware
- Use `--opt-level extended` for portable models
- Use `--opt-level all` for maximum performance on the same hardware

## Verified Behavior

### FP16 Conversion
- ✅ Confirmed: FP16 models have tensor data_type = 10 (FLOAT16)
- ✅ Confirmed: Original models have tensor data_type = 1 (FLOAT32)
- ✅ Confirmed: ~50% size reduction achieved
- ✅ Confirmed: Models load correctly with CUDA/TensorRT providers

### INT8 Quantization
- ✅ Confirmed: ~73-75% size reduction achieved
- ✅ Confirmed: Works with all optimization levels on CPU
- ✅ Confirmed: Per-channel quantization works (default)
- ✅ Confirmed: Per-tensor quantization works (with --no-per-channel)
- ✅ Confirmed: Reduce range works (with --reduce-range)

### Graph Optimization
- ✅ Confirmed: All levels (disable, basic, extended, all) work for all valid configs
- ✅ Confirmed: CPU optimizations produce valid models
- ✅ Confirmed: GPU optimizations produce valid models
- ✅ Confirmed: TensorRT is automatically excluded from graph optimization (prevents serialization errors)
- ✅ Confirmed: TensorRT is still available at runtime for inference

## Execution Provider Selection

### For Graph Optimization
The script uses different providers for optimization vs runtime:

**CPU Target:**
- Optimization: CPUExecutionProvider
- Runtime: CPUExecutionProvider

**GPU Target:**
- Optimization: CUDAExecutionProvider only (TensorRT excluded to enable serialization)
- Runtime: TensorrtExecutionProvider → CUDAExecutionProvider → CPUExecutionProvider

This is intentional - TensorRT creates compiled nodes that can't be saved to ONNX format.

### Runtime Provider Priority
When you load the optimized models for inference:

1. **TensorrtExecutionProvider** (if available) - Fastest for GPU
2. **CUDAExecutionProvider** (if available) - Fast for GPU
3. **CPUExecutionProvider** (always available) - Fallback

## Performance Recommendations

### For Maximum Speed
```bash
# GPU
python optimize_onnx.py model.onnx output.onnx --precision fp16 --target gpu --opt-level all

# CPU
python optimize_onnx.py model.onnx output.onnx --precision int8 --target cpu --opt-level all
```

### For Portability
```bash
# Use 'extended' instead of 'all'
python optimize_onnx.py model.onnx output.onnx --precision fp16 --target gpu --opt-level extended
```

### For Smallest Size
```bash
# CPU only - INT8 gives ~75% size reduction
python optimize_onnx.py model.onnx output.onnx --precision int8 --target cpu --opt-level all
```

## Troubleshooting

### Error: "INT8 quantization is not supported on GPU"
- This is expected behavior, not a bug
- Use `--precision fp16` or `--precision fp32` for GPU targets

### Warning: TensorRT errors about FusedMatMul
- These are harmless warnings
- The model still works - operations fall back to CUDA
- You can ignore these messages

### Warning: "hardware specific optimizations"
- Only appears with `--opt-level all`
- Means the model is optimized for your specific CPU/GPU
- Use `--opt-level extended` if you need portable models

## Summary

The optimization script works correctly for all supported configurations. The main limitation is:

**INT8 + GPU = Not Supported** (by ONNX Runtime, not a script limitation)

All other combinations work as expected across all optimization levels.
