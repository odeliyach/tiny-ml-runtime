# Architecture Design Document

## Overview

Tiny ML Runtime is a production-grade neural network inference engine demonstrating the performance characteristics and trade-offs between high-level ML frameworks and low-level implementations.

## Design Philosophy

### Zero-Dependency Inference
The core inference engine (`inference.c`) has no external dependencies beyond the C standard library. This design choice enables:
- **Portability**: Runs on any platform with a C compiler
- **Embeddability**: Suitable for resource-constrained environments
- **Auditability**: Entire codebase fits in memory, no hidden dependencies

### Training-Inference Separation
Training happens in PyTorch (Python), inference in pure C. This architecture reflects real-world ML deployment patterns:
- Train with high-level frameworks (ergonomics, auto-differentiation)
- Deploy with optimized runtime (performance, low resource usage)

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
│  (Python + PyTorch)                                     │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────┐   │
│  │ Data     │──▶│ Training │──▶│ Weight          │   │
│  │ Loading  │   │ Loop     │   │ Serialization   │   │
│  └──────────┘   └──────────┘   └─────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼ (binary format)
┌─────────────────────────────────────────────────────────┐
│                  Inference Engine                        │
│  (Pure C, zero dependencies)                            │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────┐   │
│  │ Weight   │──▶│ Forward  │──▶│ Softmax +       │   │
│  │ Loader   │   │ Pass     │   │ Argmax          │   │
│  └──────────┘   └──────────┘   └─────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Custom Binary Format vs. Standard Formats (ONNX, Protobuf)

**Decision**: Custom binary format

**Rationale**:
- ONNX/Protobuf require parsing libraries (defeats zero-dependency goal)
- Our format is trivial: architecture header + raw float32 arrays
- 20 lines of C vs. 2000+ lines for ONNX parser
- Target use case: demonstration/education, not production deployment

**Trade-off**: Limited interoperability (acceptable for project scope)

### 2. Naive Matrix Multiplication vs. Optimized BLAS

**Decision**: Naive triple-nested loops

**Rationale**:
- **Educational value**: Shows the performance cliff between naive and optimized code
- **Benchmark insight**: Reveals framework overhead vs. computational cost crossover
- **Code clarity**: `W[i*cols + j] * in[j]` is more readable than BLAS `sgemm` calls

**Performance impact**:
- Small networks (Iris): Framework overhead dominates, naive code still 258x faster than PyTorch
- Large networks (MNIST): BLAS matters, PyTorch 5x faster

**Production consideration**: Real deployments would use BLAS/MKL/NEON intrinsics

### 3. Single-Threaded Execution

**Decision**: No parallelism in inference engine

**Rationale**:
- Keeps code simple and portable
- Single-sample inference (batch=1) has no intra-sample parallelism
- Demonstrates that framework overhead, not parallelism, explains small-network speedup

**Scaling path**: For production, add OpenMP pragmas around matrix multiply loops

### 4. Float32 vs. Quantization

**Decision**: Full-precision float32

**Rationale**:
- Focus is on architectural patterns, not optimization techniques
- Quantization (INT8/INT16) is orthogonal to the core design
- Keeps weight format simple

**Future work**: INT8 quantization would add ~4x memory savings and 2-3x speedup on ARM

### 5. Error Handling Strategy

**Current**: Return codes (0 = error, 1 = success)

**Production upgrade**:
```c
typedef enum {
    TML_SUCCESS = 0,
    TML_ERR_FILE_NOT_FOUND,
    TML_ERR_INVALID_FORMAT,
    TML_ERR_MEMORY_ALLOCATION,
    TML_ERR_INVALID_ARCHITECTURE
} tml_error_t;
```

### 6. Memory Management

**Current**: Manual `malloc`/`free` in load_weights/free_network

**Safety considerations**:
- All allocations checked for NULL
- Ownership: Network struct owns all weight memory
- Lifetime: Caller must call `free_network()` before destruction

**Production hardening**:
- Add arena allocator for batch allocations
- Add memory pool for repeated inference calls
- Valgrind-clean (zero leaks, zero invalid accesses)

## Performance Model

The benchmark results reveal the **framework overhead crossover point**:

### Small Networks (Iris: 4→8→3)
- **Computation**: ~100 FLOPs per inference
- **Overhead**: Python dispatch, PyTorch C++ layer, tensor creation
- **Result**: Overhead >> computation → C wins 258x

### Large Networks (MNIST: 784→128→10)
- **Computation**: ~100K FLOPs per inference
- **Overhead**: Same as above (~constant)
- **Result**: Computation >> overhead → Optimized BLAS wins 5x

**Insight**: The crossover happens around 10K-50K FLOPs per inference, where:
```
T_pytorch = T_overhead + T_blas_compute
T_c_naive = T_naive_compute

Crossover when: T_overhead ≈ (T_naive_compute - T_blas_compute)
```

## Security Considerations

### Current Threat Model
- **Trusted weight files**: We assume weights are benign (no adversarial poisoning checks)
- **No input validation**: Assumes caller provides correct input dimensions
- **Stack safety**: Fixed MAX_LAYERS=10 prevents unbounded allocation

### Production Hardening
1. **Weight file validation**: Check magic bytes, size limits, checksum
2. **Input sanitization**: Verify input dimensions match architecture
3. **Bounds checking**: Add `assert()` or runtime checks on all array accesses
4. **Fuzzing**: AFL/libFuzzer to find crashes in `load_weights()`

## Deployment Patterns

### Embedded Systems
```c
// Compile once, link with application
gcc -c -O3 inference.c -o inference.o
ar rcs libtml.a inference.o

// Application links against libtml.a
gcc my_app.c -L. -ltml -lm -o my_app
```

### Python Extension (CPython C-API)
```python
import tinymlinference
class_idx, probs = tinymlinference.predict("model.bin", input_data)
```

### Docker Container
```dockerfile
FROM alpine:latest
COPY inference /usr/local/bin/
COPY model.bin /data/
CMD ["/usr/local/bin/inference", "/data/model.bin"]
```

## Testing Strategy

### Unit Tests (`test_inference.c`)
- Matrix multiplication correctness
- ReLU activation
- Softmax numerical stability
- Weight loading with various architectures

### Integration Tests
- End-to-end: Train → Export → Infer → Verify accuracy
- Cross-validation: C predictions match PyTorch predictions

### Property-Based Tests
- Softmax always sums to 1.0 (within float epsilon)
- Output always has exactly one maximum (argmax uniqueness)
- Weights are bit-for-bit identical after load/save round-trip

## Future Extensions

### 1. SIMD Vectorization
- Replace naive loops with SSE/AVX intrinsics
- Expected speedup: 4-8x on x86, 2-4x on ARM NEON

### 2. Multi-Layer Support
- Add Convolutional layers (conv2d, pooling)
- Add Batch Normalization
- Extend to ResNet-18 / MobileNetV2

### 3. Quantization
- INT8 quantization with scale/zero-point
- Dynamic range calibration
- Simulated quantization during training

### 4. WebAssembly Target
```bash
emcc inference.c -O3 -s WASM=1 -o inference.wasm
```

## References

- [ONNX Runtime Architecture](https://onnxruntime.ai/docs/reference/high-level-design.html)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [CMSIS-NN: Efficient Neural Network Kernels for Cortex-M](https://arxiv.org/abs/1801.06601)
