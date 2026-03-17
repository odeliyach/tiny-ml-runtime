# Tiny ML Runtime in C

[![CI](https://github.com/odeliyach/tiny-ml-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/odeliyach/tiny-ml-runtime/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Zero-dependency neural network inference engine in pure C** — demonstrating systems programming, memory management, and Python-C integration.

Train in Python (PyTorch) → Export weights → Run blazing-fast inference in C.
**258x faster** than PyTorch on small networks, revealing exactly where framework overhead matters.

**Key Skills**: C Systems Programming • Memory Management • CPython C-API • Performance Engineering

## Benchmark Results

**Benchmark conditions:** single-sample inference (batch size = 1), single-threaded Python loop,
no GPU, CPU-only. PyTorch 2.x, Python 3.11, Apple M1 (ARM64).
Each PyTorch call goes through the full Python → C++ dispatch path.

### Iris (4 → 8 → 3) - tiny network, overhead-bound
| Runtime                  | Predictions/sec | Time (1M iterations) |
|--------------------------|-----------------|----------------------|
| Pure C                   | 2,732,240       | 0.366 seconds        |
| PyTorch (Python, batch=1)| 10,596          | 94.374 seconds       |
| **Speedup**              | **258x**        |                      |

> ⚠️ The 258x figure measures **Python + PyTorch dispatch overhead** on a 4-feature input,
> not PyTorch's raw computational throughput. With batching or GPU this gap disappears entirely.

### MNIST (784 → 128 → 10) - larger network, compute-bound
| Runtime                  | Predictions/sec | Time (100K iterations) |
|--------------------------|-----------------|------------------------|
| Pure C (naive loops)     | 4,244           | 23.563 seconds         |
| PyTorch (Python, batch=1)| 22,502          | 0.444 seconds          |
| **Speedup**              | **PyTorch 5x faster** |                   |

### Why the numbers tell opposite stories

On **Iris (4 inputs)**, the actual matrix multiplications are trivial — just a few dozen
floating-point operations. The bottleneck is the Python interpreter and PyTorch's
C++ dispatch overhead on every call. C has zero overhead, so it wins by a wide margin.

On **MNIST (784 inputs)**, the dominant cost is the `W1` matrix multiply: 128 × 784 = ~100K
multiply-add operations per prediction. PyTorch calls into optimized BLAS (OpenBLAS / Accelerate)
with SIMD vectorization. Our C code uses naive nested loops — no SIMD, no cache tiling,
no parallelism — so PyTorch wins here.

**The crossover point is the real insight:** it reveals exactly where framework overhead ends
and where raw computation begins. A C implementation is only faster than PyTorch when
the network is small enough that dispatch overhead dominates the math.

## Architecture
```
Input layer
    ↓
Hidden layer(s) + ReLU
    ↓
Output layer + Softmax
```

Tested on:
- Iris:  4 → 8 → 3   (flower classification, 3 classes)
- MNIST: 784 → 128 → 10  (digit recognition, 10 classes)

## What's implemented from scratch in C

- Matrix multiplication (naive, row-major flat arrays)
- ReLU activation function
- Softmax with numerical stability (max subtraction)
- Generic forward pass — reads architecture from weights file
- Binary weight loader with optional StandardScaler normalization
- Benchmarking

## How it works

1. Train a network in PyTorch (~50 lines of Python)
2. Export architecture + weights to a binary file
3. Load in C — zero dependencies, just math

The binary format:
```
[num_layers: int32]
[layer_sizes: int32 x num_layers]
[W0: float32 x rows*cols]
[b0: float32 x rows]
...repeat for each layer...
[scaler_mean: float32 x input_size]  ← optional
[scaler_std:  float32 x input_size]  ← optional
```

## Sample Output
```
=== Iris (4 -> 8 -> 3) ===
Weights loaded successfully!

Flower 1: [5.1, 3.5, 1.4, 0.2]
Probabilities: [1.000, 0.000, 0.000]
Predicted: Setosa (class 0)

Flower 2: [6.0, 2.9, 4.5, 1.5]
Probabilities: [0.000, 0.993, 0.007]
Predicted: Versicolor (class 1)

Flower 3: [6.3, 3.3, 6.0, 2.5]
Probabilities: [0.000, 0.000, 1.000]
Predicted: Virginica (class 2)

--- Benchmark ---
Iterations:       1,000,000
Time:             0.366 seconds
Predictions/sec:  2,732,240
```

## Project Structure
```
inference.c           # C inference engine (the interesting part)
inference_module.c    # CPython C extension — exposes engine to Python
setup.py              # Python package build (pip install .)
test_inference.c      # Unit tests for core functions
Makefile              # Build system (make, make test, make clean)
Dockerfile            # Containerised inference engine
train.py              # Train Iris model in PyTorch
train_mnist.py        # Train MNIST model in PyTorch
export_weights.py     # Export Iris weights to binary
export_mnist.py       # Export MNIST weights to binary
benchmark.py          # PyTorch benchmark (Iris)
benchmark_mnist.py    # PyTorch benchmark (MNIST)
```

## Run it yourself

**Train and export:**
```bash
py train.py
py train_mnist.py
py export_weights.py
py export_mnist.py
```

**Compile and run:**
```bash
make            # build inference engine
./inference iris
./inference mnist
```

**Run tests:**
```bash
make test       # build and run unit tests
```

**Docker:**
```bash
docker build -t tiny-ml-runtime .
docker run --rm tiny-ml-runtime            # Iris inference
docker run --rm tiny-ml-runtime ./inference mnist  # MNIST inference
```

**Python package (CPython C-API):**
```bash
pip install .
```

```python
import tinymlinference

class_idx, probs = tinymlinference.predict("iris_weights.bin", [5.1, 3.5, 1.4, 0.2])
print(class_idx, probs)  # 0  [1.0, 0.0, 0.0]
```

## Why I built this

I wanted to understand what PyTorch actually does under the hood,
so I implemented neural network inference from scratch in C —
matrix multiplication, ReLU, Softmax, the full forward pass.

The benchmark results tell an interesting story: for single-sample
(batch=1) inference, C is 258x faster on tiny networks because
PyTorch's Python dispatch overhead dominates the actual computation.
But on larger networks (MNIST), PyTorch's BLAS/SIMD back-end wins
by 5x because our C code uses naive loops.
That crossover — the point where dispatch overhead gives way to
raw FLOP throughput — is the real insight.

## What This Project Demonstrates

### 1. **Systems Programming & Memory Management**
- Manual memory allocation with proper cleanup (no leaks)
- Buffer reuse patterns (ping-pong buffers reduce allocations)
- Understanding stack vs heap tradeoffs for different data sizes
- Pointer arithmetic for efficient matrix operations on flat arrays

### 2. **Performance Engineering**
- Identifying performance bottlenecks (overhead vs computation)
- Benchmarking methodology (1M iterations, controlled environment)
- Understanding when naive C beats optimized frameworks (and when it doesn't)
- Recognizing the crossover point between framework overhead and raw throughput

### 3. **Polyglot Programming (Python ↔ C)**
- CPython C-API for native extensions
- Cross-language type conversion and error handling
- Build system integration (setup.py, pyproject.toml)
- Knowing when to drop down to C for performance

### 4. **Production Patterns**
- Numerical stability techniques (softmax max-subtraction trick)
- Proper error handling and input validation
- Comprehensive unit testing (25 tests covering edge cases)
- CI/CD pipeline with multiple build targets

## License

MIT License — see [LICENSE](LICENSE) for details.
