<div align="center">

# Tiny ML Runtime in C

[![Build Status](https://github.com/odeliyach/tiny-ml-runtime/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/odeliyach/tiny-ml-runtime/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language](https://img.shields.io/badge/Language-C-blue.svg)]()
[![Coverage](https://codecov.io/gh/odeliyach/tiny-ml-runtime/branch/main/graph/badge.svg)](https://codecov.io/gh/odeliyach/tiny-ml-runtime)
</div>

## Technical Stack
- `C (C11)`
- `Python`
- `PyTorch`
- `Docker`
- `GitHub Actions`

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

## Technical Highlights

- **C inference core (from scratch):** Matrix multiplication (naive, row-major flat arrays), ReLU, Softmax with numerical stability, generic forward pass driven by weight files, and a binary loader with optional StandardScaler normalization.
- **Systems programming & memory management:** Manual allocation with cleanup, buffer ping-pong to avoid churn, and stack vs heap sizing tradeoffs.
- **Performance engineering:** Benchmarking discipline (1M iterations, controlled runs), separating framework overhead from math, and reading the overhead/computation crossover.
- **Polyglot programming (Python ↔ C):** CPython C-API extension, cross-language error handling, and packaging via `setup.py`/`pyproject.toml`.
- **Production patterns:** Numerical stability, input validation, unit tests (25 cases), and CI/CD coverage across targets.
- **Infrastructure:** Containerized environment (Docker) and automated CI/CD pipeline (GitHub Actions) for build and test validation.

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

## Project Structure
```
src/
  c/
    inference.c          # C inference engine
    inference_module.c   # CPython C extension
    test_inference.c     # Unit tests for core functions
  python/
    train.py             # Train Iris model in PyTorch
    train_mnist.py       # Train MNIST model in PyTorch
    export_weights.py    # Export Iris weights to binary
    export_mnist.py      # Export MNIST weights to binary
    benchmark.py         # PyTorch benchmark (Iris)
    benchmark_mnist.py   # PyTorch benchmark (MNIST)
docs/
  TECHNICAL_ANALYSIS.md  # Deeper dive on design and performance
data/                    # Generated weight files (.bin)
Makefile                 # Build system (make, make test, make clean)
Dockerfile               # Containerised inference engine
setup.py / pyproject.toml# Python package build metadata
```

## Run it yourself

**Train and export** (from the repo root so the generated binaries land in `data/`):
```bash
mkdir -p data
python3 src/python/train.py
python3 src/python/export_weights.py

python3 src/python/train_mnist.py
python3 src/python/export_mnist.py
```
The export scripts now write their `.bin` files directly into `data/`, so inference and tests can find them immediately.

**Compile and run:**
```bash
make                         # build inference engine
./inference iris             # expects data/iris_weights.bin
./inference mnist            # expects data/mnist_weights.bin
```

**Run tests:**
```bash
make c_tests    # build + run Unity + legacy C tests
pytest          # run Python wrapper tests (requires `pip install -e .`)
```

**Coverage (local):**
```bash
COVERAGE=1 make c_tests
lcov --capture --directory . --output-file coverage.info
pytest --cov=tinymlinference --cov=src/python --cov-report=xml
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

class_idx, probs = tinymlinference.predict("data/iris_weights.bin", [5.1, 3.5, 1.4, 0.2])
print(class_idx, probs)  # 0  [1.0, 0.0, 0.0]
```

## Why I built this

I wanted to see exactly what PyTorch does under the hood, so I wrote the inference path myself in C — matrix multiplication, ReLU, Softmax, the full forward pass. Building it from scratch made the tradeoffs obvious: on tiny, batch=1 networks the Python dispatch path dominates (C wins by 258x), but on MNIST the optimized BLAS/SIMD back-end in PyTorch beats my naive loops by 5x. That crossover — where overhead yields to raw computation — is the insight I was chasing.

## License

MIT License — see [LICENSE](LICENSE) for details.
