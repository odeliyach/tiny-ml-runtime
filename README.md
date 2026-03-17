# Tiny ML Runtime: C Inference Core with a Python Training Bridge

[![CI](https://github.com/odeliyach/tiny-ml-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/odeliyach/tiny-ml-runtime/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Compact neural-network inference engine in pure C, wired to a CPython extension so Python trains/exports while C owns the hot path. Direct, production-minded: clear src layout, repeatable builds, tests, Docker, and CI on every push.

## Key Technical Highlights
- **Memory discipline:** Dynamic allocation for weights/biases with `free_network()` cleanup, plus ping-pong activation buffers sized once per run to avoid per-layer allocations and keep cache-friendly hot paths.
- **C ↔ Python interface:** `src/c/inference_module.c` exposes `tinymlinference.predict()` via the CPython C-API. Python hands off a list of floats and a binary weight file; C returns `(class_idx, probs)` without interpreter overhead in the forward path.
- **Numerical stability:** Softmax uses the max-subtraction trick; scaler stats are validated before use; inputs are normalized when provided in the binary.
- **Binary contract:** PyTorch training scripts in `src/python/` export weights, biases, and optional StandardScaler statistics into a compact binary format that `load_weights()` validates before inference.
- **Automation:** Makefile builds/tests the C targets; GitHub Actions compiles, runs unit tests, executes the Iris benchmark artifact, builds the Docker image, and verifies the Python extension build.
- **Testing coverage:** `src/c/test_inference.c` exercises linear layers, ReLU, numerically-stable softmax, scaler handling, and end-to-end prediction (including real Iris weights when present).

## Benchmarks (single-thread, batch=1)

**Conditions:** single-sample inference (batch=1), single-threaded Python loop, CPU-only. PyTorch 2.x, Python 3.11, Apple M1 (ARM64). Each PyTorch call goes through the full Python → C++ dispatch path.

### Iris (4 → 8 → 3) — overhead-bound
| Runtime                  | Predictions/sec | Time (1M iterations) |
|--------------------------|-----------------|----------------------|
| Pure C                   | 2,732,240       | 0.366 seconds        |
| PyTorch (Python, batch=1)| 10,596          | 94.374 seconds       |
| **Speedup (C vs PyTorch)** | **≈258×**      |                      |

> The 258× figure measures Python + PyTorch dispatch overhead on a 4-feature input, not PyTorch's raw compute. Once you batch (dozens or more samples) or run on GPU kernels, that overhead is amortized toward parity and PyTorch becomes the faster path.

### MNIST (784 → 128 → 10) — compute-bound
| Runtime                  | Predictions/sec | Time (100K iterations) |
|--------------------------|-----------------|------------------------|
| Pure C (naive loops)     | 4,244           | 23.563 seconds         |
| PyTorch (Python, batch=1)| 22,502          | 0.444 seconds          |
| **Speedup (PyTorch vs C)** | **~5×**        |                |

**Crossover point:** Iris is dominated by Python/C++ dispatch overhead, so the C path wins. MNIST is dominated by the 128×784 matmul (~100K FLOPs) where PyTorch leans on BLAS/SIMD and beats the naive C loops. Overhead matters only until math takes over.

## System Architecture & Layout
```
Input → Linear → ReLU → ... → Linear → Softmax
```
- **src/c/inference.c**: core forward path (linear, ReLU, softmax), weight loader, benchmark.
- **src/c/inference_module.c**: CPython extension exposing `tinymlinference.predict()`.
- **src/c/test_inference.c**: unit tests for math, stability, scaling, and end-to-end predictions.
- **src/python/**: PyTorch training/export (`train*.py`, `export*.py`) and benchmarks.
- **docs/TECHNICAL_ANALYSIS.md**: math-to-code deep dive and optimization notes.
- **Dockerfile / Makefile / setup.py / pyproject.toml**: container, builds, and Python extension config.
- **data/**: binary weight artifacts (repo ignores everything except `.bin`).

## Technical Depth
### Memory discipline
- Weights/biases allocated dynamically based on the binary; released in `free_network()`.
- Ping-pong activation buffers sized once per run to avoid per-layer allocations and keep cache locality.
- Stack for small temporaries, heap for weight matrices to avoid stack blowups.

### C ↔ Python interface
- CPython C-API parses `(weights_path: str, input: list[float])`, validates shapes, normalizes input if scaler stats are present, runs the C forward pass, and returns `(class_idx, probs)`.
- `setup.py` builds the `tinymlinference` extension from `src/c/inference_module.c` with `-O2 -Wall` and links `-lm`.

### Numerical stability & validation
- Softmax uses max-subtraction to avoid overflow; covered by `test_softmax_numerical_stability`.
- Optional StandardScaler stats are sanity-checked; inputs are normalized before layer 0.
- Binary format: `num_layers`, `layer_sizes[]`, weight/bias pairs per layer, optional `scaler_mean` and `scaler_std` (float32).

### Performance engineering & production patterns
- Benchmarks are repeatable (fixed iterations, single-threaded) to separate overhead vs compute effects.
- Error handling on all allocation/parse steps; predictable cleanup paths to avoid leaks.
- Unit tests cover edge cases (uniform softmax, scaler paths, identity networks) to catch regressions early.

## Getting Started
1) **Train & export (Python):**
```bash
python src/python/train.py
python src/python/export_weights.py          # Iris
python src/python/train_mnist.py
python src/python/export_mnist.py            # MNIST
```

2) **Build and run the C inference engine:**
```bash
make inference
./inference iris
./inference mnist
```

3) **Run C unit tests:**
```bash
make test
```

4) **Use the Python extension (CPython C-API):**
```bash
pip install .
python - <<'PY'
import tinymlinference
cls, probs = tinymlinference.predict("iris_weights.bin", [5.1, 3.5, 1.4, 0.2])
print(cls, probs)
PY
```

5) **Dockerized run:**
```bash
docker build -t tiny-ml-runtime .
docker run --rm tiny-ml-runtime                 # Iris inference
docker run --rm tiny-ml-runtime ./inference mnist
```

## Automation & Quality Gates
- **GitHub Actions (`.github/workflows/ci.yml`):** builds the C binary, runs `make test`, captures the Iris benchmark artifact, builds and runs the Docker image, and installs the Python extension on Python 3.11.
- **Makefile:** single-source build commands to keep local and CI flows aligned (`make inference`, `make test`, `make clean`).
- **Artifacts:** benchmark output is uploaded from CI for traceability.

## Dive Deeper
For the math-to-code mapping, memory strategy, and CPython bridge details, see [`docs/TECHNICAL_ANALYSIS.md`](docs/TECHNICAL_ANALYSIS.md).

## License
MIT License — see [LICENSE](LICENSE) for details.
