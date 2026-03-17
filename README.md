# Tiny ML Runtime: C Inference Core with a Python Training Bridge

[![CI](https://github.com/odeliyach/tiny-ml-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/odeliyach/tiny-ml-runtime/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Compact neural-network inference engine in pure C, wired to a CPython extension so Python can train/export models while C handles the hot path. The repo is organized for production-style work: clear src layout, repeatable builds, tests, Docker image, and CI running on every push.

## Key Technical Highlights
- **Memory discipline:** Dynamic allocation for weights/biases with `free_network()` cleanup, plus ping-pong activation buffers sized once per run to avoid per-layer allocations.
- **C ↔ Python interface:** `src/c/inference_module.c` exposes `tinymlinference.predict()` via the CPython C-API. Python hands off a list of floats and a binary weight file; C returns `(class_idx, probs)` without Python overhead in the forward path.
- **Binary contract:** PyTorch training scripts in `src/python/` export weights, biases, and optional StandardScaler statistics into a compact binary format that `load_weights()` validates before inference.
- **Automation:** Makefile builds/tests the C targets; GitHub Actions compiles, runs unit tests, executes the Iris benchmark artifact, builds the Docker image, and verifies the Python extension build.
- **Testing coverage:** `src/c/test_inference.c` exercises linear layers, ReLU, numerically-stable softmax, scaler handling, and end-to-end prediction (including real Iris weights when present).

## Repository Layout
```
src/
  c/
    inference.c          # C inference engine
    inference_module.c   # CPython extension exposing predict()
    test_inference.c     # C unit tests
  python/
    train.py, train_mnist.py        # PyTorch training
    export_weights.py, export_mnist.py
    benchmark.py, benchmark_mnist.py
docs/TECHNICAL_ANALYSIS.md          # Deeper math-to-code and design notes
Dockerfile                          # Containerized runtime
Makefile                            # Build/test entrypoints
pyproject.toml, setup.py            # Python extension build config
data/                               # Weight artifacts (ignored except .bin)
```

## Getting Started (works with the new structure)
1) **Train & export weights (Python):**
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

## Benchmarks (single-thread, batch=1)
- **Iris (4→8→3):** C path avoids Python/PyTorch dispatch overhead → ~2.7M preds/sec (≈258× vs PyTorch called from Python).
- **MNIST (784→128→10):** math dominates; PyTorch with BLAS/SIMD is ~5× faster than the naive C loops.

The takeaway: C wins when overhead dominates; frameworks win when FLOPs dominate.

## Dive Deeper
For the math-to-code mapping, memory strategy, and CPython bridge details, see [`docs/TECHNICAL_ANALYSIS.md`](docs/TECHNICAL_ANALYSIS.md).

## License
MIT License — see [LICENSE](LICENSE) for details.
