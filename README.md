# Tiny ML Runtime in C

A generic neural network inference engine implemented in pure C-
no frameworks, no dependencies.

Trains in Python (PyTorch), exports weights, runs inference in C.
Supports any architecture - tested on Iris and MNIST.

## Benchmark Results

### Iris (4 → 8 → 3) - tiny network
| Runtime        | Predictions/sec | Time (1M iterations) |
|----------------|-----------------|----------------------|
| Pure C         | 2,732,240       | 0.366 seconds        |
| PyTorch Python | 10,596          | 94.374 seconds       |
| **Speedup**    | **258x faster** |                      |

### MNIST (784 → 128 → 10) - larger network
| Runtime        | Predictions/sec | Time (100K iterations) |
|----------------|-----------------|------------------------|
| Pure C         | 4,244           | 23.563 seconds         |
| PyTorch Python | 22,502          | 0.444 seconds          |
| **Speedup**    | **PyTorch 5x faster** |                   |

### Why the difference?

On **Iris**, the network is tiny - PyTorch's Python overhead dominates.
C wins because it has zero overhead.

On **MNIST**, the matrices are large (W1 = 128×784 = 100K multiplications
per prediction). PyTorch uses optimized BLAS libraries with SIMD instructions.
Our C code uses naive loops — no vectorization, no cache optimization.

The crossover point reveals exactly where framework overhead ends
and raw computation begins. That's the interesting result.

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
train.py              # Train Iris model in PyTorch
train_mnist.py        # Train MNIST model in PyTorch
export_weights.py     # Export Iris weights to binary
export_mnist.py       # Export MNIST weights to binary
benchmark.py          # PyTorch benchmark (Iris)
benchmark_mnist.py    # PyTorch benchmark (MNIST)
inference.c           # C inference engine (the interesting part)
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
gcc inference.c -o inference -lm
./inference iris
./inference mnist
```

## Why I built this

I wanted to understand what PyTorch actually does under the hood,
so I implemented neural network inference from scratch in C -
matrix multiplication, ReLU, Softmax, the full forward pass.

The benchmark results tell an interesting story: C is 258x faster
on tiny networks (overhead-bound) but slower on large ones
(compute-bound, where PyTorch's BLAS optimization wins).
That crossover is the real insight.
