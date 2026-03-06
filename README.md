# Tiny ML Runtime in C

A neural network inference engine implemented in pure C -
no frameworks, no dependencies.

Trains in Python (PyTorch), exports weights, runs inference in C.

## Benchmark Results

| Runtime        | Predictions/sec | Time (1M iterations) |
|----------------|-----------------|----------------------|
| Pure C         | 3,333,333       | 0.300 seconds        |
| PyTorch Python | 10,596          | 94.374 seconds       |
| **Speedup**    | **314x faster** |                      |

Tested on CPU, 1,000,000 iterations, Iris dataset (4 inputs → 3 outputs).

## Architecture
```
Input layer (4 neurons)
        ↓
Hidden layer (8 neurons) + ReLU
        ↓
Output layer (3 neurons) + Softmax
```

## What's implemented from scratch in C

- Matrix multiplication
- ReLU activation function
- Softmax activation function
- Full forward pass
- Binary weight loader
- Benchmarking

## How it works

1. Train a small neural network on the Iris dataset in PyTorch (~95 lines of Python)
2. Export the learned weights to a binary file (`weights.bin`)
3. Load the weights in C and run predictions — pure math, no libraries

## Sample Output
```
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
Iterations:        1,000,000
Time:              0.300 seconds
Predictions/sec:   3,333,333
```

## Project Structure
```
train.py          # Train the model in PyTorch, save weights
export_weights.py # Export weights to binary format
benchmark.py      # Python/PyTorch benchmark
inference.c       # C inference engine (the interesting part)
weights.bin       # Exported weights (generated)
```

## Run it yourself

**Train and export:**
```bash
py train.py
py export_weights.py
```

**Compile and run:**
```bash
gcc inference.c -o inference -lm
./inference
```

**Python benchmark:**
```bash
py benchmark.py
```

## Why I built this

I wanted to understand what PyTorch actually does under the hood,
so I implemented neural network inference from scratch in C —
matrix multiplication, ReLU, Softmax, the full forward pass.
Then I benchmarked it against PyTorch: the C runtime runs
314x faster on CPU.