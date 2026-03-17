import torch
import numpy as np
import time

weights = np.load('mnist_weights.npy', allow_pickle=True).item()

w1 = torch.tensor(weights['w1'])
b1 = torch.tensor(weights['b1'])
w2 = torch.tensor(weights['w2'])
b2 = torch.tensor(weights['b2'])

sample = torch.zeros(784)

def predict(x):
    x = torch.relu(x @ w1.T + b1)
    x = x @ w2.T + b2
    return x.argmax().item()

# Warmup
for _ in range(100):
    predict(sample)

# Benchmark
iterations = 10000
start = time.time()
for _ in range(iterations):
    predict(sample)
end = time.time()

seconds = end - start
print(f"Iterations:       {iterations}")
print(f"Time:             {seconds:.3f} seconds")
print(f"Predictions/sec:  {iterations/seconds:.0f}")