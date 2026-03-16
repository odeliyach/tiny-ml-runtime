import torch
import numpy as np
import time

weights = np.load('weights.npy', allow_pickle=True).item()

w1 = torch.tensor(weights['w1'])
b1 = torch.tensor(weights['b1'])
w2 = torch.tensor(weights['w2'])
b2 = torch.tensor(weights['b2'])
mean = torch.tensor(weights['scaler_mean'])
std = torch.tensor(weights['scaler_std'])

flower = torch.tensor([5.1, 3.5, 1.4, 0.2])

def predict(x):
    x = (x - mean) / std
    x = torch.relu(x @ w1.T + b1)
    x = x @ w2.T + b2
    return x.argmax().item()

# Warmup
for _ in range(1000):
    predict(flower)

# Benchmark
iterations = 1000000
start = time.time()
for _ in range(iterations):
    predict(flower)
end = time.time()

seconds = end - start
print(f"Iterations:       {iterations}")
print(f"Time:             {seconds:.3f} seconds")
print(f"Predictions/sec:  {iterations/seconds:.0f}")