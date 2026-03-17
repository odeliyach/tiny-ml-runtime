import os
import numpy as np

os.makedirs("data", exist_ok=True)
weights = np.load('mnist_weights.npy', allow_pickle=True).item()
out_path = os.path.join("data", "mnist_weights.bin")

with open(out_path, 'wb') as f:
    # Write architecture info first
    arch = np.array([3, 784, 128, 10], dtype=np.int32)  # 3 layers, then sizes
    arch.tofile(f)
    # Write weights
    weights['w1'].astype(np.float32).tofile(f)
    weights['b1'].astype(np.float32).tofile(f)
    weights['w2'].astype(np.float32).tofile(f)
    weights['b2'].astype(np.float32).tofile(f)

print(f"Exported {out_path}")
print(f"w1: {weights['w1'].shape}")
print(f"w2: {weights['w2'].shape}")
