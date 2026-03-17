import numpy as np

weights = np.load('mnist_weights.npy', allow_pickle=True).item()

with open('mnist_weights.bin', 'wb') as f:
    # Write architecture info first
    arch = np.array([3, 784, 128, 10], dtype=np.int32)  # 3 layers, then sizes
    arch.tofile(f)
    # Write weights
    weights['w1'].astype(np.float32).tofile(f)
    weights['b1'].astype(np.float32).tofile(f)
    weights['w2'].astype(np.float32).tofile(f)
    weights['b2'].astype(np.float32).tofile(f)

print("Exported mnist_weights.bin")
print(f"w1: {weights['w1'].shape}")
print(f"w2: {weights['w2'].shape}")