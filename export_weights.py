import numpy as np

weights = np.load('weights.npy', allow_pickle=True).item()

with open('iris_weights.bin', 'wb') as f:
    arch = np.array([3, 4, 8, 3], dtype=np.int32)
    arch.tofile(f)
    weights['w1'].astype(np.float32).tofile(f)
    weights['b1'].astype(np.float32).tofile(f)
    weights['w2'].astype(np.float32).tofile(f)
    weights['b2'].astype(np.float32).tofile(f)
    weights['scaler_mean'].astype(np.float32).tofile(f)
    weights['scaler_std'].astype(np.float32).tofile(f)

print("Exported iris_weights.bin")