import os
import numpy as np

os.makedirs("data", exist_ok=True)
weights = np.load('weights.npy', allow_pickle=True).item()
out_path = os.path.join("data", "iris_weights.bin")

with open(out_path, 'wb') as f:
    arch = np.array([3, 4, 8, 3], dtype=np.int32)
    arch.tofile(f)
    weights['w1'].astype(np.float32).tofile(f)
    weights['b1'].astype(np.float32).tofile(f)
    weights['w2'].astype(np.float32).tofile(f)
    weights['b2'].astype(np.float32).tofile(f)
    weights['scaler_mean'].astype(np.float32).tofile(f)
    weights['scaler_std'].astype(np.float32).tofile(f)

print(f"Exported {out_path}")
