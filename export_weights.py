import numpy as np

weights = np.load('weights.npy', allow_pickle=True).item()

with open('weights.bin', 'wb') as f:
    weights['w1'].astype(np.float32).tofile(f)
    weights['b1'].astype(np.float32).tofile(f)
    weights['w2'].astype(np.float32).tofile(f)
    weights['b2'].astype(np.float32).tofile(f)
    weights['scaler_mean'].astype(np.float32).tofile(f)
    weights['scaler_std'].astype(np.float32).tofile(f)

print("Exported weights.bin successfully")
print(f"w1 shape: {weights['w1'].shape}")
print(f"w2 shape: {weights['w2'].shape}")
