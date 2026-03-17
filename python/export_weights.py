import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

weights = np.load(ROOT / 'weights.npy', allow_pickle=True).item()

with open(DATA_DIR / 'iris_weights.bin', 'wb') as f:
    arch = np.array([3, 4, 8, 3], dtype=np.int32)
    arch.tofile(f)
    weights['w1'].astype(np.float32).tofile(f)
    weights['b1'].astype(np.float32).tofile(f)
    weights['w2'].astype(np.float32).tofile(f)
    weights['b2'].astype(np.float32).tofile(f)
    weights['scaler_mean'].astype(np.float32).tofile(f)
    weights['scaler_std'].astype(np.float32).tofile(f)

print(f"Exported {DATA_DIR / 'iris_weights.bin'}")
