import struct
from pathlib import Path

import pytest

import tinymlinference


def write_weights(path: Path, layer_sizes, weights_and_biases, scaler=None):
    """
    Create a minimal binary weights file.
    layer_sizes: list[int] including input and output sizes
    weights_and_biases: iterable of (weights, biases) per layer
    scaler: tuple(mean, std) or None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("i", len(layer_sizes)))
        f.write(struct.pack(f"{len(layer_sizes)}i", *layer_sizes))
        for W, b in weights_and_biases:
            f.write(struct.pack(f"{len(W)}f", *W))
            f.write(struct.pack(f"{len(b)}f", *b))
        if scaler is not None:
            mean, std = scaler
            f.write(struct.pack(f"{len(mean)}f", *mean))
            f.write(struct.pack(f"{len(std)}f", *std))


@pytest.fixture()
def identity_weights(tmp_path):
    path = tmp_path / "ident.bin"
    layer_sizes = [2, 2]
    weights_and_biases = [
        ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0]),
    ]
    write_weights(path, layer_sizes, weights_and_biases)
    return path


def test_predict_identity(identity_weights):
    class_idx, probs = tinymlinference.predict(str(identity_weights), [2.0, 0.0])
    assert class_idx == 0
    assert pytest.approx(sum(probs)) == 1.0
    assert probs[0] > probs[1]


def test_predict_with_scaler(tmp_path):
    path = tmp_path / "scaled.bin"
    layer_sizes = [2, 2]
    weights_and_biases = [
        ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0]),
    ]
    scaler = ([5.0, 5.0], [1.0, 1.0])
    write_weights(path, layer_sizes, weights_and_biases, scaler=scaler)

    class_idx, probs = tinymlinference.predict(str(path), [6.0, 4.0])
    assert class_idx == 0
    assert pytest.approx(sum(probs)) == 1.0


def test_input_length_mismatch(identity_weights):
    with pytest.raises(ValueError):
        tinymlinference.predict(str(identity_weights), [1.0, 2.0, 3.0])
