import gzip
import struct
import numpy as np
from pathlib import Path

def _open_maybe_gz(path: str):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rb")
    return open(p, "rb")

def load_idx_images(path: str) -> np.ndarray:
    with _open_maybe_gz(path) as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic for images: {magic} in {path}")
        data = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)
        return data.reshape(num, rows, cols)

def load_idx_labels(path: str) -> np.ndarray:
    with _open_maybe_gz(path) as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic for labels: {magic} in {path}")
        data = np.frombuffer(f.read(num), dtype=np.uint8)
        return data

def load_kmnist(
    train_images="train-images-idx3-ubyte.gz",
    train_labels="train-labels-idx1-ubyte.gz",
    test_images="t10k-images-idx3-ubyte.gz",
    test_labels="t10k-labels-idx1-ubyte.gz",
):
    x_train = load_idx_images(train_images)
    y_train = load_idx_labels(train_labels)
    x_test  = load_idx_images(test_images)
    y_test  = load_idx_labels(test_labels)

    # (N,28,28) -> (N,28,28,1)
    x_train = x_train[..., None]
    x_test  = x_test[..., None]
    return (x_train, y_train), (x_test, y_test)