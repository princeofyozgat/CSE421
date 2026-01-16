import os
import struct
import gzip
import numpy as np

# =========================
# CONFIG (burayı değiştir)
# =========================
DATA_DIR = "."  # dosyaların olduğu klasör
MODE = "gz"     # "gz" veya "npz"

# MODE="gz" ise: idx-ubyte.gz dosyalarını arar
# MODE="npz" ise: tek bir npz'den çeker
NPZ_FILE = "kmnist.npz"  # MODE="npz" ise kullanılacak

# npy output isimleri (train/test ayrı)
OUT_TRAIN_IMGS = "kmnist_train_imgs.npy"
OUT_TRAIN_LBLS = "kmnist_train_labels.npy"
OUT_TEST_IMGS  = "kmnist_test_imgs.npy"
OUT_TEST_LBLS  = "kmnist_test_labels.npy"

# =========================
# Helpers
# =========================
def find_file(folder: str, candidates: list[str]) -> str:
    for name in candidates:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Bu klasörde dosya bulunamadı: {folder}\n"
        "Aşağıdakilerden biri olmalı:\n" + "\n".join(candidates)
    )

def read_idx_images_gz(path: str) -> np.ndarray:
    # magic=2051, n, rows, cols (big-endian)
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError(f"Image header okunamadı: {path}")
        magic, n, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Image magic hatalı: {magic} (expected 2051) file={path}")
        raw = f.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        expected = n * rows * cols
        if arr.size != expected:
            raise ValueError(f"Image size mismatch: got={arr.size} expected={expected} file={path}")
        return arr.reshape(n, rows, cols)

def read_idx_labels_gz(path: str) -> np.ndarray:
    # magic=2049, n (big-endian)
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Label header okunamadı: {path}")
        magic, n = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Label magic hatalı: {magic} (expected 2049) file={path}")
        raw = f.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size != n:
            raise ValueError(f"Label size mismatch: got={arr.size} expected={n} file={path}")
        return arr

def normalize_img_shape(x: np.ndarray) -> np.ndarray:
    # hedef: (N,28,28) uint8
    if x.ndim == 3:
        return x.astype(np.uint8)
    if x.ndim == 4 and x.shape[-1] == 1:
        return x[..., 0].astype(np.uint8)
    raise ValueError(f"Beklenmeyen image shape: {x.shape}")

def normalize_lbl_shape(y: np.ndarray) -> np.ndarray:
    # hedef: (N,) uint8
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.uint8)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1).astype(np.uint8)
    raise ValueError(f"Beklenmeyen label shape: {y.shape}")

# =========================
# Main logic
# =========================
def export_from_gz():
    folder = DATA_DIR

    # En yaygın KMNIST/MNIST isimleri
    train_images_candidates = [
        "kmnist-train-images-idx3-ubyte.gz",
        "train-images-idx3-ubyte.gz",
    ]
    train_labels_candidates = [
        "kmnist-train-labels-idx1-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ]
    test_images_candidates = [
        "kmnist-test-images-idx3-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "test-images-idx3-ubyte.gz",
    ]
    test_labels_candidates = [
        "kmnist-test-labels-idx1-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "test-labels-idx1-ubyte.gz",
    ]

    train_img = find_file(folder, train_images_candidates)
    train_lbl = find_file(folder, train_labels_candidates)
    test_img  = find_file(folder, test_images_candidates)
    test_lbl  = find_file(folder, test_labels_candidates)

    print("GZ dosyaları:")
    print(" train_img:", train_img)
    print(" train_lbl:", train_lbl)
    print(" test_img :", test_img)
    print(" test_lbl :", test_lbl)

    x_train = read_idx_images_gz(train_img)
    y_train = read_idx_labels_gz(train_lbl)
    x_test  = read_idx_images_gz(test_img)
    y_test  = read_idx_labels_gz(test_lbl)

    # Sanity
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Train mismatch: x_train={x_train.shape[0]} y_train={y_train.shape[0]}")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Test mismatch: x_test={x_test.shape[0]} y_test={y_test.shape[0]}")

    np.save(OUT_TRAIN_IMGS, x_train.astype(np.uint8))
    np.save(OUT_TRAIN_LBLS, y_train.astype(np.uint8))
    np.save(OUT_TEST_IMGS,  x_test.astype(np.uint8))
    np.save(OUT_TEST_LBLS,  y_test.astype(np.uint8))

    print("Saved:")
    print(" ", OUT_TRAIN_IMGS, x_train.shape, x_train.dtype)
    print(" ", OUT_TRAIN_LBLS, y_train.shape, y_train.dtype)
    print(" ", OUT_TEST_IMGS,  x_test.shape,  x_test.dtype)
    print(" ", OUT_TEST_LBLS,  y_test.shape,  y_test.dtype)

def export_from_npz():
    path = os.path.join(DATA_DIR, NPZ_FILE)
    d = np.load(path)

    # NPZ içinde çeşitli isimler olabilir:
    # (x_train,y_train,x_test,y_test) ya da (x_train,t_train,x_test,t_test) vs.
    def pick(keys):
        for k in keys:
            if k in d.files:
                return d[k]
        raise KeyError(f"NPZ içinde şu anahtarlardan hiçbiri yok: {keys}\nMevcut: {d.files}")

    x_train = pick(["x_train", "train_images", "images_train", "X_train"])
    y_train = pick(["y_train", "t_train", "train_labels", "labels_train", "Y_train"])
    x_test  = pick(["x_test", "test_images", "images_test", "X_test"])
    y_test  = pick(["y_test", "t_test", "test_labels", "labels_test", "Y_test"])

    x_train = normalize_img_shape(x_train)
    x_test  = normalize_img_shape(x_test)
    y_train = normalize_lbl_shape(y_train)
    y_test  = normalize_lbl_shape(y_test)

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Train mismatch: x_train={x_train.shape[0]} y_train={y_train.shape[0]}")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Test mismatch: x_test={x_test.shape[0]} y_test={y_test.shape[0]}")

    np.save(OUT_TRAIN_IMGS, x_train.astype(np.uint8))
    np.save(OUT_TRAIN_LBLS, y_train.astype(np.uint8))
    np.save(OUT_TEST_IMGS,  x_test.astype(np.uint8))
    np.save(OUT_TEST_LBLS,  y_test.astype(np.uint8))

    print("Saved:")
    print(" ", OUT_TRAIN_IMGS, x_train.shape, x_train.dtype)
    print(" ", OUT_TRAIN_LBLS, y_train.shape, y_train.dtype)
    print(" ", OUT_TEST_IMGS,  x_test.shape,  x_test.dtype)
    print(" ", OUT_TEST_LBLS,  y_test.shape,  y_test.dtype)

if __name__ == "__main__":
    if MODE == "gz":
        export_from_gz()
    elif MODE == "npz":
        export_from_npz()
    else:
        raise ValueError("MODE 'gz' veya 'npz' olmalı")