import struct
import numpy as np
import os

def read_idx_images(path):
    """
    IDX3-UBYTE format:
      magic number: 2051 (0x00000803)
      num_images: 4 bytes
      num_rows:   4 bytes
      num_cols:   4 bytes
      then: num_images * num_rows * num_cols bytes (uint8)
    """
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} magic yanlış! Beklenen 2051, gelen {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        expected = num * rows * cols
        if data.size != expected:
            raise ValueError(f"{path} boyut uyuşmuyor! Beklenen {expected}, gelen {data.size}")
        return data.reshape(num, rows, cols)

def read_idx_labels(path):
    """
    IDX1-UBYTE format:
      magic number: 2049 (0x00000801)
      num_labels: 4 bytes
      then: num_labels bytes (uint8)
    """
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} magic yanlış! Beklenen 2049, gelen {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        if data.size != num:
            raise ValueError(f"{path} label sayısı uyuşmuyor! Beklenen {num}, gelen {data.size}")
        return data

def main():
    # Bu scripti hangi klasörde çalıştırırsan çalıştır, dosyaları buraya göre ayarlayacaksın.
    # Aynı klasördeysen isimler yeter.
    train_images_path = "train-images.idx3-ubyte"
    train_labels_path = "train-labels.idx1-ubyte"
    test_images_path  = "t10k-images.idx3-ubyte"
    test_labels_path  = "t10k-labels.idx1-ubyte"

    # Okuma
    x_train = read_idx_images(train_images_path)
    y_train = read_idx_labels(train_labels_path)
    x_test  = read_idx_images(test_images_path)
    y_test  = read_idx_labels(test_labels_path)

    # Basit kontroller
    print("x_train:", x_train.shape, x_train.dtype)  # (60000, 28, 28) uint8
    print("y_train:", y_train.shape, y_train.dtype)  # (60000,) uint8
    print("x_test :", x_test.shape, x_test.dtype)    # (10000, 28, 28) uint8
    print("y_test :", y_test.shape, y_test.dtype)    # (10000,) uint8

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Train image/label sayısı eşleşmiyor!")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("Test image/label sayısı eşleşmiyor!")

    # NPZ kaydet
    out_path = "mnist.npz"
    np.savez_compressed(out_path,
                        x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)
    print("OK -> yazıldı:", os.path.abspath(out_path))

if __name__ == "__main__":
    main()