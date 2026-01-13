# train/mnist_loader.py
import numpy as np
import tensorflow as tf

def load_mnist_npz(npz_path: str):
    """
    Beklenen içerik:
      x_train: (60000, 28, 28)
      y_train: (60000,)
      x_test:  (10000, 28, 28)
      y_test:  (10000,)
    """
    data = np.load(npz_path)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test  = data["x_test"]
    y_test  = data["y_test"]
    return x_train, y_train, x_test, y_test

def preprocess_for_rgb_32(x: np.ndarray) -> np.ndarray:
    """
    MNIST (28x28, grayscale) -> (32x32, 3 kanal RGB benzeri)
    Normalize 0..1 float32
    """
    # (N, 28, 28) -> (N, 28, 28, 1)
    x = x.astype(np.float32)[..., None]

    # normalize
    x /= 255.0

    # resize 32x32
    x = tf.image.resize(x, (32, 32), method="bilinear").numpy()

    # 1 kanal -> 3 kanal
    x = np.repeat(x, repeats=3, axis=-1)  # (N, 32, 32, 3)

    return x.astype(np.float32)

def one_hot(y: np.ndarray, num_classes=10) -> np.ndarray:
    return tf.keras.utils.to_categorical(y, num_classes=num_classes).astype(np.float32)