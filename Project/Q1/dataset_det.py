import numpy as np
import tensorflow as tf

def make_single_digit_detection_dataset(num_samples=25000, canvas=96, seed=42):
    rng = np.random.default_rng(seed)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype(np.uint8)
    y_train = y_train.astype(np.int32)

    images = np.zeros((num_samples, canvas, canvas, 1), dtype=np.uint8)
    boxes  = np.zeros((num_samples, 4), dtype=np.float32)  # ymin,xmin,ymax,xmax in 0..1
    labels = np.zeros((num_samples,), dtype=np.int32)

    for i in range(num_samples):
        cls = int(rng.integers(0, 10))
        idxs = np.where(y_train == cls)[0]
        idx = int(rng.choice(idxs))
        digit = x_train[idx]  # 28x28

        target = int(rng.integers(24, 40))  # 24..39
        digit_tf = tf.image.resize(digit[..., None], (target, target), method="bilinear")
        digit_tf = tf.clip_by_value(digit_tf, 0.0, 255.0)
        digit_rs = tf.cast(digit_tf, tf.uint8).numpy()

        y0 = int(rng.integers(0, canvas - target))
        x0 = int(rng.integers(0, canvas - target))

        img = np.zeros((canvas, canvas, 1), dtype=np.uint8)
        img[y0:y0+target, x0:x0+target, :] = np.maximum(
            img[y0:y0+target, x0:x0+target, :],
            digit_rs
        )

        ymin = y0 / canvas
        xmin = x0 / canvas
        ymax = (y0 + target) / canvas
        xmax = (x0 + target) / canvas

        images[i] = img
        boxes[i]  = (ymin, xmin, ymax, xmax)
        labels[i] = cls

    return images, boxes, labels