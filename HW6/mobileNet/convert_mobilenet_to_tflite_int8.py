import os
import numpy as np
import tensorflow as tf
from mnist_loader import load_mnist_npz, preprocess_for_rgb_32

def representative_dataset_gen(mnist_npz_path, num_samples=200):
    x_train, y_train, x_test, y_test = load_mnist_npz(mnist_npz_path)
    x = preprocess_for_rgb_32(x_train[:num_samples])

    def gen():
        for i in range(num_samples):
            yield [x[i:i+1].astype(np.float32)]
    return gen

def main():

    mnist_npz = os.path.join("mnist.npz")
    if not os.path.exists(mnist_npz):
        raise FileNotFoundError("mnist.npz bulunamadı.")

    h5_path = os.path.join("mobilenet_mnist_best.h5")
    if not os.path.exists(h5_path):
        h5_path = os.path.join("mobilenet_mnist_final.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError("MobileNet .h5 yok. Önce train_mobilenet.py çalıştır.") 
    model = tf.keras.models.load_model(h5_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(mnist_npz, 200)

    # INT8
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS  # fallback
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    out_path = os.path.join("mobilenet_mnist_int8.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    print("Wrote:", out_path, "bytes:", len(tflite_model))

if __name__ == "__main__":
    main()