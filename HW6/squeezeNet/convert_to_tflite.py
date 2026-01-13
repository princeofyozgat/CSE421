# train/convert_to_tflite.py
import os
import numpy as np
import tensorflow as tf

from mnist_loader import load_mnist_npz, preprocess_for_rgb_32

def make_representative_dataset(mnist_npz_path: str, num_samples=200):
    x_train, y_train, x_test, y_test = load_mnist_npz(mnist_npz_path)
    x = preprocess_for_rgb_32(x_train[:num_samples])

    def gen():
        for i in range(num_samples):
            # model input shape: (1, 32, 32, 3)
            yield [x[i:i+1].astype(np.float32)]
    return gen

def main():
    
    
    h5_path = os.path.join("squeezenet_mnist_best.h5")
    if not os.path.exists(h5_path):
        h5_path = os.path.join("squeezenet_mnist_final.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError("Önce modeli eğit. .h5 bulunamadı.")

    mnist_npz = os.path.join("mnist.npz")
    if not os.path.exists(mnist_npz):
        raise FileNotFoundError("mnist.npz bulunamadı. models/ içine koy.")

    model = tf.keras.models.load_model(h5_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(mnist_npz, num_samples=200)

    # Tam int8 (input/output dahil)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    out_path = os.path.join("squeezenet_mnist_int8.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    print("TFLite written:", out_path, "bytes:", len(tflite_model))

if __name__ == "__main__":
    main()