import tensorflow as tf
import numpy as np
from kmnist_io import load_kmnist

def representative_dataset():
    (x_train, _), _ = load_kmnist()
    x = x_train.astype(np.float32) / 255.0  # (N,28,28,1)
    for i in range(400):
        yield [x[i:i+1]]

def main():
    model_path = "lenet_kmnist_best.h5"
    if not tf.io.gfile.exists(model_path):
        model_path = "lenet_kmnist_final.h5"

    model = tf.keras.models.load_model(model_path)
    print("Loaded:", model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open("lenet_kmnist_int8.tflite", "wb") as f:
        f.write(tflite_model)

    print("Saved: lenet_kmnist_int8.tflite")

if __name__ == "__main__":
    main()