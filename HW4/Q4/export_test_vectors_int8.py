import numpy as np
import tensorflow as tf

X_test = np.load("test_features_5.npy")

interpreter = tf.lite.Interpreter(
    model_path="temperature_model_int8.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
scale, zero = input_details[0]["quantization"]

X_test_q = (X_test / scale + zero).round().astype(np.int8)

np.save("test_features_int8.npy", X_test_q)

print("INT8 test vectors exported.")
