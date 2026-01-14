import numpy as np
import tensorflow as tf

# Load test data
test_features = np.load("test_features_5.npy")
test_labels = np.load("test_labels_5.npy")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="kws_model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scale, zero_point = input_details[0]["quantization"]

print("\nPC Inference Results:")
print("---------------------")

for i in range(5):
    x = test_features[i]

    # FLOAT → INT8 (AYNI MCU'DAKİ GİBİ)
    x_q = (x / scale + zero_point).astype(np.int8)

    interpreter.set_tensor(
        input_details[0]["index"],
        x_q.reshape(1, 26)
    )

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    pred = np.argmax(output)

    print(f"Sample {i} | True = {test_labels[i]} | Pred = {pred}")
