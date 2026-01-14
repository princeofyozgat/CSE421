import numpy as np
import tensorflow as tf

X_test_q = np.load("test_features_int8.npy")
y_true = np.load("test_labels_5.npy")

interpreter = tf.lite.Interpreter(
    model_path="temperature_model_int8.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_scale, in_zero = input_details[0]["quantization"]
out_scale, out_zero = output_details[0]["quantization"]

print("Input scale:", in_scale, "zero:", in_zero)
print("Output scale:", out_scale, "zero:", out_zero)
print()

for i in range(len(X_test_q)):

    x_q = X_test_q[i:i+1].astype(np.int8)

    interpreter.set_tensor(
        input_details[0]["index"], x_q
    )
    interpreter.invoke()

    y_q = interpreter.get_tensor(
        output_details[0]["index"]
    )[0][0]

    # ⚠️ ÖNCE int32
    y_q_i32 = np.int32(y_q)
    y_pred = (y_q_i32 - out_zero) * out_scale

    print(
        f"Sample {i} | True: {y_true[i]:.3f} | Predicted: {y_pred:.3f}"
    )
