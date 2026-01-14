import numpy as np
import tensorflow as tf

# =========================
# Load trained model
# =========================
model = tf.keras.models.load_model("mlp_mnist_model.h5")

# =========================
# Representative dataset
# =========================
train_features = np.load("train_huMoments.npy")

def representative_data_gen():
    for i in range(200):
        yield [train_features[i].astype(np.float32)]

# =========================
# Converter
# =========================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# =========================
# Save model
# =========================
with open("mnist_hu_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("INT8 TFLite model generated")
