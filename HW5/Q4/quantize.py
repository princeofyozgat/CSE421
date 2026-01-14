import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("temperature_prediction_mlp.h5")

# Load representative dataset (normalized!)
rep_data = np.load("X_train.npy")

def representative_data_gen():
    for i in range(min(300, len(rep_data))):
        yield [rep_data[i].reshape(1, -1).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("temperature_model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("INT8 model exported successfully.")
