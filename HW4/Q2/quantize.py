import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("mlp_fsdd_model.h5")

def representative_data_gen():
    data = np.load("train_mfcc_features.npy")  # veya train_mfcc_features
    for i in range(100):
        yield [data[i].astype(np.float32)]


def representative_dataset():
    # MFCC feature'larından birkaç tanesi
    for _ in range(100):
        yield [np.random.rand(1,26).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()


with open("kws_model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("INT8 TFLite model generated")
