import os.path as osp
import numpy as np
import tensorflow as tf
import shutil
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from data_utils import read_data
from sklearn.preprocessing import OneHotEncoder
from feature_utils import create_features
DATA_PATH =osp.join("WISDM_ar_v1.1_raw.txt")
TIME_PERIODS =80
STEP_DISTANCE =40

data_df =read_data(DATA_PATH)
df_train =data_df[data_df["user"]<=28]
df_test =data_df[data_df["user"]>28]

train_segments_df ,train_labels =create_features(df_train , TIME_PERIODS , STEP_DISTANCE)
test_segments_df ,test_labels = create_features(df_test ,TIME_PERIODS , STEP_DISTANCE)

model =tf.keras.models.Sequential([tf.keras.layers.Dense(100, input_shape=[10],
                                                          activation="relu"),
                                                          tf.keras.layers.Dense(100, activation="relu"),
                                                          tf.keras.layers.Dense(6,activation="softmax")])

train_segments_np =train_segments_df.to_numpy()
test_segments_np =test_segments_df.to_numpy()
ohe=OneHotEncoder()

train_labels_ohe =ohe.fit_transform(train_labels.reshape(-1,1)).toarray()
categories , test_labels =np.unique(test_labels , return_inverse=True)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-3))
model.fit(train_segments_np,train_labels_ohe,epochs=50,verbose = 1)
nn_preds =model.predict(test_segments_np)
predicted_classes =np.argmax(nn_preds ,axis=1)


def rep_data_gen():
    for i in range(200):
        yield [train_segments_np[i:i+1].astype(np.float32)]

np.save("X_rep.npy", train_segments_np[:1000].astype(np.float32))
model.save("mlp_har_model.h5")
print("Saved mlp_har_model.h5 and X_rep.npy")

# 1) Float32 TFLite (debug)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float = converter.convert()
open("mlp_har_float32.tflite", "wb").write(tflite_float)
print("Wrote mlp_har_float32.tflite")

# 2) INT8 TFLite
def rep_data_gen():
    n = min(200, len(train_segments_np))
    for i in range(n):
        yield [train_segments_np[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()
open("mlp_har_int8.tflite", "wb").write(tflite_int8)
print("Wrote mlp_har_int8.tflite")

conf_matrix =confusion_matrix(test_labels , predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix , display_labels=categories)
cm_display.plot()
cm_display.ax_.set_title("NeuralNetworkConfusionMatrix")
plt.show()
model.save("mlp_har_model.h5")

N=20
X = test_segments_np[:N]
y = test_labels[:N]  # 0..5 index
print(X)
print(y)
print("\n\n")
print(categories)