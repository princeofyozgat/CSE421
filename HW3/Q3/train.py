import os
import scipy.signal as sig
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from mfcc_func import create_mfcc_features
RECORDINGS_DIR = "recordings"
recordings_list = [(RECORDINGS_DIR , recording_path) for 
                   recording_path in os.listdir(RECORDINGS_DIR)]

FFTSize = 1024

sample_rate = 8000
numOfMelFilters = 20
numOfDctOutputs = 13

window = sig.get_window("hamming",FFTSize)

test_list = {record for record in recordings_list if "yweweler" in record[1]}

train_list = set(recordings_list)-test_list

train_mfcc_features ,train_labels = create_mfcc_features(
    train_list ,FFTSize, sample_rate ,numOfMelFilters,
    numOfDctOutputs ,window)

test_mfcc_features ,test_labels = create_mfcc_features(test_list , FFTSize , 
                                sample_rate ,numOfMelFilters,numOfDctOutputs,
                                window)

model = tf.keras.models.Sequential([
tf.keras.layers.Dense(1,input_shape=[numOfDctOutputs*2], activation ="sigmoid")])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = [tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.FalseNegatives()])

train_labels[train_labels != 0] = 1
test_labels[test_labels != 0] = 1

model.fit(train_mfcc_features , train_labels ,epochs=50,verbose=1,class_weight= {0:10.,1:1.})

perceptron_preds = model.predict(test_mfcc_features)

conf_matrix = confusion_matrix(test_labels , perceptron_preds > 0.5)

cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix)
cm_display.plot()
cm_display.ax_.set_title("SingleNeuronClassifierConfusionMatrix")
plt.show()

model.save("fsdd_perceptron.h5")

weights, bias = model.layers[0].get_weights()

print("Weights shape:", weights.shape)
print("Bias:", bias)

weights_flattened = weights.flatten()
print("Weights flattened:", weights_flattened)

preds = model.predict(test_mfcc_features).flatten()

zero_idx = np.where(test_labels == 0)[0][0]
one_idx  = np.where(test_labels == 1)[0][0]

zero_feature = test_mfcc_features[zero_idx]
one_feature  = test_mfcc_features[one_idx]

def print_c_array(name, arr):
    print(f"\nstatic const float {name}[26] = {{")
    for i, v in enumerate(arr):
        end = ",\n" if i % 5 == 4 else ", "
        print(f"    {v:.6f}f", end=end)
    print("\n};")

print_c_array("test_zero_feature", zero_feature)
print_c_array("test_one_feature", one_feature)