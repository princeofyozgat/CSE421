import os
import scipy.signal as sig
from mfcc_func import create_mfcc_features
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import sklearn2c
import numpy as np
import joblib

RECORDINGS_DIR = "recordings"
FFTSize = 1024
sample_rate = 8000
numOfMelFilters = 20
numOfDctOutputs = 13


window = sig.get_window("hamming", FFTSize)


recordings_list = [
    (RECORDINGS_DIR, rec_name)
    for rec_name in os.listdir(RECORDINGS_DIR)
    if rec_name.endswith(".wav")
]

print(f"Toplam kayıt sayısı: {len(recordings_list)}")


test_list = {rec for rec in recordings_list if "yweweler" in rec[1]}
train_list = set(recordings_list) - test_list

print(f"Train kayıt sayısı: {len(train_list)}")
print(f"Test kayıt sayısı : {len(test_list)}")

train_features, train_labels = create_mfcc_features(
    list(train_list),
    FFTSize,
    sample_rate,
    numOfMelFilters,
    numOfDctOutputs,
    window,
)

test_features, test_labels = create_mfcc_features(
    list(test_list),
    FFTSize,
    sample_rate,
    numOfMelFilters,
    numOfDctOutputs,
    window,
)

print("Train feature shape:", train_features.shape)
print("Test feature shape :", test_features.shape)

knn = sklearn2c.KNNClassifier(n_neighbors=3)
knn.train(train_features, train_labels)

knn_raw = np.array(knn.predict(test_features))

print("knn_raw shape:", knn_raw.shape)

if knn_raw.ndim == 2 and knn_raw.shape[1] > 1:
    knn_preds = np.argmax(knn_raw, axis=1)
else:
    knn_preds = knn_raw.reshape(-1)

test_labels_int = np.array(test_labels, dtype=int).reshape(-1)

print("test_labels_int shape:", test_labels_int.shape)
print("knn_preds shape      :", knn_preds.shape)


conf_matrix = confusion_matrix(
    test_labels_int,
    knn_preds,
    labels=list(range(10))    
)

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=list(range(10))
)

cm_display.plot()
plt.title("KNN Classifier – Keyword Spotting (FSDD)")
plt.show()


knn.export("knn_mfcc_config")
print("Model C'ye export edildi: knn_mfcc_config.c / .h")

joblib.dump(knn, "knn_model.joblib")