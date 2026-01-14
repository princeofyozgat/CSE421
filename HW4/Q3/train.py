import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mnist import load_images, load_labels
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt

# -----------------------------
# DATA LOAD
# -----------------------------
train_img_path = "train-images.idx3-ubyte"
train_label_path = "train-labels.idx1-ubyte"
test_img_path = "t10k-images.idx3-ubyte"
test_label_path = "t10k-labels.idx1-ubyte"

train_images = load_images(train_img_path)
train_labels = load_labels(train_label_path)
test_images = load_images(test_img_path)
test_labels = load_labels(test_label_path)

# -----------------------------
# FEATURE EXTRACTION (Hu Moments)
# -----------------------------
train_huMoments = np.empty((len(train_images), 7), dtype=np.float32)
test_huMoments  = np.empty((len(test_images), 7), dtype=np.float32)

for i, img in enumerate(train_images):
    m = cv2.moments(img, True)
    train_huMoments[i] = cv2.HuMoments(m).reshape(7)

for i, img in enumerate(test_images):
    m = cv2.moments(img, True)
    test_huMoments[i] = cv2.HuMoments(m).reshape(7)

# -----------------------------
# MODEL
# -----------------------------
model = keras.models.Sequential([
    keras.layers.Dense(100, input_shape=(7,), activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(1e-4),
    metrics=["accuracy"]
)

# -----------------------------
# CALLBACKS
# -----------------------------
mc_callback = ModelCheckpoint(
    "mlp_mnist_model.h5",
    monitor="loss",
    save_best_only=True,
    verbose=1
)

es_callback = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# -----------------------------
# TRAINING
# -----------------------------
model.fit(
    train_huMoments,
    train_labels,
    epochs=1000,
    verbose=1,
    callbacks=[mc_callback, es_callback]
)

# -----------------------------
# INFERENCE (PC)
# -----------------------------
nn_preds = model.predict(test_huMoments)
predicted_classes = np.argmax(nn_preds, axis=1)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
categories = np.unique(test_labels)
conf_matrix = confusion_matrix(test_labels, predicted_classes)

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=categories
)

cm_display.plot()
plt.title("MLP MNIST Confusion Matrix")
plt.show()

# -----------------------------
# SAVE NPY FILES (PC + MCU)
# -----------------------------
np.save("train_huMoments.npy", train_huMoments)
np.save("test_huMoments.npy", test_huMoments)
np.save("test_labels.npy", test_labels)

# 5 SAMPLE FOR MCU / PC COMPARISON
np.save("test_huMoments_5.npy", test_huMoments[:5])
np.save("test_labels_5.npy", test_labels[:5])

print("✔ Training complete")
print("✔ Confusion matrix plotted")
print("✔ .npy files saved")
