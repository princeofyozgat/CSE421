import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from mnist_loader import load_mnist_npz, preprocess_for_rgb_32

def main():

    # ---- Model yükle ----
    model_path = os.path.join("resnetv1_mnist_best.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join("resnetv1_mnist_final.h5")

    model = tf.keras.models.load_model(model_path)
    print("Model yüklendi:", model_path)

    # ---- MNIST yükle ----
    mnist_npz = os.path.join("mnist.npz")
    x_train, y_train, x_test, y_test = load_mnist_npz(mnist_npz)

    # ---- Preprocess ----
    x_test_p = preprocess_for_rgb_32(x_test)

    # ---- Tahmin ----
    y_pred_prob = model.predict(x_test_p, batch_size=128)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ---- Görselleştirme ----
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10)
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("ResNet MNIST - Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()