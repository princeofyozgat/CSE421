# train/train_squeezenet_mnist.py
import os
import numpy as np
import tensorflow as tf

from mnist_loader import load_mnist_npz, preprocess_for_rgb_32, one_hot
from squeezenet import build_squeezenet

def main():
    # paths
    mnist_npz = os.path.join("mnist.npz")  # buraya mnist.npz koy
    if not os.path.exists(mnist_npz):
        raise FileNotFoundError(f"MNIST NPZ bulunamadı: {mnist_npz}")

    # load
    x_train, y_train, x_test, y_test = load_mnist_npz(mnist_npz)

    # preprocess
    x_train_p = preprocess_for_rgb_32(x_train)
    x_test_p  = preprocess_for_rgb_32(x_test)

    y_train_oh = one_hot(y_train, 10)
    y_test_oh  = one_hot(y_test, 10)

    # model
    model = build_squeezenet(input_shape=(32, 32, 3), num_classes=10, dropout=0.5)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # callbacks
    ckpt_path = os.path.join("squeezenet_mnist_best.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, save_weights_only=False, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1
        ),
    ]

    # train
    history = model.fit(
        x_train_p, y_train_oh,
        validation_split=0.1,
        epochs=20,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    # evaluate
    test_loss, test_acc = model.evaluate(x_test_p, y_test_oh, verbose=0)
    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")



if __name__ == "__main__":
    # GPU/CPU fark etmez
    main()