import os
import numpy as np
import tensorflow as tf

from mnist_loader import load_mnist_npz, preprocess_for_rgb_32, one_hot
from resnet_mnist_v1 import build_resnet_v1_mnist

def main():
   
   

    mnist_npz = os.path.join("mnist.npz")
    x_train, y_train, x_test, y_test = load_mnist_npz(mnist_npz)

    x_train_p = preprocess_for_rgb_32(x_train)  # (N,32,32,3) float32 0..1
    x_test_p  = preprocess_for_rgb_32(x_test)

    y_train_oh = one_hot(y_train, 10)
    y_test_oh  = one_hot(y_test, 10)

    # Küçük ResNet: MCU için hafif
    model = build_resnet_v1_mnist(
        input_shape=(32,32,3),
        num_classes=10,
        base_filters=16,
        blocks_per_stage=(2,2,2),   # istersen (3,3,3) yaparsın ama ağırlaşır
        include_softmax=True
    )
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    ckpt_path = os.path.join("resnetv1_mnist_best.h5")
    final_path = os.path.join("resnetv1_mnist_final.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1
        ),
    ]

    model.fit(
        x_train_p, y_train_oh,
        validation_split=0.1,
        epochs=20,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test_p, y_test_oh, verbose=0)
    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")

    model.save(final_path)
    print("Saved best :", ckpt_path)
    print("Saved final:", final_path)

if __name__ == "__main__":
    main()