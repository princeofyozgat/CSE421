# train/train_mobilenet.py
import os
import tensorflow as tf
from keras import layers, models
from mnist_loader import load_mnist_npz, preprocess_for_rgb_32, one_hot

def build_mobilenet():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None,          # MNIST için sıfırdan
        alpha=0.35             # MCU için hafifletildi
    )

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs=base.input, outputs=x)
    return model

def main():
    mnist_npz = "mnist.npz"
    if not os.path.exists(mnist_npz):
        raise FileNotFoundError("mnist.npz yok")

    # ---- Load ----
    x_train, y_train, x_test, y_test = load_mnist_npz(mnist_npz)

    x_train = preprocess_for_rgb_32(x_train)
    x_test  = preprocess_for_rgb_32(x_test)

    y_train = one_hot(y_train)
    y_test  = one_hot(y_test)

    model = build_mobilenet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=128
    )

    model.save("mobilenet_mnist_final.h5")
    print("Model saved -> mobilenet_mnist_final.h5")

if __name__ == "__main__":
    main()
