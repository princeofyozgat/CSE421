import numpy as np
import tensorflow as tf
from kmnist_io import load_kmnist

NUM_CLASSES = 10

def preprocess_np_to_float32_28(x_u8: np.ndarray) -> np.ndarray:
    # (N,28,28,1) uint8 -> float32 [0,1]
    return (x_u8.astype(np.float32) / 255.0)

def make_ds(x_u8, y_u8, batch=128, train=True):
    x = preprocess_np_to_float32_28(x_u8)
    y = y_u8.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if train:
        ds = ds.shuffle(20000)
        # çok hafif noise (istersen kapat)
        ds = ds.map(lambda im, lab: (tf.clip_by_value(im + tf.random.normal(tf.shape(im), stddev=0.02), 0.0, 1.0), lab),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def build_lenet_lite():
    inp = tf.keras.layers.Input(shape=(28, 28, 1), name="input")

    # LeNet-Lite: daha az kanal + daha küçük FC
    x = tf.keras.layers.Conv2D(4, kernel_size=5, padding="valid", activation="relu", name="conv1")(inp)
    x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name="avgpool1")(x)

    x = tf.keras.layers.Conv2D(8, kernel_size=5, padding="valid", activation="relu", name="conv2")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name="avgpool2")(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="fc2")(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    return tf.keras.Model(inp, out, name="LeNet_KMNIST")

def main():
    (x_train, y_train), (x_test, y_test) = load_kmnist()

    train_ds = make_ds(x_train, y_train, batch=128, train=True)
    val_ds   = make_ds(x_test,  y_test,  batch=128, train=False)

    model = build_lenet_lite()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            "lenet_kmnist_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=cbs)
    model.save("lenet_kmnist_final.h5")
    print("Saved: lenet_kmnist_best.h5 + lenet_kmnist_final.h5")
    model.summary()

if __name__ == "__main__":
    main()