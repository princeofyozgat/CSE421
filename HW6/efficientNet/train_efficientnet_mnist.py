import os
import numpy as np
import tensorflow as tf

from mnist_loader import load_mnist_npz, preprocess_for_rgb_32, one_hot

# ---------------- MBConv Block ----------------
def mbconv(x, out_ch, expand=2, k=3, s=1, act="swish", name="mb"):
    in_ch = x.shape[-1]
    mid_ch = int(in_ch * expand)

    Act = tf.keras.layers.Activation
    Conv = tf.keras.layers.Conv2D
    DW   = tf.keras.layers.DepthwiseConv2D
    BN   = tf.keras.layers.BatchNormalization

    shortcut = x

    # Expand
    if expand != 1:
        x = Conv(mid_ch, 1, padding="same", use_bias=False, name=f"{name}_exp")(x)
        x = BN(name=f"{name}_exp_bn")(x)
        x = Act(act, name=f"{name}_exp_act")(x)

    # Depthwise
    x = DW(k, strides=s, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = BN(name=f"{name}_dw_bn")(x)
    x = Act(act, name=f"{name}_dw_act")(x)

    # Project
    x = Conv(out_ch, 1, padding="same", use_bias=False, name=f"{name}_prj")(x)
    x = BN(name=f"{name}_prj_bn")(x)

    # Residual
    if s == 1 and in_ch == out_ch:
        x = tf.keras.layers.Add(name=f"{name}_add")([shortcut, x])

    return x

# -------------- Tiny EfficientNet --------------
def build_tiny_efficientnet_mnist(input_shape=(32,32,3), num_classes=10):
    inp = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(8, 3, strides=1, padding="same", use_bias=False, name="stem")(inp)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.Activation("swish", name="stem_act")(x)

    # Stage 1 (32x32)
    x = mbconv(x, out_ch=8,  expand=1, k=3, s=1, name="b1_0")
    x = mbconv(x, out_ch=8,  expand=2, k=3, s=1, name="b1_1")

    # Stage 2 (16x16)
    x = mbconv(x, out_ch=12, expand=2, k=3, s=2, name="b2_0")
    x = mbconv(x, out_ch=12, expand=2, k=3, s=1, name="b2_1")

    # Stage 3 (8x8)
    x = mbconv(x, out_ch=16, expand=2, k=3, s=2, name="b3_0")
    x = mbconv(x, out_ch=16, expand=2, k=3, s=1, name="b3_1")

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(32, activation="swish", name="fc1")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    return tf.keras.Model(inp, out, name="tiny_efficientnet_mnist")

def main():
    # ---- data ----
    mnist_npz = os.path.join("mnist.npz")
    x_train, y_train, x_test, y_test = load_mnist_npz(mnist_npz)

    x_train = x_train[:20000]   # HIZ için (istersen kaldır)
    y_train = y_train[:20000]

    x_train_p = preprocess_for_rgb_32(x_train)
    x_test_p  = preprocess_for_rgb_32(x_test)

    y_train_oh = one_hot(y_train, 10)
    y_test_oh  = one_hot(y_test, 10)

    model = build_tiny_efficientnet_mnist((32,32,3), 10)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    ckpt = "tiny_efficientnet_mnist_best.h5"
    final = "tiny_efficientnet_mnist_final.h5"

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1)
    ]

    model.fit(
        x_train_p, y_train_oh,
        validation_split=0.1,
        epochs=10,
        batch_size=256,
        callbacks=cbs,
        verbose=1
    )

    loss, acc = model.evaluate(x_test_p, y_test_oh, verbose=0)
    print(f"[TEST] loss={loss:.4f} acc={acc:.4f}")

    model.save(final)
    print("Saved:", ckpt, final)

if __name__ == "__main__":
    main()