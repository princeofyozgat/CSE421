import os
import numpy as np
import tensorflow as tf

from dataset_det import make_single_digit_detection_dataset
from model_mobilenet_ssd import build_mobilenet_ssd

def train_and_export(
    canvas=96,
    num_samples=25000,
    epochs=5,
    batch_size=64,
    num_anchors=6,
    lr=1e-3,
    rep_calib=200,
):
    base_dir = os.path.abspath(os.path.dirname(__file__))

    print("Generating dataset...")
    images, boxes, labels = make_single_digit_detection_dataset(
        num_samples=num_samples,
        canvas=canvas,
        seed=42
    )

    # shuffle
    n = len(images)
    perm = np.random.permutation(n)
    images = images[perm]
    boxes  = boxes[perm]
    labels = labels[perm]

    # split
    split = int(n * 0.9)
    x_tr, x_va = images[:split], images[split:]
    b_tr, b_va = boxes[:split],  boxes[split:]
    y_tr, y_va = labels[:split], labels[split:]

    print("Building MobileNet+SSD model...")
    model = build_mobilenet_ssd(
        input_shape=(canvas, canvas, 1),
        num_classes=10,
        num_anchors=num_anchors
    )

    huber = tf.keras.losses.Huber()
    sce   = tf.keras.losses.SparseCategoricalCrossentropy()
    opt   = tf.keras.optimizers.Adam(lr)

    @tf.function
    def train_step(x, b, y):
        with tf.GradientTape() as tape:
            pb, ps = model(x, training=True)  # pb:(B,N,4) ps:(B,N,K)
            N = tf.shape(pb)[1]
            b_rep = tf.repeat(b[:, None, :], repeats=N, axis=1)
            y_rep = tf.repeat(y[:, None], repeats=N, axis=1)
            loss = huber(b_rep, pb) + sce(y_rep, ps)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    tr_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, b_tr, y_tr))
        .shuffle(5000)
        .batch(batch_size)
        .prefetch(2)
    )

    print("Training...")
    for ep in range(epochs):
        losses = []
        for xb, bb, yb in tr_ds:
            xb = tf.cast(xb, tf.float32)
            bb = tf.cast(bb, tf.float32)
            yb = tf.cast(yb, tf.int32)
            losses.append(float(train_step(xb, bb, yb).numpy()))
        print(f"Epoch {ep+1}/{epochs}  loss={np.mean(losses):.4f}")

    # ===============================
    # Keras -> INT8 TFLite
    # ===============================
    print("Converting to INT8 TFLite (MobileNet+SSD)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_gen():
        # representative dataset float32 (0..255 uint8 cast -> float32)
        for i in range(min(rep_calib, len(x_tr))):
            yield [x_tr[i:i+1].astype(np.float32)]

    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = os.path.join(base_dir, "mobilenet_ssd_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print("DONE ✅")
    print("Output:", tflite_path)

if __name__ == "__main__":
    train_and_export()