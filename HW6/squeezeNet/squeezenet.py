# train/squeezenet.py
import tensorflow as tf
from tensorflow import keras 
from keras import layers, models

def fire_module(x, squeeze_filters: int, expand1x1_filters: int, expand3x3_filters: int, name: str):
    """
    Fire module:
      squeeze: 1x1 conv
      expand: 1x1 conv + 3x3 conv (concat)
    """
    squeeze = layers.Conv2D(
        squeeze_filters, (1, 1),
        activation="relu", padding="valid",
        name=f"{name}_squeeze1x1"
    )(x)

    expand1x1 = layers.Conv2D(
        expand1x1_filters, (1, 1),
        activation="relu", padding="valid",
        name=f"{name}_expand1x1"
    )(squeeze)

    expand3x3 = layers.Conv2D(
        expand3x3_filters, (3, 3),
        activation="relu", padding="same",
        name=f"{name}_expand3x3"
    )(squeeze)

    x = layers.Concatenate(name=f"{name}_concat")([expand1x1, expand3x3])
    return x

def build_squeezenet(input_shape=(32, 32, 3), num_classes=10, dropout=0.5):
    """
    SqueezeNet benzeri hafif CNN.
    Çıkış: num_classes softmax
    """
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPool2D((3, 3), strides=(2, 2), padding="same", name="maxpool1")(x)

    x = fire_module(x, 16, 64, 64, name="fire2")
    x = fire_module(x, 16, 64, 64, name="fire3")
    x = layers.MaxPool2D((3, 3), strides=(2, 2), padding="same", name="maxpool3")(x)

    x = fire_module(x, 32, 128, 128, name="fire4")
    x = fire_module(x, 32, 128, 128, name="fire5")
    x = layers.MaxPool2D((3, 3), strides=(2, 2), padding="same", name="maxpool5")(x)

    x = fire_module(x, 48, 192, 192, name="fire6")
    x = fire_module(x, 48, 192, 192, name="fire7")
    x = fire_module(x, 64, 256, 256, name="fire8")
    x = fire_module(x, 64, 256, 256, name="fire9")

    x = layers.Dropout(dropout, name="dropout")(x)

    # classification head (Conv -> GAP -> Softmax)
    x = layers.Conv2D(num_classes, (1, 1), padding="valid", activation="relu", name="conv10")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Activation("softmax", name="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SqueezeNet_MNIST")
    return model