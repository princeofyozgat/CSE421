import tensorflow as tf
from keras import layers, models

def _conv3x3(filters, stride=1):
    return layers.Conv2D(
        filters, kernel_size=3, strides=stride, padding="same",
        use_bias=False, kernel_initializer="he_normal"
    )

def _bn_relu(x):
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def basic_block(x, filters, stride=1):
    """
    ResNet v1 BasicBlock:
      y = Conv3x3(stride) -> BN -> ReLU -> Conv3x3 -> BN
      shortcut: identity OR 1x1 conv if shape mismatch
      out = ReLU(y + shortcut)
    """
    shortcut = x

    # 1st conv
    y = _conv3x3(filters, stride=stride)(x)
    y = _bn_relu(y)

    # 2nd conv
    y = _conv3x3(filters, stride=1)(y)
    y = layers.BatchNormalization()(y)

    # projection shortcut if needed
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(
            filters, kernel_size=1, strides=stride, padding="same",
            use_bias=False, kernel_initializer="he_normal"
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    out = layers.Add()([y, shortcut])
    out = layers.ReLU()(out)
    return out

def build_resnet_v1_mnist(input_shape=(32, 32, 3), num_classes=10,
                          base_filters=16, blocks_per_stage=(2, 2, 2),
                          include_softmax=True):
    """
    Küçük ResNet v1:
      Stem: Conv3x3(base_filters) + BN+ReLU
      Stage1: filters=base_filters, stride=1
      Stage2: filters=base_filters*2, stride=2 (downsample)
      Stage3: filters=base_filters*4, stride=2 (downsample)
      Head : GAP -> Dense(num_classes) -> (Softmax)

    blocks_per_stage=(2,2,2) => toplam BasicBlock sayısı 6 (MCU-friendly)
    """
    inp = layers.Input(shape=input_shape)

    x = _conv3x3(base_filters, stride=1)(inp)
    x = _bn_relu(x)

    # Stage 1
    for _ in range(blocks_per_stage[0]):
        x = basic_block(x, base_filters, stride=1)

    # Stage 2
    x = basic_block(x, base_filters * 2, stride=2)
    for _ in range(blocks_per_stage[1] - 1):
        x = basic_block(x, base_filters * 2, stride=1)

    # Stage 3
    x = basic_block(x, base_filters * 4, stride=2)
    for _ in range(blocks_per_stage[2] - 1):
        x = basic_block(x, base_filters * 4, stride=1)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    logits = layers.Dense(num_classes, kernel_initializer="he_normal")(x)

    if include_softmax:
        out = layers.Softmax()(logits)
    else:
        out = logits

    model = models.Model(inputs=inp, outputs=out, name="resnet_v1_mnist")
    return model