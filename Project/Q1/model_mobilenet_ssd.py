import tensorflow as tf

def _dw_pw_block(x, pw_filters, stride, name):
    x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = tf.keras.layers.ReLU(6.0, name=f"{name}_dw_relu")(x)

    x = tf.keras.layers.Conv2D(pw_filters, 1, padding="same", use_bias=False, name=f"{name}_pw")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = tf.keras.layers.ReLU(6.0, name=f"{name}_pw_relu")(x)
    return x

def build_mobilenet_ssd(input_shape=(96, 96, 1), num_classes=10, num_anchors=6):
    inp = tf.keras.Input(shape=input_shape, name="image")

    # input: uint8 0..255 -> float32 0..1
    x = tf.keras.layers.Rescaling(1.0/255.0, name="rescale")(inp)

    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(6.0, name="stem_relu")(x)

    x = _dw_pw_block(x, 32,  stride=1, name="mb1")   # 48x48
    x = _dw_pw_block(x, 64,  stride=2, name="mb2")   # 24x24
    x = _dw_pw_block(x, 96,  stride=2, name="mb3")   # 12x12
    x = _dw_pw_block(x, 128, stride=2, name="mb4")   # 6x6

    fm = x  # 6x6x128

    # boxes
    box = tf.keras.layers.Conv2D(num_anchors * 4, 3, padding="same", name="box_head")(fm)
    box = tf.keras.layers.Reshape((-1, 4), name="raw_boxes")(box)  # (B,216,4)

    # class probs
    cls = tf.keras.layers.Conv2D(num_anchors * num_classes, 3, padding="same", name="cls_head")(fm)
    cls = tf.keras.layers.Reshape((-1, num_classes), name="raw_scores")(cls)  # (B,216,10)
    cls = tf.keras.layers.Softmax(axis=-1, name="scores")(cls)

    return tf.keras.Model(inp, [box, cls], name="MobileNet_SSD_Lite")