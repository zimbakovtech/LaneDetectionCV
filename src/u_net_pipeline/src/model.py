from typing import Tuple
import tensorflow as tf
from keras import layers, models


def conv_block(x, filters, kernel_size=(3,3), padding='same', activation='relu', use_bn=True, dropout=0.0):
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def up_block(x, skip, filters, kernel_size=(3,3), padding='same', activation='relu', use_bn=True, dropout=0.0):
    x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters, kernel_size, padding, activation, use_bn, dropout)
    return x


def build_unet(input_shape: Tuple[int, int, int] = (256, 512, 3), base_filters: int = 32, depth: int = 4,
               use_bn: bool = True, dropout: float = 0.0) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    # Encoder
    skips = []
    x = inputs
    filters = base_filters
    for d in range(depth):
        x = conv_block(x, filters, use_bn=use_bn, dropout=dropout)
        skips.append(x)
        x = layers.MaxPooling2D((2,2))(x)
        filters *= 2

    # Bottleneck
    x = conv_block(x, filters, use_bn=use_bn, dropout=dropout)

    # Decoder
    for d in reversed(range(depth)):
        filters //= 2
        x = up_block(x, skips[d], filters, use_bn=use_bn, dropout=dropout)

    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
