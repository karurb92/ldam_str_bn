# model class here
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.strat_bn_simplified import StratBN

# source:
# https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/


def res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu',
                      padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation('relu')(x)
    return x


def test_stratbn_model(n_strat_classes, n_classes=7,):
    inputs1 = keras.Input(shape=(450, 600, 3))
    inputs2 = keras.Input(shape=(n_strat_classes,))
    x = StratBN()([inputs1, inputs2])

    x = layers.Conv2D(1, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(1, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(1, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(1, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(n_classes)(x)
    return keras.Model([inputs1, inputs2], output)


def res_net_model(n_strat_classes, n_classes=7, num_res_net_blocks=2, use_stratification=False):

    inputs1 = keras.Input(shape=(450, 600, 3))
    inputs2 = keras.Input(shape=(n_strat_classes,))

    if use_stratification:
        x = StratBN()([inputs1, inputs2])
    else:
        x = layers.BatchNormalization()(inputs1)

    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    for _ in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes)(x)  # no softmax

    return keras.Model([inputs1, inputs2], outputs)
