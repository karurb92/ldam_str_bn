# model class here
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from strat_bn import StratBN

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



def res_net_model(num_res_net_blocks=10):
    # CIFAR-10 image size

    inputs = keras.Input(shape=(450, 600, 3))
    '''
    probably we need to have 2 inputs like this:
    inputs1 = keras.Input(shape=(450, 600, 3))
    inputs2 = keras.Input(shape=(2,)) #or whatever shape stratification requires, probably needs to be universal
    which we then pass to strat bn layer as [inputs1, inputs2]
    to be tested.
    '''
    n_classes = 7

    # here we can have an addition vector

    # ======================================
    # add stratified batch norm here
    # ======================================

    # here we should just have the batched data

    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    for _ in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)

# str batch norm class here
