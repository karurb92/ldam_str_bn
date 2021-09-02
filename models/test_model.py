from tensorflow import keras
from tensorflow.keras import layers
from models.strat_bn_simplified import StratBN

# just a really simple model with almost no parameters
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
