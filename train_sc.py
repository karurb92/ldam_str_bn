'''params for training will be:
y - (string) which y to use ('dx' or 'dx_alternative')
strat_dims - (list) of dimensions to stratify on (dx_type / sex / age_mapped / localization). maybe localization makes more sense because it affects photos the most? distr of ys is heavily dependent on sex and age
imb_ratio - (float) imbalance ratio (main class count / all other count)
etc. (all the learning rates, depth, batch size etc)
'''

from config_sc import *
from utils_sc import *
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from models.resnet import res_net_model


def main():
    # parse args here

    metadf = load_metadf()
    data = draw_data(metadf, imb_ratio, strat_dims, y)

    # callbacks = [
    #     # Write TensorBoard logs to `./logs` directory
    #     keras.callbacks.TensorBoard(
    #         log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True),
    # ]


    # model = res_net_model()
    # model.compile(optimizer=keras.optimizers.Adam(),
    #               loss='TODO',
    #               metrics=['TODO'])
    # model.fit(train_dataset, epochs=30, steps_per_epoch=195,
    #           validation_data=valid_dataset,
    #           validation_steps=3, callbacks=callbacks)

# further training from here

if __name__ == '__main__':
    main()
