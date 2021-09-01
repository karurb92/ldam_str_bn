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
from models.resnet import res_net_model, test_stratbn_model
from strat_data_generator import DataGenerator
from losses import *
import datetime as dt

# parse args here
#imb_ratio, strat_dims, data_path, train_split, batch_size


def main():

    # before we feed all imgs here, we should earlier decide on some constant test set
    # assess on imbalance or balance?
    imb_ratio = 10
    strat_dims = ['age_mapped']
    train_split = 0.8
    batch_size = 1
    data_path = project_path

    metadf = load_metadf(data_path)
    data_train, data_val, labels, strat_classes_num, cls_num_list = draw_data(
        metadf, imb_ratio, strat_dims, train_split)

    print(strat_classes_num)
    print(cls_num_list)

    params_generator = {'dim': (450, 600, 3),
                        'batch_size': batch_size,
                        'n_classes': 7,
                        'shuffle': True}

    training_generator = DataGenerator(
        data_train, labels, strat_classes_num, imgs_path, **params_generator)
    validation_generator = DataGenerator(
        data_val, labels, strat_classes_num, imgs_path, **params_generator)

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True, histogram_freq=1),
    ]

    # test_stratbn_model
    model = res_net_model(strat_classes_num, num_res_net_blocks=2)
    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=LDAMLoss(cls_num_list),
                  metrics=['accuracy'])

    model.fit(training_generator, epochs=10,
              validation_data=validation_generator, callbacks=callbacks)

    model.save(data_path)


if __name__ == '__main__':
    main()
