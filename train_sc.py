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
from strat_data_generator import DataGenerator
from losses import *

# parse args here
#imb_ratio, strat_dims, data_path, train_split, batch_size

def main():

    ### before we feed all imgs here, we should earlier decide on some constant test set
    ### assess on imbalance or balance?

    metadf = load_metadf(data_path)
    data_train, data_val, labels = draw_data(metadf, imb_ratio, strat_dims, train_split)
    
    params_generator = {'dim': (450, 600, 3),
          'batch_size': batch_size,
          'n_classes': 7,
          'shuffle': True}

    training_generator = DataGenerator(data_train, labels, data_path, **params_generator)
    validation_generator = DataGenerator(data_val, labels, data_path, **params_generator)


    # callbacks = [
    #     # Write TensorBoard logs to `./logs` directory
    #     keras.callbacks.TensorBoard(
    #         log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True),
    # ]


    # model = res_net_model()
    # model.compile(optimizer=keras.optimizers.Adam(),
    #               loss='LDAMLoss()',
    #               metrics=['TODO'])
    ###KAROL: WE HAVE TO USE fit_generator()
    # model.fit(train_dataset, epochs=30, steps_per_epoch=195,
    #           validation_data=valid_dataset,
    #           validation_steps=3, callbacks=callbacks)

# further training from here

#save data_test? use it for final assessment

if __name__ == '__main__':
    main()
