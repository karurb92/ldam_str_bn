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
from strat_data_generator import StratifiedDataGenerator


def main():
    # parse args here
    #OBLIGATORY: imb_ratio
    #NOT OBL: strat_dims, y, 

    metadf = load_metadf()
    data = draw_data(metadf, imb_ratio, strat_dims, y)
    #TODO draw IDs according to train-val-test split, save to numpy? data_train, data_val, data_test as lists of image ids. labels as dictionary of id:class_NUMBER (0 to 7)
    #somehow we need to save their strat_dim values as well (list of triplets (not always triplets)). ADD DIMENSION EVEN IF THERE IS NO STRAT
    
    params_generator = {'dim': (450, 600, 3),
          'batch_size': 64,
          'n_classes': 7,
          'shuffle': True}

    training_generator = StratifiedDataGenerator(data_train, labels, strat_dims, **params_generator)
    validation_generator = StratifiedDataGenerator(data_val, labels, strat_dims, **params_generator)


    # callbacks = [
    #     # Write TensorBoard logs to `./logs` directory
    #     keras.callbacks.TensorBoard(
    #         log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True),
    # ]


    # model = res_net_model()
    # model.compile(optimizer=keras.optimizers.Adam(),
    #               loss='TODO',
    #               metrics=['TODO'])
    ###KAROL: WE HAVE TO USE fit_generator()
    # model.fit(train_dataset, epochs=30, steps_per_epoch=195,
    #           validation_data=valid_dataset,
    #           validation_steps=3, callbacks=callbacks)

# further training from here

if __name__ == '__main__':
    main()
