# definition of the data generator used for training
# loosely inspired by: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

# our contributions:
# - yielding metadata for stratification purposes
# - performing img->numpy step

import numpy as np
from tensorflow import keras
import PIL
import PIL.Image
import os


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_imgs, labels, strat_classes_num, data_path, batch_size=32, dim=(450, 600, 3), n_classes=7, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_imgs = list_imgs
        self.strat_classes_num = strat_classes_num
        self.data_path = data_path
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    # returns number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.list_imgs) / self.batch_size))

    # generate one batch of data
    def __getitem__(self, index):
        
        # generates indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # takes list of image IDs
        list_imgs_temp = [self.list_imgs[k] for k in indexes]

        # generates data for given IDs
        X, y = self.__data_generation(list_imgs_temp)

        return X, y

    # updates indexes after each epoch. thanks to shuffling the training loop doesnt see the same batches in each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_imgs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # generates data containing batch_size samples
    def __data_generation(self, list_imgs_temp):

        # sets up empty arrays in proper shapes
        X = np.empty((self.batch_size, *self.dim))
        meta_X = np.empty((self.batch_size, self.strat_classes_num))
        y = np.empty((self.batch_size), dtype=int)

        # generates data, according to what was earlier returned by draw_data()
        for i, img in enumerate(list_imgs_temp):
            X[i, ] = self.__get_img_to_numpy(img[0])
            meta_X[i, ] = [img[1] == el for el in range(self.strat_classes_num)]
            y[i] = self.labels[img[0]]

        X = X / 255.0

        return [X, meta_X], keras.utils.to_categorical(y, num_classes=self.n_classes)

    # takes one image, spits out 450x600x3 numpy
    def __get_img_to_numpy(self, img):
        pic = PIL.Image.open(os.path.join(self.data_path, f'{img}.jpg'))
        return np.array(pic)
