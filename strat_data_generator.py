# We set up data generator for later use by .fit_generator()
# It also performs reformatting pics
# loosely inspired by: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

'''
#test it with:
list_imgs = [['ISIC_0024306', 'male', '<50;inf)'], ['ISIC_0024307', 'female', '<50;inf)'], ['ISIC_0024308', 'female', '<0;50>']]
labels = {'ISIC_0024306': 2, 'ISIC_0024307': 3, 'ISIC_0024308': 4}
#make sure batch_size <= number of images
'''

import numpy as np
from tensorflow import keras
import PIL
import PIL.Image
import os


class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
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

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_imgs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_imgs_temp = [self.list_imgs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_imgs_temp)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_imgs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_imgs_temp):
        # Generates data containing batch_size samples

        X = np.empty((self.batch_size, *self.dim))
        meta_X = np.empty((self.batch_size, self.strat_classes_num))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, img in enumerate(list_imgs_temp):
            X[i, ] = self.__get_img_to_numpy(img[0])
            meta_X[i, ] = [img[1] == el for el in range(self.strat_classes_num)]
            y[i] = self.labels[img[0]]

        X = X / 255.0

        return [X, meta_X], keras.utils.to_categorical(y, num_classes=self.n_classes)


    def __get_img_to_numpy(self, img):
        pic = PIL.Image.open(os.path.join(self.data_path, f'{img}.jpg'))
        return np.array(pic)
