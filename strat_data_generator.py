#This is one of our main contributions. We set up data generator for later use by .fit_generator()
#It performs stratified normalization of the minibatch it is yielding
#loosely inspired by: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

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
import config_sc

class StratifiedDataGenerator(keras.utils.Sequence):
    #Generates data for Keras
    def __init__(self, list_imgs, labels, batch_size=32, dim=(450, 600, 3), n_classes=7, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_imgs = list_imgs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        #Denotes the number of batches per epoch
        return int(np.floor(len(self.list_imgs) / self.batch_size))

    def __getitem__(self, index):
        #Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_imgs_temp = [self.list_imgs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_imgs_temp)

        return X, y

    def on_epoch_end(self):
        #Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_imgs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_imgs_temp):
        #Generates data containing batch_size samples

        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, img in enumerate(list_imgs_temp):
            X[i,] = self.__get_img_to_numpy(img[0])
            y[i] = self.labels[img[0]]

        X = self.__str_normalize_batch(X, list_imgs_temp)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __str_normalize_batch(self, X, list_imgs_temp):
        ###TBD
        #X here is np array of dim (batch_size, 450, 600, 3)
        #list_imgs_temp is a list like that: [['ISIC_0024306', 'male', '<50;inf)'], ['ISIC_0024307', 'female', '<50;inf)'], ['ISIC_0024308', 'female', '<0;50>']]
        #we need to normalize X stratifying by values from list_imgs_temp
        return X

    def __get_img_to_numpy(self, img):
        ###TBD: set path as parameter instead of stupid global (import config_sc then unnecessary)
        pic = PIL.Image.open(f'{config_sc.project_path}\\all_imgs\\{img}.jpg')
        return np.array(pic)