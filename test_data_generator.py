
'''
Main purpose of testing is to check that the metadata (about stratification classes) is yielded properly
'''

import unittest
import strat_data_generator as sdg
import config_sc
import numpy as np


class TestDataGenerator(unittest.TestCase):

    def test_metadata(self):

        # we drew 6 imgs to be given to data generator to later generate from
        # there are 6 different strat classes, but only 5 are covered by out imgs (0,1,2,3,4)
        list_imgs = [['ISIC_0024306', 0], ['ISIC_0024307', 2], ['ISIC_0024308', 3], ['ISIC_0024309', 4], ['ISIC_0024310', 1], ['ISIC_0024311', 4]]
        labels = {'ISIC_0024306': 2, 'ISIC_0024307': 3, 'ISIC_0024308': 4, 'ISIC_0024309': 2, 'ISIC_0024310': 3, 'ISIC_0024311': 4}
        strat_classes_num = 8
        params_generator = {'dim': (450, 600, 3),
                        'batch_size': 6,
                        'n_classes': 7,
                        'shuffle': False}

        # after feeding the above to the generator we expect, that apart from batch's X and y it will spit out such one-hot encoded vector of stratification classes 
        benchmark = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0.]])

        gen = sdg.DataGenerator(
            list_imgs, labels, strat_classes_num, config_sc.imgs_path, **params_generator)

        # indexes mean respectively: batch, first output, second element of the first output
        # because the output is of the form '[X, meta_X], y' and we want to check meta_X
        self.assertTrue((gen[0][0][1]==benchmark).all())

if __name__ == '__main__':
    unittest.main()