import unittest

import tensorflow as tf
from models import strat_bn_simplified
import numpy as np


class TestStratBNLoss(unittest.TestCase):
    m = strat_bn_simplified.StratBN(axis=[-1])

    def test_same_strat(self):
        # create two different images
        input_shape = [2, 2, 1]
        ex1 = tf.fill(input_shape, 1.0)
        ex2 = tf.fill(input_shape, 2.0)
        batch = tf.stack([ex1, ex2])

        # stratify together
        strat = tf.constant([[1, 0], [1, 0]])

        # so the batch norm should just be computed over the entire batch
        # therefore the two images should still be different
        # as the mean will be 1.5

        # we set training to true to compute the mean and var over the batch
        out = TestStratBNLoss.m([batch, strat], training=True)
        out = out.numpy()

        # everything should be false
        comp = out[0] == out[1]

        self.assertFalse(comp.any())

        # more precisely example 1 should all have values of -1 because:
        # mean = 1.5
        # std = .5
        # so (1 - 1.5) / (.5 + epsilon) = ca. -1

        # we use round because of th epsilon and float imprecisions
        # so actualy values are more like -0.997
        self.assertTrue((np.around(out[0]) == -1).all())
        self.assertTrue((np.around(out[1]) == 1).all())

    # same as above but now we stratify the images separately

    def test_diff_strat(self):
        # create two different images
        input_shape = [2, 2, 1]
        ex1 = tf.fill(input_shape, 1.0)
        ex2 = tf.fill(input_shape, 2.0)
        batch = tf.stack([ex1, ex2])

        # stratify separately
        strat = tf.constant([[1, 0], [0, 1]])

        # we set training to true to compute the mean and var over the batch
        out = TestStratBNLoss.m([batch, strat], training=True)
        out = out.numpy()

        # so now we expect the two normalized images to be the same because their
        # means were subtracted separately
        comp = out[0] == out[1]

        self.assertTrue(comp.all())


if __name__ == '__main__':
    unittest.main()
