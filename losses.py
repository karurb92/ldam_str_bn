import numpy as np
import tensorflow as tf


class LDAMLoss():

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = tf.convert_to_tensor(m_list, dtype=tf.float32)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.n_classes = len(cls_num_list)

    def __call__(self, x, target):
        # this performs one hot encoding of target
        # index = tf.zeros_like(x, dtype=tf.float32)
        # update = tf.constant([[1]], dtype=tf.float32)
        # index = tf.reshape(index, (-1, 1))
        # print("!!!!!!!!!!!!!!!!!!!!!!", index)
        # index_float = tf.tensor_scatter_nd_update(index, target, update)
        index_float = tf.one_hot(target, self.n_classes)

        batch_m = tf.matmul(self.m_list[None, :], tf.transpose(index_float))
        print("batch_m :", batch_m)
        batch_m = tf.reshape(batch_m, (-1, 1))
        print("batch_m :", batch_m)
        x_m = x - batch_m
        print("x_m :", x_m)

        # if condition is true, return x_m[index], otherwise return x[index]
        index_bool = tf.cast(index_float, bool)
        # index_bool = tf.reshape(index_bool, (1, -1))
        output = tf.where(index_bool, x_m, x)
        print("output : ", output)

        print(index_bool)
        print(index_float)

        labels = index_float
        logits = output
        print("labels : \n", labels, "\n logits : \n", logits)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits*self.s)
        return tf.math.reduce_mean(loss)
