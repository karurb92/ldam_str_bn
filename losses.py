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

    def __call__(self, x, target):
        index = tf.zeros_like(x, dtype=tf.dtypes.uint8)
        # a little bit confused to convert parameters in torch.scatter_ to tf.scatter_nd
        index = tf.scatter_nd(tf.reshape(target.data, (-1, 1)), index, 1)
        index_float = tf.convert_to_tensor(index, dtype=tf.float32)
        batch_m = tf.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = tf.reshape(batch_m, (-1, 1))
        x_m = x - batch_m

        # if condition is true, return x_m[index], otherwise return x[index]
        output = tf.where(index, x_m, x)
        # Continue - For using tf softmax cross_entropy, we need labels and logits
        labels = target
        logits = output
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        '''
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
        '''
