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
        index = tf.zeros_like(x, dtype=tf.float32)
        update = tf.constant([[1]],dtype=tf.float32)
        # fix the problem
        index = tf.reshape(index,(-1,1))
        print("!!!!!!!!!!!!!!!!!!!!!!",index)
        index_float = tf.tensor_scatter_nd_update(index, target, update)
        batch_m = tf.matmul(self.m_list[None, :], index_float)
        print("batch_m :",batch_m)
        batch_m = tf.reshape(batch_m, (-1, 1))
        print("batch_m :",batch_m)
        x_m = x - batch_m
        print("x_m :",x_m)

        # if condition is true, return x_m[index], otherwise return x[index]
        index_bool = tf.cast(index_float, bool)
        index_bool = tf.reshape(index_bool,(1,-1))
        print("index :\n",index)
        output = tf.where(index_bool, x_m, x)
        print("output : ",output)
        # Continue - For using tf softmax cross_entropy, we need labels and logits
        labels = index_bool
        logits = output
        print("labels : \n",labels,"\n logits : \n", logits)
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits*self.s)

        '''
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
        '''
