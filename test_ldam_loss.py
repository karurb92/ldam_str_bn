import tensorflow as tf
import losses
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TorchLDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(TorchLDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        # print(m_list, s)
        m_list = m_list * (max_m / np.max(m_list))
        # print(m_list, s)
        m_list = torch.FloatTensor(m_list)
        # print(m_list, s)
        self.m_list = m_list
        # print("self.m_list :\n", self.m_list)
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        # print(x, target, target.data, target.data.view(-1, 1))
        index = torch.zeros_like(x, dtype=torch.uint8)
        # print("index :", index)
        index.scatter_(1, target.data.view(-1, 1), 1)
        # print("index :", index)

        index_float = index.type(torch.FloatTensor)
        # print("self.m_list[None, :] : \n", self.m_list[None, :],
        # "\n index_float.transpose(0,1) :\n", index_float.transpose(0, 1))
        batch_m = torch.matmul(
            self.m_list[None, :], index_float.transpose(0, 1))
        # print("batch_m :", batch_m)
        batch_m = batch_m.view((-1, 1))
        # print("batch_m :", batch_m)
        x_m = x - batch_m
        # print("x_m :", x_m)

        output = torch.where(index, x_m, x)
        # print("output :", output)
        # print(target)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


class TestLDAMLoss(unittest.TestCase):

    cls_num_list = [11, 100, 5, 17, 9, 10, 6, 3, 4, 2]
    torch_loss = TorchLDAMLoss(cls_num_list=cls_num_list)
    tf_loss = losses.LDAMLoss(cls_num_list=cls_num_list)

    # Note that the argument order for the TF and Torch loss are flipped

    def test_single(self):
        predictions = [[.3, .7, 0, 0, 0, 0, 0, 0, 0, 0]]
        labels = [1]

        tf_out = TestLDAMLoss.tf_loss(tf.one_hot(
            labels, len(TestLDAMLoss.cls_num_list)), tf.constant(predictions))
        torch_out = TestLDAMLoss.torch_loss(
            torch.tensor(predictions), torch.tensor(labels))

        self.assertAlmostEqual(tf_out.numpy(), torch_out.numpy(), places=5)

    def test_batch(self):
        predictions = [[.3, .7, 0, 0, 0, 0, 0, 0, 0, 0],
                       [.3, .7, 0, 0, 0, 0, 0, 0, 0, 0]]
        labels = [1, 0]

        tf_out = TestLDAMLoss.tf_loss(tf.one_hot(
            labels, len(TestLDAMLoss.cls_num_list)), tf.constant(predictions))
        torch_out = TestLDAMLoss.torch_loss(
            torch.tensor(predictions), torch.tensor(labels))

        print(tf_out)
        print(torch_out)

        self.assertAlmostEqual(tf_out.numpy(), torch_out.numpy(), places=5)


if __name__ == '__main__':
    unittest.main()

'''
input space is R^d 
label space is {1,...,k}, y is corresponding to the label (Here k is 10)
model f : R^d -> R^k and outputs k logits
the inputs for Ldam loss is x,y 
cls_num_list is [n1,n2,...,nk] -> the number of instances of each classes
'''
