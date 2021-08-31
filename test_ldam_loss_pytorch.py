import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        print("index :",index)
        index.scatter_(1, target.data.view(-1, 1), 1)
        print("index :",index)
        
        index_float = index.type(torch.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        print("output :",output)
        return F.cross_entropy(self.s*output, target, weight=self.weight)



cls_num_list = [10, 1]
ldam = LDAMLoss(cls_num_list=cls_num_list)

network_outputs = torch.tensor([[.5, .5]])
targets = torch.tensor([[1, 0]])

loss = ldam(network_outputs, targets)
print(loss)