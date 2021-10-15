# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
from torch import nn
from torch.nn import BatchNorm2d
from torch.utils.data import DataLoader

# With Learnable Parameters
m = nn.BatchNorm2d(3)
# Without Learnable Parameters
m = nn.BatchNorm2d(3, affine=False)
input = torch.randn(1, 3, 2, 2)
print(input)
input[0, 2, :, :].mean()
output = m(input)
print(output)
output[0, 2, :, :].mean()