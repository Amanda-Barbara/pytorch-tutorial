import torch
import random
from torch import nn
from math import log
import math
#
# random.seed(0)
# torch.manual_seed(0)
# input = torch.randn(3, 3)
# out = torch.sigmoid(input)
# target = torch.FloatTensor([[0, 1, 1],
#                             [0, 0, 1],
#                             [1, 0, 1]])
# l1 = nn.BCELoss()
# loss1 = l1(out, target)
# print(input)
# print(loss1)  # tensor(1.1805)
#
# input2 = torch.FloatTensor([[1.5410, -0.2934, -2.1788],
#                             [0.5684, -1.0845, -1.3986],
#                             [0.4033,  0.8380, -0.7193]])
# l2 = nn.BCEWithLogitsLoss()
# loss2 = l2(input2, target)
# print(loss2)  # tensor(1.1805)
# # 可以发现loss1和loss2相等


# https://flyfish.blog.csdn.net/article/details/118909723

a = 1 /(1+math.exp(-1 * 0.7))
b = 1 /(1+math.exp(-1 * 0.2))
c = 1 /(1+math.exp(-1 * 0.1))
print(((1 * log (a)+(1-1) * log (1-a))+(0 * log (b)+(1-0) * log (1-b))+(0 * log (c)+(1-0) * log (1-c))) /(-3))

pred=torch.tensor([0.7,0.2,0.1],dtype=torch.float)
target=torch.tensor([1,0,0],dtype=torch.float)
criterion = torch.nn.BCEWithLogitsLoss()
loss=criterion(pred, target)
print(loss)#tensor(0.6486)

