import numpy as np
import torch

# 操作数据维度
x = torch.Tensor(2, 2, 2)
print(x.shape)
y = x.view(1, 8)    #输出维度：1*8
print(y.shape)
z = x.view(-1, 4)  # -1表示维数自动判断，此输出的维度为：2*4
print(z.shape)
t = x.view(8)      #输出维度 ：   8*1
print(t.shape)
t = x.view(-1)   #输出维度： 1*8
print(t.shape)
# result:
'''
torch.Size([2, 2, 2])
torch.Size([1, 8])
torch.Size([2, 4])
torch.Size([8])
torch.Size([8])
'''

x = torch.tensor([1, 2, 3])
print(x.repeat(3, 2)) # 把x看成一个整体然后沿着特定的维度重复这个整体，
print(x.expand(4, 3)) # x是一个1X3的矩阵，保持列的维度不变，把x看成一个整体，沿着行的方向进行复制，
print(x.shape)
'''
tensor([[1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3]])
tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])
torch.Size([3])
'''


