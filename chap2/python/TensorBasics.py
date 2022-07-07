# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   CreateTensor.py
@Time    :   2022/07/06
@Author  :   Han Hui
@Contact :   clearhanhui@gmail.com
'''


import torch

# 创建张量
def tensor_init():
    a = torch.zeros((2,3))
    b = torch.ones((2,3))
    c = torch.eye(4)
    d = torch.full_like(a, 10)
    e = torch.randn((2,3))
    f = torch.arange(10)
    g = torch.tensor([[1,2], [3,4]])
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)
    print(g)


# 张量索引切片
def tensor_index_slice():
    a = torch.randn((2,3,4))
    b = a[1]
    c = a[1,2,3]
    d = a[:, 2]
    e = a[:, :, ::2]
    f = a.index_select(-1, torch.tensor([1,1,0]))
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)

# 张量属性
def tensor_attribute():
    a = torch.randn(2,3)
    print(a.size(1))
    print(a.shape)
    print(a[0].shape)
    print(a[0][0].item())
    print(a.data)
    print(a.dtype)
    print(a.device)

# 张量变换
def tensor_transform():
    a = torch.randn((2,3))
    b = a.T
    c = a.reshape(-1, 1)
    d = a.view((3,2))
    print(a)
    print(b)
    print(c)
    print(d)


# 张量计算
def tensor_calculate():
    a = torch.ones((3,3))
    b = torch.randn((3,3))
    c = a.matmul(b)
    d = a.mul(b)
    e = torch.cat([a,b], 1)
    f = torch.stack([a,b])
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)


if __name__ == "__main__":
    tensor_init()
    tensor_index_slice()
    tensor_attribute()
    tensor_transform()
    tensor_calculate()