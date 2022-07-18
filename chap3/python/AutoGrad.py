#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   AutoGrad.py
@Time    :   2022/07/07
@Author  :   Han Hui
@Contact :   clearhanhui@gmail.com
'''


import torch

x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
y = w * x + b
y.backward(retain_graph=True)
print(w.grad)
print(b.grad)
print(x.requires_grad)
x.requires_grad_(True) # x.requires_grad=True
print(x.requires_grad)

xx = torch.randn(3, requires_grad=True)
yy = xx * xx + 2 * xx
yy.backward(torch.tensor([1,2,3]))
print(xx)
print(xx.grad)
print((xx*torch.tensor([2,4,6])+torch.tensor([2,4,6])).data)
