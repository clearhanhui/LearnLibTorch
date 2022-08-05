#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   check_profile.py
@Time    :   2022/07/28
@Author  :   Han Hui
@Contact :   clearhanhui@gmail.com
'''


import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from python.extend import GCNLayerFunction, LinearFunction, GCNLayer, AddSelf


addself = AddSelf.apply
x = torch.randn(1, dtype=torch.double, requires_grad=True)
y = addself(x)
print(gradcheck(addself, x, eps=1e-6, atol=1e-4))


linear = LinearFunction.apply
input = torch.randn(20, 20, dtype=torch.double, requires_grad=True)
weight = torch.randn(20, 30, dtype=torch.double, requires_grad=True)
print(gradcheck(linear, (input, weight), eps=1e-6, atol=1e-4))


use_cpp = False
gc = GCNLayerFunction.apply
a = torch.randn((10, 10), dtype=torch.double)
x = torch.randn((10, 20), dtype=torch.double)
w = torch.randn((20, 30), dtype=torch.double, requires_grad=True)
b = torch.randn(30, dtype=torch.double, requires_grad=True)
print(gradcheck(gc, (a, x, w, b, use_cpp), eps=1e-6, atol=1e-4))


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gc1 = GCNLayer(1433, 16)
        self.gc2 = GCNLayer(16, 7)
        self.dropout = nn.Dropout(0.5)

    def forward(self, a, x):
        x = F.relu(self.gc1(a, x, use_cpp))
        x = self.dropout(x)
        x = self.gc2(a, x, use_cpp)
        return x

start_t = time.time()
gcn = GCN()
loss_fn = torch.nn.CrossEntropyLoss()
adam = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
a = torch.randn((2708, 2708))
x = torch.randn((2708, 1433))
y = torch.randint(0, 2, (2708,), dtype=torch.long)
for i in range(200):
    y_prob = gcn(a, x)
    loss = loss_fn(y_prob, y)
    adam.zero_grad()
    loss.backward()
    adam.step()
end_t = time.time()
print(end_t - start_t)
