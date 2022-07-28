#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   check.py
@Time    :   2022/07/28
@Author  :   Han Hui
@Contact :   clearhanhui@gmail.com
'''


import torch
import numpy as np
from python.AutoGrad import make_array, mm, mul, add, BackwardEngine


a = make_array(np.random.randn(2,2), requires_grad=True)
b = make_array(np.random.randn(2,2), requires_grad=True)
c = make_array(np.random.randn(2,2), requires_grad=True)

d = mul(add(a, b), c)
e = mm(d, add(a, c))
engine = BackwardEngine(e)
engine.run_backward(np.ones_like(e))

# print(a.grad)
# print(b.grad)
# print(c.grad)


aa = torch.tensor(a, requires_grad=True)
bb = torch.tensor(b, requires_grad=True)
cc = torch.tensor(c, requires_grad=True)

dd = torch.mul(torch.add(aa, bb), cc)
ee = torch.mm(dd, torch.add(aa, cc))
ee.backward(torch.ones_like(ee))

# print(aa.grad.numpy())
# print(bb.grad.numpy())
# print(cc.grad.numpy())

print(np.allclose(a.grad, aa.grad.numpy()))
print(np.allclose(b.grad, bb.grad.numpy()))
print(np.allclose(c.grad, cc.grad.numpy()))
