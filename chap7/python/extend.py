#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   extend.py
@Time    :   2022/07/21
@Author  :   Han Hui
@Contact :   clearhanhui@gmail.com
'''


import torch
import torch.nn as nn
from torch.autograd.function import Function


class LinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


linear = LinearFunction.apply
from torch.autograd import gradcheck

input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))
test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
print(test)


class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)


        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

lin = Linear(2,3)
x = torch.randn((4,2))
x.backward()
y = lin(x)
y.backward(gradient = torch.ones_like(y))
print(x)
print(y.shape)
print(lin.weight.grad)


class GCNLayerFunction(Function):

    @staticmethod
    def forward(ctx, a, x, w, b):
        ctx.sava_for_backward(a, x, w, b)
        output = torch.spmm(a, torch.mm(x, w)) + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        a, x, w, b = ctx.saved_tensors
        grada = gradx = gradw = gradb = None
        gradw = grad_output.t().mm(torch.spmm(a,x))
        gradb = grad_output.sum(0)
        return grada, gradx, gradw, gradb
