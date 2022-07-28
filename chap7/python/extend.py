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
        output = input.mm(weight)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = input.t().mm(grad_output)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class GCNLayerFunction(Function):
    @staticmethod
    def forward(ctx, a, x, w, b):
        ctx.save_for_backward(a, x, w, b)
        output = a.mm(x).mm(w) + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        a, x, w, b = ctx.saved_tensors
        grada = gradx = gradw = gradb = None
        gradw = a.mm(x).t().mm(grad_output)
        gradb = grad_output.sum(0)
        return grada, gradx, gradw, gradb


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, a, x):
        return GCNLayerFunction.apply(a, x, self.weight, self.bias)
