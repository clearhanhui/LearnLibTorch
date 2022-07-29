#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   AutoGrad.py
@Time    :   2022/07/27
@Author  :   Han Hui
@Contact :   clearhanhui@gmail.com
'''


import numpy as np


class MyArray(np.ndarray):
    """
    Same with `np.ndarray` but `__setattr__` fucntion.
    """
    def __setattr__(self, __name, __value):
        self.__dict__[__name] = __value


def make_array(input_list, requires_grad=False):
    nparray = np.array(input_list)
    myarray = MyArray(nparray.shape)
    myarray[:] = nparray[:]
    if requires_grad:
        myarray.requires_grad = True
    return myarray


class NodeBase:
    """
    Base class of Function
    """

    def __init__(self):
        """
        initialize the function.
        """
        self._edges = []
        self._depends = 0
        self._inputs = []
        self._outputs = []

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def _backward(self):
        """
        calculate the gradient of inputs.
        this is local.
        """
        raise NotImplementedError

    def forward(self, inputs):
        """
        forward propagation
        """
        raise NotImplementedError

    def _save_inputs(self, *args):
        for a in args:
            self._inputs.append(a)
            if not hasattr(a, "requires_grad"):
                self._edges.append(a.fn)

    def _save_outputs(self, *args):
        for a in args:
            self._outputs.append(a)


class MulOP(NodeBase):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        self._save_inputs(a, b)
        output = a * b
        output.fn = self
        self._save_outputs(output)
        return output

    def _backward(self):
        if not hasattr(self._inputs[0], "grad"):
            self._inputs[0].grad = 0.0
        self._inputs[0].grad += self._outputs[0].grad * self._inputs[1]
        if not hasattr(self._inputs[1], "grad"):
            self._inputs[1].grad = 0.0
        self._inputs[1].grad += self._outputs[0].grad * self._inputs[0]


class MatMulOP(NodeBase):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        self._save_inputs(a, b)
        output = a @ b
        output.fn = self
        self._save_outputs(output)
        return output

    def _backward(self):
        if not hasattr(self._inputs[0], "grad"):
            self._inputs[0].grad = np.zeros(
                self._inputs[0].shape, dtype=np.float32)
        self._inputs[0].grad += self._outputs[0].grad @ self._inputs[1].T
        if not hasattr(self._inputs[1], "grad"):
            self._inputs[1].grad = np.zeros(
                self._inputs[1].shape, dtype=np.float32)
        self._inputs[1].grad += self._inputs[0].T @ self._outputs[0].grad


class AddOP(NodeBase):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        self._save_inputs(a, b)
        output = a + b
        output.fn = self
        self._save_outputs(output)
        return output

    def _backward(self):
        if not hasattr(self._inputs[0], "grad"):
            self._inputs[0].grad = 0.0
        self._inputs[0].grad += self._outputs[0].grad
        if not hasattr(self._inputs[1], "grad"):
            self._inputs[1].grad = 0.0
        self._inputs[1].grad += self._outputs[0].grad


class ExpOP(NodeBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def _backward(self):
        pass


class BackwardEngine:
    """
    Single thread backward.
    """

    def __init__(self, arr):
        self._target_arr = arr
        self._root = arr.fn
        self.topo_queue = []
        self._topological_sorting()

    def _topological_sorting(self):
        def _dfs(node):
            for next_node in node._edges:
                _dfs(next_node)
            node._depends += 1
            self.topo_queue.append(node)
        _dfs(self._root)
        self._root._depends = 0

    def run_backward(self, gradient=None):
        # Following check codes are simulated to PyTorch
        if not hasattr(self._root, "grad"):
            if isinstance(gradient, np.ndarray):
                assert gradient.shape == self._target_arr.shape
                self._target_arr.grad = gradient
            elif gradient is None:
                if self._target_arr.size == 1:
                    self._target_arr.grad = np.ones_like(self._target_arr)
                else:
                    raise RuntimeError(
                        "grad can be implicitly created only for scalar outputs")
            else:
                raise TypeError("gradients can be either `np.ndarray` or `None`, but got `{}`"
                                .format(type(gradient).__name__))
        else:
            raise Exception("root node cannot has grad attribute")


        while len(self.topo_queue):
            task_node = self.topo_queue.pop()
            if task_node._depends == 0:
                task_node._backward()
                for n in task_node._edges:
                    n._depends -= 1


# Wrap again, every time call the func will create an NODE instance.
def add(a, b):
    op = AddOP()
    return op(a, b)


def mul(a, b):
    op = MulOP()
    return op(a, b)


def mm(a, b):
    op = MatMulOP()
    return op(a, b)
