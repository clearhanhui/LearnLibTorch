#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   BasicModels.py
@Time    :   2022/07/07
@Author  :   Han Hui
@Contact :   clearhanhui@gmail.com
'''

import torch

# 生成线性数据
w = torch.tensor([[1.0, 2.0]])
x = torch.rand((20, 2))
b = torch.randn((20, 1)) + 3
y = x.mm(w.t()) + b

# 生成图像数据
img0 = torch.randn((10, 1, 28, 28)) * 100 + 100
label0 = torch.zeros(10,  dtype=torch.long)
img1 = torch.randn((10, 1, 28, 28)) * 100 + 150
label1 = torch.ones(10, dtype=torch.long)
img = torch.cat([img0, img1])# .clamp(0, 255)
label = torch.cat([label0, label1])


# 线性回归
def train_lr(x, y):
    lin = torch.nn.Linear(2, 1)
    loss_fn = torch.nn.MSELoss()
    sgd = torch.optim.SGD(lin.parameters(), lr=0.1)

    for i in range(10):
        y_ = lin(x)
        loss = loss_fn(y_, y)
        sgd.zero_grad()
        loss.backward()
        sgd.step()
        print("Epoch [{:0>2d}]  loss={:.4f}".format(i, loss.item()))


# 非线性激活的多层感知机，三层
def train_mlp(x, y):
    class MLP(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.lin1 = torch.nn.Linear(in_dim, hidden_dim)
            self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.lin3 = torch.nn.Linear(hidden_dim, out_dim)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.lin1(x)
            x = self.relu(x)
            x = self.lin2(x)
            x = self.relu(x)
            x = self.lin3(x)
            return x

    mlp = MLP(x.shape[1], 4, 1)
    rms_prop = torch.optim.RMSprop(mlp.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()

    for i in range(10):
        y_ = mlp(x)
        loss = loss_fn(y_, y)
        rms_prop.zero_grad()
        loss.backward()
        rms_prop.step()
        print("Epoch [{:0>2d}]  loss={:.4f}".format(i, loss.item()))


# 2层 卷积网络
def train_cnn(img, label):
    class CNN(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()

            self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
            self.bn = torch.nn.BatchNorm2d(16)
            self.max_pool = torch.nn.MaxPool2d(2)
            self.relu = torch.nn.ReLU()
            self.lin = torch.nn.Linear(7*7*16, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.max_pool(x)

            x = self.conv2(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.max_pool(x)

            x = self.lin(x.reshape(x.size(0), -1))
            return x

    cnn = CNN(2)
    loss_fn = torch.nn.CrossEntropyLoss()
    adam = torch.optim.Adam(cnn.parameters(), lr=0.01)

    for i in range(10):
        label_ = cnn(img)
        loss = loss_fn(label_, label)
        adam.zero_grad()
        loss.backward()
        adam.step()
        print("Epoch [{:0>2d}]  loss={:.4f}".format(i, loss.item()))


if __name__ == "__main__":
    print("\n============= train_lr  ==============")
    train_lr(x, y)
    print("\n============= train_mlp ==============")
    train_mlp(x, y)
    # print("\n============ train_mlpp ==============")
    # train_mlp(img.reshape(20, -1), label.float())
    print("\n============= train_cnn ==============")
    train_cnn(img, label)
