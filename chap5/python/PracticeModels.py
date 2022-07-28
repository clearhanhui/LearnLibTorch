#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   Lenet5MNIST.py
@Time    :   2022/07/13
@Author  :   Han Hui
@Contact :   clearhanhui@gmail.com
'''


import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
from utils import load_data, accuracy


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.reshape((x.size(0), -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_lenet5():
    datapath = sys.path[0] + "/../data/"
    train_dataset = torchvision.datasets.MNIST(root=datapath,
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root=datapath,
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False)

    lenet5 = LeNet5()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lenet5.parameters(), lr=0.001)

    for i in range(5):
        total_loss = 0
        for x, y in train_loader:
            y_prob = lenet5(x)
            loss = loss_fn(y_prob, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch [{:0>2d}]  total_loss = {:.4f}".format(i, total_loss))

    # torch.save(lenet5.state_dict(), 'lenet5.pt')
    # lenet5.load_state_dict(torch.load('lenet5.pt'))
    lenet5.eval()

    correct = 0
    total = 0
    for x, y in test_loader:
        y_prob = lenet5(x)
        y_ = y_prob.argmax(1)
        correct += (y_ == y).sum().item()
        total += y.size(0)

    print('Test Accuracy = {:.2f} %'.format(100 * correct / total))


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        w = torch.empty(in_features, out_features)
        b = torch.empty(out_features)
        stdv = 1. / math.sqrt(out_features)
        w.uniform_(-stdv, stdv)
        b.uniform_(-stdv, stdv)
        self.w = Parameter(w)
        self.b = Parameter(b)
        # self.w = Parameter(nn.init.trunc_normal_(
        #     torch.empty(in_features, out_features), std=0.05))
        # self.b = Parameter(nn.init.trunc_normal_(
        #     torch.empty(out_features), std=0.05))

    def forward(self, x, a):
        out = torch.spmm(a, torch.mm(x, self.w)) + self.b
        # x = a@self.w@x + self.b
        return out


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gc1 = GCNLayer(1433, 16)
        self.gc2 = GCNLayer(16, 7)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, a):
        x = F.relu(self.gc1(x, a))
        x = self.dropout(x)
        x = self.gc2(x, a)
        return x


def train_gcn():
    datapath = sys.path[0] + "/../data/cora/"
    a, x, y, idx_train, idx_val, idx_test = load_data(path=datapath)
    gcn = GCN()
    loss_fn = torch.nn.CrossEntropyLoss()
    adam = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)

    for i in range(200):
        y_prob = gcn(x, a)
        loss = loss_fn(y_prob[idx_train], y[idx_train])
        adam.zero_grad()
        loss.backward()
        adam.step()
        print("Epoch [{:0>3d}]  loss={:.4f} train_acc={:.2f}% val_acc={:.2f}%"
              .format(i, loss.item(), 100*accuracy(y_prob[idx_train], y[idx_train]),
                      100*accuracy(y_prob[idx_val], y[idx_val])))
    y_prob = gcn(x, a)
    print("test_acc={:.2f}%".format(
        100*accuracy(y_prob[idx_test], y[idx_test])))


if __name__ == "__main__":
    # train_lenet5()
    train_gcn()
