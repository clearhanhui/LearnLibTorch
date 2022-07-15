# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Lenet5MNIST.py
@Time    :   2022/07/13
@Author  :   Han Hui
@Contact :   clearhanhui@gmail.com
'''


import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


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
