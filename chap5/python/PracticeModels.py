# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Lenet5MNIST.py
@Time    :   2022/07/13
@Author  :   Han Hui
@Contact :   clearhanhui@gmail.com
'''


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


train_dataset = torchvision.datasets.MNIST(root='../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=128,
                                          shuffle=False)

model = LeNet5()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(5):
    loss = 0
    for x, y in train_loader:
        y_prob = model(x)
        loss = loss_fn(y_prob, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("Epoch [{:0>2d}]  loss={:.4f}".format(i, loss.item()))

# torch.save(model, 'lenet5.pt')
# model = torch.load('lenet5.pt')
# model.eval()

correct = 0
total = 0
for x, y in test_loader:
    y_prob = model(x)
    _, y_ = torch.max(y_prob, 1)
    correct += (y_ == y).sum().item()
    total += y.size(0)

print('Test Accuracy = {:.2f} %'.format(100 * correct / total))
