#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   train_export_model.py
@Time    :   2022/07/18
@Author  :   Han Hui
@Contact :   clearhanhui@gmail.com
'''

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def infer(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.reshape((x.size(0), -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.infer(x)
        return x


class LeNet5_pos(LeNet5):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = self.infer(x)
        if x.sum() > 0:
            return x
        else:
            return -x


model = LeNet5()
model_pos = LeNet5_pos()
example = torch.rand(1, 1, 28, 28)

traced_script_module = torch.jit.trace(model, example)
# traced_script_module_pos = torch.jit.trace(model_pos, example) # cause warning
traced_script_module_pos = torch.jit.script(model_pos)

save_path = sys.path[0] + "/"
traced_script_module.save(save_path+"traced_resnet_model.pt")
traced_script_module_pos.save(save_path+"traced_resnet_model_pos.pt")
