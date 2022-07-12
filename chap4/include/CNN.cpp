/*
 * @File    :   CNN.cpp
 * @Time    :   2022/07/11
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include "CNN.h"
#include <iostream>

CNN::CNN(int num_classes) {
  conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).padding(1));
  conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 16, 3).padding(1));
  bn = torch::nn::BatchNorm2d(16);
  relu = torch::nn::ReLU();
  max_pool = torch::nn::MaxPool2d(2);
  lin = torch::nn::Linear(7 * 7 * 16, num_classes);

  register_module("conv1", conv1);
  register_module("conv2", conv2);
  register_module("bn", bn);
  register_module("relu", relu);
  register_module("max_pool", max_pool);
  register_module("lin", lin);
}

torch::Tensor CNN::forward(torch::Tensor x) {
  x = conv1(x);
  x = bn(x);
  x = relu(x);
  x = max_pool2d(x, 2);

  x = conv2(x);
  x = bn(x);
  x = relu(x);
  x = max_pool2d(x, 2);
  x = lin(x.reshape({x.size(0), -1}));
  
  return x;
}
