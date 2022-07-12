/*
 * @File    :   LR.cpp
 * @Time    :   2022/07/11
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

// #include "include/CNN.h"
#include "include/CNN.h"
#include "include/MLP.h"
#include <iostream>
#include <torch/torch.h>

int main() {
  // 生成数据
  torch::Tensor w = torch::tensor({{1.0, 2.0}});
  torch::Tensor x = torch::rand({20, 2});
  torch::Tensor b = torch::randn({20, 1}) + 3;
  torch::Tensor y = x.mm(w.t()) + b;

  torch::Tensor img0 = torch::randn({10, 1, 28, 28}) * 100 + 100;
  torch::Tensor label0 = torch::zeros({10}, torch::kLong);
  torch::Tensor img1 = torch::randn({10, 1, 28, 28}) * 100 + 150;
  torch::Tensor label1 = torch::ones({10}, torch::kLong);
  torch::Tensor img = torch::cat({img0, img1});
  torch::Tensor label = torch::cat({label0, label1});

  // 线性回归
  std::cout << "\n============= train_lr  ==============\n";
  torch::nn::Linear lin(2, 1);
  torch::optim::SGD sgd(lin->parameters(), 0.1);
  for (int i = 0; i < 10; i++) {
    torch::Tensor y_ = lin(x);
    torch::Tensor loss = torch::mse_loss(y_, y);
    sgd.zero_grad();
    loss.backward();
    sgd.step();
    std::cout << "Epoch " << i << " loss=" << loss.item() << std::endl;
  }

  //多层感知机
  std::cout << "\n============= train_mlp ==============\n";
  MLP mlp(x.size(1), 4, 1);
  torch::optim::RMSprop rms_prop(mlp.parameters(), 0.1);
  for (int i = 0; i < 10; i++) {
    torch::Tensor y_ = mlp.forward(x);
    torch::Tensor loss = torch::mse_loss(y_, y);
    rms_prop.zero_grad();
    loss.backward();
    rms_prop.step();
    std::cout << "Epoch " << i << " loss=" << loss.item() << std::endl;
  }

  // 卷积网络
  std::cout << "\n============= train_cnn ==============\n";
  CNN cnn(2);
  torch::optim::Adam adam(cnn.parameters(), 0.01);
  torch::nn::CrossEntropyLoss cross_entropy;
  for (int i = 0; i < 10; i++) {
    torch::Tensor label_ = cnn.forward(img);
    torch::Tensor loss = cross_entropy(label_, label);
    adam.zero_grad();
    loss.backward();
    adam.step();
    std::cout << "Epoch " << i << " loss=" << loss.item() << std::endl;
  }

  return 0;
}
