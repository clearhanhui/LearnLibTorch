/*
 * @File    :   Lenet5.h
 * @Time    :   2022/07/13
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <torch/torch.h>

class Lenet5 : public torch::nn::Module {
private:
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::ReLU relu{nullptr};
  torch::nn::MaxPool2d max_pool{nullptr};
  torch::nn::Linear lin1{nullptr};
  torch::nn::Linear lin2{nullptr};
  torch::nn::Linear lin3{nullptr};

public:
  Lenet5();
  torch::Tensor forward(torch::Tensor x);
};
