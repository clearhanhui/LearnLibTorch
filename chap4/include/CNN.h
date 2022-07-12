/*
 * @File    :   CNN.h
 * @Time    :   2022/07/11
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <torch/torch.h>

class CNN : public torch::nn::Module {
public:
  CNN(int num_classes);
  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::ReLU relu{nullptr};
  torch::nn::MaxPool2d max_pool{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  torch::nn::Linear lin{nullptr};
};