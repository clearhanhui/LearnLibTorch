/*
 * @File    :   LR.h
 * @Time    :   2022/07/09
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <torch/torch.h>

class LinearRegression : public torch::nn::Module {
public:
  LinearRegression(int in_dim, int out_dim);
  torch::Tensor forward(torch::Tensor);

private:
  torch::nn::Linear lin{nullptr};
};