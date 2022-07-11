/* 
 * @File    :   MLP.h
 * @Time    :   2022/07/07
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <iostream>
#include <torch/torch.h>

class MLP : public torch::nn::Module {
public:
  MLP(int in_dim, int hidden_dim,int out_dim);
  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Linear lin1{nullptr};
  torch::nn::Linear lin2{nullptr};
  torch::nn::Linear lin3{nullptr};
};