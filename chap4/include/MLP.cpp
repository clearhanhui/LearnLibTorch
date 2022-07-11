/*
 * @File    :   MLP.cpp
 * @Time    :   2022/07/11
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include "MLP.h"

MLP::MLP(int in_dim, int hidden_dim, int out_dim) {
  lin1 = torch::nn::Linear(in_dim, hidden_dim);
  lin2 = torch::nn::Linear(hidden_dim, hidden_dim);
  lin3 = torch::nn::Linear(hidden_dim, out_dim);
  
  register_module("lin1", lin1);
  register_module("lin2", lin2);
  register_module("lin3", lin3);
};

torch::Tensor MLP::forward(torch::Tensor x) {
  x = lin1(x);
  x = torch::relu(x);
  x = lin2(x);
  x = torch::relu(x);
  x = lin3(x);
  return x;
}
