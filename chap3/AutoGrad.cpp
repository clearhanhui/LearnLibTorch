/*
 * @File    :   AutoGrad.cpp
 * @Time    :   2022/07/07
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <iostream>
#include <torch/torch.h>

int main() {
  // y = w * x + b
  torch::Tensor x = torch::tensor({2.0});
  torch::Tensor w = torch::tensor({3.0}, torch::requires_grad());
  torch::Tensor b = torch::tensor({4.0}, torch::requires_grad());
  torch::Tensor y = w * x + b;
  y.backward();
  std::cout << w.grad() << std::endl;
  std::cout << b.grad() << std::endl;
  std::cout << x.requires_grad() << std::endl;
  x.requires_grad_(true);
  std::cout << x.requires_grad() << std::endl;

  // yy = xx * xx + 2 * xx
  torch::Tensor xx = torch::randn({3}, torch::requires_grad());
  torch::Tensor yy = xx * xx + 2 * xx;
  yy.backward(torch::tensor({1, 2, 3}));
  std::cout << xx << std::endl;
  std::cout << xx.grad() << std::endl;
  std::cout << xx * torch::tensor({2, 4, 6}) + torch::tensor({2, 4, 6})
            << std::endl;

      return 0;
}