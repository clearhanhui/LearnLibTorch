/*
 * @File    :   TensorCalculate.h
 * @Time    :   2022/07/07
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <iostream>
#include <torch/torch.h>

void tensor_calculate() {
  torch::Tensor a = torch::ones({3, 3});
  torch::Tensor b = torch::randn({3, 3});
  torch::Tensor c = a.matmul(b);
  torch::Tensor d = a.mul(b);
  torch::Tensor e = torch::cat({a, b}, 0);
  torch::Tensor f = torch::stack({a, b});

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;
  std::cout << d << std::endl;
  std::cout << e << std::endl;
  std::cout << f << std::endl;
}