/*
 * @File    :   TensorInit.h
 * @Time    :   2022/07/06
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <iostream>
#include <torch/torch.h>

void tensor_init() {
  torch::Tensor a = torch::zeros({2, 3});
  torch::Tensor b = torch::ones({2, 3});
  torch::Tensor c = torch::eye(3);
  torch::Tensor d = torch::full_like(a, 10);
  torch::Tensor e = torch::randn({2, 3});
  torch::Tensor f = torch::arange(10);
  torch::Tensor g = torch::tensor({{1, 2}, {3, 4}});
  // torch::Tensor h = torch::from_blob({1, 2, 3, 4}, {4}); // 大坑

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;
  std::cout << d << std::endl;
  std::cout << e << std::endl;
  std::cout << f << std::endl;
  std::cout << g << std::endl;
  // std::cout << h << std::endl;
}