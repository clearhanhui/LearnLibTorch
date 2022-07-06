/*
 * @File    :   TensorTransform.h
 * @Time    :   2022/07/07
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <iostream>
#include <torch/torch.h>

void tensor_transform() {
  torch::Tensor a = torch::randn({2, 3});
  torch::Tensor b = a.transpose(0, 1);
  torch::Tensor c = a.reshape({-1, 1});
  torch::Tensor d = a.view({3, 2});
  torch::Tensor e = a.toType(torch::kFloat32);

  std::cout <<a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;
  std::cout << d << std::endl;
  std::cout << e << std::endl;
}
