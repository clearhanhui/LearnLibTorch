/*
 * @File    :   HelloWorld.cpp
 * @Time    :   2022/07/06 16:10:05
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <iostream>
#include <torch/torch.h>

int main() {
  // 创建一个(2,3)张量
  torch::Tensor tensor = torch::zeros({2, 3});
  std::cout << tensor << std::endl;
  std::cout << "\nWelcome to LibTorch!" << std::endl;

  return 0;
}