/*
 * @File    :   TensorCuda.h
 * @Time    :   2023/08/31
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <iostream>
#include <torch/torch.h>

void tensor_cuda() {
  torch::Device device = torch::Device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
  }
  // torch::Tensor a = torch::randn({3, 3}).cuda();
  torch::Tensor b = torch::randn({3, 3}).to(device);
  torch::Tensor c = torch::randn({3, 3}, device); // 似乎是发生了 Device -> TensorOptions 的隐式转换

  // std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;
}