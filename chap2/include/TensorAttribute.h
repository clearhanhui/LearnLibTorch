/*
 * @File    :   TensorAttribute.h
 * @Time    :   2022/07/06
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <iostream>
#include <torch/torch.h>

void tensor_attribute() {
  torch::Tensor a = torch::randn({2, 3});
  std::cout << a.size(1) << std::endl;
  std::cout << a.sizes() << std::endl;
  std::cout << a[0].sizes() << std::endl;
  std::cout << a[0][0].item<float>() << std::endl;
  std::cout << a.data() << std::endl;
  std::cout << a.dtype() << std::endl;
  std::cout << a.device() << std::endl;
}
