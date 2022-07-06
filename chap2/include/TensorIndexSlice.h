/*
 * @File    :   TensorIndexSlice.h
 * @Time    :   2022/07/07
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <iostream>
#include <torch/torch.h>
using namespace torch::indexing; //不然一行代码真长啊

void tensor_index_slice() {
  torch::Tensor a = torch::randn({2, 3, 4});
  torch::Tensor b = a[1];
  torch::Tensor c = a.index({1, 2, 3});        // 等同 a[1][2][3]
  torch::Tensor d = a.index({Slice(None), 2}); // 等同 a.index({"...", 2})
  torch::Tensor e = a.index({Slice(None), Slice(None), Slice(None, None, 2)});
  torch::Tensor f = a.index_select(-1, torch::tensor({1, 1, 0}));

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;
  std::cout << d << std::endl;
  std::cout << e << std::endl;
  std::cout << f << std::endl;
}