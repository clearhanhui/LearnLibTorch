#include "include/CNN.h"
#include "include/LR.h"
#include "include/MLP.h"
#include "include/data.h"

#include <iostream>
#include <torch/torch.h>

int main() {
  // 生成数据
  torch::Tensor w = torch::tensor({{1.0, 2.0}});
  torch::Tensor x = torch::rand({10, 2});
  torch::Tensor b = torch::randn({10, 1}) + 3;
  torch::Tensor y = x.mm(w.t()) + b;

  torch::Tensor img0 = torch::randn({10, 1, 28, 28}) * 100 + 100;
  torch::Tensor label0 = torch::zeros({10}, torch::kLong);
  torch::Tensor img1 = torch::randn({10, 1, 28, 38}) * 100 + 150;
  torch::Tensor label1 = torch::ones({10}, torch::kLong);
  torch::Tensor img = torch::cat({img0, img1});
  torch::Tensor label = torch::cat({label0, label1});

  //

  return 0;
}
