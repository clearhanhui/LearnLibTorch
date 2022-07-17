/*
 * @File    :   TorchScript.cpp
 * @Time    :   2022/07/18
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

int main() {
  std::string module_path = "../python/traced_resnet_model_pos.pt";

  torch::jit::script::Module module;
  try {
    module = torch::jit::load(module_path);
  } catch (const c10::Error &e) {
    std::cout << "error loading the model\n";
    return -1;
  }

  std::vector<torch::jit::IValue> x;
  x.push_back(torch::ones({1, 1, 28, 28}));

  at::Tensor output = module.forward(x).toTensor();
  std::cout << output.sum() << std::endl;

  return 0;
}