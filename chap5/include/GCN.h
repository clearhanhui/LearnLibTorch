/*
 * @File    :   GCN.h
 * @Time    :   2022/07/21
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include <torch/torch.h>
namespace nn = torch::nn;
namespace F = torch::nn::functional;

class GCNLayerImpl : public nn::Module {
public:

  GCNLayerImpl(int in_features, int out_features);
  torch::Tensor forward(torch::Tensor x, torch::Tensor a);

private:
  torch::Tensor w;
  torch::Tensor b;
};
TORCH_MODULE(GCNLayer); // 注册


class GCN : public nn::Module {
public:
  GCN();
  torch::Tensor forward(torch::Tensor x, torch::Tensor a);

private:
  torch::nn::Dropout dropout = nullptr;
  GCNLayer gc1 = nullptr;
  GCNLayer gc2 = nullptr;
};