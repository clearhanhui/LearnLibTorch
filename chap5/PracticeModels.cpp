/*
 * @File    :   PracticeModels.cpp
 * @Time    :   2022/07/13
 * @Author  :   Han Hui
 * @Contact :   clearhanhui@gmail.com
 */

#include "include/Lenet5.h"
#include <iostream>

int main() {
  /// The supplied `root` path should contain the *content* of the unzipped
  /// MNIST dataset, available from http://yann.lecun.com/exdb/mnist.
  auto train_dataset =
      torch::data::datasets::MNIST("../data/MNIST/raw",
                                   torch::data::datasets::MNIST::Mode::kTrain)
          .map(torch::data::transforms::Stack<>());

  auto test_dataset =
      torch::data::datasets::MNIST("../data/MNIST/raw",
                                   torch::data::datasets::MNIST::Mode::kTest)
          .map(torch::data::transforms::Stack<>());

  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), 128);
  auto test_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_dataset), 128);

  Lenet5 lenet5;
  torch::optim::Adam adam(lenet5.parameters(), 0.001);
  torch::nn::CrossEntropyLoss cross_entropy;
  for (int i = 0; i < 5; i++) {
    torch::Tensor loss;
    for (auto &batch : *test_loader) {
      torch::Tensor x = batch.data;
      torch::Tensor y = batch.target;
      torch::Tensor y_prob = lenet5.forward(x);
      loss = cross_entropy(y_prob, y);
      adam.zero_grad();
      loss.backward();
      adam.step();
    }
    std::cout << "Epoch " << i << " loss=" << loss.item() << std::endl;
  }

  return 0;
}