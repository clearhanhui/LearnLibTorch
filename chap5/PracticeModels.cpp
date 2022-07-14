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
      torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
          std::move(train_dataset), 128);
  auto test_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_dataset), 128);

  Lenet5 lenet5;
  torch::optim::Adam adam(lenet5.parameters(), 0.001);
  torch::nn::CrossEntropyLoss cross_entropy;
  for (int i = 0; i < 5; i++) {
    float total_loss = 0.0;
    for (auto &batch : *train_loader) {
      torch::Tensor x = batch.data;
      torch::Tensor y = batch.target;
      torch::Tensor y_prob = lenet5.forward(x);
      torch::Tensor loss = cross_entropy(y_prob, y);
      total_loss += loss.item<float>();
      adam.zero_grad();
      loss.backward();
      adam.step();
    }
    std::cout << "Epoch " << i << "  total_loss = " << total_loss << std::endl;
  }

  // torch::serialize::OutputArchive output_archive;
  // lenet5.save(output_archive);
  // output_archive.save_to("lenet5.pt");
  lenet5.eval();

  int correct = 0;
  int total = 0;
  for (auto &batch : *test_loader) {
    torch::Tensor x = batch.data;
    torch::Tensor y = batch.target;
    torch::Tensor y_prob = lenet5.forward(x);
    correct += y_prob.argmax(1).eq(y).sum().item<int>();
    total += y.size(0);
  }
  std::cout << "Test Accuracy = " << (float)correct / (float)total * 100 << " %"
            << std::endl;
  return 0;
}