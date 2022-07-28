- [基本模型](#基本模型)
  - [1. 数据准备](#1-数据准备)
  - [2. 线性回归](#2-线性回归)
  - [3. 多层感知机](#3-多层感知机)
  - [4. 卷积网络](#4-卷积网络)
  - [5. 长短时记忆网络](#5-长短时记忆网络)
  

# 基本模型

这一章介绍几种使用 LibTorch 和 C++ 实现的基础模型，其实大部分的操作和 PyTorch 相比来说都是很相似的。

> torch::nn::Module 的第一句注释：The design and implementation of this class is largely based on the Python API.


## 1. 数据准备
生成两个数据集，每个数据集各20个样本。第一个是一个带有噪声的线性分布数据集，然后分别应用[线性回归](#2-线性回归)和[多层感知机](#3-多层感知机)拟合这一条曲线，其实后者添加了非线性激活，更适合拟合非线性曲线。

``` cpp
torch::Tensor w = torch::tensor({{1.0, 2.0}});
torch::Tensor x = torch::rand({20, 2});
torch::Tensor b = torch::randn({20, 1}) + 3;
torch::Tensor y = x.mm(w.t()) + b;
```

第二个数据集是图像数据集，在[卷积网络](#4-卷积网络)中应用，这里数据仿照的是 [MNIST](http://yann.lecun.com/exdb/mnist/) 将数据维度设定为 [1, 28, 28] 。

```cpp
torch::Tensor img0 = torch::randn({10, 1, 28, 28}) * 100 + 100;
torch::Tensor label0 = torch::zeros({10}, torch::kLong);
torch::Tensor img1 = torch::randn({10, 1, 28, 28}) * 100 + 150;
torch::Tensor label1 = torch::ones({10}, torch::kLong);
torch::Tensor img = torch::cat({img0, img1});
torch::Tensor label = torch::cat({label0, label1});
```

> 注意1：保证数据类型正确。


## 2. 线性回归
高中知识，最小二乘法可以求精确解，这里采用梯度下降法拟合。如果采用 Sigmoid 激活，并将激活更换为负对数损失函数，就变成了逻辑回归模型。由于模型很简单，这里可以直接应用 `torch::nn::Linear`，下面是采用 `MSE` 损失函数和 `SGD` 优化器。

```cpp
torch::nn::Linear lin(2, 1);
torch::optim::SGD sgd(lin->parameters(), 0.1);
for (int i = 0; i < 10; i++) {
  torch::Tensor y_ = lin(x);
  torch::Tensor loss = torch::mse_loss(y_, y);
  sgd.zero_grad();
  loss.backward();
  sgd.step();
  std::cout << "Epoch " << i << " loss=" << loss.item() << std::endl;
}
```

在默认情况下 `torch::nn::Linear` 附带 `bias`，如果不需要可以用对应的 `(ModuleName)Options` 类，在卷积网络中也有同样的设置。

```cpp
torch::nn::Linear lin_no_bias(torch::nn::LinearOptions(2,1).bias(false));
```

> 注意2： `torch::nn::Linear` 会声明**一个经由封装过的指针，而不是对象**，所以语法上调用 `forward()` 或者 `parameters()` 函数的时候用 `->` 而不是 `.` ，在 [include](./include/) 目录中也实现了对应的 LR 类，可以应用 `.` 调用相关方法。

> 注意3：在 LibTorch 中可以像 PyTorch 语法那样直接通过形如 `lin(x)` 方式前向传播（仅限 LibTorch 原生定义的模块）。PyTorch 中是因为在 `__call__()` 函数中调用了 `forward()` ,LibTorch 是因为重写了运算符 `()` 相关内容在 TORCH_MODULE_IMPL 和 ModuleHolder 中。


## 3. 多层感知机

多层感知机 MLP 的结构设计如下：

```
(lin1): Linear(2, 4)
(relu): ReLU()
(lin2): Linear(4, 4)
(relu): ReLU()
(lin3): Linear(4, 1)
```

在 [`include`](./include/) 目录下创建两个文件 `MLP.h` 和 `MLP.cpp`，创建 `MLP` 类并实现构建函数和 `forward()`（其实套路和 PyTorch 也差不多）。

```cpp
// MLP.h
class MLP : public torch::nn::Module {
public:
  MLP(int in_dim, int hidden_dim,int out_dim);
  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Linear lin1{nullptr};
  torch::nn::Linear lin2{nullptr};
  torch::nn::Linear lin3{nullptr};
};
```

头文件中声明类名，继承自 `torch::nn::Module`，并在类内的 `public` 关键字下声明构建函数和 `forward()`，`private` 关键字下面声明三个空指针。

```cpp
// MLP.cpp
MLP::MLP(int in_dim, int hidden_dim, int out_dim) {
  lin1 = torch::nn::Linear(in_dim, hidden_dim);
  lin2 = torch::nn::Linear(hidden_dim, hidden_dim);
  lin3 = torch::nn::Linear(hidden_dim, out_dim);
  
  register_module("lin1", lin1);
  register_module("lin2", lin2);
  register_module("lin3", lin3);
};

torch::Tensor MLP::forward(torch::Tensor x) {
  x = lin1(x);
  x = torch::relu(x);
  x = lin2(x);
  x = torch::relu(x);
  x = lin3(x);
  return x;
}
```

源文件分别实现两个函数。`register_module` 会为创建 键-值 对以供可以嵌套调用。PyTorch 中不需要显示的声明是因为 python 的类天然支持通过 键-值 的方式查找内部对象。

此外，为了能在目录外部调用子文件夹的内容，工程上需要在 include 目录下创建一个 `CMakeLists.txt`。将需要的源文件添加到 `libchap4` 库中。

```cmake
# chap4/include/CMakeLists.txt
add_library(libchap4 MLP.cpp CNN.cpp)
target_link_libraries(libchap4 ${TORCH_LIBRARIES})
```

主目录下面的 `CMakeLists.txt` 也需要做一些修改。
```cmake
# chap4/CMakeLists.txt
cmake_minimum_required(VERSION 3.21)
project(BasicModels)

find_package(Torch REQUIRED)
add_subdirectory(include)
add_executable(BasicModels BasicModels.cpp)
target_link_libraries(BasicModels ${TORCH_LIBRARIES} libchap4)
```

训练过程基本和 [2.](#2-线性回归) 中一致。


## 4. 卷积网络

卷积网络 CNN 的结构设计如下， 主要包含两层 3 $\times$ 3 卷积层和一层线性变换层。每层卷积层之后一次执行归一、池化、激活的操作，由于他们没有科学系参数，只需定义一个即可，也可以用 `torch::nn::functional` 中对应的函数。

```
(conv1): Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
(bn): BatchNorm2d(16)
(max_pool): MaxPool2d(2)
(relu): ReLU()
(conv2): Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
(bn): BatchNorm2d(16)
(max_pool): MaxPool2d(2)
(relu): ReLU()
(lin): Linear(783, 2)
```

同样创建两个文件 `CNN.h` 和 `CNN.cpp` 。

```cpp
// CNN.h
class CNN : public torch::nn::Module {
public:
  CNN(int num_classes);
  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::ReLU relu{nullptr};
  torch::nn::MaxPool2d max_pool{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  torch::nn::Linear lin{nullptr};
};
```

```cpp
// CNN.cpp
CNN::CNN(int num_classes) {
  conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).padding(1));
  conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 16, 3).padding(1));
  bn = torch::nn::BatchNorm2d(16);
  relu = torch::nn::ReLU();
  max_pool = torch::nn::MaxPool2d(2);
  lin = torch::nn::Linear(7 * 7 * 16, num_classes);

  register_module("conv1", conv1);
  register_module("conv2", conv2);
  register_module("bn", bn);
  register_module("relu", relu);
  register_module("max_pool", max_pool);
  register_module("lin", lin);
}

torch::Tensor CNN::forward(torch::Tensor x) {
  x = conv1(x);
  x = bn(x);
  x = relu(x);
  x = max_pool2d(x, 2);

  x = conv2(x);
  x = bn(x);
  x = relu(x);
  x = max_pool2d(x, 2);
  x = lin(x.reshape({x.size(0), -1}));
  
  return x;
}
```

卷积类需要通过 `Conv2dOptions` 设置 `padding` 参数，默认值是 0。 `BatchNorm2d` 的 `stride` 参数默认会和 `kernel_size` 保持一致。

>  注意4：在 LibTorch 和 PyTorch 中，不同损失函数对输入数据要求不同，如 `MSE` 一般可以交换 `target` 和 `input` ，虽然留了自动广播机制，但要求输入维度匹配且都是浮点型。`CrossEntropy` 要求 `target` 是长整型，如果输入 `target` 维度是 1，会自动生成对应的 onehot 张量。


## 5. 长短时记忆网络
（留坑）
