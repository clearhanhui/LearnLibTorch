- [张量基础](#张量基础)
  - [1. 创建张量](#1-创建张量)
  - [2. 张量索引和切片](#2-张量索引和切片)
  - [3. 张量属性](#3-张量属性)
  - [4. 张量变换](#4-张量变换)
  - [5. 张量计算](#5-张量计算)
  - [6. CUDA](#6-cuda)

# 张量基础

`include` 目录中包含了各种头文件，每个头文件对一类操作。  
`python` 目录中有对应的 python 脚本。 

## 1. 创建张量

和 PyTorch 相比不能说一模一样，只能说极为相似。

```cpp
torch::Tensor a = torch::zeros({2, 3});
torch::Tensor b = torch::ones({2, 3});
torch::Tensor c = torch::eye(3);
torch::Tensor d = torch::full_like(a, 10);
torch::Tensor e = torch::randn({2, 3});
torch::Tensor f = torch::arange(10);
torch::Tensor g = torch::tensor({{1, 2}, {3, 4}});
```


## 2. 张量索引和切片
张量支持运算符 `[]` 但是不能如 `[i,j,k]` 这样使用。如果想要更加灵活的切片需要使用 `torch::indexing` 空间中的相关类如 `Slice`。`Slice(None)` 类似 Numpy 中的 `:` 或 `...` 操作符。

```cpp
torch::Tensor a = torch::randn({2, 3, 4});
torch::Tensor b = a[1];
torch::Tensor c = a.index({1, 2, 3});        // 等同 a[1][2][3]
torch::Tensor d = a.index({Slice(None), 2}); // 等同 a.index({"...", 2})
torch::Tensor e = a.index({Slice(None), Slice(None), Slice(None, None, 2)});
torch::Tensor f = a.index_select(-1, torch::tensor({1, 1, 0}));
```

## 3. 张量属性
除了 `shape` 属性变味了 `sizes()` 函数，item其余常用的基本上和 PyTorch 一致。
```cpp
torch::Tensor a = torch::randn({2, 3});
std::cout << a.size(1) << std::endl;
std::cout << a.sizes() << std::endl;
std::cout << a[0].sizes() << std::endl;
std::cout << a[0][0].item<float>() << std::endl;
std::cout << a.data() << std::endl;
std::cout << a.dtype() << std::endl;
std::cout << a.device() << std::endl;
```

## 4. 张量变换

不支持形如 Tensor.T 的操作，另外 `transpose()` 函数的用法也与 python 稍有区别。
```cpp
torch::Tensor a = torch::randn({2, 3});
torch::Tensor b = a.transpose(0, 1);
torch::Tensor c = a.reshape({-1, 1});
torch::Tensor d = a.view({3, 2});
torch::Tensor e = a.toType(torch::kFloat32);
```

## 5. 张量计算

基本与 PyTorch 一致。
```cpp
torch::Tensor a = torch::ones({3, 3});
torch::Tensor b = torch::randn({3, 3});
torch::Tensor c = a.matmul(b);
torch::Tensor d = a.mul(b);
torch::Tensor e = torch::cat({a, b}, 0);
torch::Tensor f = torch::stack({a, b});
```

## 6. CUDA
留坑。

LibTorch 大部分 api 和 PyTorch 都基本一致，稍加熟悉就可以了。