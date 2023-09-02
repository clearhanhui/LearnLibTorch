- [自动微分](#自动微分)
  - [1. 普通微分](#1-普通微分)
  - [2. vector-Jacobian product(vjp)](#2-vector-jacobian-productvjp)


# 自动微分


## 1. 普通微分

LibTorch 支持自动微分,  一般情况下可以调用 `backward()` 函数直接计算，和 python 操作基本一致。

$$
\begin{aligned}
& y = wx+b \\
& \frac{\partial y}{\partial w} = x
\end{aligned}
$$

```cpp
torch::Tensor x = torch::tensor({2.0});
torch::Tensor w = torch::tensor({3.0}, torch::requires_grad());
torch::Tensor b = torch::tensor({4.0}, torch::requires_grad());
torch::Tensor y = w * x + b;
y.backward();
std::cout << w.grad() << std::endl;
std::cout << b.grad() << std::endl;
```

> 注意：只有浮点和复数可以获取梯度。

已经计算的张量也是可以通过修改 `required_grad` 属性，但获取梯度需要重新执行计算，接着上面的代码：

```cpp
std::cout << x.requires_grad() << std::endl;
x.requires_grad_(true);
std::cout << x.grad() << std::endl;
y.backward();
std::cout << x.requires_grad() << std::endl;
```

> 注意：libtorch 默认不保留计算图，如果想要得到正确的结果需要第一次计算梯度时候设置相关参数，并且清除已有梯度。


## 2. vector-Jacobian product(vjp)

雅可比矩阵 $J$ 计算的是向量 $y$ 对于向量 $w$ 的导数，这里假设向量 $w=[w_1, w_2, w_3]$ 是当前某个中间层的权重，$y=[y_1, y_2, y_3]$ 由 $w$ 经过某个可导函数产生。反向传播的时候，实际的梯度向量就是本层的导数 $J$ 与上层的梯度向量 $v$ 的乘积。

对于 $y = x^2+2x$，雅可比矩阵为：
$$
J=\left(
\begin{array}{ccc}
\frac{\partial y_{1}}{\partial x_{1}} & \frac{\partial y_{1}}{\partial x_{2}} & \frac{\partial y_{1}}{\partial x_{3}} \\
\frac{\partial y_{2}}{\partial x_{1}} & \frac{\partial y_{2}}{\partial x_{2}} & \frac{\partial y_{2}}{\partial x_{3}} \\
\frac{\partial y_{3}}{\partial x_{1}} & \frac{\partial y_{3}}{\partial x_{2}} & \frac{\partial y_{3}}{\partial x_{3}}
\end{array}
\right) =
\left(
\begin{array}{ccc}
2 x_{1}+2 & 0 & 0 \\
0 & 2 x_{2}+2 & 0 \\
0 & 0 & 2 x_{3}+2
\end{array}\right)
$$

如果向量是$v=[1, 2, 3]$，那么实际的梯度值为：

$$
vJ=[2 x_{1}+2, 4 x_{2}+4, 6 x_{3}+6]
$$

这个向量其实就可以理解为通过**链式法则**传递的上一层的梯度值，也可以理解为在投影方向，实现的时候传入 `backward()` 函数中即可。在 python 也有同样的特性，但一般各种运算算子一般都是封装好的，所以很少用到。查看下面的代码：

```cpp
torch::Tensor xx = torch::randn({3}, torch::requires_grad());
torch::Tensor yy = xx * xx + 2 * xx;
yy.backward(torch::tensor({1, 2, 3}));
```
