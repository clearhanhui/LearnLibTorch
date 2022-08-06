- [PyTorch拓展](#pytorch拓展)
  

# PyTorch拓展
这一章节介绍用 LibTorch 拓展 PyTorch 的算子或者模块，提升性能。

首先需要写一个 `gc_layer.cpp`，里面包含了用 LibTorch 写的前向传播和反向传播函数。理论上计算过程是不需要必须用 LibTorch 实现的，但是需要返回对应的输入和输出。代码如下：
```cpp
torch::Tensor gc_forward(torch::Tensor a, 
                         torch::Tensor x, 
                         torch::Tensor w,
                         torch::Tensor b) {
  return a.mm(x).mm(w) + b;
}

torch::Tensor gc_backward(torch::Tensor a, 
                          torch::Tensor x, 
                          torch::Tensor g) {
  return a.mm(x).t().mm(g);
}
```

然后需要使用 PyBind11 的一个包，需要提前安装，安装过程可以参考[这个文档](https://pybind11.readthedocs.io/en/stable/installing.html)。在上面的文件下面写下面的几行代码：
```cpp
PYBIND11_MODULE(gc_cpp, m) {
  m.def("forward", &gc_forward, "gc forward");
  m.def("backward", &gc_backward, "gc backward");
}
```
`PYBIND11_MODULE` 的第一个参数 `gc_cpp` 是我们后面需要 `import` 的包名，`m` 是一个 `library` 对象，`def` 的三个参数分别是暴露给 python 的函数名，函数的指针，和描述。关于宏 `PYBIND11_MODULE` 的详细文档可以看[这里](https://pybind11.readthedocs.io/en/stable/reference.html#c.PYBIND11_MODULE)。
> 提一句，这个宏其实帮我们做了很多事情，如果不使用 PyBind11 的话，可以参考这两个文档：[链接1](https://docs.python.org/3/extending/extending.html)，[链接2](https://docs.microsoft.com/zh-cn/visualstudio/python/working-with-c-cpp-python-in-visual-studio?view=vs-2022#use-cpython-extensions)。当然这个除了封装函数，像类、属性、等都可以封装，这其实利用了就是 python 一切皆对象的特性。

然后在同一个目录下创建 `setup.py` ，最简单的只需要这几行代码（对我提供的代码做了简化）：
```python
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
setup(
    name='gc_cpp',
    ext_modules=[CppExtension('gc_cpp', ['gc_layer.cpp'])],
    cmdclass={'build_ext': BuildExtension}
)
```
这里需要特别注意，CppExtension 中的 `gc_cpp` 必须要和 `PYBIND11_MODULE` 中保持一致，而 `setup` 函数的第一个名字并不需要，这是安装包的名字。如果我们不想维护两个名字可以在 c++ 的代码中使用 `TORCH_EXTENSION_NAME` 宏代替 `gc_cpp`，前提是使用 `BuildExtension`，他会帮我们在 c++ 的世界里创建一个系统变量。此外 `setup` 还有很多其他的参数，例如可以指定作者和版本信息等。`CppExtension` 是 PyTorch 中提供的继承了 `setuptools.Extension` 的一个类，为我们节省了一些工作。

然后执行下面的命令就可以在当前 python 环境中安装我们的 `gc_cpp` 包了。
```bash
python3 setup.py install 
```
可以通过下面的命令测试安装是否成功
```bash
python3 -c "import torch; import gc_cpp"
```
> 注意：上述过程虽然编写代码引用到了 LibTorch，但是安装和运行并不依赖 LibTorch。

简单做了一个测试：
* PyTorch: 27.52 s
* Py_ext:  88.47 s
* Cpp_ext: 78.84 s

我们的 c++ 拓展虽然比用 python 写的拓展速度快一些，但是和直接使用 PyTorch 的相比还是查了很多，我猜测可能 PyTorch 本身优化就很好，使用 c++ 写的拓展反而增加了加载动态链接库和数据从 python 转换到 c++ 的开销。

我还提供了一个 `CMakeLists.txt` 文件，可以使用 `cmake` 构建出和 `setup.py` 同样的动态链接库，并且也可以创建为 c++ 使用的库。

关于类和注册算子的拓展，总体上逻辑差不多，可以去查看 PyTorch 官网文档（[注册算子](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)，[注册类](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html)），和上面的区别在于注册的方式，我们可以直接在 PyTorch 中使用，而不需要额外导入一个包。