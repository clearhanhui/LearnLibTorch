
# TorchScript

TorchScript 允许对 Python 中定义的 PyTorch 模型进行序列化，然后在 C++ 中加载和运行，通过 `torch.jit.script`或 `torch.jit.trace` 其执行来捕获模型代码。

> The C++ interface to TorchScript encompasses three primary pieces of functionality:
> * A mechanism for loading and executing serialized TorchScript models defined in Python;
> * An API for defining custom operators that extend the TorchScript standard library of operations;
> * Just-in-time compilation of TorchScript programs from C++.



------------

内容主要来自于这篇官方文档，内容稍有修改，感兴趣可以阅读原文档。

[LOADING A TORCHSCRIPT MODEL IN C++](https://pytorch.org/tutorials/advanced/cpp_export.html)