- [LibTorch 教程](#libtorch-教程)
  - [简介](#简介)
  - [目录](#目录)
  - [软件环境](#软件环境)


#  LibTorch 教程 

## 简介

LibTorch 是什么呢，简单来讲可以认为它提供了一种 c++ 前端（同理 PyTorch 是一种 python 前端）。在其[设计哲学](https://pytorch.org/cppdocs/frontend.html#philosophy)中提到

> PyTorch’s C++ frontend was designed with the idea that the Python frontend is great, and should be used when possible; but in some settings, performance and portability requirements make the use of the Python interpreter infeasible. 

简单来说它可以提供更高的效率，此外由于在部署和拓展上面也可以和 PyTorch 很好的结合。然而网上关于 LibTorch 的中文教程和资料太少了，最近也要用到就好好学习总结下。

本教程每个目录是独立一个 [CMake](https://cmake.org/) 项目，每个项目主要参考内容是 [LibTorch 官方文档](https://pytorch.org/cppdocs/)和网上的一些中英文博客资料。


## 目录
* [0--LibTorch 配置](./chap0/)
* [1--LibTorch HelloWorld](./chap1/)
* [2--张量基础](./chap2/)
* [3--自动微分（1）](./chap3/)
* [4--基本模型](./chap4/)
* [5--模型实践](./chap5/)
* [6--TorchScript](./chap6/)
* [7--拓展](./chap7/)
* [8--自动微分（2）](./chap8/)
  

## 软件环境

必要的软件和详细的环境配置可以参考[这里](./chap0/)。


----------------
----------------

如有 bug 欢迎 [issue](https://github.com/clearhanhui/LearnLibTorch/issues)，喜欢的话给个免费的 star 。

对在 CV 的应用 LibTorch 感兴趣可以去看 AllentDan 大佬的 LibTorch 系列，我也从中学习到很多。
* [https://github.com/AllentDan/LibtorchTutorials](https://github.com/AllentDan/LibtorchTutorials)
* [https://github.com/AllentDan/LibtorchDetection](https://github.com/AllentDan/LibtorchDetection)
* [https://github.com/AllentDan/LibtorchSegmentation](https://github.com/AllentDan/LibtorchSegmentation)

