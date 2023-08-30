- [LibTorch 配置](#libtorch-配置)
  - [1. 必要软件](#1-必要软件)
  - [2. 安装 PyTorch](#2-安装-pytorch)
  - [3. 下载解压 LibTorch](#3-下载解压-libtorch)
  - [4. 配置环境](#4-配置环境)
  - [5. 运行 demo](#5-运行-demo)

# LibTorch 配置

## 1. 必要软件

* wget
* unzip
* python3
* pip3
* cmake
* gcc (or clang)
* make (or ninja)

软件版本尽量用新的。
根据 (issue)[https://github.com/clearhanhui/LearnLibTorch/issues/1] 的提醒，可能某些 LibTorch 版本使用了C++17语法，编译的时候加上 flag `-std=c++17`。


## 2. 安装 PyTorch

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
推荐使用 [miniconda](https://docs.conda.io/en/latest/miniconda.html) 管理 python 环境。


## 3. 下载解压 LibTorch

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

> 注意1：本教程适用 Linux 系统，Windows 等我有时间，MacOS 等我有钱。

> 注意2：以上均是 CPU 版本，有 GPU 并配置了 CUDA 的土豪移步[官网](https://pytorch.org/get-started/locally/)根据版本下载。


## 4. 配置环境

四种方式（影响范围由大到小）：
1. 将解压目录 `/path/to/libtorch`（注意替换）添加到系统 `PATH` 变量中：

```bash
# 临时使用
export PATH=/path/to/libtorch:$PATH

# 永久使用
echo "export PATH=/path/to/libtorch:\$PATH" >> ~/.bashrc && . ~/.bashrc
```

2. 设置环境变量 `Torch_ROOT`，方法参考上面。
3. 在 `CMakeLists.txt` 中通过 `set` 函数临时设置。
4. 执行 `cmake` 命令的时候，设置参数 `-DCMAKE_PREFIX_PATH=/path/to/libtorch`。

推荐使用第2种。


## 5. 运行 demo

下载项目
```bash 
git clone https://github.com/clearhanhui/LearnLibTorch.git
cd LearnLibTorch/chap1/
```

使用 make 构建
```bash
mkdir build-make && cd build-make
cmake .. 
make
./HelloWorld
```

使用 ninja 构建
```bash
mkdir build-ninja && cd build-ninja
cmake .. -G Ninja
ninja -v
./HelloWorld
```

输出：
```
 0  0  0
 0  0  0
[ CPUFloatType{2,3} ]

Welcome to LibTorch!
```