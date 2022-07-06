- [LibTorch Installation](#libtorch-installation)
    - [1. 首先安装PyTorch](#1-首先安装pytorch)
    - [2. 下载解压LibTorch](#2-下载解压libtorch)
    - [3. 配置环境](#3-配置环境)

# LibTorch Installation

### 1. 首先安装PyTorch

```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. 下载解压LibTorch

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

> 注意1：本教程适用 Linux 系统，Windows 等我有时间，MacOS 等我有钱。

> 注意2：以上均是 CPU 版本，有 GPU 并配置了 CUDA 的土豪移步[官网](https://pytorch.org/get-started/locally/)根据版本下载。


### 3. 配置环境

只需要讲解压目录 `/path/to/libtorch`（注意替换）添加到系统 `PATH` 变量中即可。

```bash
# 临时使用
export PATH="/path/to/libtorch:$PATH"

# 永久使用
echo "export PATH=\"/path/to/libtorch:\$PATH\"" >> ~/.bashrc && . ~/.bashrc
```

在 CMake 项目中配置变量也是可以的，详情参考 [Chap1](../Chap1/)（推荐）。

