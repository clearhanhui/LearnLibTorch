- [LibTorch 配置](#libtorch-配置)
  - [1. 首先安装 PyTorch](#1-首先安装-pytorch)
  - [2. 下载解压 LibTorch](#2-下载解压-libtorch)
  - [3. 配置环境](#3-配置环境)

# LibTorch 配置

## 1. 首先安装 PyTorch

```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. 下载解压 LibTorch

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

> 注意1：本教程适用 Linux 系统，Windows 等我有时间，MacOS 等我有钱。

> 注意2：以上均是 CPU 版本，有 GPU 并配置了 CUDA 的土豪移步[官网](https://pytorch.org/get-started/locally/)根据版本下载。


## 3. 配置环境

三种方式（影响范围由大到小）：
1. 将讲解压目录 `/path/to/libtorch`（注意替换）添加到系统 `PATH` 变量中：

```bash
# 临时使用
export PATH="/path/to/libtorch:$PATH"

# 永久使用
echo "export PATH=\"/path/to/libtorch:\$PATH\"" >> ~/.bashrc && . ~/.bashrc
```

2. 设置环境变量 `Torch_ROOT`，方法参考上面。
3. 在 `CMakeLists.txt` 中通过 `set` 函数临时设置。
4. 执行 `cmake` 命令的时候，设置参数 `-DCMAKE_PREFIX_PATH=/path/to/libtorch`。

推荐使用第2种。
