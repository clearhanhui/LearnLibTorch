- [HelloWorld](#helloworld)
  - [1. 创建 CMake 项目](#1-创建-cmake-项目)
  - [2. 编写 HelloWorld.cpp](#2-编写-helloworldcpp)
  - [3. 编译运行](#3-编译运行)

# HelloWorld

## 1. 创建 CMake 项目

`CMakelists.txt` 文件如下：

```cmake 
cmake_minimum_required(VERSION 3.21)
project(HelloWorld)

find_package(Torch REQUIRED)
add_executable(HelloWorld HelloWorld.cpp)
target_link_libraries(HelloWorld $TORCH_LIBRARIES)
```
逐行解释一下  
`cmake_minimum_required` 设置了最小的 cmake 版本。  
`project` 声明了项目名称，会在全局增加一个 `$PROJECT_NAME` 的变量。  
`find_package` 用于查找 LibTorch，并且会根据 LibTorch 中的内容生成对应的变量如 `$TORCH_LIBRARIES`。  
`add_executable` 声明了生成的可执行文件和对应的源文件。  
`target_link_libraries` 链接库。


## 2. 编写 HelloWorld.cpp

创建并打印一个全零的张量，代码如下：
```cpp
#include <iostream>
#include <torch/torch.h>

int main() {
    // 创建一个(2,3)张量
    torch::Tensor tensor = torch::zeros({2, 3});
    std::cout << tensor << std::endl;
    std::cout << "Welcome to LibTorch" << std::endl;
    
    return 0;
}
```


## 3. 编译运行

依次执行下列命令

```bash
mkdir build && cd build
cmake .. 
make
./HelloWorld
```

输出如下：

```
 0  0  0
 0  0  0
[ CPUFloatType{2,3} ]

Welcome to LibTorch!
```

如果可以正确得到以上输出就说明基本环境没问题了。

> 大坑：执行 `cmake ..` 的时候一直报错找不到 `TorchConfig.cmake`，但可以通过 `-DCMAKE_PREFIX_PATH=/path/to/libtorch` 编译通过，说明 LibTorch 没问题。我是通过 `apt` 命令安装的 3.18，`find_package()` [官方文档](https://cmake.org/cmake/help/latest/command/find_package.html)上面说 3.12 之后的版本可以通过 `<PackageName>_ROOT` 变量查找，按理说版本应该没问题，百思不得其解，最后无奈从官网手动下载了 3.23 竟然可以了，实测 3.21 也可以。