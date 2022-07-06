- [LibTorch Hello World](#libtorch-hello-world)
  - [1. 创建 CMake 项目](#1-创建-cmake-项目)
  - [2. 编写 HelloWorld.cpp](#2-编写-helloworldcpp)
  - [3. 编译运行](#3-编译运行)

# LibTorch Hello World

## 1. 创建 CMake 项目

`CMakelists.txt` 文件如下：
```cmake 
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(HelloWorld)

set(Torch_ROOT /path/to/your/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(HelloWorld HelloWorld.cpp)
target_link_libraries(HelloWorld "${TORCH_LIBRARIES}")
set_property(TARGET HelloWorld PROPERTY CXX_STANDARD 14)
```
注意修改第三行 `path/to/your/libtorch` 。


## 2. 编写 HelloWorld.cpp

代码如下：
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

> 大坑：执行 `cmake ..` 的时候一直报错找不到 `TorchConfig.cmake` ，`find_package()` [官方文档](https://cmake.org/cmake/help/latest/command/find_package.html)是在说明 3.12 之后的版本可以通过 `<PackageName>_ROOT` 变量查找，我是通过 `apt install cmake` 安装的 3.18，按理说版本应该没问题，也可以通过 `-DCMAKE_PREFIX_PATH=/path/to/libtorch` 编译通过，说明 `libtorch` 没问题百思不得其解。最后无奈从官网手动下载了 3.23 竟然可以了。