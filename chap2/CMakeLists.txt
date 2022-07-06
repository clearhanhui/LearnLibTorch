cmake_minimum_required(VERSION 3.21)
project(TensorBasics)

set(Torch_ROOT /home/clear/libtorch)
find_package(Torch REQUIRED)

add_executable(TensorBasics TensorBasics.cpp)
target_link_libraries(TensorBasics "${TORCH_LIBRARIES}")
