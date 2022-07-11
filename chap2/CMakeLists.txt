cmake_minimum_required(VERSION 3.21)
project(TensorBasics)

find_package(Torch REQUIRED)
add_executable(TensorBasics TensorBasics.cpp)
target_link_libraries(TensorBasics "${TORCH_LIBRARIES}")
