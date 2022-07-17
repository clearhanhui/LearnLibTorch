cmake_minimum_required(VERSION 3.21)
project(TorchScript)

find_package(Torch REQUIRED)
add_executable(TorchScript TorchScript.cpp)
target_link_libraries(TorchScript ${TORCH_LIBRARIES})

