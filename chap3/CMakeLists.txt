cmake_minimum_required(VERSION 3.21)
project(AutoGrad)

find_package(Torch REQUIRED)
add_executable(AutoGrad AutoGrad.cpp)
target_link_libraries(AutoGrad "${TORCH_LIBRARIES}")
