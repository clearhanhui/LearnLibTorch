cmake_minimum_required(VERSION 3.21)
project(BasicModels)

find_package(Torch REQUIRED)
add_executable(BasicModels BasicModels.cpp)
target_link_libraries(BasicModels "${TORCH_LIBRARIES}")
