cmake_minimum_required(VERSION 3.21)
project(HelloWorld)

find_package(Torch REQUIRED)
add_executable(HelloWorld HelloWorld.cpp)
target_link_libraries(HelloWorld ${TORCH_LIBRARIES})
