cmake_minimum_required(VERSION 3.15)
project(HelloWorld)

set(Torch_ROOT /home/clear/libtorch)
find_package(Torch REQUIRED)



add_executable(${PROJECT_NAME} HelloWorld.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
# set_property(TARGET HelloWorld PROPERTY CXX_STANDARD 14)