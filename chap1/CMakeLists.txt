cmake_minimum_required(VERSION 3.21)
project(HelloWorld)

find_package(Torch REQUIRED)
add_executable(${PROJECT_NAME} HelloWorld.cpp)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
