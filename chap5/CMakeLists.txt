cmake_minimum_required(VERSION 3.21)
project(PracticeModels)

find_package(Torch REQUIRED)
add_subdirectory(include)
add_executable(PracticeModels PracticeModels.cpp)
target_link_libraries(PracticeModels ${TORCH_LIBRARIES} libchap5)
