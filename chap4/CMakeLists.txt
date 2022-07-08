project(BasicModels)

add_executable(BasicModels BasicModels.cpp)
target_link_libraries(BasicModels "${TORCH_LIBRARIES}")
