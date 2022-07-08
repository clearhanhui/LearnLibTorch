project(AutoGrad)

add_executable(AutoGrad AutoGrad.cpp)
target_link_libraries(AutoGrad "${TORCH_LIBRARIES}")
