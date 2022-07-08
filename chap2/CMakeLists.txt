project(TensorBasics)

add_executable(TensorBasics TensorBasics.cpp)
target_link_libraries(TensorBasics "${TORCH_LIBRARIES}")
