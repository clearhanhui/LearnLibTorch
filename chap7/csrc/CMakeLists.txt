cmake_minimum_required(VERSION 3.21)
project(lltm_cpp)



## c++
# find_package(Python3 COMPONENTS Development)
# find_package(Torch REQUIRED)
# add_library(lltm_cpp SHARED lltm.cpp)
# target_compile_features(lltm_cpp PRIVATE cxx_std_14)
# target_link_libraries(lltm_cpp ${TORCH_LIBRARIES} Python3::Python)


## python
## https://github.com/pytorch/pytorch/issues/38122
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(${TORCH_PYTHON_LIBRARY})
pybind11_add_module(lltm_cpp MODULE lltm.cpp)
target_link_libraries(lltm_cpp
    PRIVATE ${TORCH_LIBRARIES}
    PRIVATE ${TORCH_PYTHON_LIBRARY}
)
