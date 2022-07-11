add_library(libchap4 MLP.cpp CNN.cpp)
target_link_libraries(libchap4 ${TORCH_LIBRARIES})