project(HelloWorld)

add_executable(${PROJECT_NAME} HelloWorld.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
