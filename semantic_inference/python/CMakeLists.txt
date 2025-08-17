cmake_minimum_required(VERSION 3.14)
project(semantic_inference_python)

find_package(semantic_inference REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wno-maybe-uninitialized)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_semantic_inference_bindings bindings.cpp)
target_link_libraries(_semantic_inference_bindings PRIVATE semantic_inference::semantic_inference)
target_include_directories(
  _semantic_inference_bindings PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bindings/include
)

install(TARGETS _semantic_inference_bindings DESTINATION semantic_inference)
