cmake_minimum_required(VERSION 3.0.2)
project(semantic_recolor_nodelet)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

find_package(catkin REQUIRED roscpp)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp
  DEPENDS OpenCV
  INCLUDE_DIRS
  LIBRARIES
)

add_library(
  ${PROJECT_NAME}
  src/segmenter.cpp src/utilities.cpp
)
target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  nvinfer
  nvonnxparser
  nvinfer_plugin
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
target_include_directories(
  ${PROJECT_NAME} PUBLIC include ${catkin_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}
                         ${OpenCV_INCLUDE_DIRS}
)

add_executable(demo_segmenter src/demo_segmenter.cpp)
target_link_libraries(demo_segmenter ${PROJECT_NAME})
