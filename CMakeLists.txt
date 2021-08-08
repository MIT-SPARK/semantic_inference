cmake_minimum_required(VERSION 3.0.2)
project(semantic_recolor_nodelet)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

# TODO(nathan) compile options
find_package(catkin REQUIRED roscpp)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp
  DEPENDS
  INCLUDE_DIRS include
  LIBRARIES
)

add_executable(${PROJECT_NAME} src/segmenter.cpp)
target_link_libraries(
  ${PROJECT_NAME} ${catkin_LIBRARIES} nvinfer ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES}
)
target_include_directories(
  ${PROJECT_NAME} PRIVATE ${catkin_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}
                          ${OpenCV_INCLUDE_DIRS}
)
