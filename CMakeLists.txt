cmake_minimum_required(VERSION 3.0.2)
project(semantic_recolor)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

add_compile_options(-Wall -Wextra)

find_package(catkin REQUIRED roscpp nodelet cv_bridge image_transport)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp nodelet cv_bridge image_transport
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

add_library(
  ${PROJECT_NAME}_nodelet
  src/segmentation_nodelet.cpp
)
target_include_directories(
  ${PROJECT_NAME}_nodelet PUBLIC include ${catkin_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_nodelet ${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(demo_segmentation src/demo_segmentation.cpp)
target_link_libraries(demo_segmentation ${PROJECT_NAME})
