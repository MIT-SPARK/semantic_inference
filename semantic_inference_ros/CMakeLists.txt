cmake_minimum_required(VERSION 3.16)
project(semantic_inference_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

find_package(catkin REQUIRED roscpp nodelet cv_bridge image_transport)
find_package(semantic_inference REQUIRED)

catkin_package(
  CATKIN_DEPENDS cv_bridge image_transport nodelet roscpp
  DEPENDS semantic_inference
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

add_library(
  ${PROJECT_NAME}
  src/mask_nodelet.cpp src/rgbd_segmentation_nodelet.cpp src/segmentation_nodelet.cpp
  src/output_publisher.cpp src/recolor_nodelet.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(
  ${PROJECT_NAME} ${catkin_LIBRARIES} semantic_inference::semantic_inference
)
