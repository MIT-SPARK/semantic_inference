cmake_minimum_required(VERSION 3.0.2)
project(semantic_recolor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

find_package(catkin REQUIRED roscpp nodelet cv_bridge image_transport)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp nodelet cv_bridge image_transport
  DEPENDS OpenCV CUDA
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME} nvinfer nvonnxparser nvinfer_plugin
)

add_library(
  ${PROJECT_NAME}
  src/image_utilities.cpp
  src/model_config.cpp
  src/rgbd_segmenter.cpp
  src/ros_utilities.cpp
  src/segmenter.cpp
  src/semantic_color_config.cpp
  src/trt_utilities.cpp
  src/yaml_utilities.cpp
)
target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  nvinfer
  nvonnxparser
  nvinfer_plugin
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  yaml-cpp
)
target_include_directories(
  ${PROJECT_NAME} PUBLIC include ${catkin_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}
                         ${OpenCV_INCLUDE_DIRS}
)

add_library(
  ${PROJECT_NAME}_nodelet
  src/nodelet_output_publisher.cpp src/rgbd_segmentation_nodelet.cpp
  src/segmentation_nodelet.cpp
)
target_include_directories(
  ${PROJECT_NAME}_nodelet PUBLIC include ${catkin_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_nodelet ${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(demo_segmentation src/demo_segmentation.cpp)
target_link_libraries(demo_segmentation ${PROJECT_NAME})

add_executable(demo_rgbd_segmentation src/demo_rgbd_segmentation.cpp)
target_link_libraries(demo_rgbd_segmentation ${PROJECT_NAME})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(
    utest_${PROJECT_NAME} tests/utest_main.cpp tests/utest_config.cpp
                          tests/utest_image_utilities.cpp
  )
  target_link_libraries(utest_${PROJECT_NAME} ${PROJECT_NAME})
endif()
