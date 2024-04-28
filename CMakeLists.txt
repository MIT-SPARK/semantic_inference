cmake_minimum_required(VERSION 3.0.2)
project(semantic_recolor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

find_package(config_utilities REQUIRED)
find_package(catkin REQUIRED roscpp nodelet cv_bridge image_transport)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp nodelet cv_bridge image_transport
  DEPENDS OpenCV config_utilities
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

add_library(
  ${PROJECT_NAME}
  src/image_recolor.cpp
  src/image_utilities.cpp
  src/model.cpp
  src/model_config.cpp
  src/segmenter.cpp
  src/trt_utilities.cpp
  src/output_publisher.cpp
)
target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC config_utilities::config_utilities ${catkin_LIBRARIES} ${OpenCV_LIBRARIES}
  PRIVATE nvinfer nvonnxparser nvinfer_plugin ${CUDA_LIBRARIES}
)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> ${CUDA_INCLUDE_DIRS}
)

add_library(
  ${PROJECT_NAME}_nodelet
  src/nodelets/mask_nodelet.cpp src/nodelets/rgbd_segmentation_nodelet.cpp
  src/nodelets/segmentation_nodelet.cpp src/nodelets/recolor_nodelet.cpp
)
target_include_directories(
  ${PROJECT_NAME}_nodelet PUBLIC include ${catkin_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_nodelet ${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(demo_segmentation app/demo_segmentation.cpp)
target_link_libraries(demo_segmentation ${PROJECT_NAME})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(
    utest_${PROJECT_NAME} tests/utest_main.cpp tests/utest_config.cpp
                          tests/utest_image_utilities.cpp
  )
  target_link_libraries(utest_${PROJECT_NAME} ${PROJECT_NAME})
endif()
